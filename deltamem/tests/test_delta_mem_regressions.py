from __future__ import annotations

import copy
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from deltamem.chat_templates import (
    apply_chat_template as apply_project_chat_template,
    qwen3_enable_thinking_override,
    smollm3_enable_thinking_override,
)
from deltamem.core.delta import (
    DeltaMemAttention,
    HFDeltaMemConfig,
    attach_delta_mem,
    freeze_non_delta_mem_params,
    get_delta_mem_write_regularization,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
import deltamem.runtime.session as pipeline
import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.runtime.session import DeltaMemChatSession, DeltaMemSessionSnapshot
from deltamem.train.delta_sft import (
    DeltaMemTrainer,
    EpisodeCausalLMCollator,
    build_episode_training_examples,
    parse_layer_indices,
    parse_args,
    prepare_tokenized_dataset,
    tokenize_messages_for_sft,
)
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.smollm3 import SmolLM3Config
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Attention


class FakeTokenizer:
    def __init__(
        self,
        response_text: str = "",
        eos_token_id: int | None = None,
        *,
        name_or_path: str = "fake-tokenizer",
    ) -> None:
        self.response_text = response_text
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
        self.name_or_path = name_or_path
        self.last_apply_kwargs: dict[str, object] | None = None

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.response_text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.last_apply_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": return_tensors,
            "enable_thinking": enable_thinking,
        }
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if enable_thinking is False:
            rendered += "<think></think>"
        if not tokenize:
            return rendered
        token_ids = [ord(ch) for ch in rendered] or [0]
        return torch.tensor([token_ids], dtype=torch.long)


class AlternationStrictTokenizer(FakeTokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chat_template = (
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
            "{% endif %}"
            "<{{ message['role'] }}>{{ message['content'] }}"
            "{% endfor %}"
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.last_apply_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": return_tensors,
            "enable_thinking": enable_thinking,
        }
        roles = [message["role"] for message in messages if message["role"] != "system"]
        if (
            "Conversation roles must alternate user/assistant/user/assistant/..." in self.chat_template
            and any(left == right for left, right in zip(roles, roles[1:]))
        ):
            raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        token_ids = [ord(ch) for ch in rendered] or [0]
        return torch.tensor([token_ids], dtype=torch.long)


class SentenceSpanUnstableTokenizer(FakeTokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._piece_to_id: dict[str, int] = {}
        self._next_piece_id = 1000

    def _piece_id(self, piece: str) -> int:
        token_id = self._piece_to_id.get(piece)
        if token_id is None:
            token_id = self._next_piece_id
            self._piece_to_id[piece] = token_id
            self._next_piece_id += 1
        return token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces: list[int] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                start = index
                while index < len(text) and text[index].isspace():
                    index += 1
                if index < len(text) and text[index].isalnum():
                    word_start = index
                    while index < len(text) and text[index].isalnum():
                        index += 1
                    pieces.append(self._piece_id(text[start:index]))
                else:
                    pieces.append(self._piece_id(text[start:index]))
                continue
            if char.isalnum():
                start = index
                while index < len(text) and text[index].isalnum():
                    index += 1
                pieces.append(self._piece_id(text[start:index]))
                continue
            pieces.append(self._piece_id(char))
            index += 1
        return pieces

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.last_apply_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": return_tensors,
            "enable_thinking": enable_thinking,
        }
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if enable_thinking is False:
            rendered += "<think></think>"
        if not tokenize:
            return rendered
        token_ids: list[int] = []
        for message in messages:
            token_ids.extend([
                self._piece_id(f"<role:{message['role']}>"),
                self._piece_id("\\n"),
            ])
            token_ids.extend(self.encode(message["content"], add_special_tokens=False))
            token_ids.extend([
                self._piece_id("<end>"),
                self._piece_id("\\n"),
            ])
        if add_generation_prompt:
            token_ids.append(self._piece_id("<assistant>"))
        return torch.tensor([token_ids or [0]], dtype=torch.long)


def test_qwen3_chat_template_override_respects_model_type() -> None:
    unified = FakeTokenizer(name_or_path="Qwen/Qwen3-8B")
    instruct = FakeTokenizer(name_or_path="Qwen/Qwen3-4B-Instruct-2507")
    thinking = FakeTokenizer(name_or_path="Qwen/Qwen3-4B-Thinking-2507")

    assert qwen3_enable_thinking_override(unified) is False
    assert qwen3_enable_thinking_override(instruct) is None
    assert qwen3_enable_thinking_override(thinking) is None


def test_qwen3_chat_template_does_not_break_instruct_prompt() -> None:
    tokenizer = FakeTokenizer(name_or_path="Qwen/Qwen3-4B-Instruct-2507")
    rendered = apply_project_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert tokenizer.last_apply_kwargs is not None
    assert tokenizer.last_apply_kwargs["enable_thinking"] is None
    assert "<think></think>" not in rendered


def test_smollm3_chat_template_disables_thinking_by_default() -> None:
    tokenizer = FakeTokenizer(name_or_path="HuggingFaceTB/SmolLM3-3B")
    rendered = apply_project_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert smollm3_enable_thinking_override(tokenizer) is False
    assert tokenizer.last_apply_kwargs is not None
    assert tokenizer.last_apply_kwargs["enable_thinking"] is False
    assert "<think></think>" in rendered


def test_qwen3_chat_template_injects_no_think_marker_for_unified_model() -> None:
    tokenizer = FakeTokenizer(name_or_path="Qwen/Qwen3-8B")
    rendered = apply_project_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert tokenizer.last_apply_kwargs is not None
    assert tokenizer.last_apply_kwargs["enable_thinking"] is False
    assert "<think></think>" in rendered


def test_chat_template_falls_back_for_non_alternating_roles() -> None:
    tokenizer = AlternationStrictTokenizer(name_or_path="example/non-qwen-chat-template")
    rendered = apply_project_chat_template(
        tokenizer,
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert rendered == "<system>sys<user>u1<user>u2<assistant>a1"
    assert "Conversation roles must alternate" in tokenizer.chat_template


def test_chat_template_keeps_default_path_for_alternating_roles() -> None:
    tokenizer = AlternationStrictTokenizer(name_or_path="example/non-qwen-chat-template")
    rendered = apply_project_chat_template(
        tokenizer,
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == "<user>u1<assistant>a1<assistant>"
    assert "Conversation roles must alternate" in tokenizer.chat_template


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        return_dict: bool = True,
    ):
        self.calls += 1
        vocab_size = 8
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size)
        return type(
            "FakeOutputs",
            (),
            {
                "logits": logits,
                "past_key_values": ("pkv", self.calls),
            },
        )()


class StubSession(DeltaMemChatSession):
    def _ingest_full_ids(self, full_ids: torch.Tensor) -> torch.Tensor:
        self.processed_input_ids = full_ids.detach().cpu()
        self.last_ingest_stats = {
            "prev_tokens": 0,
            "full_tokens": int(full_ids.size(1)),
            "prefix_tokens": 0,
            "suffix_tokens": int(full_ids.size(1)),
            "rebuilt": False,
            "elapsed_ms": 0.0,
        }
        return torch.zeros(1, 4)

    def _greedy_generate(self, next_token_logits: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        generated_ids = torch.tensor([[11, 12]], dtype=torch.long)
        generated_cpu = generated_ids.detach().cpu()
        self.processed_input_ids = torch.cat([self.processed_input_ids, generated_cpu], dim=1)
        self.last_decode_stats = {
            "generated_tokens": int(generated_ids.size(1)),
            "elapsed_ms": 0.0,
        }
        return generated_ids

    def state_stats(self) -> dict[str, float]:
        return {"num_modules": 0}


def make_qwen3_attention(
    layer_idx: int = 0,
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    head_dim: int = 4,
) -> Qwen3Attention:
    config = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=max(layer_idx + 1, 1),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3Attention(config, layer_idx)


def make_smollm3_attention(
    layer_idx: int = 0,
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    head_dim: int = 4,
) -> SmolLM3Attention:
    config = SmolLM3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=max(layer_idx + 1, 1),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_dropout=0.0,
        attention_bias=False,
        use_sliding_window=False,
    )
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.no_rope_layers = [1] * config.num_hidden_layers
    config._attn_implementation = "eager"
    return SmolLM3Attention(config, layer_idx)


def make_position_embeddings(
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = torch.ones(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.zeros_like(cos)
    return cos, sin


def make_causal_attention_mask(attention_mask_2d: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask_2d.shape
    mask = torch.full(
        (batch_size, 1, seq_len, seq_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    for batch_idx in range(batch_size):
        valid_prefix = int(attention_mask_2d[batch_idx].sum().item())
        for query_idx in range(seq_len):
            if query_idx < valid_prefix:
                mask[batch_idx, 0, query_idx, : query_idx + 1] = 0.0
            else:
                mask[batch_idx, 0, query_idx, :valid_prefix] = 0.0
    return mask


def make_causal_attention_mask_with_past(
    *,
    batch_size: int,
    query_len: int,
    past_len: int,
) -> torch.Tensor:
    total_k_len = past_len + query_len
    mask = torch.full(
        (batch_size, 1, query_len, total_k_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    for query_idx in range(query_len):
        mask[:, :, query_idx, : past_len + query_idx + 1] = 0.0
    return mask


def make_delta_module(
    *,
    output_init: str = "base_slice",
    rank: int = 2,
    num_state_heads: int = 1,
    rankwise_gates: bool = True,
    slot_read_top_k: int = 0,
    memory_readout_mode: str = "delta",
    synthetic_memory_slots: int = 1,
    delta_heads: tuple[str, ...] | str = ("q", "k", "v", "o"),
    delta_o_rmsnorm: bool = False,
) -> DeltaMemAttention:
    torch.manual_seed(0)
    base = make_qwen3_attention()
    return DeltaMemAttention(
        base,
        HFDeltaMemConfig(
            rank=rank,
            num_state_heads=num_state_heads,
            output_init=output_init,
            rankwise_gates=rankwise_gates,
            slot_read_top_k=slot_read_top_k,
            memory_readout_mode=memory_readout_mode,
            synthetic_memory_slots=synthetic_memory_slots,
            delta_heads=delta_heads,
            delta_o_rmsnorm=delta_o_rmsnorm,
        ),
    )


def naive_delta_forward(
    module: DeltaMemAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    state = torch.zeros(
        hidden_states.size(0),
        module.rank,
        module.rank,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    reads = []
    for token_idx in range(hidden_states.size(1)):
        token = hidden_states[:, token_idx, :]
        q = torch.nn.functional.linear(token, module.memory_q_proj)
        k = torch.nn.functional.linear(token, module.memory_k_proj)
        v = torch.nn.functional.linear(token, module.memory_v_proj)
        if module.normalize_qk:
            q = torch.tanh(q)
            q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
            k = torch.tanh(k)
            k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
        read = torch.bmm(state, q.unsqueeze(-1)).squeeze(-1)
        v_old = torch.bmm(state, k.unsqueeze(-1)).squeeze(-1)
        beta = torch.sigmoid(torch.nn.functional.linear(token, module.beta_proj) + module.beta_bias).view(
            token.size(0), module.gate_dim, 1
        )
        if module.couple_lambda:
            lam = 1.0 - beta
        else:
            lam = torch.sigmoid(
                torch.nn.functional.linear(token, module.lambda_proj) + module.lambda_bias
            ).view(token.size(0), module.gate_dim, 1)
        state = lam * state + beta * torch.einsum("bi,bj->bij", v - v_old, k)
        reads.append(read)

    reads = torch.stack(reads, dim=1)
    packed_delta = torch.nn.functional.linear(
        reads,
        torch.cat(
            [
                module.delta_q_proj,
                module.delta_k_proj,
                module.delta_v_proj,
                module.delta_o_proj,
            ],
            dim=0,
        ),
    ) * module.delta_scaling
    delta_q, delta_k, delta_v, delta_o = torch.split(
        packed_delta,
        [
            module.base.q_proj.out_features,
            module.base.k_proj.out_features,
            module.base.v_proj.out_features,
            module.base.o_proj.out_features,
        ],
        dim=-1,
    )
    delta_o = module._apply_delta_o_rmsnorm(delta_o)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.base.head_dim)
    query_states = module.base.q_proj(hidden_states) + delta_q
    key_states = module.base.k_proj(hidden_states) + delta_k
    value_states = module.base.v_proj(hidden_states) + delta_v
    query_states = module.base.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.base.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        position_embeddings[0],
        position_embeddings[1],
    )
    attn_output, attn_weights = eager_attention_forward(
        module.base,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=module.base.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = module.base.o_proj(attn_output) + delta_o
    return attn_output, attn_weights


class ToyAttentionModel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.self_attn = make_qwen3_attention() if module is None else module
        self.residual = torch.nn.Linear(8, 8)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.self_attn, DeltaMemAttention):
            attn_output, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        else:
            attn_output, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        return self.residual(attn_output)


def test_two_training_steps_succeed_with_state_reset() -> None:
    module = make_delta_module(output_init="base_slice")
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)

    for _ in range(2):
        module.reset_state()
        x = torch.randn(1, 3, 8)
        position_embeddings = make_position_embeddings(
            batch_size=x.size(0),
            seq_len=x.size(1),
            head_dim=module.base.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        loss = module(x, position_embeddings, None)[0].sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@pytest.mark.parametrize("rankwise_gates", [False, True])
def test_sequence_scan_matches_naive_reference(rankwise_gates: bool) -> None:
    module = make_delta_module(output_init="random", rank=2, rankwise_gates=rankwise_gates)
    x = torch.randn(2, 5, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    with torch.inference_mode():
        optimized, optimized_weights = module(x, position_embeddings, None)
    module.reset_state()
    reference, reference_weights = naive_delta_forward(module, x, position_embeddings)
    assert torch.allclose(optimized, reference, atol=1e-5, rtol=1e-5)
    assert torch.allclose(optimized_weights, reference_weights, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton scan test")
@pytest.mark.parametrize("rankwise_gates", [False, True])
def test_triton_sequence_scan_matches_torch_forward_and_backward(rankwise_gates: bool) -> None:
    module = make_delta_module(output_init="random", rank=2, rankwise_gates=rankwise_gates).cuda().to(torch.float32)
    batch_size = 2
    seq_len = 5
    rank = module.rank
    token_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        device="cuda",
        dtype=torch.bool,
    )

    inputs = {
        "state": torch.randn(batch_size, rank, rank, device="cuda", dtype=torch.float32),
        "q": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "k": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "v": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
        "lam": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
    }

    ref_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    keep_ref, erase_ref, write_ref = module._memory_update_coefficients(
        ref_inputs["beta"],
        ref_inputs["lam"],
    )
    state_ref, reads_ref = module._memory_affine_scan_torch(
        ref_inputs["state"],
        ref_inputs["q"],
        ref_inputs["k"],
        ref_inputs["v"],
        keep_ref,
        erase_ref,
        write_ref,
        token_mask=token_mask,
    )
    ref_loss = state_ref.sum() + reads_ref.sum()
    ref_grads = torch.autograd.grad(
        ref_loss,
        tuple(ref_inputs.values()),
        retain_graph=False,
    )

    module.scan_impl = "triton"
    opt_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    state_opt, reads_opt = module._memory_affine_scan(
        opt_inputs["state"],
        opt_inputs["q"],
        opt_inputs["k"],
        opt_inputs["v"],
        opt_inputs["beta"],
        opt_inputs["lam"],
        token_mask=token_mask,
    )
    opt_loss = state_opt.sum() + reads_opt.sum()
    opt_grads = torch.autograd.grad(
        opt_loss,
        tuple(opt_inputs.values()),
        retain_graph=False,
    )

    assert torch.allclose(state_opt, state_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reads_opt, reads_ref, atol=1e-5, rtol=1e-5)
    for grad_ref, grad_opt in zip(ref_grads, opt_grads):
        assert torch.allclose(grad_opt, grad_ref, atol=1e-5, rtol=1e-5)


def test_zero_init_attention_wrapper_matches_base_attention() -> None:
    torch.manual_seed(0)
    base = make_qwen3_attention()
    wrapped_base = make_qwen3_attention()
    wrapped_base.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped = DeltaMemAttention(
        wrapped_base,
        HFDeltaMemConfig(rank=2, output_init="zero", rankwise_gates=True),
    )
    x = torch.randn(2, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )

    base_out, base_weights = base(x, position_embeddings, None)
    wrapped_out, wrapped_weights = wrapped(x, position_embeddings, None)

    assert torch.allclose(base_out, wrapped_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_weights, wrapped_weights, atol=1e-6, rtol=1e-6)


def test_high_rank_scan_supports_large_rank() -> None:
    module = make_delta_module(rank=64)
    module.scan_impl = "auto"
    batch_size = 1
    seq_len = 4
    state = torch.randn(batch_size, 64, 64)
    q = torch.randn(batch_size, seq_len, 64)
    k = torch.randn(batch_size, seq_len, 64)
    v = torch.randn(batch_size, seq_len, 64)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, module.gate_dim, 1))
    lam = torch.sigmoid(torch.randn(batch_size, seq_len, module.gate_dim, 1))

    final_state, reads = module._memory_affine_scan(
        state,
        q,
        k,
        v,
        beta,
        lam,
    )

    assert final_state.shape == (batch_size, 64, 64)
    assert reads.shape == (batch_size, seq_len, 64)
    assert torch.isfinite(final_state).all()
    assert torch.isfinite(reads).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton high-rank scan test")
@pytest.mark.parametrize("rank", [64, 128])
def test_triton_scan_supports_high_rank(rank: int) -> None:
    module = make_delta_module(rank=rank).cuda().to(torch.float32)
    module.scan_impl = "triton"
    batch_size = 1
    seq_len = 4
    token_mask = torch.tensor([[1, 1, 1, 1]], device="cuda", dtype=torch.bool)

    inputs = {
        "state": torch.randn(batch_size, rank, rank, device="cuda", dtype=torch.float32),
        "q": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "k": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "v": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
        "lam": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
    }

    ref_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    keep_ref, erase_ref, write_ref = module._memory_update_coefficients(
        ref_inputs["beta"],
        ref_inputs["lam"],
    )
    state_ref, reads_ref = module._memory_affine_scan_torch(
        ref_inputs["state"],
        ref_inputs["q"],
        ref_inputs["k"],
        ref_inputs["v"],
        keep_ref,
        erase_ref,
        write_ref,
        token_mask=token_mask,
    )
    ref_loss = state_ref.sum() + reads_ref.sum()
    ref_grads = torch.autograd.grad(ref_loss, tuple(ref_inputs.values()), retain_graph=False)

    triton_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    state_opt, reads_opt = module._memory_affine_scan(
        triton_inputs["state"],
        triton_inputs["q"],
        triton_inputs["k"],
        triton_inputs["v"],
        triton_inputs["beta"],
        triton_inputs["lam"],
        token_mask=token_mask,
    )
    opt_loss = state_opt.sum() + reads_opt.sum()
    opt_grads = torch.autograd.grad(opt_loss, tuple(triton_inputs.values()), retain_graph=False)

    assert torch.allclose(state_opt, state_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(reads_opt, reads_ref, atol=1e-4, rtol=1e-4)
    for grad_ref, grad_opt in zip(ref_grads, opt_grads):
        assert torch.allclose(grad_opt, grad_ref, atol=2e-2, rtol=5e-4)


def test_attach_delta_mem_wraps_attention_modules_not_projection_modules() -> None:
    model = ToyAttentionModel()
    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(rank=2, output_init="zero", target_modules=("self_attn",)),
    )

    assert replaced == ["self_attn"]
    assert isinstance(model.self_attn, DeltaMemAttention)
    assert isinstance(model.self_attn.base.q_proj, torch.nn.Linear)


def test_attach_delta_mem_wraps_smollm3_attention_modules() -> None:
    model = ToyAttentionModel(module=make_smollm3_attention())
    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(rank=2, output_init="zero", target_modules=("self_attn",)),
    )

    assert replaced == ["self_attn"]
    assert isinstance(model.self_attn, DeltaMemAttention)
    assert isinstance(model.self_attn.base, SmolLM3Attention)


def test_reset_state_restores_independent_sample_behavior() -> None:
    module = make_delta_module(output_init="random")
    fresh = make_delta_module(output_init="random")
    fresh.load_state_dict(copy.deepcopy(module.state_dict()))

    first_sample = torch.randn(1, 3, 8)
    second_sample = torch.randn(1, 3, 8)
    first_position_embeddings = make_position_embeddings(
        batch_size=first_sample.size(0),
        seq_len=first_sample.size(1),
        head_dim=module.base.head_dim,
        device=first_sample.device,
        dtype=first_sample.dtype,
    )
    second_position_embeddings = make_position_embeddings(
        batch_size=second_sample.size(0),
        seq_len=second_sample.size(1),
        head_dim=module.base.head_dim,
        device=second_sample.device,
        dtype=second_sample.dtype,
    )

    _ = module(first_sample, first_position_embeddings, None)
    module.reset_state()
    output_after_reset = module(second_sample, second_position_embeddings, None)[0]
    fresh_output = fresh(second_sample, second_position_embeddings, None)[0]

    assert torch.allclose(output_after_reset, fresh_output)


def test_padding_mask_prevents_padded_tokens_from_updating_state() -> None:
    module = make_delta_module(output_init="random")
    standalone = make_delta_module(output_init="random")
    standalone.load_state_dict(copy.deepcopy(module.state_dict()))

    short = torch.randn(1, 3, 8)
    long = torch.randn(1, 5, 8)
    padded_short = torch.cat([short, torch.randn(1, 2, 8)], dim=1)
    batch = torch.cat([padded_short, long], dim=0)
    attention_mask = make_causal_attention_mask(torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]))
    standalone_attention_mask = make_causal_attention_mask(torch.tensor([[1, 1, 1]]))
    batch_position_embeddings = make_position_embeddings(
        batch_size=batch.size(0),
        seq_len=batch.size(1),
        head_dim=module.base.head_dim,
        device=batch.device,
        dtype=batch.dtype,
    )
    standalone_position_embeddings = make_position_embeddings(
        batch_size=short.size(0),
        seq_len=short.size(1),
        head_dim=standalone.base.head_dim,
        device=short.device,
        dtype=short.dtype,
    )

    batch_output = module(batch, batch_position_embeddings, attention_mask)[0]
    standalone_output = standalone(
        short,
        standalone_position_embeddings,
        standalone_attention_mask,
    )[0]

    assert torch.allclose(batch_output[0, : short.size(1)], standalone_output[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(module.delta_state[0], standalone.delta_state[0], atol=1e-5, rtol=1e-5)


def test_disable_writes_preserves_delta_state_during_read_phase() -> None:
    module = make_delta_module(output_init="random")
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = module(x, position_embeddings, None)
    state_before = module.delta_state.clone()

    set_delta_mem_write_enabled(module, False)
    _ = module(x, position_embeddings, None)
    set_delta_mem_write_enabled(module, True)

    assert torch.allclose(module.delta_state, state_before)


def test_base_slice_initialization_gives_controller_gradients() -> None:
    module = make_delta_module(output_init="base_slice")
    x = torch.randn(1, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    loss = module(x, position_embeddings, None)[0].sum()
    loss.backward()

    assert module.memory_q_proj.grad is not None and module.memory_q_proj.grad.norm().item() > 0
    assert module.memory_k_proj.grad is not None and module.memory_k_proj.grad.norm().item() > 0
    assert module.memory_v_proj.grad is not None and module.memory_v_proj.grad.norm().item() > 0
    assert module.beta_proj.grad is not None and module.beta_proj.grad.norm().item() > 0


def test_base_slice_zero_state_matches_base_even_at_high_rank() -> None:
    torch.manual_seed(0)
    common_kwargs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
    }
    base = make_qwen3_attention(**common_kwargs)
    wrapped_base_rank8 = make_qwen3_attention(**common_kwargs)
    wrapped_base_rank64 = make_qwen3_attention(**common_kwargs)
    wrapped_base_rank8.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped_base_rank64.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped_rank8 = DeltaMemAttention(
        wrapped_base_rank8,
        HFDeltaMemConfig(rank=8, output_init="base_slice", rankwise_gates=True),
    )
    wrapped_rank64 = DeltaMemAttention(
        wrapped_base_rank64,
        HFDeltaMemConfig(rank=64, output_init="base_slice", rankwise_gates=True),
    )
    x = torch.randn(2, 1, common_kwargs["hidden_size"])
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )

    base_out, _ = base(x, position_embeddings, None)
    rank8_out, _ = wrapped_rank8(x, position_embeddings, None)
    rank64_out, _ = wrapped_rank64(x, position_embeddings, None)

    assert torch.allclose(base_out, rank8_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_out, rank64_out, atol=1e-6, rtol=1e-6)



def test_base_slice_initialization_strength_grows_with_rank() -> None:
    torch.manual_seed(0)
    common_kwargs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
    }
    base_rank8 = make_qwen3_attention(**common_kwargs)
    base_rank64 = make_qwen3_attention(**common_kwargs)
    base_rank64.load_state_dict(copy.deepcopy(base_rank8.state_dict()))
    rank8_module = DeltaMemAttention(
        base_rank8,
        HFDeltaMemConfig(rank=8, output_init="base_slice", rankwise_gates=True),
    )
    rank64_module = DeltaMemAttention(
        base_rank64,
        HFDeltaMemConfig(rank=64, output_init="base_slice", rankwise_gates=True),
    )

    rank8_q_nonzero = int((rank8_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_q_nonzero = int((rank64_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank8_o_nonzero = int((rank8_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_o_nonzero = int((rank64_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())

    assert rank8_q_nonzero == 8
    assert rank64_q_nonzero == 64
    assert rank8_o_nonzero == 8
    assert rank64_o_nonzero == 64
    assert rank64_module.delta_q_proj.float().norm().item() > rank8_module.delta_q_proj.float().norm().item()
    assert rank64_module.delta_o_proj.float().norm().item() > rank8_module.delta_o_proj.float().norm().item()


def test_base_slice_fixed_keeps_initialization_width_constant_across_ranks() -> None:
    torch.manual_seed(0)
    common_kwargs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
    }
    base_rank8 = make_qwen3_attention(**common_kwargs)
    base_rank64 = make_qwen3_attention(**common_kwargs)
    base_rank64.load_state_dict(copy.deepcopy(base_rank8.state_dict()))
    rank8_module = DeltaMemAttention(
        base_rank8,
        HFDeltaMemConfig(
            rank=8,
            output_init="base_slice_fixed",
            base_slice_ref_width=8,
            rankwise_gates=True,
        ),
    )
    rank64_module = DeltaMemAttention(
        base_rank64,
        HFDeltaMemConfig(
            rank=64,
            output_init="base_slice_fixed",
            base_slice_ref_width=8,
            rankwise_gates=True,
        ),
    )

    rank8_q_nonzero = int((rank8_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_q_nonzero = int((rank64_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank8_o_nonzero = int((rank8_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_o_nonzero = int((rank64_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())

    assert rank8_q_nonzero == 8
    assert rank64_q_nonzero == 8
    assert rank8_o_nonzero == 8
    assert rank64_o_nonzero == 8
    assert torch.allclose(
        rank8_module.delta_q_proj[:, :8],
        rank64_module.delta_q_proj[:, :8],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(rank64_module.delta_q_proj[:, 8:], torch.zeros_like(rank64_module.delta_q_proj[:, 8:]))
    assert torch.allclose(rank64_module.delta_o_proj[:, 8:], torch.zeros_like(rank64_module.delta_o_proj[:, 8:]))


def test_delta_o_rmsnorm_adds_trainable_weight() -> None:
    module = make_delta_module(output_init="zero", delta_heads=("o",), delta_o_rmsnorm=True)

    assert module.delta_o_rmsnorm is True
    assert hasattr(module, "delta_o_rmsnorm_weight")
    assert module.delta_o_rmsnorm_weight.shape == (module.base.o_proj.out_features,)
    assert module.is_trainable_parameter("delta_o_rmsnorm_weight") is True


def test_delta_o_rmsnorm_matches_manual_rms_norm() -> None:
    module = make_delta_module(output_init="zero", delta_heads=("o",), delta_o_rmsnorm=True)
    delta_o = torch.randn(2, 3, module.base.o_proj.out_features)

    actual = module._apply_delta_o_rmsnorm(delta_o)
    expected = torch.nn.functional.rms_norm(
        delta_o.float(),
        (delta_o.shape[-1],),
        weight=module.delta_o_rmsnorm_weight.float(),
        eps=module.delta_o_rmsnorm_eps,
    ).to(dtype=delta_o.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_memory_reader_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_reader_layers"):
        HFDeltaMemConfig(
            rank=2,
            output_init="base_slice",
            memory_reader_layers=(1,),
            memory_reader_hidden_size=4,
        )


def test_synthetic_kv_readout_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_readout_mode='delta'"):
        HFDeltaMemConfig(
            rank=8,
            output_init="zero",
            delta_heads=(),
            memory_readout_mode="synthetic_kv",
            synthetic_memory_slots=4,
        )


def test_latent_context_readout_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_readout_mode='delta'"):
        HFDeltaMemConfig(
            rank=8,
            output_init="zero",
            delta_heads=(),
            memory_readout_mode="latent_context",
            latent_memory_layers=(0,),
            latent_memory_slots=4,
            latent_gate_init=0.01,
        )


def test_training_cli_rejects_latent_context_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["delta_sft.py", "--memory-readout-mode", "latent_context"])
    with pytest.raises(SystemExit):
        parse_args()


def test_training_cli_rejects_latent_prefix_margin_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["delta_sft.py", "--memory-readout-mode", "latent_context"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_collect_latent_stats_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_readout_mode='delta'"):
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            delta_heads=(),
            memory_readout_mode="latent_context",
            latent_memory_layers=(0,),
            latent_memory_slots=2,
        )


def test_training_cli_rejects_synthetic_kv_readout_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft.py",
            "--memory-readout-mode",
            "synthetic_kv",
            "--synthetic-memory-slots",
            "4",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_layer_indices_supports_off_and_csv() -> None:
    assert parse_layer_indices("off") == ()
    assert parse_layer_indices(" 24, 25,26 ") == (24, 25, 26)


def test_rankwise_gates_expand_gate_parameters() -> None:
    module = make_delta_module(output_init="base_slice", rank=3, rankwise_gates=True)
    assert module.beta_proj.shape == (3, 8)
    assert module.beta_bias.shape == (3,)


def test_write_sparsity_regularization_is_positive_after_forward() -> None:
    model = ToyAttentionModel(
        DeltaMemAttention(
            make_qwen3_attention(),
            HFDeltaMemConfig(rank=2, output_init="base_slice", rankwise_gates=True),
        )
    )
    x = torch.randn(1, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=model.self_attn.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = model(x, position_embeddings)
    penalty = get_delta_mem_write_regularization(model, target=0.0)
    assert penalty.item() > 0


def test_tokenize_messages_supervises_all_assistant_turns() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    tokenized = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="all_assistant_turns",
    )
    final_only = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="final_assistant_only",
    )

    prefix_user = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    prefix_assistant_1 = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    prefix_user_2 = tokenizer.apply_chat_template(
        messages[:3],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    prefix_assistant_2 = tokenizer.apply_chat_template(
        messages[:4],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    assistant_1_span = len(prefix_assistant_1) - len(prefix_user)
    assistant_2_span = len(prefix_assistant_2) - len(prefix_user_2)

    supervised_all = sum(label != -100 for label in tokenized["labels"])
    supervised_final = sum(label != -100 for label in final_only["labels"])

    assert supervised_all == assistant_1_span + assistant_2_span
    assert supervised_final == assistant_2_span


def test_tokenize_messages_keeps_tail_supervision_when_truncated() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "u" * 40},
        {"role": "assistant", "content": "a" * 10},
        {"role": "user", "content": "v" * 40},
        {"role": "assistant", "content": "b" * 12},
    ]

    full = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=4096,
        assistant_loss_mode="all_assistant_turns",
    )
    truncated = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=80,
        assistant_loss_mode="all_assistant_turns",
    )

    assert truncated["input_ids"] == full["input_ids"][-80:]
    assert truncated["labels"] == full["labels"][-80:]
    assert any(label != -100 for label in truncated["labels"])

    last_assistant_tokens = len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).squeeze(0).tolist()
    ) - len(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).squeeze(0).tolist()
    )
    assert sum(label != -100 for label in truncated["labels"][-last_assistant_tokens:]) > 0


def test_build_episode_training_examples_splits_write_and_visible_context() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    episodes = build_episode_training_examples(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="all_assistant_turns",
        episode_recent_messages=1,
        max_write_length=1024,
        include_sentence_ids=True,
    )

    assert len(episodes) == 2
    assert episodes[0]["write_input_ids"] == []

    expected_write = tokenizer.apply_chat_template(
        messages[:3],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    expected_read = tokenize_messages_for_sft(
        tokenizer,
        [messages[0], messages[3], messages[4]],
        max_length=1024,
        assistant_loss_mode="final_assistant_only",
    )

    assert episodes[1]["write_input_ids"] == expected_write
    assert episodes[1]["input_ids"] == expected_read["input_ids"]
    assert episodes[1]["labels"] == expected_read["labels"]
    assert episodes[1]["episode_target_message_index"] == 4
    assert len(episodes[1]["write_message_ids"]) == len(episodes[1]["write_input_ids"])
    assert len(episodes[1]["write_sentence_ids"]) == len(episodes[1]["write_input_ids"])
    assert set(episodes[1]["write_message_ids"]) == {-1, 0, 1}
    assert set(episodes[1]["write_sentence_ids"]) == {-1, 0, 1}
    assert episodes[1]["state_only_write_input_ids"] == expected_write
    assert episodes[1]["state_only_input_ids"] == expected_read["input_ids"]
    assert len(episodes[1]["state_only_write_message_ids"]) == len(episodes[1]["state_only_write_input_ids"])
    assert len(episodes[1]["state_only_write_sentence_ids"]) == len(episodes[1]["state_only_write_input_ids"])
    assert set(episodes[1]["state_only_write_message_ids"]) == {-1, 0, 1}
    assert set(episodes[1]["state_only_write_sentence_ids"]) == {-1, 0, 1}


def test_experimental_sentence_span_builder_handles_non_prefix_stable_sentence_chunks() -> None:
    tokenizer = SentenceSpanUnstableTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Sentence one. Sentence two."},
        {"role": "assistant", "content": "ok"},
    ]

    episodes = experimental_train.build_episode_training_examples(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="all_assistant_turns",
        episode_recent_messages=0,
        max_write_length=1024,
        include_sentence_ids=True,
    )

    assert len(episodes) == 1
    assert len(episodes[0]["write_sentence_ids"]) == len(episodes[0]["write_input_ids"])
    active_sentence_ids = [value for value in episodes[0]["write_sentence_ids"] if value >= 0]
    assert active_sentence_ids
    assert set(active_sentence_ids) == {0, 1}


def test_runtime_sentence_span_builder_handles_non_prefix_stable_sentence_chunks() -> None:
    tokenizer = SentenceSpanUnstableTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Sentence one. Sentence two."},
    ]

    input_ids, message_ids, sentence_ids = pipeline._tokenize_chat_messages_with_write_span_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
        include_sentence_ids=True,
    )

    assert len(input_ids) == len(message_ids) == len(sentence_ids)
    active_sentence_ids = [value for value in sentence_ids if value >= 0]
    assert active_sentence_ids
    assert set(active_sentence_ids) == {0, 1}


def test_prepare_tokenized_dataset_reuses_saved_cache(tmp_path: Path) -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        ]
    )
    args = Namespace(
        tokenized_cache=True,
        tokenized_dataset_root=tmp_path / "tokenized",
        dataset_name="fake",
        dataset_split="train",
        train_file=None,
        training_mode="episode",
        assistant_loss_mode="all_assistant_turns",
        episode_recent_messages=1,
        max_length=128,
        max_write_length=64,
        group_by_length=True,
        dataset_num_proc=1,
        memory_write_granularity="token",
    )
    tokenizer = FakeTokenizer()

    tokenized_1, cache_hit_1, cache_dir_1 = prepare_tokenized_dataset(
        args,
        dataset,
        tokenizer,
        distributed=False,
        local_rank=-1,
    )
    tokenized_2, cache_hit_2, cache_dir_2 = prepare_tokenized_dataset(
        args,
        dataset,
        tokenizer,
        distributed=False,
        local_rank=-1,
    )

    assert cache_hit_1 is False
    assert cache_hit_2 is True
    assert cache_dir_1 == cache_dir_2
    assert tokenized_1[0]["input_ids"] == tokenized_2[0]["input_ids"]
    assert "write_input_ids" in tokenized_1.column_names


def test_episode_collator_builds_teacher_inputs() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = 0
    collator = EpisodeCausalLMCollator(tokenizer)
    features = [
        {
            "write_input_ids": [1, 2],
            "write_attention_mask": [1, 1],
            "write_message_ids": [-1, 0],
            "write_sentence_ids": [-1, 0],
            "input_ids": [3, 4, 5],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 4, 5],
            "state_only_write_input_ids": [1, 2],
            "state_only_write_attention_mask": [1, 1],
            "state_only_write_message_ids": [-1, 0],
            "state_only_write_sentence_ids": [-1, 0],
            "state_only_input_ids": [3, 4],
            "state_only_attention_mask": [1, 1],
            "state_only_labels": [-100, 4],
        },
        {
            "write_input_ids": [6],
            "write_attention_mask": [1],
            "write_message_ids": [0],
            "write_sentence_ids": [0],
            "input_ids": [7, 8],
            "attention_mask": [1, 1],
            "labels": [-100, 8],
            "state_only_write_input_ids": [6],
            "state_only_write_attention_mask": [1],
            "state_only_write_message_ids": [0],
            "state_only_write_sentence_ids": [0],
            "state_only_input_ids": [7],
            "state_only_attention_mask": [1],
            "state_only_labels": [-100],
        },
    ]

    batch = collator(features)

    assert torch.equal(batch["write_lengths"], torch.tensor([2, 1], dtype=torch.long))
    assert torch.equal(batch["read_lengths"], torch.tensor([3, 2], dtype=torch.long))
    assert batch["write_message_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["write_sentence_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["state_only_write_message_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["state_only_write_sentence_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["full_input_ids"].tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]]
    assert batch["full_attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch["full_labels"].tolist() == [[-100, -100, -100, 4, 5], [-100, -100, 8, -100, -100]]


def test_ssw_writer_forward_runs() -> None:
    module = DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            output_init="base_slice",
            delta_heads=("q", "o"),
            memory_write_granularity="message_mean",
        ),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 2, 2)


def test_sentence_mean_writer_forward_runs() -> None:
    module = DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            output_init="base_slice",
            delta_heads=("q", "o"),
            memory_write_granularity="sentence_mean",
        ),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    set_delta_mem_write_sentence_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 2, 2)


def test_multi_head_state_forward_runs() -> None:
    module = make_delta_module(
        rank=2,
        num_state_heads=4,
        delta_heads=("q", "o"),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 4, 2, 2)
    assert module.delta_q_proj.shape[-1] == 8
    assert module.delta_o_proj.shape[-1] == 8


def test_msw_state_with_ssw_writer_forward_runs() -> None:
    module = DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            num_state_heads=4,
            output_init="base_slice",
            delta_heads=("q", "o"),
            memory_write_granularity="message_mean",
        ),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 4, 2, 2)
    assert module.delta_q_proj.shape[-1] == 8
    assert module.delta_o_proj.shape[-1] == 8


def test_runtime_ingest_sets_write_message_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    write_message_ids_calls = []
    monkeypatch.setattr(
        pipeline,
        "set_delta_mem_write_message_ids",
        lambda model, message_ids: write_message_ids_calls.append(
            None if message_ids is None else message_ids.detach().cpu().clone()
        ),
    )
    session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    session.messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    full_ids = session._tokenize_messages(session.messages, add_generation_prompt=False)
    session._ingest_full_ids(full_ids)

    non_none_calls = [call for call in write_message_ids_calls if call is not None]
    assert len(non_none_calls) == 1
    assert set(non_none_calls[0].squeeze(0).tolist()) == {-1, 0, 1}
    assert write_message_ids_calls[-1] is None


def test_runtime_write_message_ids_reuses_snapshot_cache_for_appended_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_calls = 0
    original_helper = pipeline._tokenize_chat_messages_with_write_span_ids

    def counted_helper(*args, **kwargs):
        nonlocal helper_calls
        helper_calls += 1
        return original_helper(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_tokenize_chat_messages_with_write_span_ids", counted_helper)

    history_session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    history_session.messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    history_ids = history_session._tokenize_messages(history_session.messages, add_generation_prompt=False)
    history_session._ingest_full_ids(history_ids)
    assert helper_calls == 1

    snapshot = DeltaMemSessionSnapshot(
        messages=[dict(message) for message in history_session.messages],
        processed_input_ids=history_ids.squeeze(0).tolist(),
        delta_state={},
        past_key_values=None,
        write_message_ids=list(history_session.cached_write_message_ids or []),
        write_sentence_ids=list(history_session.cached_write_sentence_ids or []),
    )
    followup_session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    followup_session.load_snapshot(snapshot)
    followup_session.messages.append({"role": "user", "content": "u2"})

    prompt_ids = followup_session._tokenize_messages(
        followup_session.messages,
        add_generation_prompt=True,
    )
    prompt_message_ids = followup_session._runtime_write_message_ids(prompt_ids)

    assert prompt_message_ids is not None
    assert helper_calls == 1
    assert prompt_message_ids.shape[1] == prompt_ids.shape[1]
    assert set(prompt_message_ids.squeeze(0).tolist()) == {-1, 0, 1, 2}



def test_runtime_generate_sets_assistant_message_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    write_message_ids_calls = []
    monkeypatch.setattr(
        pipeline,
        "set_delta_mem_write_message_ids",
        lambda model, message_ids: write_message_ids_calls.append(
            None if message_ids is None else message_ids.detach().cpu().clone()
        ),
    )
    session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    session.messages = [{"role": "user", "content": "u1"}]

    session._decode_generate(
        torch.zeros(1, 8),
        max_new_tokens=2,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )

    assistant_calls = [call for call in write_message_ids_calls if call is not None]
    assert len(assistant_calls) == 2
    assert all(torch.equal(call, torch.ones((1, 1), dtype=torch.long)) for call in assistant_calls)
    assert write_message_ids_calls[-1] is None


def test_generate_reply_preserves_raw_assistant_text() -> None:
    session = StubSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(response_text=" hello\n"),
        device="cpu",
    )

    result = session.generate_reply("remember this", max_new_tokens=2)

    assert result["assistant"] == " hello\n"
    assert result["assistant_display"] == "hello"
    assert session.messages[-1]["content"] == " hello\n"


def test_generate_reply_restores_write_enabled_after_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_enabled_calls = []
    monkeypatch.setattr(
        pipeline,
        "set_delta_mem_write_enabled",
        lambda model, enabled: write_enabled_calls.append(enabled),
    )
    model = FakeModel()
    session = StubSession(
        model=model,
        tokenizer=FakeTokenizer(response_text="ok"),
        device="cpu",
    )

    session.generate_reply("probe", max_new_tokens=2, write_enabled=False)

    assert write_enabled_calls == [False, True]


def test_decode_generate_does_not_commit_eos_to_history() -> None:
    model = FakeModel()
    session = DeltaMemChatSession(
        model=model,
        tokenizer=FakeTokenizer(eos_token_id=5),
        device="cpu",
    )
    session.processed_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    next_token_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    generated_ids = session._decode_generate(
        next_token_logits,
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )

    assert generated_ids.shape == (1, 0)
    assert torch.equal(session.processed_input_ids, torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert model.calls == 0


def test_reset_delta_mem_states_clears_wrapped_model_state() -> None:
    wrapped = ToyAttentionModel(make_delta_module(output_init="random"))
    x = torch.randn(1, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=wrapped.self_attn.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = wrapped(x, position_embeddings)
    reset_delta_mem_states(wrapped)
    assert wrapped.self_attn.delta_state is None
