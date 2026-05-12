from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from deltamem.chat_templates import apply_chat_template as apply_project_chat_template
from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_state_stats,
    get_delta_mem_online_state,
    load_delta_mem_adapter,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
from deltamem.core.write_segmentation import split_text_into_sentence_token_chunks


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _chat_template_input_ids(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> torch.Tensor:
    tokenized = apply_project_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    return tokenized


def _tokenize_chat_messages_to_list(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    tokenized = _chat_template_input_ids(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    return tokenized.squeeze(0).tolist()


def _tokenize_text_no_special_tokens(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _find_subsequence_start(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return 0
    max_start = len(haystack) - len(needle)
    for start in range(max_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


_MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS = 16


def _longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _chat_template_delta(
    previous_ids: list[int],
    current_ids: list[int],
    *,
    error_message: str,
) -> tuple[int, list[int]]:
    prefix_len = _longest_common_prefix_length(previous_ids, current_ids)
    rollback_tokens = len(previous_ids) - prefix_len
    if rollback_tokens > _MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS:
        raise ValueError(error_message)
    return prefix_len, current_ids[prefix_len:]


def _sentence_ids_for_message_delta(
    tokenizer,
    message_content: str,
    delta_ids: list[int],
    next_sentence_id: int,
) -> tuple[list[int], int]:
    sentence_ids = [-1] * len(delta_ids)
    content_ids = _tokenize_text_no_special_tokens(tokenizer, message_content)
    if not content_ids:
        return sentence_ids, next_sentence_id
    content_start = _find_subsequence_start(delta_ids, content_ids)
    if content_start is None:
        return sentence_ids, next_sentence_id
    sentence_chunks = split_text_into_sentence_token_chunks(message_content)
    sentence_chunk_ids = [
        _tokenize_text_no_special_tokens(tokenizer, sentence_chunk)
        for sentence_chunk in sentence_chunks
    ]
    flat_sentence_ids = [token_id for chunk_ids in sentence_chunk_ids for token_id in chunk_ids]
    if flat_sentence_ids != content_ids:
        sentence_ids[content_start : content_start + len(content_ids)] = [next_sentence_id] * len(content_ids)
        return sentence_ids, next_sentence_id + 1
    position = content_start
    for chunk_ids in sentence_chunk_ids:
        if not chunk_ids:
            continue
        sentence_ids[position : position + len(chunk_ids)] = [next_sentence_id] * len(chunk_ids)
        position += len(chunk_ids)
        next_sentence_id += 1
    return sentence_ids, next_sentence_id


def _tokenize_chat_messages_with_write_span_ids(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    include_sentence_ids: bool,
) -> tuple[list[int], list[int], list[int]]:
    input_ids: list[int] = []
    message_ids: list[int] = []
    sentence_ids: list[int] = []
    previous_ids: list[int] = []
    next_message_id = 0
    next_sentence_id = 0
    for index, message in enumerate(messages):
        current_ids = _tokenize_chat_messages_to_list(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=False,
        )
        prefix_len, delta_ids = _chat_template_delta(
            previous_ids,
            current_ids,
            error_message="Chat template tokenization is not prefix-stable; cannot recover runtime write spans safely.",
        )
        if prefix_len < len(input_ids):
            del input_ids[prefix_len:]
            del message_ids[prefix_len:]
            del sentence_ids[prefix_len:]
        input_ids.extend(delta_ids)
        if message["role"] == "system":
            message_ids.extend([-1] * len(delta_ids))
            sentence_ids.extend([-1] * len(delta_ids))
            previous_ids = current_ids
            continue

        message_id = next_message_id
        next_message_id += 1
        message_ids.extend([message_id] * len(delta_ids))
        if include_sentence_ids:
            message_sentence_ids, next_sentence_id = _sentence_ids_for_message_delta(
                tokenizer,
                message["content"],
                delta_ids,
                next_sentence_id,
            )
            sentence_ids.extend(message_sentence_ids)
        else:
            sentence_ids.extend([-1] * len(delta_ids))
        previous_ids = current_ids

    if add_generation_prompt:
        prompted_ids = _tokenize_chat_messages_to_list(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        prefix_len, prompt_suffix = _chat_template_delta(
            previous_ids,
            prompted_ids,
            error_message="Chat template tokenization is not prefix-stable with generation prompt; cannot recover runtime write spans safely.",
        )
        if prefix_len < len(input_ids):
            del input_ids[prefix_len:]
            del message_ids[prefix_len:]
            del sentence_ids[prefix_len:]
        input_ids.extend(prompt_suffix)
        message_ids.extend([-1] * len(prompt_suffix))
        sentence_ids.extend([-1] * len(prompt_suffix))
    return input_ids, message_ids, sentence_ids


@dataclass
class DeltaMemSessionSnapshot:
    messages: list[dict[str, str]]
    processed_input_ids: list[int]
    delta_state: dict[str, torch.Tensor]
    past_key_values: object | None = None
    write_message_ids: list[int] = field(default_factory=list)
    write_sentence_ids: list[int] = field(default_factory=list)


@dataclass
class DeltaMemChatSession:
    model: torch.nn.Module
    tokenizer: object
    device: str
    messages: list[dict[str, str]] = field(default_factory=list)
    processed_input_ids: torch.Tensor | None = None
    past_key_values: object | None = None
    last_ingest_stats: dict[str, object] = field(default_factory=dict)
    last_decode_stats: dict[str, object] = field(default_factory=dict)
    last_turn_stats: dict[str, object] = field(default_factory=dict)
    cached_message_input_ids: list[int] | None = None
    cached_write_message_ids: list[int] | None = None
    cached_write_sentence_ids: list[int] | None = None
    cached_message_count: int = 0

    def reset(self) -> None:
        self.messages = []
        self.processed_input_ids = None
        self.past_key_values = None
        self.last_ingest_stats = {}
        self.last_decode_stats = {}
        self.last_turn_stats = {}
        self.cached_message_input_ids = None
        self.cached_write_message_ids = None
        self.cached_write_sentence_ids = None
        self.cached_message_count = 0
        reset_delta_mem_states(self.model)

    def state_stats(self) -> dict[str, float]:
        return collect_delta_mem_state_stats(self.model)

    def snapshot(self) -> DeltaMemSessionSnapshot:
        _, write_message_ids, write_sentence_ids = self._current_message_token_cache()
        return DeltaMemSessionSnapshot(
            messages=[dict(message) for message in self.messages],
            processed_input_ids=[]
            if self.processed_input_ids is None
            else self.processed_input_ids.squeeze(0).tolist(),
            delta_state=get_delta_mem_online_state(self.model),
            past_key_values=_move_nested_tensors(self.past_key_values, "cpu"),
            write_message_ids=write_message_ids,
            write_sentence_ids=write_sentence_ids,
        )

    def load_snapshot(self, snapshot: DeltaMemSessionSnapshot) -> None:
        self.messages = [dict(message) for message in snapshot.messages]
        if snapshot.processed_input_ids:
            self.processed_input_ids = torch.tensor(
                [snapshot.processed_input_ids],
                dtype=torch.long,
            )
        else:
            self.processed_input_ids = None
        if (
            snapshot.write_message_ids
            and snapshot.write_sentence_ids
            and len(snapshot.write_message_ids) == len(snapshot.processed_input_ids)
            and len(snapshot.write_sentence_ids) == len(snapshot.processed_input_ids)
        ):
            self.cached_message_input_ids = list(snapshot.processed_input_ids)
            self.cached_write_message_ids = list(snapshot.write_message_ids)
            self.cached_write_sentence_ids = list(snapshot.write_sentence_ids)
            self.cached_message_count = len(self.messages)
        else:
            self.cached_message_input_ids = None
            self.cached_write_message_ids = None
            self.cached_write_sentence_ids = None
            self.cached_message_count = 0
        past_key_values = _move_nested_tensors(snapshot.past_key_values, self.device)
        if isinstance(past_key_values, tuple):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        self.past_key_values = past_key_values
        self.last_ingest_stats = {}
        self.last_decode_stats = {}
        self.last_turn_stats = {}
        reset_delta_mem_states(self.model)
        load_delta_mem_online_state(self.model, snapshot.delta_state)

    def save_snapshot(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        meta = {
            "messages": snapshot.messages,
            "processed_input_ids": snapshot.processed_input_ids,
            "write_message_ids": snapshot.write_message_ids,
            "write_sentence_ids": snapshot.write_sentence_ids,
        }
        (output_path / "session.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )
        torch.save(snapshot.delta_state, output_path / "delta_state.pt")
        torch.save(snapshot.past_key_values, output_path / "past_key_values.pt")

    def load_snapshot_dir(self, input_dir: str | Path) -> None:
        input_path = Path(input_dir)
        meta = json.loads((input_path / "session.json").read_text())
        delta_state = torch.load(
            input_path / "delta_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        past_key_values = torch.load(
            input_path / "past_key_values.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.load_snapshot(
            DeltaMemSessionSnapshot(
                messages=meta["messages"],
                processed_input_ids=meta["processed_input_ids"],
                delta_state=delta_state,
                past_key_values=past_key_values,
                write_message_ids=meta.get("write_message_ids", []),
                write_sentence_ids=meta.get("write_sentence_ids", []),
            )
        )

    def _tokenize_messages(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> torch.Tensor:
        return _chat_template_input_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
        ).to(self.device)

    def _include_sentence_write_ids(self) -> bool:
        for module in self.model.modules():
            if getattr(module, "memory_write_granularity", None) is not None:
                return getattr(module, "memory_write_granularity") == "sentence_mean"
        return False

    def _set_message_token_cache(
        self,
        token_ids: list[int],
        message_ids: list[int],
        sentence_ids: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        self.cached_message_input_ids = list(token_ids)
        self.cached_write_message_ids = list(message_ids)
        self.cached_write_sentence_ids = list(sentence_ids)
        self.cached_message_count = len(self.messages)
        return (
            self.cached_message_input_ids,
            self.cached_write_message_ids,
            self.cached_write_sentence_ids,
        )

    def _rebuild_message_token_cache_from_scratch(self) -> tuple[list[int], list[int], list[int]]:
        if not self.messages:
            return self._set_message_token_cache([], [], [])
        token_ids, message_ids, sentence_ids = _tokenize_chat_messages_with_write_span_ids(
            self.tokenizer,
            self.messages,
            add_generation_prompt=False,
            include_sentence_ids=self._include_sentence_write_ids(),
        )
        return self._set_message_token_cache(token_ids, message_ids, sentence_ids)

    def _current_message_token_cache(self) -> tuple[list[int], list[int], list[int]]:
        if not self.messages:
            return self._set_message_token_cache([], [], [])
        if (
            self.cached_message_input_ids is None
            or self.cached_write_message_ids is None
            or self.cached_write_sentence_ids is None
        ):
            return self._rebuild_message_token_cache_from_scratch()
        if self.cached_message_count == len(self.messages):
            return (
                self.cached_message_input_ids,
                self.cached_write_message_ids,
                self.cached_write_sentence_ids,
            )
        if self.cached_message_count + 1 == len(self.messages):
            current_ids = _tokenize_chat_messages_to_list(
                self.tokenizer,
                self.messages,
                add_generation_prompt=False,
            )
            cached_ids = self.cached_message_input_ids
            if current_ids[: len(cached_ids)] == cached_ids:
                last_message = self.messages[-1]
                current_message_ids = list(self.cached_write_message_ids)
                current_sentence_ids = list(self.cached_write_sentence_ids)
                delta_len = len(current_ids) - len(cached_ids)
                if last_message["role"] == "system":
                    current_message_ids.extend([-1] * delta_len)
                    current_sentence_ids.extend([-1] * delta_len)
                    return self._set_message_token_cache(
                        current_ids,
                        current_message_ids,
                        current_sentence_ids,
                    )

                last_message_id = self._next_runtime_message_id() - 1
                current_message_ids.extend([last_message_id] * delta_len)
                if not self._include_sentence_write_ids():
                    current_sentence_ids.extend([-1] * delta_len)
                    return self._set_message_token_cache(
                        current_ids,
                        current_message_ids,
                        current_sentence_ids,
                    )

                # Sentence-mean mode uses message-content/token alignment; rebuild once
                # from scratch instead of relying on prefix-stable partial chat renders.
                return self._rebuild_message_token_cache_from_scratch()
        return self._rebuild_message_token_cache_from_scratch()

    def _common_prefix_len(self, full_ids: torch.Tensor) -> int:
        if self.processed_input_ids is None:
            return 0
        lhs = self.processed_input_ids.squeeze(0)
        rhs = full_ids.squeeze(0).detach().cpu()
        max_len = min(lhs.numel(), rhs.numel())
        if max_len == 0:
            return 0
        matches = lhs[:max_len].eq(rhs[:max_len])
        mismatch = (~matches).nonzero(as_tuple=False)
        if mismatch.numel() == 0:
            return int(max_len)
        return int(mismatch[0].item())

    def _runtime_write_message_ids(self, full_ids: torch.Tensor) -> torch.Tensor | None:
        message_ids, _ = self._runtime_write_span_ids(full_ids)
        return message_ids

    def _runtime_write_span_ids(
        self,
        full_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.messages:
            return None, None
        target_ids = full_ids.squeeze(0).detach().cpu().tolist()

        current_token_ids, current_message_ids, current_sentence_ids = self._current_message_token_cache()
        if target_ids == current_token_ids:
            return (
                torch.tensor([current_message_ids], dtype=torch.long, device=full_ids.device),
                torch.tensor([current_sentence_ids], dtype=torch.long, device=full_ids.device),
            )
        if len(target_ids) >= len(current_token_ids) and target_ids[: len(current_token_ids)] == current_token_ids:
            prompt_message_ids = current_message_ids + [-1] * (len(target_ids) - len(current_token_ids))
            prompt_sentence_ids = current_sentence_ids + [-1] * (len(target_ids) - len(current_token_ids))
            return (
                torch.tensor([prompt_message_ids], dtype=torch.long, device=full_ids.device),
                torch.tensor([prompt_sentence_ids], dtype=torch.long, device=full_ids.device),
            )

        current_token_ids, current_message_ids, current_sentence_ids = self._rebuild_message_token_cache_from_scratch()
        if target_ids == current_token_ids:
            return (
                torch.tensor([current_message_ids], dtype=torch.long, device=full_ids.device),
                torch.tensor([current_sentence_ids], dtype=torch.long, device=full_ids.device),
            )
        if len(target_ids) >= len(current_token_ids) and target_ids[: len(current_token_ids)] == current_token_ids:
            prompt_message_ids = current_message_ids + [-1] * (len(target_ids) - len(current_token_ids))
            prompt_sentence_ids = current_sentence_ids + [-1] * (len(target_ids) - len(current_token_ids))
            return (
                torch.tensor([prompt_message_ids], dtype=torch.long, device=full_ids.device),
                torch.tensor([prompt_sentence_ids], dtype=torch.long, device=full_ids.device),
            )

        for add_generation_prompt in (False, True):
            token_ids, message_ids, sentence_ids = _tokenize_chat_messages_with_write_span_ids(
                self.tokenizer,
                self.messages,
                add_generation_prompt=add_generation_prompt,
                include_sentence_ids=self._include_sentence_write_ids(),
            )
            if token_ids == target_ids:
                if not add_generation_prompt:
                    self._set_message_token_cache(token_ids, message_ids, sentence_ids)
                return (
                    torch.tensor([message_ids], dtype=torch.long, device=full_ids.device),
                    torch.tensor([sentence_ids], dtype=torch.long, device=full_ids.device),
                )
        return None, None

    def _next_runtime_message_id(self) -> int:
        return sum(1 for message in self.messages if message["role"] != "system")

    def _next_runtime_sentence_id(self) -> int:
        _, _, current_sentence_ids = self._current_message_token_cache()
        active_ids = [sentence_id for sentence_id in current_sentence_ids if sentence_id >= 0]
        if not active_ids:
            return 0
        return max(active_ids) + 1

    def _ingest_full_ids(self, full_ids: torch.Tensor) -> torch.Tensor | None:
        started_at = time.perf_counter()
        prefix_len = self._common_prefix_len(full_ids)
        prev_len = 0 if self.processed_input_ids is None else self.processed_input_ids.size(1)
        rebuilt = False
        if self.processed_input_ids is not None and prefix_len < self.processed_input_ids.size(1):
            # Fallback: rebuild state/caches from scratch if history diverged.
            self.processed_input_ids = None
            self.past_key_values = None
            reset_delta_mem_states(self.model)
            prefix_len = 0
            rebuilt = True

        full_write_message_ids, full_write_sentence_ids = self._runtime_write_span_ids(full_ids)
        suffix = full_ids[:, prefix_len:]
        suffix_message_ids = None
        suffix_sentence_ids = None
        if full_write_message_ids is not None:
            suffix_message_ids = full_write_message_ids[:, prefix_len:]
        if full_write_sentence_ids is not None:
            suffix_sentence_ids = full_write_sentence_ids[:, prefix_len:]
        suffix_tokens = int(suffix.size(1))
        self.last_ingest_stats = {
            "prev_tokens": int(prev_len),
            "full_tokens": int(full_ids.size(1)),
            "prefix_tokens": int(prefix_len),
            "suffix_tokens": suffix_tokens,
            "rebuilt": rebuilt,
        }
        if suffix.numel() == 0:
            self.last_ingest_stats["elapsed_ms"] = round(
                (time.perf_counter() - started_at) * 1000.0, 3
            )
            return None
        set_delta_mem_write_message_ids(self.model, suffix_message_ids)
        set_delta_mem_write_sentence_ids(self.model, suffix_sentence_ids)
        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=suffix,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        finally:
            set_delta_mem_write_message_ids(self.model, None)
            set_delta_mem_write_sentence_ids(self.model, None)
        self.past_key_values = outputs.past_key_values
        self.processed_input_ids = full_ids.detach().cpu()
        self.last_ingest_stats["elapsed_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0, 3
        )
        return outputs.logits[:, -1, :]

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0 when do_sample=True")

        filtered = logits / temperature
        if 0 < top_k < filtered.size(-1):
            top_values = torch.topk(filtered, k=top_k, dim=-1).values
            threshold = top_values[..., -1:].expand_as(filtered)
            filtered = filtered.masked_fill(filtered < threshold, torch.finfo(filtered.dtype).min)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_remove, torch.finfo(sorted_logits.dtype).min)
            filtered = torch.full_like(filtered, torch.finfo(filtered.dtype).min)
            filtered.scatter_(-1, sorted_indices, sorted_logits)
        probs = torch.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _decode_generate(
        self,
        next_token_logits: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        started_at = time.perf_counter()
        generated = []
        generated_cpu = []
        eos_id = self.tokenizer.eos_token_id
        next_token = self._sample_next_token(
            next_token_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        assistant_message_id = self._next_runtime_message_id()
        assistant_sentence_id = self._next_runtime_sentence_id()
        for _ in range(max_new_tokens):
            if eos_id is not None and next_token.item() == eos_id:
                break
            generated.append(next_token)
            generated_cpu.append(next_token.detach().cpu())
            token_message_ids = torch.full(
                next_token.shape,
                assistant_message_id,
                dtype=torch.long,
                device=next_token.device,
            )
            token_sentence_ids = torch.full(
                next_token.shape,
                assistant_sentence_id,
                dtype=torch.long,
                device=next_token.device,
            )
            set_delta_mem_write_message_ids(self.model, token_message_ids)
            set_delta_mem_write_sentence_ids(self.model, token_sentence_ids)
            try:
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=self.past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
            finally:
                set_delta_mem_write_message_ids(self.model, None)
                set_delta_mem_write_sentence_ids(self.model, None)
            self.past_key_values = outputs.past_key_values
            next_token = self._sample_next_token(
                outputs.logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        if generated:
            generated_ids = torch.cat(generated, dim=1)
            generated_ids_cpu = torch.cat(generated_cpu, dim=1)
            if self.processed_input_ids is None:
                self.processed_input_ids = generated_ids_cpu
            else:
                self.processed_input_ids = torch.cat(
                    [self.processed_input_ids, generated_ids_cpu], dim=1
                )
        else:
            generated_ids = next_token.new_empty((next_token.size(0), 0))
        self.last_decode_stats = {
            "generated_tokens": int(generated_ids.size(1)),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
        }
        return generated_ids

    def generate_reply(
        self,
        user_text: str,
        max_new_tokens: int = 2048,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        sample_seed: int | None = None,
        write_enabled: bool = True,
        include_debug: bool = False,
    ) -> dict[str, object]:
        started_at = time.perf_counter()
        set_delta_mem_write_enabled(self.model, write_enabled)
        try:
            self.messages.append({"role": "user", "content": user_text})
            prompt_ids = self._tokenize_messages(
                self.messages,
                add_generation_prompt=True,
            )
            next_token_logits = self._ingest_full_ids(prompt_ids)
            if next_token_logits is None:
                raise RuntimeError("Prompt suffix was empty; cannot start generation.")
            prompt_ingest_stats = dict(self.last_ingest_stats)
            rng_devices = []
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                rng_devices = [torch.device(self.device)]
            with torch.random.fork_rng(devices=rng_devices):
                if sample_seed is not None:
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)
                generated_ids = self._decode_generate(
                    next_token_logits,
                    max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            response_raw = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            self.messages.append({"role": "assistant", "content": response_raw})

            # Materialize any template suffix introduced by storing the assistant turn.
            full_ids = self._tokenize_messages(
                self.messages,
                add_generation_prompt=False,
            )
            self._ingest_full_ids(full_ids)
            materialize_stats = dict(self.last_ingest_stats)

            self.last_turn_stats = {
                "prompt_ingest": prompt_ingest_stats,
                "decode": dict(self.last_decode_stats),
                "assistant_materialize": materialize_stats,
                "sample_seed": sample_seed,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            }

            result = {
                "user": user_text,
                "assistant": response_raw,
                "assistant_display": response_raw.strip(),
                "state_stats": self.state_stats(),
            }
            if include_debug:
                result["turn_stats"] = dict(self.last_turn_stats)
            return result
        finally:
            set_delta_mem_write_enabled(self.model, True)


def load_delta_mem_chat_model(
    *,
    model_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    adapter_dir: str | Path,
) -> tuple[torch.nn.Module, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=get_dtype(dtype),
        device_map={"": device},
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).eval()
    config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    attach_delta_mem(model, config)
    load_delta_mem_adapter(model, adapter_dir)
    return model, tokenizer


def _move_nested_tensors(obj, device: str):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_nested_tensors(item, device) for item in obj)
    if isinstance(obj, list):
        return [_move_nested_tensors(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: _move_nested_tensors(value, device) for key, value in obj.items()}
    if hasattr(obj, "__dict__"):
        cloned = copy.copy(obj)
        for key, value in vars(obj).items():
            setattr(cloned, key, _move_nested_tensors(value, device))
        return cloned
    return obj
