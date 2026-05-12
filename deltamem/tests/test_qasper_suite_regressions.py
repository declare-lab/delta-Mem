from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from deltamem.eval import benchmark_compare


def test_benchmark_compare_skip_delta_does_not_require_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_compare.py",
            "--model-path",
            "/tmp/model",
            "--skip-base",
            "--skip-delta",
            "--lora-adapter-dir",
            "/tmp/lora",
            "--output-json",
            "/tmp/out.json",
        ],
    )

    args = benchmark_compare.parse_args()

    assert args.skip_delta is True
    assert args.delta_adapter_dir is None
    assert args.lora_adapter_dir == Path("/tmp/lora")


class _TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    model_max_length = 4096

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def __call__(self, text, return_tensors: str = "pt", padding: bool = False, add_special_tokens: bool = True):
        del return_tensors, add_special_tokens
        if isinstance(text, list):
            rows = [self.encode(item) for item in text]
            if padding:
                width = max(len(row) for row in rows)
                rows = [[0] * (width - len(row)) + row for row in rows]
            input_ids = torch.tensor(rows, dtype=torch.long)
            return SimpleNamespace(input_ids=input_ids, attention_mask=(input_ids != 0).long())
        input_ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return SimpleNamespace(input_ids=input_ids)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            chars.append(chr(token_id))
        return "".join(chars)


class _FakeCache:
    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len


class _SharedPrefixModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[int, int]] = []
        self.generation_config = SimpleNamespace(eos_token_id=0)

    def forward(self, input_ids, past_key_values=None, use_cache: bool = True, return_dict: bool = True, **kwargs):
        del use_cache, return_dict, kwargs
        prev_len = 0 if past_key_values is None else int(past_key_values.seq_len)
        current_len = int(input_ids.shape[1])
        self.calls.append((prev_len, current_len))
        logits = torch.full((1, current_len, 256), -1e9)
        last_token = int(input_ids[0, -1].item())
        next_token = 0 if last_token == ord("O") else ord("O")
        logits[:, -1, next_token] = 0.0
        return {"logits": logits, "past_key_values": _FakeCache(prev_len + current_len)}


def test_memory_agent_bench_row_tasks_keep_rows_on_one_rank() -> None:
    items = [
        {
            "row_id": "row0",
            "context": "A" * 400,
            "selected_questions": [{"eval_index": 0}, {"eval_index": 1}, {"eval_index": 2}],
        },
        {
            "row_id": "row1",
            "context": "B" * 800,
            "selected_questions": [{"eval_index": 3}],
        },
        {
            "row_id": "row2",
            "context": "C" * 1200,
            "selected_questions": [{"eval_index": 4}, {"eval_index": 5}],
        },
    ]

    owners: dict[int, int] = {}
    for rank in range(3):
        context = benchmark_compare.DistributedContext(
            enabled=True,
            rank=rank,
            world_size=3,
            local_rank=rank,
            device="cpu",
        )
        row_tasks = benchmark_compare.local_memory_agent_bench_row_tasks(
            items,
            context,
            default_max_new_tokens=128,
            use_official_generation_lengths=True,
        )
        for row_task in row_tasks:
            assert row_task.row_index not in owners
            owners[row_task.row_index] = rank

    assert owners == {0: owners[0], 1: owners[1], 2: owners[2]}
    assert len(set(owners.values())) >= 2


def test_memory_agent_bench_shared_prefix_reuses_prefill() -> None:
    tokenizer = _TinyTokenizer()
    model = _SharedPrefixModel()
    prompts = [
        "Use only the memory context below.\n\nshared context block\n\nQuestion: alpha\nAnswer:",
        "Use only the memory context below.\n\nshared context block\n\nQuestion: beta\nAnswer:",
    ]
    prompt_token_lists = [
        benchmark_compare._tokenize_prompt_to_token_list(
            tokenizer,
            prompt,
            use_chat_template=False,
        )
        for prompt in prompts
    ]

    predictions = benchmark_compare._generate_predictions_with_shared_prefix(
        model,
        tokenizer,
        "cpu",
        prompt_token_lists,
        generation_settings={
            "max_new_tokens": 4,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "use_chat_template": False,
        },
        reset_delta_state=False,
    )

    common_prefix_len = benchmark_compare._longest_common_token_prefix_length(prompt_token_lists)
    assert predictions == ["O", "O"]
    assert model.calls[0] == (0, common_prefix_len)
    assert model.calls[1][0] == common_prefix_len
    assert model.calls[3][0] == common_prefix_len
    assert model.calls[1][1] < len(prompt_token_lists[0])
    assert model.calls[3][1] < len(prompt_token_lists[1])
    assert len(model.calls) == 5
