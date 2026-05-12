from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from datasets import DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download
import torch

from deltamem.chat_templates import apply_chat_template as apply_project_chat_template
from deltamem.core.delta import reset_delta_mem_states
from deltamem.eval.common import (
    DistributedContext,
    gather_indexed_records,
    init_distributed,
    load_base_model_and_tokenizer,
    load_delta_model_and_tokenizer,
    load_lora_model_and_tokenizer,
    set_all_seeds,
)
from deltamem.model_loading import DEFAULT_DATASETS_CACHE_DIR, DEFAULT_HF_HOME, DEFAULT_LOCAL_MODEL_PATH


DEFAULT_MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH
DEFAULT_HF_HUB_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
DEFAULT_MEMORY_AGENT_BENCH_ROOT = Path("external/MemoryAgentBench")
RESUME_PROMPT_PREFIX_CHARS = 1000


def dataset_download_config(*, local_files_only: bool) -> DownloadConfig | None:
    if not local_files_only:
        return None
    return DownloadConfig(local_files_only=True)


def load_dataset_cached(*args, cache_dir: Path, local_files_only: bool, **kwargs):
    return load_dataset(
        *args,
        cache_dir=str(cache_dir),
        download_config=dataset_download_config(local_files_only=local_files_only),
        **kwargs,
    )


def candidate_hub_cache_dirs(primary_cache_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for cache_dir in (primary_cache_dir, DEFAULT_HF_HUB_CACHE_DIR):
        resolved = Path(cache_dir)
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)
    return candidates


def find_cached_hub_file(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    hub_cache_dir: Path,
) -> Path | None:
    repo_prefix = "datasets--" if repo_type == "dataset" else "models--"
    repo_dir_name = repo_prefix + repo_id.replace("/", "--")
    for cache_root in candidate_hub_cache_dirs(hub_cache_dir):
        snapshot_root = cache_root / repo_dir_name / "snapshots"
        if not snapshot_root.is_dir():
            continue
        for snapshot_dir in sorted(snapshot_root.iterdir(), key=lambda path: path.name, reverse=True):
            if not snapshot_dir.is_dir():
                continue
            candidate = snapshot_dir / filename
            if candidate.is_file():
                return candidate
    return None


def resolve_hub_file(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    hub_cache_dir: Path,
    local_files_only: bool,
) -> Path:
    cached_file = find_cached_hub_file(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        hub_cache_dir=hub_cache_dir,
    )
    if cached_file is not None:
        return cached_file
    if local_files_only:
        raise FileNotFoundError(
            f"Could not find cached {repo_type} file {filename!r} from {repo_id!r} in {hub_cache_dir}"
        )
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        cache_dir=str(hub_cache_dir),
        local_files_only=False,
    )
    return Path(downloaded)


def infer_model_context_window(model, tokenizer) -> int:
    direct_max_model_len = getattr(model, "max_model_len", None)
    if isinstance(direct_max_model_len, int) and 0 < direct_max_model_len < 10**7:
        return direct_max_model_len
    config = getattr(model, "config", None)
    config_candidates = [
        model,
        config,
        getattr(model, "text_config", None),
        getattr(config, "text_config", None),
    ]
    for candidate in config_candidates:
        if candidate is None:
            continue
        for attr in ("max_position_embeddings", "sliding_window"):
            value = getattr(candidate, attr, None)
            if isinstance(value, int) and 0 < value < 10**7:
                return value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 10**7:
        return value
    return 32768


def load_model_for_eval(args, context: DistributedContext):
    adapter_dir = None
    if args.delta_adapter_dir:
        adapter_dir = Path(args.delta_adapter_dir)
    if args.lora_adapter_dir:
        adapter_dir = Path(args.lora_adapter_dir)

    if args.delta_adapter_dir:
        model, tokenizer, adapter_config = load_delta_model_and_tokenizer(
            model_path=args.model_path,
            adapter_dir=Path(args.delta_adapter_dir),
            device=context.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        return "delta", model, tokenizer, adapter_config
    if args.lora_adapter_dir:
        model, tokenizer = load_lora_model_and_tokenizer(
            model_path=args.model_path,
            adapter_dir=Path(args.lora_adapter_dir),
            device=context.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        return "lora", model, tokenizer, {"adapter_dir": str(adapter_dir)}
    model, tokenizer = load_base_model_and_tokenizer(
        model_path=args.model_path,
        device=context.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    return "base", model, tokenizer, None


@lru_cache(maxsize=32)
def load_external_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {file_path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def truncate_text_by_tokens(
    text: str,
    *,
    tokenizer,
    max_tokens: int,
    keep: Literal["head", "tail"],
) -> str:
    if max_tokens <= 0 or not text:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    if keep == "head":
        kept = token_ids[:max_tokens]
    else:
        kept = token_ids[-max_tokens:]
    return tokenizer.decode(kept, skip_special_tokens=True).strip()


def render_messages(tokenizer, messages: list[dict[str, str]], *, use_chat_template: bool) -> str:
    if use_chat_template:
        return apply_project_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def make_resume_prompt_prefix(text: str, *, max_chars: int = RESUME_PROMPT_PREFIX_CHARS) -> str:
    return str(text)[:max_chars]


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_resume_record(
    existing_records: list[dict[str, Any]],
    *,
    prompt_text: str,
    question_text: str,
) -> dict[str, Any] | None:
    prompt_prefix = make_resume_prompt_prefix(prompt_text)
    for record in existing_records:
        record_prefix = str(
            record.get("resume_prompt_prefix")
            or make_resume_prompt_prefix(record.get("prompt") or record.get("user_prompt") or "")
        )
        if record_prefix != prompt_prefix:
            continue
        record_question = str(
            record.get("resume_question")
            or record.get("question")
            or record.get("query")
            or ""
        )
        if record_question == str(question_text):
            return record
    return None


def generate_from_message_batches(
    model,
    tokenizer,
    device: str,
    message_batches: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    batch_size: int,
    reset_delta_state: bool,
    use_chat_template: bool,
) -> list[dict[str, object]]:
    if not message_batches:
        return []
    if reset_delta_state:
        reset_delta_mem_states(model)

    results: list[dict[str, object]] = []
    effective_batch_size = max(1, batch_size)
    for start in range(0, len(message_batches), effective_batch_size):
        chunk = message_batches[start : start + effective_batch_size]
        rendered_prompts = [render_messages(tokenizer, messages, use_chat_template=use_chat_template) for messages in chunk]

        old_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            tokenized = tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not use_chat_template,
            )
        finally:
            tokenizer.padding_side = old_padding_side

        input_ids = tokenized.input_ids.to(device)
        attention_mask = getattr(tokenized, "attention_mask", None)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        else:
            attention_mask = attention_mask.to(device)

        generation_config = copy.deepcopy(getattr(model, "generation_config", None))
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if generation_config is not None:
            generation_config.do_sample = do_sample
            generation_config.max_new_tokens = max_new_tokens
            generation_config.use_cache = True
            if tokenizer.pad_token_id is not None:
                generation_config.pad_token_id = tokenizer.pad_token_id
            if do_sample:
                generation_config.temperature = temperature
                generation_config.top_p = top_p
                generation_config.top_k = top_k
            else:
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None
            generate_kwargs["generation_config"] = generation_config
        else:
            generate_kwargs["do_sample"] = do_sample
            generate_kwargs["max_new_tokens"] = max_new_tokens
            generate_kwargs["use_cache"] = True
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
                generate_kwargs["top_k"] = top_k
            if tokenizer.pad_token_id is not None:
                generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

        started = time.time()
        outputs = model.generate(**generate_kwargs)
        elapsed = time.time() - started
        prompt_width = input_ids.shape[1]
        generated_ids = outputs[:, prompt_width:]
        batch_query_time = elapsed / max(1, len(chunk))
        for prompt_ids, row in zip(input_ids, generated_ids):
            text = tokenizer.decode(row, skip_special_tokens=True).strip()
            results.append(
                {
                    "output": text,
                    "input_len": int(prompt_ids.shape[-1]),
                    "output_len": int(row.shape[-1]),
                    "memory_construction_time": 0.0,
                    "query_time_len": batch_query_time,
                }
            )
    return results


def local_indexed_items(items: list[dict[str, object]], context: DistributedContext) -> list[tuple[int, dict[str, object]]]:
    if not context.enabled:
        return list(enumerate(items))
    return [
        (item_idx, item)
        for item_idx, item in enumerate(items)
        if item_idx % context.world_size == context.rank
    ]


def build_common_arg_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--delta-adapter-dir")
    parser.add_argument("--lora-adapter-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--datasets-cache-dir", type=Path, default=Path(DEFAULT_DATASETS_CACHE_DIR))
    parser.add_argument("--hub-cache-dir", type=Path, default=Path(DEFAULT_HF_HUB_CACHE_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


__all__ = [
    "DEFAULT_MEMORY_AGENT_BENCH_ROOT",
    "dataset_download_config",
    "generate_from_message_batches",
    "gather_indexed_records",
    "infer_model_context_window",
    "append_jsonl_record",
    "find_resume_record",
    "load_jsonl_records",
    "make_resume_prompt_prefix",
    "init_distributed",
    "build_common_arg_parser",
    "load_dataset_cached",
    "load_external_module",
    "load_model_for_eval",
    "local_indexed_items",
    "render_messages",
    "resolve_hub_file",
    "set_all_seeds",
    "truncate_text_by_tokens",
]
