from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import re
import threading
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from deltamem.chat_templates import apply_chat_template as apply_project_chat_template
from deltamem.core.delta import (
    get_delta_mem_online_state,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
from deltamem.eval.common import (
    DistributedContext,
    attach_delta_adapter_in_place,
    barrier_distributed,
    finalize_distributed,
    gather_indexed_records,
    init_distributed,
    load_base_model_and_tokenizer,
    load_delta_model_and_tokenizer,
    load_lora_model_and_tokenizer,
    maybe_empty_cache,
    set_all_seeds,
)
from deltamem.model_loading import DEFAULT_DATASETS_CACHE_DIR, DEFAULT_HF_HOME, DEFAULT_LOCAL_MODEL_PATH
from deltamem.eval.ifeval_compat import evaluate_ifeval_response
from deltamem.eval.official_memory_agent_bench import (
    OFFICIAL_SOURCE_CONFIGS as MEMORY_AGENT_BENCH_OFFICIAL_SOURCE_CONFIGS,
    build_context_chunks as build_official_mab_context_chunks,
    build_memorized_context as build_official_mab_memorized_context,
    build_query_answer_pairs as build_official_mab_query_answer_pairs,
    load_mab_eval_utils,
    truncate_memory_context as truncate_official_mab_memory_context,
)
from deltamem.eval.official_eval_utils import DEFAULT_MEMORY_AGENT_BENCH_ROOT
from deltamem.eval.official_memory_agent_bench_templates import get_template as get_official_mab_template
from deltamem.runtime.session import _tokenize_chat_messages_with_write_span_ids as tokenize_chat_messages_with_write_span_ids

warnings.filterwarnings(
    "ignore",
    message=r"Position ids are not supported for parameter efficient tuning\. Ignoring position ids\.",
    category=UserWarning,
    module=r"peft\.peft_model",
)

DEFAULT_MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH
DEFAULT_HF_HUB_CACHE_DIR = Path(os.environ.get("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub")))
FALLBACK_HF_HUB_CACHE_DIRS = (
    DEFAULT_HF_HUB_CACHE_DIR,
    Path.home() / ".cache" / "huggingface" / "hub",
)


def _distributed_rank_label() -> str:
    if dist.is_available() and dist.is_initialized():
        try:
            return f"rank{dist.get_rank()}"
        except RuntimeError:
            pass
    return f"rank{os.environ.get('RANK', '0')}"


def _gpqa_heartbeat_interval_seconds() -> float:
    raw_value = os.environ.get("DELTALORA_GPQA_HEARTBEAT_SECONDS", "0").strip()
    if not raw_value:
        return 0.0
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 0.0


def _log_eval_debug(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[eval-debug][{timestamp}][{_distributed_rank_label()}] {message}", flush=True)


def _run_with_gpqa_heartbeat(label: str, fn):
    interval_seconds = _gpqa_heartbeat_interval_seconds()
    if interval_seconds <= 0:
        return fn()

    start_time = time.time()
    stop_event = threading.Event()

    def _heartbeat_worker() -> None:
        while not stop_event.wait(interval_seconds):
            elapsed = time.time() - start_time
            _log_eval_debug(f"{label} still running after {elapsed:.1f}s")

    _log_eval_debug(f"{label} started")
    heartbeat_thread = threading.Thread(target=_heartbeat_worker, name="gpqa-heartbeat", daemon=True)
    heartbeat_thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=0.1)
        elapsed = time.time() - start_time
        _log_eval_debug(f"{label} finished in {elapsed:.1f}s")

HOTPOTQA_PROMPT_TEMPLATE = (
    "Answer the question using only the passages below.\n"
    "Reply with a short span or yes/no only.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
MEMORY_CONTEXT_QA_PROMPT_TEMPLATE = (
    "Use only the memory context below to answer the question.\n"
    "Reply with a short entity, phrase, number, or sentence only.\n"
    "If the answer is not supported by the context, reply exactly: I don't know.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
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
    for cache_dir in (primary_cache_dir, *FALLBACK_HF_HUB_CACHE_DIRS):
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
        searched = ", ".join(str(path) for path in candidate_hub_cache_dirs(hub_cache_dir))
        raise FileNotFoundError(
            f"Could not find cached {repo_type} file {repo_id}:{filename} in local hub caches: {searched}"
        )
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            cache_dir=str(hub_cache_dir),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base vs Delta-Mem vs LoRA on external benchmarks."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--delta-adapter-dir", type=Path, default=None)
    parser.add_argument("--lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--base-inference-backend",
        default="transformers",
        choices=["transformers"],
        help="Inference backend for base-model evaluation only.",
    )
    parser.add_argument(
        "--lora-inference-backend",
        default="transformers",
        choices=["transformers"],
        help="Inference backend for LoRA evaluation only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets-cache-dir",
        type=Path,
        default=Path(DEFAULT_DATASETS_CACHE_DIR),
    )
    parser.add_argument(
        "--hub-cache-dir",
        type=Path,
        default=DEFAULT_HF_HUB_CACHE_DIR,
        help="Cache root for Hugging Face hub snapshots used by manual benchmark datasets.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require benchmark datasets to come from local caches only.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["hotpotqa", "ifeval", "gpqa_diamond", "memory_agent_bench"],
        choices=[
            "ifeval",
            "gpqa_diamond",
            "hotpotqa",
            "memory_agent_bench",
        ],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch prompts together for stateless generation tasks. Tasks with custom stop criteria still run one-by-one.",
    )
    parser.add_argument("--ifeval-max-new-tokens", type=int, default=1500)
    parser.add_argument("--gpqa-max-new-tokens", type=int, default=8192)
    parser.add_argument("--hotpotqa-max-new-tokens", type=int, default=32)
    parser.add_argument("--memory-agent-bench-max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--memory-agent-bench-use-official-generation-lengths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use official per-source generation_max_length for MemoryAgentBench instead of one shared max_new_tokens budget.",
    )
    parser.add_argument(
        "--memory-agent-bench-use-official-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use source-specific official MemoryAgentBench system/memorize/query templates. Disable to use the older unified fast QA prompt path.",
    )
    parser.add_argument(
        "--gpqa-official-decoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force GPQA Diamond to use official-style greedy decoding instead of the shared benchmark sampling config.",
    )
    parser.add_argument(
        "--hotpotqa-official-decoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force HotpotQA to use official-style greedy decoding instead of the shared benchmark sampling config.",
    )
    parser.add_argument(
        "--eval-do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the same sampling configuration for every benchmark task.",
    )
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-top-k", type=int, default=0)
    parser.add_argument(
        "--memory-agent-bench-splits",
        nargs="+",
        default=[
            "Accurate_Retrieval",
            "Test_Time_Learning",
            "Long_Range_Understanding",
            "Conflict_Resolution",
        ],
    )
    parser.add_argument(
        "--memory-agent-bench-sources",
        nargs="+",
        default=None,
        choices=sorted(MEMORY_AGENT_BENCH_OFFICIAL_SOURCE_CONFIGS),
        help="Optional subset of official MemoryAgentBench sources to evaluate. Defaults to all sources present in the selected splits.",
    )
    parser.add_argument(
        "--external-memory-agent-bench-root",
        type=Path,
        default=DEFAULT_MEMORY_AGENT_BENCH_ROOT,
        help="Checked-out MemoryAgentBench repository root used by official metric helpers.",
    )
    parser.add_argument(
        "--memory-agent-bench-eval-batch-size",
        type=int,
        default=1,
        help="Task-specific batch size for MemoryAgentBench.",
    )
    parser.add_argument(
        "--memory-agent-bench-max-context-chars",
        type=int,
        default=120000,
        help="Optional fast preclip before official-style prompt-budget truncation for MemoryAgentBench.",
    )
    parser.add_argument(
        "--memory-agent-bench-max-questions-per-row-task",
        type=int,
        default=0,
        help="Split very large MemoryAgentBench rows into smaller question shards for better multi-rank load balance. 0 disables sharding.",
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-delta", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/qwen3_benchmark_compare.json"),
    )
    args = parser.parse_args()
    if args.skip_base and args.skip_delta and args.skip_lora:
        parser.error("at least one of base, delta, or lora evaluation must be enabled")
    if not args.skip_delta and args.delta_adapter_dir is None:
        parser.error("--delta-adapter-dir is required unless --skip-delta is set")
    if not args.skip_lora and args.lora_adapter_dir is None:
        parser.error("--lora-adapter-dir is required unless --skip-lora is set")
    return args


def local_indexed_items(items: list[dict], context: DistributedContext) -> list[tuple[int, dict]]:
    return [
        (item_idx, item)
        for item_idx, item in enumerate(items)
        if item_idx % context.world_size == context.rank
    ]


def local_memory_agent_bench_question_tasks(
    items: list[dict],
    context: DistributedContext,
) -> list[tuple[int, dict, dict]]:
    tasks: list[tuple[int, dict, dict]] = []
    for item in items:
        for question_meta in item.get("selected_questions", []):
            eval_index = int(question_meta["eval_index"])
            if eval_index % context.world_size == context.rank:
                tasks.append((eval_index, item, question_meta))
    return tasks


@dataclass(frozen=True)
class MemoryAgentBenchRowTask:
    row_index: int
    item: dict
    question_count: int
    estimated_cost: int
    question_start: int
    question_end: int


def memory_agent_bench_source(item: dict) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("source", "")).strip()


def memory_agent_bench_source_config(item: dict) -> dict[str, int]:
    source = memory_agent_bench_source(item)
    config = MEMORY_AGENT_BENCH_OFFICIAL_SOURCE_CONFIGS.get(source)
    if config is None:
        return {}
    return dict(config)


def memory_agent_bench_effective_max_new_tokens(
    item: dict,
    *,
    default_max_new_tokens: int,
    use_official_generation_lengths: bool,
) -> int:
    if use_official_generation_lengths:
        official_max_new_tokens = int(memory_agent_bench_source_config(item).get("generation_max_length") or 0)
        if official_max_new_tokens > 0:
            return official_max_new_tokens
    return max(1, default_max_new_tokens)



def _memory_agent_bench_metric_display_name(metric_key: str) -> str:
    if metric_key == "f1":
        return "F1-Score"
    if metric_key == "recsys_recall@5":
        return "Recall@5"
    return "Accuracy"



def _memory_agent_bench_source_spec(source: str) -> dict[str, str]:
    source = str(source or "")
    if source == "ruler_qa1_197K":
        metric_key = "accuracy"
        dataset_name = "SH-Doc QA"
    elif source == "ruler_qa2_421K":
        metric_key = "accuracy"
        dataset_name = "MH-Doc QA"
    elif source.startswith("longmemeval_"):
        metric_key = "accuracy"
        dataset_name = "LongMemEval (S*)"
    elif source.startswith("eventqa_"):
        metric_key = "accuracy"
        dataset_name = "EventQA"
    elif source.startswith("icl_banking77_"):
        metric_key = "accuracy"
        dataset_name = "BANKING77"
    elif source.startswith("icl_clinic150_"):
        metric_key = "accuracy"
        dataset_name = "CLINIC150"
    elif source.startswith("icl_nlu_"):
        metric_key = "accuracy"
        dataset_name = "NLU"
    elif source.startswith("icl_trec_coarse_"):
        metric_key = "accuracy"
        dataset_name = "TREC Coarse"
    elif source.startswith("icl_trec_fine_"):
        metric_key = "accuracy"
        dataset_name = "TREC Fine"
    elif source.startswith("recsys_"):
        metric_key = "recsys_recall@5"
        dataset_name = "Movie Recommendation"
    elif source == "infbench_sum_eng_shots2":
        metric_key = "f1"
        dataset_name = "InfinityBench-Sum"
    elif source == "detective_qa":
        metric_key = "accuracy"
        dataset_name = "Detective QA"
    elif source.startswith("factconsolidation_sh_"):
        metric_key = "accuracy"
        dataset_name = "FactConsolidation-SH"
    elif source.startswith("factconsolidation_mh_"):
        metric_key = "accuracy"
        dataset_name = "FactConsolidation-MH"
    else:
        metric_key = "accuracy"
        dataset_name = source or "unknown"
    return {
        "dataset_name": dataset_name,
        "metric_key": metric_key,
        "metric": _memory_agent_bench_metric_display_name(metric_key),
    }



def memory_agent_bench_primary_metric_name(source: str) -> str:
    return _memory_agent_bench_source_spec(source)["metric_key"]


def _memory_agent_bench_row_estimated_cost(
    item: dict,
    *,
    default_max_new_tokens: int,
    use_official_generation_lengths: bool,
) -> int:
    question_count = len(item.get("selected_questions", []))
    context_chars = len(str(item.get("context", "")))
    context_token_estimate = max(1, context_chars // 4)
    generation_token_estimate = question_count * memory_agent_bench_effective_max_new_tokens(
        item,
        default_max_new_tokens=default_max_new_tokens,
        use_official_generation_lengths=use_official_generation_lengths,
    )
    return context_token_estimate + generation_token_estimate


def local_memory_agent_bench_row_tasks(
    items: list[dict],
    context: DistributedContext,
    *,
    default_max_new_tokens: int,
    use_official_generation_lengths: bool,
    max_questions_per_row_task: int = 0,
) -> list[MemoryAgentBenchRowTask]:
    candidates: list[MemoryAgentBenchRowTask] = []
    for row_index, item in enumerate(items):
        selected_questions = list(item.get("selected_questions", []))
        if not selected_questions:
            continue
        row_stride = max(1, int(max_questions_per_row_task)) if int(max_questions_per_row_task) > 0 else len(selected_questions)
        for question_start in range(0, len(selected_questions), row_stride):
            question_end = min(len(selected_questions), question_start + row_stride)
            task_item = dict(item)
            task_item["selected_questions"] = selected_questions[question_start:question_end]
            candidates.append(
                MemoryAgentBenchRowTask(
                    row_index=row_index,
                    item=task_item,
                    question_count=question_end - question_start,
                    estimated_cost=_memory_agent_bench_row_estimated_cost(
                        task_item,
                        default_max_new_tokens=default_max_new_tokens,
                        use_official_generation_lengths=use_official_generation_lengths,
                    ),
                    question_start=question_start,
                    question_end=question_end,
                )
            )
    if not context.enabled:
        return candidates

    rank_loads = [0] * context.world_size
    task_to_rank: dict[tuple[int, int, int], int] = {}
    for row_task in sorted(
        candidates,
        key=lambda task: (
            -task.estimated_cost,
            -task.question_count,
            str(task.item.get("row_id", "")),
            task.row_index,
            task.question_start,
        ),
    ):
        target_rank = min(range(context.world_size), key=lambda rank: (rank_loads[rank], rank))
        task_key = (row_task.row_index, row_task.question_start, row_task.question_end)
        task_to_rank[task_key] = target_rank
        rank_loads[target_rank] += row_task.estimated_cost

    return [
        row_task
        for row_task in candidates
        if task_to_rank[(row_task.row_index, row_task.question_start, row_task.question_end)] == context.rank
    ]


def load_gpqa_diamond(
    *,
    cache_dir: Path,
    max_samples: int | None,
    seed: int,
    local_files_only: bool,
) -> list[dict]:
    dataset = load_dataset_cached(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        split="train",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))
    return [dict(row) for row in dataset]

def load_ifeval(
    *,
    cache_dir: Path,
    max_samples: int | None,
    seed: int,
    local_files_only: bool,
) -> list[dict]:
    dataset = load_dataset_cached(
        "google/IFEval",
        split="train",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))
    return [dict(row) for row in dataset]

def load_hotpotqa(
    *,
    cache_dir: Path,
    max_samples: int | None,
    seed: int,
    local_files_only: bool,
) -> list[dict]:
    dataset = load_dataset_cached(
        "hotpotqa/hotpot_qa",
        name="distractor",
        split="validation",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))
    return [dict(row) for row in dataset]


def clip_context_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[... context truncated ...]\n\n"
    if max_chars <= len(marker) + 32:
        return text[-max_chars:]
    head_chars = max(1, (max_chars - len(marker)) // 3)
    tail_chars = max(1, max_chars - len(marker) - head_chars)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def normalize_answer_aliases(values: list[object] | object) -> list[str]:
    raw_values = values if isinstance(values, list) else [values]
    aliases: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value).strip()
        if not text:
            continue
        normalized = normalize_qa_span(text)
        if not normalized or normalized in seen:
            continue
        aliases.append(text)
        seen.add(normalized)
    return aliases


def memory_qa_is_correct(prediction: str, aliases: list[str]) -> bool:
    normalized_prediction = normalize_qa_span(prediction)
    normalized_first_line = normalize_qa_span(extract_first_line(prediction))
    if not normalized_prediction and not normalized_first_line:
        return False
    for alias in aliases:
        normalized_alias = normalize_qa_span(alias)
        if not normalized_alias:
            continue
        if normalized_alias == normalized_first_line or normalized_alias == normalized_prediction:
            return True
        if normalized_alias in normalized_prediction:
            return True
    return False


def qa_alias_max_f1(prediction: str, aliases: list[str]) -> float:
    candidates = [prediction, extract_first_line(prediction)]
    best = 0.0
    for candidate in candidates:
        if not str(candidate).strip():
            continue
        for alias in aliases:
            if not str(alias).strip():
                continue
            best = max(best, hotpotqa_f1(str(candidate), str(alias)))
    return best


def _metadata_list_value(metadata: dict, key: str, index: int, default=None):
    values = metadata.get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return default


def load_memory_agent_bench(
    *,
    cache_dir: Path,
    hub_cache_dir: Path,
    splits: list[str],
    sources: list[str] | None,
    max_samples: int | None,
    seed: int,
    local_files_only: bool,
) -> list[dict]:
    all_rows: list[dict] = []
    flat_refs: list[tuple[int, int]] = []
    selected_sources = None if not sources else set(sources)
    for split in splits:
        parquet_path = resolve_hub_file(
            repo_id="ai-hyz/MemoryAgentBench",
            repo_type="dataset",
            filename=f"data/{split}-00000-of-00001.parquet",
            hub_cache_dir=hub_cache_dir,
            local_files_only=local_files_only,
        )
        dataset = load_dataset_cached(
            "parquet",
            data_files=str(parquet_path),
            split="train",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        for row_idx, row in enumerate(dataset):
            materialized = dict(row)
            metadata = dict(materialized.get("metadata") or {})
            source = str(metadata.get("source", "")).strip()
            if selected_sources is not None and source not in selected_sources:
                continue
            questions = [str(value).strip() for value in materialized.get("questions", []) or []]
            answers = list(materialized.get("answers", []) or [])
            all_rows.append(
                {
                    "split": split,
                    "row_id": f"{split}:{row_idx}",
                    "context": str(materialized.get("context", "")),
                    "questions": questions,
                    "answers": answers,
                    "metadata": metadata,
                    "source": source,
                    "official_source_config": memory_agent_bench_source_config({"metadata": metadata}),
                }
            )
            current_row_index = len(all_rows) - 1
            for question_idx in range(len(questions)):
                flat_refs.append((current_row_index, question_idx))

    if max_samples is not None and len(flat_refs) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(flat_refs)
        flat_refs = flat_refs[:max_samples]

    row_to_selected: dict[int, list[int]] = defaultdict(list)
    for eval_index, (row_index, question_index) in enumerate(flat_refs):
        row_to_selected[row_index].append(question_index)

    items: list[dict] = []
    running_index = 0
    for row_index, row in enumerate(all_rows):
        selected_question_indices = row_to_selected.get(row_index)
        if not selected_question_indices:
            continue
        selected_question_indices = sorted(selected_question_indices)
        selected_questions: list[dict] = []
        for question_index in selected_question_indices:
            metadata = row["metadata"]
            selected_questions.append(
                {
                    "eval_index": running_index,
                    "question_index": question_index,
                    "question": row["questions"][question_index],
                    "answer_raw": row["answers"][question_index] if question_index < len(row["answers"]) else [],
                    "answer_aliases": normalize_answer_aliases(
                        row["answers"][question_index] if question_index < len(row["answers"]) else []
                    ),
                    "question_id": _metadata_list_value(metadata, "question_ids", question_index),
                    "question_type": _metadata_list_value(metadata, "question_types", question_index),
                    "question_date": _metadata_list_value(metadata, "question_dates", question_index),
                    "previous_event": _metadata_list_value(metadata, "previous_events", question_index),
                    "qa_pair_id": _metadata_list_value(metadata, "qa_pair_ids", question_index),
                }
            )
            running_index += 1
        items.append(
            {
                **row,
                "selected_questions": selected_questions,
            }
        )
    return items


def build_hotpotqa_context(item: dict) -> str:
    parts: list[str] = []
    context = item["context"]
    for idx, (title, sentences) in enumerate(zip(context["title"], context["sentences"]), start=1):
        text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip()).strip()
        if not text:
            continue
        parts.append(f"Passage {idx} - {title}:\n{text}")
    if not parts:
        return "No passages provided."
    return "\n\n".join(parts)


def normalize_qa_span(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_hotpotqa_answer(text: str) -> str:
    return normalize_qa_span(text)


def hotpotqa_exact_match(prediction: str, gold: str) -> bool:
    return normalize_hotpotqa_answer(prediction) == normalize_hotpotqa_answer(gold)


def hotpotqa_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_hotpotqa_answer(prediction).split()
    gold_tokens = normalize_hotpotqa_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def stable_choice_order(record_id: str, seed: int) -> list[int]:
    digest = hashlib.md5(f"{record_id}:{seed}".encode("utf-8")).hexdigest()
    order = [0, 1, 2, 3]
    rng = random.Random(int(digest[:8], 16))
    rng.shuffle(order)
    return order


def build_gpqa_prompt(item: dict, *, seed: int) -> tuple[str, str]:
    answers = [
        item["Correct Answer"],
        item["Incorrect Answer 1"],
        item["Incorrect Answer 2"],
        item["Incorrect Answer 3"],
    ]
    order = stable_choice_order(str(item["Record ID"]), seed)
    letters = ["A", "B", "C", "D"]
    shuffled = [answers[idx] for idx in order]
    correct_letter = letters[order.index(0)]
    options = "\n".join(f"{letter}. {answer.strip()}" for letter, answer in zip(letters, shuffled))
    prompt = (
        "Answer the following multiple-choice question.\n"
        "Please show your choice in the `answer` field with only the choice letter, "
        "for example: {\"answer\": \"C\"}.\n\n"
        f"Question: {item['Question'].strip()}\n\n"
        f"{options}"
    )
    return prompt, correct_letter


def extract_gpqa_letter(text: str) -> str:
    # Conservative GPQA extraction.
    # Prefer explicit final-answer markers, but still allow obvious option-style
    # outputs such as `C`, `C.`, or `C ...` on the first/last non-empty line.
    # Do not fall back to scanning the whole reasoning body for the first
    # standalone A/B/C/D, which can spuriously match symbols or option mentions.
    patterns = [
        r'```json\s*\{[^`]*?"answer"\s*:\s*"([ABCD])"[^`]*\}\s*```',
        r'"answer"\s*:\s*"([ABCD])"',
        r'\\boxed\{\s*(?:\\text\{)?\s*([ABCD])\s*\}?\s*\}',
        r'(?:final answer|correct answer|answer)\s*[:：]\s*\**\s*([ABCD])\b',
        r'(?:final answer|correct answer)\s*(?:is)?\s*\**\s*([ABCD])\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return str(matches[-1]).upper()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    line_only_pattern = re.compile(r'^\s*[\*\(\[]*([ABCD])[\*\)\]\.\:]*\s*$', flags=re.IGNORECASE)
    line_start_pattern = re.compile(r'^\s*([ABCD])(?:[\.)\:]|\s{2,}|$)', flags=re.IGNORECASE)

    for candidate in (lines[0], lines[-1]):
        match = line_only_pattern.match(candidate)
        if match:
            return match.group(1).upper()
        match = line_start_pattern.match(candidate)
        if match:
            return match.group(1).upper()
    return ""


def extract_first_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text.strip()


def shared_generation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "do_sample": args.eval_do_sample,
        "temperature": args.eval_temperature if args.eval_do_sample else None,
        "top_p": args.eval_top_p if args.eval_do_sample else None,
        "top_k": args.eval_top_k if args.eval_do_sample else None,
    }


def greedy_generation_kwargs() -> dict[str, object]:
    return {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }


def task_generation_kwargs(task_name: str, args: argparse.Namespace) -> dict[str, object]:
    shared_generation = shared_generation_kwargs(args)
    if task_name == "ifeval":
        return {
            "max_new_tokens": args.ifeval_max_new_tokens,
            "use_chat_template": True,
            **greedy_generation_kwargs(),
        }
    if task_name == "gpqa_diamond":
        generation = greedy_generation_kwargs() if args.gpqa_official_decoding else shared_generation
        return {
            "max_new_tokens": args.gpqa_max_new_tokens,
            "use_chat_template": True,
            **generation,
        }
    if task_name == "hotpotqa":
        generation = greedy_generation_kwargs() if args.hotpotqa_official_decoding else shared_generation
        return {
            "max_new_tokens": args.hotpotqa_max_new_tokens,
            "use_chat_template": True,
            **generation,
        }
    if task_name == "memory_agent_bench":
        # Match the Qwen3-8B zero-temperature decode regime used for MemoryAgentBench.
        return {
            "max_new_tokens": args.memory_agent_bench_max_new_tokens,
            "use_chat_template": True,
            "do_sample": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
        }
    raise ValueError(f"Unsupported task: {task_name}")


def _chat_template_input_ids(tokenizer, messages: list[dict[str, str]]) -> torch.Tensor:
    tokenized = apply_project_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    return tokenized


def _delta_write_granularity(model) -> str | None:
    for module in model.modules():
        granularity = getattr(module, "memory_write_granularity", None)
        if granularity is not None:
            return str(granularity)
    return None


def _requires_delta_write_span_generation(model, reset_delta_state: bool) -> bool:
    return reset_delta_state and _delta_write_granularity(model) in {"message_mean", "sentence_mean"}


def _manual_prompt_tokens_and_write_ids(
    tokenizer,
    prompt: str,
    *,
    use_chat_template: bool,
    include_sentence_ids: bool,
) -> tuple[list[int], list[int], list[int]]:
    if use_chat_template:
        return tokenize_chat_messages_with_write_span_ids(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            include_sentence_ids=include_sentence_ids,
        )
    input_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    message_ids = [0] * len(input_ids)
    sentence_ids = [-1] * len(input_ids)
    if include_sentence_ids:
        sentence_ids = [0] * len(input_ids)
    return input_ids, message_ids, sentence_ids


def generate_chat_answer_with_delta_write_spans(
    model,
    tokenizer,
    device: str,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    reset_delta_state: bool,
    use_chat_template: bool,
) -> str:
    if reset_delta_state:
        reset_delta_mem_states(model)
    granularity = _delta_write_granularity(model)
    include_sentence_ids = granularity == "sentence_mean"
    input_id_list, message_id_list, sentence_id_list = _manual_prompt_tokens_and_write_ids(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
        include_sentence_ids=include_sentence_ids,
    )
    input_ids = torch.tensor([input_id_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    write_message_ids = torch.tensor([message_id_list], dtype=torch.long, device=device)
    write_sentence_ids = torch.tensor([sentence_id_list], dtype=torch.long, device=device)

    set_delta_mem_write_message_ids(model, write_message_ids)
    set_delta_mem_write_sentence_ids(model, write_sentence_ids)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
    finally:
        set_delta_mem_write_message_ids(model, None)
        set_delta_mem_write_sentence_ids(model, None)

    past_key_values = _output_value(outputs, "past_key_values")
    next_token_logits = _output_value(outputs, "logits")[:, -1, :]
    eos_token_ids = _generation_eos_token_ids(model, tokenizer)
    generated_token_ids: list[int] = []
    assistant_message_id = 1
    assistant_sentence_id = (
        max([sid for sid in sentence_id_list if sid >= 0], default=-1) + 1
        if include_sentence_ids
        else -1
    )
    for _ in range(max_new_tokens):
        next_token = _sample_next_token_from_logits(
            next_token_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_id = int(next_token[0, 0].item())
        generated_token_ids.append(token_id)
        if eos_token_ids and token_id in eos_token_ids:
            break
        token_message_ids = torch.full_like(next_token, assistant_message_id)
        token_sentence_ids = torch.full_like(next_token, assistant_sentence_id)
        set_delta_mem_write_message_ids(model, token_message_ids)
        set_delta_mem_write_sentence_ids(model, token_sentence_ids)
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        finally:
            set_delta_mem_write_message_ids(model, None)
            set_delta_mem_write_sentence_ids(model, None)
        past_key_values = _output_value(outputs, "past_key_values")
        next_token_logits = _output_value(outputs, "logits")[:, -1, :]
    return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def infer_model_context_window(model, tokenizer) -> int:
    direct_max_model_len = getattr(model, "max_model_len", None)
    if isinstance(direct_max_model_len, int) and direct_max_model_len > 0 and direct_max_model_len < 10**7:
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
            if isinstance(value, int) and value > 0 and value < 10**7:
                return value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and value > 0 and value < 10**7:
        return value
    return 32768


def _estimate_chars_per_token(tokenizer, text: str, *, sample_chars: int = 8192) -> float:
    sample = text[: min(len(text), sample_chars)]
    if not sample:
        return 1.0
    token_count = len(tokenizer.encode(sample, add_special_tokens=False))
    return max(1.0, len(sample) / max(1, token_count))


def _rough_preclip_context_for_tokenizer(
    context_text: str,
    *,
    tokenizer,
    max_prompt_tokens: int,
) -> str:
    if not context_text:
        return context_text
    chars_per_token = _estimate_chars_per_token(tokenizer, context_text)
    rough_char_cap = int(max_prompt_tokens * chars_per_token * 1.25)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit > 0 and tokenizer_limit < 10**7:
        rough_char_cap = min(rough_char_cap, int(tokenizer_limit * chars_per_token * 0.9))
    rough_char_cap = max(32768, rough_char_cap)
    if len(context_text) <= rough_char_cap:
        return context_text
    return clip_context_text(context_text, rough_char_cap)


def prompt_token_count(
    tokenizer,
    prompt: str,
    *,
    use_chat_template: bool,
) -> int:
    if use_chat_template:
        input_ids = _chat_template_input_ids(tokenizer, [{"role": "user", "content": prompt}])
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return int(input_ids.shape[-1])


def render_clipped_context_from_tokens(
    original_text: str,
    context_token_ids: list[int],
    *,
    tokenizer,
    keep_tokens: int,
    marker: str,
    marker_token_count: int,
) -> str:
    if keep_tokens <= 0:
        return ""
    if len(context_token_ids) <= keep_tokens:
        return original_text
    if keep_tokens <= marker_token_count + 2:
        return tokenizer.decode(context_token_ids[-keep_tokens:], skip_special_tokens=True).strip()
    head_tokens = max(1, (keep_tokens - marker_token_count) // 3)
    tail_tokens = max(1, keep_tokens - marker_token_count - head_tokens)
    head_text = tokenizer.decode(context_token_ids[:head_tokens], skip_special_tokens=True).rstrip()
    tail_text = tokenizer.decode(context_token_ids[-tail_tokens:], skip_special_tokens=True).lstrip()
    return head_text + marker + tail_text


def truncate_context_text_fast(
    context_text: str,
    *,
    tokenizer,
    max_prompt_tokens: int,
    prompt_overhead_tokens: int,
    max_context_chars: int = 0,
    keep: str = "head_tail",
) -> str:
    if max_context_chars > 0 and len(context_text) > max_context_chars:
        if keep == "tail":
            context_text = context_text[-max_context_chars:]
        else:
            context_text = clip_context_text(context_text, max_context_chars)
    if not context_text:
        return context_text

    available_context_tokens = max(1, max_prompt_tokens - prompt_overhead_tokens - 32)
    context_text = _rough_preclip_context_for_tokenizer(
        context_text,
        tokenizer=tokenizer,
        max_prompt_tokens=available_context_tokens,
    )
    context_token_ids = tokenizer.encode(context_text, add_special_tokens=False)
    if len(context_token_ids) <= available_context_tokens:
        return context_text

    if keep == "tail":
        return tokenizer.decode(
            context_token_ids[-available_context_tokens:],
            skip_special_tokens=True,
        ).strip()

    marker = "\n\n[... context truncated ...]\n\n"
    marker_token_count = len(tokenizer.encode(marker, add_special_tokens=False))
    return render_clipped_context_from_tokens(
        context_text,
        context_token_ids,
        tokenizer=tokenizer,
        keep_tokens=available_context_tokens,
        marker=marker,
        marker_token_count=marker_token_count,
    )


def clip_context_text_to_model_limit(
    context_text: str,
    *,
    tokenizer,
    prompt_builder,
    use_chat_template: bool,
    model_context_window: int,
    max_new_tokens: int,
    max_context_chars: int = 0,
    keep: str = "head_tail",
) -> str:
    if not context_text:
        return context_text

    max_prompt_tokens = max(1, model_context_window - max_new_tokens)
    prompt_overhead_tokens = prompt_token_count(
        tokenizer,
        prompt_builder(""),
        use_chat_template=use_chat_template,
    )
    clipped_context = truncate_context_text_fast(
        context_text,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        prompt_overhead_tokens=prompt_overhead_tokens,
        max_context_chars=max_context_chars,
        keep=keep,
    )
    prompt_tokens = prompt_token_count(
        tokenizer,
        prompt_builder(clipped_context),
        use_chat_template=use_chat_template,
    )
    if prompt_tokens <= max_prompt_tokens:
        return clipped_context

    overflow = prompt_tokens - max_prompt_tokens + 32
    context_token_ids = tokenizer.encode(clipped_context, add_special_tokens=False)
    keep_tokens = max(1, len(context_token_ids) - overflow)
    if keep == "tail":
        return tokenizer.decode(
            context_token_ids[-keep_tokens:],
            skip_special_tokens=True,
        ).strip()

    marker = "\n\n[... context truncated ...]\n\n"
    marker_token_count = len(tokenizer.encode(marker, add_special_tokens=False))
    return render_clipped_context_from_tokens(
        clipped_context,
        context_token_ids,
        tokenizer=tokenizer,
        keep_tokens=keep_tokens,
        marker=marker,
        marker_token_count=marker_token_count,
    )


def generate_chat_answer(
    model,
    tokenizer,
    device: str,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    reset_delta_state: bool,
    use_chat_template: bool,
) -> str:
    if _requires_delta_write_span_generation(model, reset_delta_state):
        return generate_chat_answer_with_delta_write_spans(
            model,
            tokenizer,
            device,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            reset_delta_state=reset_delta_state,
            use_chat_template=use_chat_template,
        )

    if reset_delta_state:
        reset_delta_mem_states(model)
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        input_ids = _chat_template_input_ids(tokenizer, messages).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generation_config = copy.deepcopy(getattr(model, "generation_config", None))
    zero_temperature_sampling = do_sample and temperature is not None and float(temperature) <= 0.0
    effective_do_sample = False if zero_temperature_sampling else do_sample
    generate_kwargs: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": input_ids.new_ones(input_ids.shape),
    }
    if generation_config is not None:
        generation_config.do_sample = effective_do_sample
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = True
        if effective_do_sample:
            generation_config.temperature = temperature
            generation_config.top_p = top_p
            generation_config.top_k = top_k
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        generate_kwargs["generation_config"] = generation_config
    else:
        generate_kwargs["do_sample"] = effective_do_sample
        generate_kwargs["max_new_tokens"] = max_new_tokens
        generate_kwargs["use_cache"] = True
        if effective_do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            generate_kwargs["top_k"] = top_k
    outputs = model.generate(**generate_kwargs)
    generated_ids = outputs[0][input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_chat_answers_batched(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    reset_delta_state: bool,
    use_chat_template: bool,
) -> list[str]:
    if not prompts:
        return []
    if _requires_delta_write_span_generation(model, reset_delta_state):
        return [
            generate_chat_answer_with_delta_write_spans(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                reset_delta_state=reset_delta_state,
                use_chat_template=use_chat_template,
            )
            for prompt in prompts
        ]

    if reset_delta_state:
        reset_delta_mem_states(model)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        if use_chat_template:
            rendered_prompts = [
                apply_project_chat_template(
                    tokenizer,
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            tokenized = tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        else:
            tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = tokenized.input_ids.to(device)
    attention_mask = getattr(tokenized, "attention_mask", None)
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)
    else:
        attention_mask = attention_mask.to(device)

    generation_config = copy.deepcopy(getattr(model, "generation_config", None))
    zero_temperature_sampling = do_sample and temperature is not None and float(temperature) <= 0.0
    effective_do_sample = False if zero_temperature_sampling else do_sample
    generate_kwargs: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if generation_config is not None:
        generation_config.do_sample = effective_do_sample
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = True
        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if effective_do_sample:
            generation_config.temperature = temperature
            generation_config.top_p = top_p
            generation_config.top_k = top_k
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        generate_kwargs["generation_config"] = generation_config
    else:
        generate_kwargs["do_sample"] = effective_do_sample
        generate_kwargs["max_new_tokens"] = max_new_tokens
        generate_kwargs["use_cache"] = True
        if effective_do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            generate_kwargs["top_k"] = top_k
        if tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    outputs = model.generate(**generate_kwargs)
    prompt_width = input_ids.shape[1]
    generated_ids = outputs[:, prompt_width:]
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated_ids]


def _coerce_generation_settings(task_kwargs: dict[str, object]) -> dict[str, object]:
    return {
        "max_new_tokens": int(task_kwargs["max_new_tokens"]),
        "do_sample": bool(task_kwargs["do_sample"]),
        "temperature": None if task_kwargs.get("temperature") is None else float(task_kwargs["temperature"]),
        "top_p": None if task_kwargs.get("top_p") is None else float(task_kwargs["top_p"]),
        "top_k": None if task_kwargs.get("top_k") is None else int(task_kwargs["top_k"]),
        "use_chat_template": bool(task_kwargs["use_chat_template"]),
    }


def _generate_predictions_for_prompts(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    *,
    generation_settings: dict[str, object],
    reset_delta_state: bool,
    batch_size: int,
) -> list[str]:
    effective_batch_size = max(1, batch_size)
    if effective_batch_size <= 1:
        return [
            generate_chat_answer(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=int(generation_settings["max_new_tokens"]),
                do_sample=bool(generation_settings["do_sample"]),
                temperature=None if generation_settings["temperature"] is None else float(generation_settings["temperature"]),
                top_p=None if generation_settings["top_p"] is None else float(generation_settings["top_p"]),
                top_k=None if generation_settings["top_k"] is None else int(generation_settings["top_k"]),
                reset_delta_state=reset_delta_state,
                use_chat_template=bool(generation_settings["use_chat_template"]),
            )
            for prompt in prompts
        ]

    predictions: list[str] = []
    for start in range(0, len(prompts), effective_batch_size):
        predictions.extend(
            generate_chat_answers_batched(
                model,
                tokenizer,
                device,
                prompts[start : start + effective_batch_size],
                max_new_tokens=int(generation_settings["max_new_tokens"]),
                do_sample=bool(generation_settings["do_sample"]),
                temperature=None if generation_settings["temperature"] is None else float(generation_settings["temperature"]),
                top_p=None if generation_settings["top_p"] is None else float(generation_settings["top_p"]),
                top_k=None if generation_settings["top_k"] is None else int(generation_settings["top_k"]),
                reset_delta_state=reset_delta_state,
                use_chat_template=bool(generation_settings["use_chat_template"]),
            )
        )
    return predictions


def _tokenize_prompt_to_token_list(
    tokenizer,
    prompt: str,
    *,
    use_chat_template: bool,
) -> list[int]:
    if use_chat_template:
        input_ids = _chat_template_input_ids(tokenizer, [{"role": "user", "content": prompt}])
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return input_ids.squeeze(0).tolist()


def _supports_manual_shared_prefix_decode(model) -> bool:
    return bool(getattr(model, "supports_manual_shared_prefix_decode", True))


def _longest_common_token_prefix_length(token_sequences: list[list[int]]) -> int:
    if not token_sequences:
        return 0
    shortest = min(len(token_ids) for token_ids in token_sequences)
    prefix_length = 0
    while prefix_length < shortest:
        candidate = token_sequences[0][prefix_length]
        if any(token_ids[prefix_length] != candidate for token_ids in token_sequences[1:]):
            break
        prefix_length += 1
    return prefix_length


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


def _clone_nested_tensors(obj):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_nested_tensors(item) for item in obj)
    if isinstance(obj, list):
        return [_clone_nested_tensors(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _clone_nested_tensors(value) for key, value in obj.items()}
    if hasattr(obj, "__dict__"):
        cloned = copy.copy(obj)
        for key, value in vars(obj).items():
            setattr(cloned, key, _clone_nested_tensors(value))
        return cloned
    return obj


def _output_value(outputs, key: str):
    if hasattr(outputs, key):
        return getattr(outputs, key)
    return outputs[key]


def _generation_eos_token_ids(model, tokenizer) -> set[int]:
    eos_token_ids: set[int] = set()
    generation_config = getattr(model, "generation_config", None)
    generation_eos = getattr(generation_config, "eos_token_id", None)
    if isinstance(generation_eos, int):
        eos_token_ids.add(generation_eos)
    elif isinstance(generation_eos, (list, tuple, set)):
        eos_token_ids.update(int(token_id) for token_id in generation_eos)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, int):
        eos_token_ids.add(tokenizer_eos)
    return eos_token_ids


def _sample_next_token_from_logits(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    resolved_temperature = float(temperature if temperature is not None else 1.0)
    if resolved_temperature <= 0.0:
        # Treat zero-temperature sampling as the deterministic limit of sampling.
        return logits.argmax(dim=-1, keepdim=True)
    filtered = logits / resolved_temperature
    if top_k is not None and 0 < top_k < filtered.size(-1):
        top_values = torch.topk(filtered, k=top_k, dim=-1).values
        threshold = top_values[..., -1:].expand_as(filtered)
        filtered = filtered.masked_fill(filtered < threshold, torch.finfo(filtered.dtype).min)
    resolved_top_p = float(top_p if top_p is not None else 1.0)
    if 0.0 < resolved_top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_remove = cumulative_probs > resolved_top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_remove, torch.finfo(sorted_logits.dtype).min)
        filtered = torch.full_like(filtered, torch.finfo(filtered.dtype).min)
        filtered.scatter_(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _shared_prefix_snapshot_device(prefix_token_count: int, *, device: str, keep_on_gpu: bool = False) -> str:
    if not device.startswith("cuda"):
        return device
    # Prompt-learning models such as Prefix Tuning become PCIe-bound if we offload the
    # shared-prefix cache snapshot to CPU and move it back for every question.
    if keep_on_gpu:
        return device
    # Large prefixes can double KV memory if we keep both the master snapshot and the working copy on-GPU.
    return "cpu" if prefix_token_count >= 32768 else device


def _is_prompt_learning_model(model) -> bool:
    peft_config = getattr(model, "active_peft_config", None)
    return bool(peft_config is not None and getattr(peft_config, "is_prompt_learning", False))


def _prompt_learning_prefill(model, input_ids: torch.Tensor):
    peft_config = getattr(model, "active_peft_config", None)
    if peft_config is None:
        raise ValueError("Prompt-learning prefill requested for a non-PEFT model")
    prompt_cache = model.get_prompt(
        batch_size=input_ids.shape[0],
        max_cache_len=input_ids.shape[1] + int(peft_config.num_virtual_tokens),
    )
    prefix_attention_mask = torch.ones(
        input_ids.shape[0],
        int(peft_config.num_virtual_tokens),
        dtype=torch.long,
        device=input_ids.device,
    )
    input_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    attention_mask = torch.cat((prefix_attention_mask, input_attention_mask), dim=1)
    return model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=prompt_cache,
        use_cache=True,
        return_dict=True,
    )


def _shared_prefix_forward(
    model,
    *,
    input_ids: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor | None = None,
):
    if _is_prompt_learning_model(model):
        return model.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    if hasattr(model, "_reasoner_forward_call"):
        forward_kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "use_cache": True,
            "return_dict": True,
        }
        if past_key_values is not None:
            forward_kwargs["past_key_values"] = past_key_values
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        return model._reasoner_forward_call(**forward_kwargs)
    return model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )


def _generate_from_prompt_suffix(
    model,
    tokenizer,
    device: str,
    prompt_suffix_token_ids: list[int],
    past_key_values,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    initial_next_token_logits: torch.Tensor | None = None,
) -> str:
    next_token_logits = initial_next_token_logits
    working_past_key_values = past_key_values
    with torch.inference_mode():
        if prompt_suffix_token_ids:
            suffix_input_ids = torch.tensor([prompt_suffix_token_ids], dtype=torch.long, device=device)
            outputs = _shared_prefix_forward(
                model,
                input_ids=suffix_input_ids,
                past_key_values=working_past_key_values,
            )
            working_past_key_values = _output_value(outputs, "past_key_values")
            next_token_logits = _output_value(outputs, "logits")[:, -1, :]
        elif next_token_logits is None:
            raise ValueError("shared-prefix generation requires suffix tokens or initial logits")

        eos_token_ids = _generation_eos_token_ids(model, tokenizer)
        generated_token_ids: list[int] = []
        for _ in range(max_new_tokens):
            next_token = _sample_next_token_from_logits(
                next_token_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            token_id = int(next_token[0, 0].item())
            generated_token_ids.append(token_id)
            if eos_token_ids and token_id in eos_token_ids:
                break
            outputs = _shared_prefix_forward(
                model,
                input_ids=next_token,
                past_key_values=working_past_key_values,
            )
            working_past_key_values = _output_value(outputs, "past_key_values")
            next_token_logits = _output_value(outputs, "logits")[:, -1, :]
    return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def _generate_predictions_with_shared_prefix(
    model,
    tokenizer,
    device: str,
    prompt_token_lists: list[list[int]],
    *,
    generation_settings: dict[str, object],
    reset_delta_state: bool,
) -> list[str]:
    prefix_token_count = _longest_common_token_prefix_length(prompt_token_lists)
    if prefix_token_count <= 0:
        raise ValueError("shared-prefix generation requires a non-empty common prefix")

    prefix_input_ids = torch.tensor(
        [prompt_token_lists[0][:prefix_token_count]],
        dtype=torch.long,
        device=device,
    )
    if reset_delta_state:
        reset_delta_mem_states(model)
    prefix_attention_mask = torch.ones_like(prefix_input_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        if _is_prompt_learning_model(model):
            prefix_outputs = _prompt_learning_prefill(model, prefix_input_ids)
        else:
            prefix_outputs = _shared_prefix_forward(
                model,
                input_ids=prefix_input_ids,
                past_key_values=None,
                attention_mask=prefix_attention_mask,
            )
    prefix_past_key_values = _output_value(prefix_outputs, "past_key_values")
    prefix_last_logits = _output_value(prefix_outputs, "logits")[:, -1, :].detach().cpu()
    prefix_delta_state = get_delta_mem_online_state(model) if reset_delta_state else None

    snapshot_device = _shared_prefix_snapshot_device(
        prefix_token_count,
        device=device,
        keep_on_gpu=_is_prompt_learning_model(model),
    )
    if snapshot_device == device:
        prefix_cache_snapshot = _clone_nested_tensors(prefix_past_key_values)
    else:
        prefix_cache_snapshot = _move_nested_tensors(prefix_past_key_values, snapshot_device)
    del prefix_past_key_values
    del prefix_outputs
    if snapshot_device != device:
        maybe_empty_cache(device)

    predictions: list[str] = []
    for prompt_token_ids in prompt_token_lists:
        if reset_delta_state:
            reset_delta_mem_states(model)
            if prefix_delta_state:
                load_delta_mem_online_state(model, prefix_delta_state)
        if snapshot_device == device:
            working_past_key_values = _clone_nested_tensors(prefix_cache_snapshot)
        else:
            working_past_key_values = _move_nested_tensors(prefix_cache_snapshot, device)
        predictions.append(
            _generate_from_prompt_suffix(
                model,
                tokenizer,
                device,
                prompt_token_ids[prefix_token_count:],
                working_past_key_values,
                max_new_tokens=int(generation_settings["max_new_tokens"]),
                do_sample=bool(generation_settings["do_sample"]),
                temperature=None if generation_settings["temperature"] is None else float(generation_settings["temperature"]),
                top_p=None if generation_settings["top_p"] is None else float(generation_settings["top_p"]),
                top_k=None if generation_settings["top_k"] is None else int(generation_settings["top_k"]),
                initial_next_token_logits=prefix_last_logits,
            )
        )
    if reset_delta_state:
        reset_delta_mem_states(model)
    return predictions


def evaluate_ifeval(
    *,
    model,
    tokenizer,
    device: str,
    indexed_items: list[tuple[int, dict]],
    task_kwargs: dict[str, object],
    reset_delta_state: bool,
    batch_size: int = 1,
    progress_bar=None,
) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    generation_settings = _coerce_generation_settings(task_kwargs)
    entries = [
        (item_idx, item, str(item.get("prompt", "")).strip())
        for item_idx, item in indexed_items
    ]
    for start in range(0, len(entries), max(1, batch_size)):
        chunk = entries[start : start + max(1, batch_size)]
        prompts = [prompt for _, _, prompt in chunk]
        predictions = _generate_predictions_for_prompts(
            model,
            tokenizer,
            device,
            prompts,
            generation_settings=generation_settings,
            reset_delta_state=reset_delta_state,
            batch_size=batch_size,
        )
        for (item_idx, item, prompt), prediction in zip(chunk, predictions):
            evaluation = evaluate_ifeval_response(item, prediction)
            records.append(
                (
                    item_idx,
                    {
                        "key": int(item.get("key", 0)),
                        "prompt": prompt,
                        "instruction_id_list": [str(value) for value in item.get("instruction_id_list", [])],
                        "kwargs": list(item.get("kwargs", []) or []),
                        "prediction": prediction,
                        **evaluation,
                    },
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
    return records


def evaluate_gpqa(
    *,
    model,
    tokenizer,
    device: str,
    indexed_items: list[tuple[int, dict]],
    task_kwargs: dict[str, object],
    seed: int,
    reset_delta_state: bool,
    batch_size: int = 1,
    progress_bar=None,
) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    generation_settings = _coerce_generation_settings(task_kwargs)
    effective_batch_size = max(1, batch_size)
    entries = [
        (item_idx, item, *build_gpqa_prompt(item, seed=seed))
        for item_idx, item in indexed_items
    ]
    total_chunks = (len(entries) + effective_batch_size - 1) // effective_batch_size
    for start in range(0, len(entries), effective_batch_size):
        chunk = entries[start : start + effective_batch_size]
        chunk_index = (start // effective_batch_size) + 1
        prompts = [prompt for _, _, prompt, _ in chunk]
        heartbeat_label = (
            f"gpqa chunk {chunk_index}/{total_chunks} generating {len(chunk)} prompts "
            f"(batch_size={batch_size}, max_new_tokens={generation_settings['max_new_tokens']})"
        )
        if _gpqa_heartbeat_interval_seconds() > 0:
            _log_eval_debug(
                f"Dispatching gpqa chunk {chunk_index}/{total_chunks} with {len(chunk)} prompts "
                f"and max_new_tokens={generation_settings['max_new_tokens']}"
            )
        predictions = _run_with_gpqa_heartbeat(
            heartbeat_label,
            lambda: _generate_predictions_for_prompts(
                model,
                tokenizer,
                device,
                prompts,
                generation_settings=generation_settings,
                reset_delta_state=reset_delta_state,
                batch_size=batch_size,
            ),
        )
        if _gpqa_heartbeat_interval_seconds() > 0:
            _log_eval_debug(
                f"Completed gpqa chunk {chunk_index}/{total_chunks}; entering record post-processing for {len(predictions)} predictions"
            )
        for (item_idx, item, _, correct_letter), prediction in zip(chunk, predictions):
            predicted_letter = extract_gpqa_letter(prediction)
            records.append(
                (
                    item_idx,
                    {
                        "record_id": item["Record ID"],
                        "domain": item["High-level domain"],
                        "subdomain": item["Subdomain"],
                        "question": item["Question"],
                        "correct_letter": correct_letter,
                        "prediction": prediction,
                        "predicted_letter": predicted_letter,
                        "correct": predicted_letter == correct_letter,
                    },
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
    return records


def evaluate_hotpotqa(
    *,
    model,
    tokenizer,
    device: str,
    indexed_items: list[tuple[int, dict]],
    task_kwargs: dict[str, object],
    reset_delta_state: bool,
    batch_size: int = 1,
    progress_bar=None,
) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    generation_settings = _coerce_generation_settings(task_kwargs)
    entries = [
        (
            item_idx,
            item,
            HOTPOTQA_PROMPT_TEMPLATE.format(
                context=build_hotpotqa_context(item),
                question=str(item["question"]).strip(),
            ),
        )
        for item_idx, item in indexed_items
    ]
    for start in range(0, len(entries), max(1, batch_size)):
        chunk = entries[start : start + max(1, batch_size)]
        prompts = [prompt for _, _, prompt in chunk]
        predictions = _generate_predictions_for_prompts(
            model,
            tokenizer,
            device,
            prompts,
            generation_settings=generation_settings,
            reset_delta_state=reset_delta_state,
            batch_size=batch_size,
        )
        for (item_idx, item, _), prediction in zip(chunk, predictions):
            first_line = extract_first_line(prediction)
            gold = str(item["answer"]).strip()
            exact_match = hotpotqa_exact_match(first_line, gold)
            f1 = hotpotqa_f1(first_line, gold)
            records.append(
                (
                    item_idx,
                    {
                        "id": item["id"],
                        "level": item.get("level"),
                        "type": item.get("type"),
                        "question": item["question"],
                        "answer": gold,
                        "prediction": prediction,
                        "extracted_answer": first_line,
                        "exact_match": exact_match,
                        "f1": round(f1, 4),
                        "correct": exact_match,
                    },
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
    return records


def _render_message_prompt(tokenizer, messages: list[dict[str, str]], *, use_chat_template: bool) -> str:
    if use_chat_template:
        return apply_project_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def evaluate_memory_agent_bench(
    *,
    model,
    tokenizer,
    device: str,
    row_tasks: list[MemoryAgentBenchRowTask],
    task_kwargs: dict[str, object],
    reset_delta_state: bool,
    max_context_chars: int,
    use_official_generation_lengths: bool,
    use_official_prompt: bool,
    external_memory_agent_bench_root: Path,
    batch_size: int = 16,
    progress_bar=None,
) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    model_context_window = infer_model_context_window(model, tokenizer)
    use_chat_template = bool(task_kwargs.get("use_chat_template", True))
    default_max_new_tokens = int(task_kwargs["max_new_tokens"])
    base_generation_settings = _coerce_generation_settings(task_kwargs)
    eval_utils = load_mab_eval_utils(external_memory_agent_bench_root) if use_official_prompt else None
    official_buffer_length = 4000
    supports_manual_shared_prefix = _supports_manual_shared_prefix_decode(model)

    def append_record(
        *,
        item: dict[str, object],
        source: str,
        row_max_new_tokens: int,
        raw_context: str,
        entry: dict[str, object],
        prediction: str,
    ) -> None:
        nonlocal eval_utils
        question_meta = dict(entry["question_meta"])
        context = str(entry["context"])
        eval_index = int(entry["eval_index"])
        first_line = extract_first_line(prediction)
        answer_aliases = list(question_meta.get("answer_aliases", []))
        correct = memory_qa_is_correct(prediction, answer_aliases)
        f1 = qa_alias_max_f1(prediction, answer_aliases)
        primary_metric = memory_agent_bench_primary_metric_name(source)
        extra_metric_fields: dict[str, float] = {}
        if primary_metric == "recsys_recall@5":
            if eval_utils is None:
                eval_utils = load_mab_eval_utils(external_memory_agent_bench_root)
            recomputed_metrics = _memory_agent_bench_recompute_metrics(
                {
                    "source": source,
                    "prediction": prediction,
                    "answer_aliases": answer_aliases,
                },
                eval_utils_module=eval_utils,
                repo_root=external_memory_agent_bench_root,
            )
            for metric_key in ("recsys_recall@1", "recsys_recall@5", "recsys_recall@10"):
                metric_value = recomputed_metrics.get(metric_key)
                if isinstance(metric_value, (int, float, bool)):
                    extra_metric_fields[metric_key] = round(float(metric_value), 4)
        if primary_metric == "f1":
            primary_score = round(f1, 4)
        elif primary_metric == "recsys_recall@5":
            primary_score = float(extra_metric_fields.get("recsys_recall@5", 0.0))
        else:
            primary_score = float(correct)
        records.append(
            (
                eval_index,
                {
                    "row_id": item.get("row_id"),
                    "split": item.get("split"),
                    "source": source,
                    "question_id": question_meta.get("question_id"),
                    "qa_pair_id": question_meta.get("qa_pair_id"),
                    "question_type": question_meta.get("question_type"),
                    "question_date": question_meta.get("question_date"),
                    "question": question_meta["question"],
                    "query": str(entry["query"]),
                    "answer_aliases": answer_aliases,
                    "prediction": prediction,
                    "extracted_answer": first_line,
                    "context_chars": len(raw_context),
                    "prompt_context_chars": len(context),
                    "max_new_tokens": row_max_new_tokens,
                    "prompt_style": str(entry["prompt_style"]),
                    "correct": correct,
                    "f1": round(f1, 4),
                    "primary_metric": primary_metric,
                    "primary_score": primary_score,
                    **extra_metric_fields,
                },
            )
        )
        if progress_bar is not None:
            progress_bar.update(1)

    for row_task in row_tasks:
        item = row_task.item
        raw_context = str(item.get("context", ""))
        if max_context_chars > 0 and len(raw_context) > max_context_chars:
            # Fast evaluation preclip: limit very long MemoryAgentBench inputs before the
            # official/token-budget truncation so benchmark runs finish in a practical time.
            raw_context = clip_context_text(raw_context, max_context_chars)
        source = memory_agent_bench_source(item)
        selected_questions = list(item.get("selected_questions", []))
        row_max_new_tokens = memory_agent_bench_effective_max_new_tokens(
            item,
            default_max_new_tokens=default_max_new_tokens,
            use_official_generation_lengths=use_official_generation_lengths,
        )
        row_generation_settings = {
            **base_generation_settings,
            "max_new_tokens": row_max_new_tokens,
        }
        can_use_shared_prefix = supports_manual_shared_prefix and len(selected_questions) > 1

        row_entries: list[dict[str, object]] = []
        if use_official_prompt:
            row_generation_settings["use_chat_template"] = False
            source_config = dict(item.get("official_source_config") or memory_agent_bench_source_config(item))
            context_max_length = int(source_config.get("context_max_length") or 0)
            chunk_size = int(source_config.get("chunk_size") or 4096)
            query_answer_pairs = build_official_mab_query_answer_pairs(
                {
                    "questions": [question_meta["question"] for question_meta in selected_questions],
                    "answers": [question_meta.get("answer_raw", []) for question_meta in selected_questions],
                    "metadata": {
                        "source": source,
                        "question_dates": [question_meta.get("question_date") for question_meta in selected_questions],
                        "question_types": [question_meta.get("question_type") for question_meta in selected_questions],
                        "question_ids": [question_meta.get("question_id") for question_meta in selected_questions],
                        "previous_events": [question_meta.get("previous_event") for question_meta in selected_questions],
                        "qa_pair_ids": [question_meta.get("qa_pair_id") for question_meta in selected_questions],
                    },
                },
                source=source,
            )
            context_chunks = build_official_mab_context_chunks(
                [{"context": raw_context}],
                chunk_size=chunk_size,
                eval_utils_module=eval_utils,
            )
            memorized_context = build_official_mab_memorized_context(source, context_chunks[0])
            truncated_context = truncate_official_mab_memory_context(
                memorized_context,
                tokenizer=tokenizer,
                context_max_length=context_max_length,
                raw_input_length_limit=model_context_window,
                buffer_length=official_buffer_length,
                generation_max_length=row_max_new_tokens,
            )
            system_message = get_official_mab_template(
                source,
                "system",
                "Long_context_agent_deltamem",
            )

            for question_meta, (query, _, _) in zip(selected_questions, query_answer_pairs):
                prompt_builder = lambda candidate_context, query=query: _render_message_prompt(
                    tokenizer,
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"{candidate_context}\n{query}".strip()},
                    ],
                    use_chat_template=use_chat_template,
                )
                final_context = clip_context_text_to_model_limit(
                    truncated_context,
                    tokenizer=tokenizer,
                    prompt_builder=prompt_builder,
                    use_chat_template=False,
                    model_context_window=model_context_window,
                    max_new_tokens=row_max_new_tokens,
                    max_context_chars=0,
                    keep="tail",
                )
                rendered_prompt = prompt_builder(final_context)
                entry = {
                    "eval_index": int(question_meta["eval_index"]),
                    "question_meta": dict(question_meta),
                    "context": final_context,
                    "query": query,
                    "prompt": rendered_prompt,
                    "prompt_style": "official_memorize_query_templates",
                }
                if can_use_shared_prefix:
                    entry["prompt_token_ids"] = _tokenize_prompt_to_token_list(
                        tokenizer,
                        rendered_prompt,
                        use_chat_template=False,
                    )
                row_entries.append(entry)
        else:
            for question_meta in selected_questions:
                question_text = str(question_meta["question"]).strip()
                prompt_builder = lambda candidate_context, question_text=question_text: MEMORY_CONTEXT_QA_PROMPT_TEMPLATE.format(
                    context=candidate_context,
                    question=question_text,
                )
                final_context = clip_context_text_to_model_limit(
                    raw_context,
                    tokenizer=tokenizer,
                    prompt_builder=prompt_builder,
                    use_chat_template=use_chat_template,
                    model_context_window=model_context_window,
                    max_new_tokens=row_max_new_tokens,
                    max_context_chars=0,
                    keep="head_tail",
                )
                prompt = MEMORY_CONTEXT_QA_PROMPT_TEMPLATE.format(
                    context=final_context,
                    question=question_text,
                )
                entry = {
                    "eval_index": int(question_meta["eval_index"]),
                    "question_meta": dict(question_meta),
                    "context": final_context,
                    "query": question_text,
                    "prompt": prompt,
                    "prompt_style": "unified_memory_context_qa",
                }
                if can_use_shared_prefix:
                    entry["prompt_token_ids"] = _tokenize_prompt_to_token_list(
                        tokenizer,
                        prompt,
                        use_chat_template=use_chat_template,
                    )
                row_entries.append(entry)

        effective_batch_size = max(1, batch_size)
        if len(row_entries) <= 1 or not can_use_shared_prefix:
            for start in range(0, len(row_entries), effective_batch_size):
                chunk_entries = row_entries[start : start + effective_batch_size]
                chunk_prompts = [str(entry["prompt"]) for entry in chunk_entries]
                predictions = _generate_predictions_for_prompts(
                    model,
                    tokenizer,
                    device,
                    chunk_prompts,
                    generation_settings=row_generation_settings,
                    reset_delta_state=reset_delta_state,
                    batch_size=effective_batch_size,
                )
                for entry, prediction in zip(chunk_entries, predictions):
                    append_record(
                        item=item,
                        source=source,
                        row_max_new_tokens=row_max_new_tokens,
                        raw_context=raw_context,
                        entry=entry,
                        prediction=prediction,
                    )
            continue

        prompt_token_lists = [list(entry["prompt_token_ids"]) for entry in row_entries]
        prefix_token_count = _longest_common_token_prefix_length(prompt_token_lists)
        if prefix_token_count <= 0:
            for start in range(0, len(row_entries), effective_batch_size):
                chunk_entries = row_entries[start : start + effective_batch_size]
                chunk_prompts = [str(entry["prompt"]) for entry in chunk_entries]
                predictions = _generate_predictions_for_prompts(
                    model,
                    tokenizer,
                    device,
                    chunk_prompts,
                    generation_settings=row_generation_settings,
                    reset_delta_state=reset_delta_state,
                    batch_size=effective_batch_size,
                )
                for entry, prediction in zip(chunk_entries, predictions):
                    append_record(
                        item=item,
                        source=source,
                        row_max_new_tokens=row_max_new_tokens,
                        raw_context=raw_context,
                        entry=entry,
                        prediction=prediction,
                    )
            continue

        predictions = _generate_predictions_with_shared_prefix(
            model,
            tokenizer,
            device,
            prompt_token_lists,
            generation_settings=row_generation_settings,
            reset_delta_state=reset_delta_state,
        )
        for entry, prediction in zip(row_entries, predictions):
            append_record(
                item=item,
                source=source,
                row_max_new_tokens=row_max_new_tokens,
                raw_context=raw_context,
                entry=entry,
                prediction=prediction,
            )
    return records


def summarize_accuracy(records: list[dict[str, object]]) -> dict[str, object]:
    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    return {
        "accuracy": 0.0 if total == 0 else round(correct / total, 4),
        "num_samples": total,
        "num_correct": correct,
    }


def summarize_group_accuracy(
    records: list[dict[str, object]],
    *,
    group_key: str,
) -> dict[str, dict[str, object]]:
    buckets: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get(group_key, "unknown"))].append(record)
    return {
        key: summarize_accuracy(bucket)
        for key, bucket in sorted(buckets.items(), key=lambda item: item[0])
    }


def summarize_ifeval(records: list[dict[str, object]]) -> dict[str, object]:
    total_prompts = len(records)
    prompt_strict = sum(1 for record in records if record["prompt_level_strict_acc"])
    prompt_loose = sum(1 for record in records if record["prompt_level_loose_acc"])

    instruction_pairs: list[tuple[str, bool, bool]] = []
    for record in records:
        instruction_ids = [str(value) for value in record.get("instruction_id_list", [])]
        strict_flags = [bool(value) for value in record.get("inst_level_strict_acc", [])]
        loose_flags = [bool(value) for value in record.get("inst_level_loose_acc", [])]
        for instruction_id, strict_flag, loose_flag in zip(instruction_ids, strict_flags, loose_flags):
            instruction_pairs.append((instruction_id, strict_flag, loose_flag))

    total_instructions = len(instruction_pairs)
    strict_instruction_correct = sum(1 for _, strict_flag, _ in instruction_pairs if strict_flag)
    loose_instruction_correct = sum(1 for _, _, loose_flag in instruction_pairs if loose_flag)

    by_instruction_id: dict[str, dict[str, object]] = {}
    grouped: defaultdict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for instruction_id, strict_flag, loose_flag in instruction_pairs:
        grouped[instruction_id].append((strict_flag, loose_flag))
    for instruction_id, flags in sorted(grouped.items(), key=lambda item: item[0]):
        instruction_total = len(flags)
        instruction_strict = sum(1 for strict_flag, _ in flags if strict_flag)
        instruction_loose = sum(1 for _, loose_flag in flags if loose_flag)
        by_instruction_id[instruction_id] = {
            "inst_level_strict_acc": 0.0 if instruction_total == 0 else round(instruction_strict / instruction_total, 4),
            "inst_level_loose_acc": 0.0 if instruction_total == 0 else round(instruction_loose / instruction_total, 4),
            "num_instructions": instruction_total,
            "num_strict_correct": instruction_strict,
            "num_loose_correct": instruction_loose,
        }

    return {
        "prompt_level_strict_acc": 0.0 if total_prompts == 0 else round(prompt_strict / total_prompts, 4),
        "prompt_level_loose_acc": 0.0 if total_prompts == 0 else round(prompt_loose / total_prompts, 4),
        "inst_level_strict_acc": 0.0 if total_instructions == 0 else round(strict_instruction_correct / total_instructions, 4),
        "inst_level_loose_acc": 0.0 if total_instructions == 0 else round(loose_instruction_correct / total_instructions, 4),
        "num_samples": total_prompts,
        "num_prompt_strict_correct": prompt_strict,
        "num_prompt_loose_correct": prompt_loose,
        "num_instructions": total_instructions,
        "num_instruction_strict_correct": strict_instruction_correct,
        "num_instruction_loose_correct": loose_instruction_correct,
        "by_instruction_id": by_instruction_id,
    }


def summarize_gpqa(records: list[dict[str, object]]) -> dict[str, object]:
    summary = summarize_accuracy(records)
    summary["by_domain"] = summarize_group_accuracy(records, group_key="domain")
    return summary

def summarize_hotpotqa_group(records: list[dict[str, object]], *, group_key: str) -> dict[str, dict[str, object]]:
    buckets: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get(group_key, "unknown"))].append(record)
    grouped: dict[str, dict[str, object]] = {}
    for key, bucket in sorted(buckets.items(), key=lambda item: item[0]):
        total = len(bucket)
        exact_matches = sum(1 for record in bucket if record["exact_match"])
        mean_f1 = 0.0 if total == 0 else round(sum(float(record["f1"]) for record in bucket) / total, 4)
        grouped[key] = {
            "exact_match": 0.0 if total == 0 else round(exact_matches / total, 4),
            "f1": mean_f1,
            "num_samples": total,
            "num_exact_match": exact_matches,
        }
    return grouped


def summarize_hotpotqa(records: list[dict[str, object]]) -> dict[str, object]:
    total = len(records)
    exact_matches = sum(1 for record in records if record["exact_match"])
    mean_f1 = 0.0 if total == 0 else round(sum(float(record["f1"]) for record in records) / total, 4)
    summary = {
        "exact_match": 0.0 if total == 0 else round(exact_matches / total, 4),
        "f1": mean_f1,
        "num_samples": total,
        "num_exact_match": exact_matches,
    }
    summary["by_level"] = summarize_hotpotqa_group(records, group_key="level")
    summary["by_type"] = summarize_hotpotqa_group(records, group_key="type")
    return summary


MEMORY_AGENT_BENCH_SPLIT_LABELS = {
    "Accurate_Retrieval": "Accurate Retrieval",
    "Test_Time_Learning": "Test-time Learning",
    "Long_Range_Understanding": "Long Range Understanding",
    "Conflict_Resolution": "Selective Forgetting",
    "Selective Forgetting": "Selective Forgetting",
}


MEMORY_AGENT_BENCH_CATEGORY_ORDER = [
    "Accurate Retrieval",
    "Test-time Learning",
    "Long Range Understanding",
    "Selective Forgetting",
]



def _memory_agent_bench_recompute_metrics(
    record: dict[str, object],
    *,
    eval_utils_module=None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    source = str(record.get("source") or "")
    prediction = str(record.get("prediction") or "")
    answer_field = record.get("answer_aliases") or record.get("answer_raw") or []
    if isinstance(answer_field, str):
        answers: object = [answer_field]
    elif isinstance(answer_field, list):
        answers = answer_field
    else:
        answers = list(answer_field) if answer_field else []

    repo_root = repo_root or DEFAULT_MEMORY_AGENT_BENCH_ROOT
    if eval_utils_module is None:
        eval_utils_module = load_mab_eval_utils(repo_root)

    previous_cwd = os.getcwd()
    os.chdir(str(repo_root))
    try:
        metrics, _ = eval_utils_module.post_process(
            {"output": prediction},
            answers,
            {"sub_dataset": source, "debug": False},
        )
    finally:
        os.chdir(previous_cwd)
    return metrics



def _memory_agent_bench_record_score(
    record: dict[str, object],
    *,
    eval_utils_module=None,
    repo_root: Path | None = None,
) -> tuple[dict[str, str], float, bool, str]:
    source = str(record.get("source") or "")
    spec = _memory_agent_bench_source_spec(source)
    metric_key = spec["metric_key"]

    if metric_key == "accuracy":
        return spec, float(bool(record.get("correct"))), False, metric_key

    stored_value = record.get(metric_key)
    if isinstance(stored_value, (int, float)):
        return spec, float(stored_value), False, metric_key

    if metric_key == "f1":
        f1_value = record.get("f1")
        if isinstance(f1_value, (int, float)):
            return spec, float(f1_value), False, "f1"

    recomputed_metrics = _memory_agent_bench_recompute_metrics(
        record,
        eval_utils_module=eval_utils_module,
        repo_root=repo_root,
    )
    recomputed_value = recomputed_metrics.get(metric_key)
    if isinstance(recomputed_value, (int, float, bool)):
        return spec, float(recomputed_value), True, metric_key

    return spec, 0.0, True, metric_key



def _summarize_memory_agent_bench_dataset(
    records: list[dict[str, object]],
    *,
    eval_utils_module=None,
    repo_root: Path | None = None,
) -> tuple[dict[str, object], float]:
    if not records:
        return (
            {
                "metric": "Accuracy",
                "metric_key": "accuracy",
                "metric_key_used": "accuracy",
                "score": 0.0,
                "num_samples": 0,
                "fallback_used": False,
            },
            0.0,
        )

    spec, _, _, _ = _memory_agent_bench_record_score(
        records[0],
        eval_utils_module=eval_utils_module,
        repo_root=repo_root,
    )
    scores: list[float] = []
    used_metric_keys: set[str] = set()
    fallback_used = False
    for record in records:
        _, score, record_fallback_used, metric_key_used = _memory_agent_bench_record_score(
            record,
            eval_utils_module=eval_utils_module,
            repo_root=repo_root,
        )
        scores.append(score)
        used_metric_keys.add(metric_key_used)
        fallback_used = fallback_used or record_fallback_used

    metric_key_used = spec["metric_key"] if len(used_metric_keys) != 1 else next(iter(used_metric_keys))
    score_sum = sum(scores)
    return (
        {
            "metric": spec["metric"],
            "metric_key": spec["metric_key"],
            "metric_key_used": metric_key_used,
            "score": 0.0 if not scores else round(score_sum / len(scores), 4),
            "num_samples": len(records),
            "fallback_used": fallback_used,
        },
        score_sum,
    )



def summarize_memory_agent_bench(
    records: list[dict[str, object]],
    *,
    external_memory_agent_bench_root: Path | None = None,
) -> dict[str, object]:
    eval_utils_module = None
    categories: defaultdict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        category_name = MEMORY_AGENT_BENCH_SPLIT_LABELS.get(
            str(record.get("split") or ""),
            str(record.get("split") or "unknown"),
        )
        dataset_name = _memory_agent_bench_source_spec(str(record.get("source") or ""))["dataset_name"]
        categories[category_name][dataset_name].append(record)

    category_payload: dict[str, dict[str, object]] = {}
    category_overall: dict[str, float] = {}
    category_sample_weights: dict[str, int] = {}
    dataset_scores: dict[str, dict[str, object]] = {}
    total_weighted_score = 0.0
    total_num_samples = 0

    for category_name in MEMORY_AGENT_BENCH_CATEGORY_ORDER:
        datasets = categories.get(category_name)
        if not datasets:
            continue
        dataset_payload: dict[str, dict[str, object]] = {}
        category_weighted_sum = 0.0
        category_num_samples = 0
        for dataset_name, dataset_records in sorted(datasets.items(), key=lambda item: item[0]):
            dataset_summary, dataset_score_sum = _summarize_memory_agent_bench_dataset(
                dataset_records,
                eval_utils_module=eval_utils_module,
                repo_root=external_memory_agent_bench_root,
            )
            dataset_payload[dataset_name] = dataset_summary
            dataset_scores[dataset_name] = {
                "category": category_name,
                **dataset_summary,
            }
            dataset_num_samples = int(dataset_summary["num_samples"])
            category_weighted_sum += dataset_score_sum
            category_num_samples += dataset_num_samples
        overall = 0.0 if category_num_samples == 0 else round(category_weighted_sum / category_num_samples, 4)
        category_payload[category_name] = {
            "overall": overall,
            "num_datasets": len(dataset_payload),
            "num_scored_datasets": len(dataset_payload),
            "num_samples": category_num_samples,
            "num_scored_samples": category_num_samples,
            "datasets": dataset_payload,
        }
        category_overall[category_name] = overall
        category_sample_weights[category_name] = category_num_samples
        total_weighted_score += category_weighted_sum
        total_num_samples += category_num_samples

    overall = 0.0 if total_num_samples == 0 else round(total_weighted_score / total_num_samples, 4)
    return {
        "overall": overall,
        "primary_metric": "sample_weighted_category_overall",
        "primary_score": overall,
        "num_categories": len(category_overall),
        "num_samples": len(records),
        "num_scored_samples": total_num_samples,
        "dataset_scores": dataset_scores,
        "category_overall": category_overall,
        "category_sample_weights": category_sample_weights,
        "categories": category_payload,
    }


def task_num_items(task_name: str, items: list[dict]) -> int:
    if task_name == "memory_agent_bench":
        return sum(len(item.get("selected_questions", [])) for item in items)
    return len(items)


def summarize_task(
    task_name: str,
    records: list[dict[str, object]],
    *,
    args: argparse.Namespace | None = None,
) -> dict[str, object]:
    if task_name == "ifeval":
        return summarize_ifeval(records)
    if task_name == "gpqa_diamond":
        return summarize_gpqa(records)
    if task_name == "hotpotqa":
        return summarize_hotpotqa(records)
    if task_name == "memory_agent_bench":
        return summarize_memory_agent_bench(
            records,
            external_memory_agent_bench_root=(
                args.external_memory_agent_bench_root if args is not None else None
            ),
        )
    raise ValueError(f"Unsupported task: {task_name}")


def load_task_items(task_name: str, args: argparse.Namespace) -> list[dict]:
    if task_name == "ifeval":
        return load_ifeval(
            cache_dir=args.datasets_cache_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
    if task_name == "gpqa_diamond":
        return load_gpqa_diamond(
            cache_dir=args.datasets_cache_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
    if task_name == "hotpotqa":
        return load_hotpotqa(
            cache_dir=args.datasets_cache_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
    if task_name == "memory_agent_bench":
        return load_memory_agent_bench(
            cache_dir=args.datasets_cache_dir,
            hub_cache_dir=args.hub_cache_dir,
            splits=args.memory_agent_bench_splits,
            sources=args.memory_agent_bench_sources,
            max_samples=args.max_samples,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
    raise ValueError(f"Unsupported task: {task_name}")


def evaluate_task(
    *,
    task_name: str,
    items: list[dict],
    model,
    tokenizer,
    device: str,
    context: DistributedContext,
    seed: int,
    reset_delta_state: bool,
    args: argparse.Namespace,
    progress_desc: str,
) -> list[dict[str, object]] | None:
    indexed_items = local_indexed_items(items, context)
    memory_agent_bench_row_tasks: list[MemoryAgentBenchRowTask] = []
    progress_total = len(indexed_items)
    if task_name == "memory_agent_bench":
        memory_agent_bench_row_tasks = local_memory_agent_bench_row_tasks(
            items,
            context,
            default_max_new_tokens=args.memory_agent_bench_max_new_tokens,
            use_official_generation_lengths=args.memory_agent_bench_use_official_generation_lengths,
            max_questions_per_row_task=args.memory_agent_bench_max_questions_per_row_task,
        )
        progress_total = sum(row_task.question_count for row_task in memory_agent_bench_row_tasks)
    progress_bar = None
    if context.rank == 0:
        progress_bar = tqdm(total=progress_total, desc=progress_desc, dynamic_ncols=True)
    task_kwargs = task_generation_kwargs(task_name, args)
    if task_name == "ifeval":
        local_records = evaluate_ifeval(
            model=model,
            tokenizer=tokenizer,
            device=device,
            indexed_items=indexed_items,
            task_kwargs=task_kwargs,
            reset_delta_state=reset_delta_state,
            batch_size=args.eval_batch_size,
            progress_bar=progress_bar,
        )
    elif task_name == "gpqa_diamond":
        local_records = evaluate_gpqa(
            model=model,
            tokenizer=tokenizer,
            device=device,
            indexed_items=indexed_items,
            task_kwargs=task_kwargs,
            seed=seed,
            reset_delta_state=reset_delta_state,
            batch_size=args.eval_batch_size,
            progress_bar=progress_bar,
        )
    elif task_name == "hotpotqa":
        local_records = evaluate_hotpotqa(
            model=model,
            tokenizer=tokenizer,
            device=device,
            indexed_items=indexed_items,
            task_kwargs=task_kwargs,
            reset_delta_state=reset_delta_state,
            batch_size=args.eval_batch_size,
            progress_bar=progress_bar,
        )
    elif task_name == "memory_agent_bench":
        local_records = evaluate_memory_agent_bench(
            model=model,
            tokenizer=tokenizer,
            device=device,
            row_tasks=memory_agent_bench_row_tasks,
            task_kwargs=task_kwargs,
            reset_delta_state=reset_delta_state,
            max_context_chars=args.memory_agent_bench_max_context_chars,
            use_official_generation_lengths=args.memory_agent_bench_use_official_generation_lengths,
            use_official_prompt=args.memory_agent_bench_use_official_prompt,
            external_memory_agent_bench_root=args.external_memory_agent_bench_root,
            batch_size=args.memory_agent_bench_eval_batch_size,
            progress_bar=progress_bar,
        )
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    if progress_bar is not None:
        progress_bar.close()
    return gather_indexed_records(local_records, context)


def _task_output_json_path(output_json: Path, section: str, task_name: str) -> Path:
    return output_json.with_name(f"{output_json.stem}.{section}.{task_name}{output_json.suffix}")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def persist_benchmark_payload(
    *,
    output_json: Path,
    payload: dict[str, object],
    section: str | None = None,
    task_name: str | None = None,
) -> None:
    _write_json_atomic(output_json, payload)
    if section is None or task_name is None:
        return
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return
    task_payload = section_payload.get(task_name)
    if not isinstance(task_payload, dict):
        return
    per_task_payload = {
        "model_path": payload.get("model_path"),
        "seed": payload.get("seed"),
        "tasks": [task_name],
        "task_settings": {task_name: (payload.get("task_settings") or {}).get(task_name)},
        "num_items": {task_name: (payload.get("num_items") or {}).get(task_name)},
        section: {
            key: value
            for key, value in section_payload.items()
            if key == task_name or key in {"backend", "backend_requested", "config"}
        },
    }
    _write_json_atomic(_task_output_json_path(output_json, section, task_name), per_task_payload)


def main() -> None:
    args = parse_args()
    context = init_distributed(args.device)
    try:
        set_all_seeds(args.seed)
        args.datasets_cache_dir.mkdir(parents=True, exist_ok=True)
        args.hub_cache_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "model_path": args.model_path,
            "delta_adapter_dir": None if args.delta_adapter_dir is None else str(args.delta_adapter_dir),
            "lora_adapter_dir": None if args.lora_adapter_dir is None else str(args.lora_adapter_dir),
            "seed": args.seed,
            "tasks": args.tasks,
            "datasets_cache_dir": str(args.datasets_cache_dir),
            "hub_cache_dir": str(args.hub_cache_dir),
            "local_files_only": args.local_files_only,
            "distributed": {
                "enabled": context.enabled,
                "world_size": context.world_size,
            },
            "inference_backends": {
                "base_requested": args.base_inference_backend,
                "base": args.base_inference_backend,
                "delta": "transformers",
                "lora": args.lora_inference_backend,
            },
            "generation": shared_generation_kwargs(args),
            "task_settings": {
                "gpqa_diamond": {
                    "max_new_tokens": args.gpqa_max_new_tokens,
                    "prompt_style": "JSON answer letter prompt",
                    "decoding": "official_greedy" if args.gpqa_official_decoding else "shared_generation",
                },
                "hotpotqa": {
                    "split": "validation",
                    "max_new_tokens": args.hotpotqa_max_new_tokens,
                    "scoring": "token-level exact-match and F1",
                    "prompt_template": HOTPOTQA_PROMPT_TEMPLATE,
                    "decoding": "official_greedy" if args.hotpotqa_official_decoding else "shared_generation",
                },
                "memory_agent_bench": {
                    "splits": args.memory_agent_bench_splits,
                    "sources": args.memory_agent_bench_sources,
                    "max_new_tokens": args.memory_agent_bench_max_new_tokens,
                    "use_official_generation_lengths": args.memory_agent_bench_use_official_generation_lengths,
                    "use_official_prompt": args.memory_agent_bench_use_official_prompt,
                    "eval_batch_size": args.memory_agent_bench_eval_batch_size,
                    "max_context_chars": args.memory_agent_bench_max_context_chars,
                    "scoring": "normalized alias match against acceptable answers",
                    "prompt_template": (
                        "official_memorize_query_templates"
                        if args.memory_agent_bench_use_official_prompt
                        else MEMORY_CONTEXT_QA_PROMPT_TEMPLATE
                    ),
                },
            },
        }
        items_by_task = {task_name: load_task_items(task_name, args) for task_name in args.tasks}
        payload["num_items"] = {task_name: task_num_items(task_name, items) for task_name, items in items_by_task.items()}

        base_model = None
        base_tokenizer = None
        if not args.skip_base:
            base_model, base_tokenizer = load_base_model_and_tokenizer(
                model_path=args.model_path,
                device=context.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
            )
            if context.rank == 0:
                payload["base"] = {
                    "backend": args.base_inference_backend,
                    "backend_requested": args.base_inference_backend,
                }
            for task_name in args.tasks:
                records = evaluate_task(
                    task_name=task_name,
                    items=items_by_task[task_name],
                    model=base_model,
                    tokenizer=base_tokenizer,
                    device=context.device,
                    context=context,
                    seed=args.seed,
                    reset_delta_state=False,
                    args=args,
                    progress_desc=f"base_{task_name}",
                )
                if context.rank == 0:
                    payload["base"][task_name] = {
                        "records": records,
                        "summary": summarize_task(task_name, records, args=args),
                    }
                    persist_benchmark_payload(
                        output_json=args.output_json,
                        payload=payload,
                        section="base",
                        task_name=task_name,
                    )
            if args.skip_delta:
                del base_model
                base_model = None
                base_tokenizer = None
                maybe_empty_cache(context.device)

        if context.enabled:
            barrier_distributed(context)

        if not args.skip_delta:
            if base_model is not None and base_tokenizer is not None:
                delta_model = base_model
                delta_tokenizer = base_tokenizer
                delta_config = attach_delta_adapter_in_place(delta_model, args.delta_adapter_dir)
                base_model = None
                base_tokenizer = None
            else:
                delta_model, delta_tokenizer, delta_config = load_delta_model_and_tokenizer(
                    model_path=args.model_path,
                    adapter_dir=args.delta_adapter_dir,
                    device=context.device,
                    dtype=args.dtype,
                    attn_implementation=args.attn_implementation,
                )
            if context.rank == 0:
                payload["delta"] = {"config": delta_config}
            for task_name in args.tasks:
                records = evaluate_task(
                    task_name=task_name,
                    items=items_by_task[task_name],
                    model=delta_model,
                    tokenizer=delta_tokenizer,
                    device=context.device,
                    context=context,
                    seed=args.seed,
                    reset_delta_state=True,
                    args=args,
                    progress_desc=f"delta_{task_name}",
                )
                if context.rank == 0:
                    payload["delta"][task_name] = {
                        "records": records,
                        "summary": summarize_task(task_name, records, args=args),
                    }
                    persist_benchmark_payload(
                        output_json=args.output_json,
                        payload=payload,
                        section="delta",
                        task_name=task_name,
                    )
            del delta_model
            maybe_empty_cache(context.device)

        if context.enabled:
            barrier_distributed(context)

        if not args.skip_lora and args.lora_adapter_dir is not None:
            lora_model, lora_tokenizer = load_lora_model_and_tokenizer(
                model_path=args.model_path,
                adapter_dir=args.lora_adapter_dir,
                device=context.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
            )
            if context.rank == 0:
                payload["lora"] = {"backend": args.lora_inference_backend}
            for task_name in args.tasks:
                records = evaluate_task(
                    task_name=task_name,
                    items=items_by_task[task_name],
                    model=lora_model,
                    tokenizer=lora_tokenizer,
                    device=context.device,
                    context=context,
                    seed=args.seed,
                    reset_delta_state=False,
                    args=args,
                    progress_desc=f"lora_{task_name}",
                )
                if context.rank == 0:
                    payload["lora"][task_name] = {
                        "records": records,
                        "summary": summarize_task(task_name, records, args=args),
                    }
                    persist_benchmark_payload(
                        output_json=args.output_json,
                        payload=payload,
                        section="lora",
                        task_name=task_name,
                    )
            del lora_model
            maybe_empty_cache(context.device)

        if context.rank == 0:
            persist_benchmark_payload(output_json=args.output_json, payload=payload)
            print(json.dumps(payload, indent=2))
    finally:
        finalize_distributed(context)


if __name__ == "__main__":
    main()
