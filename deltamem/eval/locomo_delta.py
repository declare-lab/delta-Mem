from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.eval.common import (
    DistributedContext,
    finalize_distributed,
    gather_indexed_records,
    init_distributed,
    maybe_empty_cache,
    set_all_seeds,
)
from deltamem.eval.locomo_protocol import (
    ALL_LOCOMO_MECHANISMS,
    CATEGORY_NAMES,
    DEFAULT_LOCOMO_MECHANISMS,
    OFFICIAL_ANSWER_RESERVE_TOKENS,
    OFFICIAL_MAX_NEW_TOKENS,
    OFFICIAL_TEMPERATURE,
    OFFICIAL_TOP_K,
    OFFICIAL_TOP_P,
    build_official_question_prompt,
    build_locomo_history_messages,
    build_official_full_history_messages,
    canonicalize_locomo_prediction,
    infer_model_context_window,
    prepare_locomo_question,
    score_locomo_prediction,
    summarize_locomo_records,
)
from deltamem.chat_templates import apply_chat_template as apply_project_chat_template
from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
)
from deltamem.model_loading import DEFAULT_LOCAL_MODEL_PATH, resolve_attn_implementation
from deltamem.runtime.session import DeltaMemChatSession, get_dtype


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


def attach_delta_adapter_in_place(
    model,
    *,
    adapter_dir: Path | None,
    rank: int,
    alpha: float,
    beta_bias_init: float,
    rankwise_gates: bool,
    output_init: str,
    online_gain: float,
    config_override: HFDeltaMemConfig | None = None,
    load_adapter: bool = True,
) -> HFDeltaMemConfig:
    if config_override is not None:
        config = config_override
    elif adapter_dir is not None:
        config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    else:
        config = HFDeltaMemConfig(
            rank=rank,
            alpha=alpha,
            beta_bias_init=beta_bias_init,
            rankwise_gates=rankwise_gates,
            output_init=output_init,
            online_gain=online_gain,
        )

    attach_delta_mem(model, config)
    if adapter_dir is not None and load_adapter:
        load_delta_mem_adapter(model, adapter_dir)
    return config


def load_base_model(
    *,
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str | None,
) -> tuple[torch.nn.Module, object]:
    resolved_attn_implementation = resolve_attn_implementation(
        model_path,
        attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=get_dtype(dtype),
        device_map={"": device},
        attn_implementation=resolved_attn_implementation,
        local_files_only=True,
    ).eval()
    return model, tokenizer


def load_delta_model(
    *,
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str | None,
    adapter_dir: Path | None,
    rank: int,
    alpha: float,
    beta_bias_init: float,
    rankwise_gates: bool,
    output_init: str,
    online_gain: float,
    config_override: HFDeltaMemConfig | None = None,
    load_adapter: bool = True,
) -> tuple[torch.nn.Module, object, HFDeltaMemConfig]:
    model, tokenizer = load_base_model(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    config = attach_delta_adapter_in_place(
        model,
        adapter_dir=adapter_dir,
        rank=rank,
        alpha=alpha,
        beta_bias_init=beta_bias_init,
        rankwise_gates=rankwise_gates,
        output_init=output_init,
        online_gain=online_gain,
        config_override=config_override,
        load_adapter=load_adapter,
    )
    return model, tokenizer, config


def build_teacher_forced_snapshot(
    model,
    tokenizer,
    device: str,
    history: list[dict[str, str]],
):
    reset_delta_mem_states(model)
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)
    session.messages = [dict(message) for message in history]
    full_ids = session._tokenize_messages(session.messages, add_generation_prompt=False)
    session._ingest_full_ids(full_ids)
    return session.snapshot()

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_DATA_FILE = Path("data/locomo10.json")
DEFAULT_MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH
DEFAULT_LOCOMO_CATEGORIES = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base and Delta-Mem Qwen3 models on the LoCoMo long-conversation QA benchmark."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--data-url", default=DEFAULT_DATA_URL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--beta-bias-init", type=float, default=-1.5)
    parser.add_argument("--rankwise-gates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-init",
        default="base_slice",
        choices=["zero", "base_slice", "random"],
    )
    parser.add_argument("--online-gain", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS)
    parser.add_argument(
        "--answer-reserve-tokens",
        type=int,
        default=OFFICIAL_ANSWER_RESERVE_TOKENS,
        help="Reserved token budget for official full-history generation.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Number of LoCoMo questions to decode together per forward batch.",
    )
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-questions-per-conversation", type=int, default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        type=int,
        choices=sorted(CATEGORY_NAMES),
        default=list(DEFAULT_LOCOMO_CATEGORIES),
        help="LoCoMo categories to evaluate. Defaults to 1 2 3 4, excluding adversarial.",
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--delta-conditions",
        nargs="+",
        default=list(DEFAULT_LOCOMO_MECHANISMS),
        choices=list(ALL_LOCOMO_MECHANISMS),
    )
    parser.add_argument(
        "--full-history-mode",
        default="history_replay",
        choices=["official_prompt", "history_replay"],
        help=(
            "How to evaluate `full_history_replay`. `official_prompt` keeps the old LoCoMo-style "
            "single-prompt generation. `history_replay` replays the full chat history so Delta-Mem "
            "state is built from the same history messages used by the stateful conditions."
        ),
    )
    parser.add_argument(
        "--history-message-granularity",
        default="session",
        choices=["session", "turn"],
        help=(
            "How to chunk LoCoMo history for history-replay/stateful evaluation. `session` keeps one "
            "message per LoCoMo session. `turn` splits each session into per-turn messages so cyclic "
            "routing operates at a finer granularity."
        ),
    )
    parser.add_argument(
        "--history-do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode history-replay probes with sampling instead of greedy decoding.",
    )
    parser.add_argument("--history-temperature", type=float, default=0.4)
    parser.add_argument("--history-top-p", type=float, default=0.9)
    parser.add_argument("--history-top-k", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/qwen3_delta_mem_locomo_eval.json"),
    )
    return parser.parse_args()


def local_question_tasks(
    samples: list[dict],
    context: DistributedContext,
) -> list[tuple[int, int, int]]:
    tasks: list[tuple[int, int, int]] = []
    global_question_idx = 0
    for sample_idx, sample in enumerate(samples):
        for question_idx, _ in enumerate(sample["qa"]):
            if global_question_idx % context.world_size == context.rank:
                tasks.append((global_question_idx, sample_idx, question_idx))
            global_question_idx += 1
    return tasks


def gather_question_records(
    question_records: list[tuple[int, int, dict[str, object]]],
    context: DistributedContext,
) -> list[tuple[int, int, dict[str, object]]] | None:
    if not context.enabled:
        return list(question_records)
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, question_records)
    if context.rank != 0:
        return None
    merged = [item for rank_records in gathered for item in rank_records]
    merged.sort(key=lambda item: (item[0], item[1]))
    return merged


def build_sample_record(
    sample: dict,
    *,
    supported_mechanisms: list[str] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "sample_id": sample["sample_id"],
        "speakers": {
            "speaker_a": sample["conversation"]["speaker_a"],
            "speaker_b": sample["conversation"]["speaker_b"],
        },
        "num_sessions": len(build_locomo_history_messages(sample)) - 1,
        "qa": [],
    }
    if supported_mechanisms is not None:
        record["supported_mechanisms"] = list(supported_mechanisms)
    return record


def merge_question_records(
    samples: list[dict],
    question_records: list[tuple[int, int, dict[str, object]]],
    *,
    supported_mechanisms: list[str] | None = None,
) -> list[dict[str, object]]:
    merged = [
        build_sample_record(sample, supported_mechanisms=supported_mechanisms)
        for sample in samples
    ]
    for sample_idx, _, qa_record in sorted(question_records, key=lambda item: (item[0], item[1])):
        cast(list[dict[str, object]], merged[sample_idx]["qa"]).append(qa_record)
    return merged


def progress_jsonl_path(output_json: Path, context: DistributedContext) -> Path:
    suffix = ".jsonl" if not context.enabled else f".rank{context.rank}.jsonl"
    return output_json.parent / f"{output_json.stem}{suffix}"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def ensure_locomo_data_file(data_file: Path, data_url: str) -> Path:
    if data_file.exists():
        return data_file
    data_file.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(data_url) as response:
        payload = response.read()
    data_file.write_bytes(payload)
    return data_file


def load_locomo_samples(
    data_file: Path,
    *,
    max_conversations: int | None,
    max_questions_per_conversation: int | None,
    categories: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    samples = json.loads(data_file.read_text())
    if max_conversations is not None:
        samples = samples[:max_conversations]

    allowed_categories = None if categories is None else {int(category) for category in categories}
    filtered = []
    for sample in samples:
        sample_copy = dict(sample)
        qa_items = list(sample_copy["qa"])
        if allowed_categories is not None:
            qa_items = [qa for qa in qa_items if int(qa["category"]) in allowed_categories]
        if max_questions_per_conversation is not None:
            qa_items = qa_items[:max_questions_per_conversation]
        sample_copy["qa"] = qa_items
        if qa_items:
            filtered.append(sample_copy)
    return filtered


def infer_expected_question_count(data_file: Path) -> int | None:
    match = re.search(r"(\d+)q(?:_|\b)", data_file.stem.lower())
    if match is None:
        return None
    return int(match.group(1))


def validate_locomo_question_count(
    data_file: Path,
    samples: list[dict],
    *,
    categories: list[int] | tuple[int, ...] | None = None,
) -> None:
    # Category filtering intentionally changes subset size, so filename-derived counts no longer apply.
    if categories is not None and set(int(category) for category in categories) != set(CATEGORY_NAMES):
        return
    expected = infer_expected_question_count(data_file)
    if expected is None:
        return
    actual = sum(len(sample["qa"]) for sample in samples)
    if actual != expected:
        raise ValueError(
            f"LoCoMo subset file {data_file} implies {expected} questions, but loaded {actual}. "
            "Regenerate the subset or pass a correctly-sized data file."
        )


def build_question_prompt(question: dict) -> str:
    compat_question = dict(question)
    if int(compat_question.get("category", 0)) == 5 and "answer" not in compat_question:
        compat_question["answer"] = "No information available"
    spec = prepare_locomo_question(
        compat_question,
        sample_id="compat",
        question_index=0,
        seed=0,
    )
    return build_official_question_prompt(spec)


def generate_official_full_history_answer(
    model,
    tokenizer,
    device: str,
    sample: dict,
    question: dict,
    *,
    question_index: int,
    seed: int,
    max_new_tokens: int,
    answer_reserve_tokens: int,
    do_sample: bool = True,
    temperature: float | None = OFFICIAL_TEMPERATURE,
    top_p: float | None = OFFICIAL_TOP_P,
    top_k: int | None = OFFICIAL_TOP_K,
) -> tuple[str, str]:
    question_spec = prepare_locomo_question(
        question,
        sample_id=str(sample["sample_id"]),
        question_index=question_index,
        seed=seed,
    )
    max_context_tokens = infer_model_context_window(model, tokenizer)
    prompt_messages = build_official_full_history_messages(
        sample,
        tokenizer,
        question_spec,
        max_context_tokens=max_context_tokens,
        answer_reserve_tokens=answer_reserve_tokens,
    )
    input_ids = _chat_template_input_ids(tokenizer, prompt_messages).to(device)
    # Snapshot construction for other mechanisms mutates the model's online state.
    # Reset here so full-history replay always starts from a clean state.
    reset_delta_mem_states(model)
    rng_devices = []
    if torch.cuda.is_available() and device.startswith("cuda"):
        rng_devices = [torch.device(device)]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed + question_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + question_index)
        with torch.inference_mode():
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
            }
            if do_sample:
                generate_kwargs["top_k"] = OFFICIAL_TOP_K if top_k in (None, 0) else int(top_k)
                generate_kwargs["top_p"] = OFFICIAL_TOP_P if top_p is None else float(top_p)
                generate_kwargs["temperature"] = (
                    OFFICIAL_TEMPERATURE if temperature is None else float(temperature)
                )
            outputs = model.generate(**generate_kwargs)
    generated_ids = outputs[0][input_ids.shape[1] :]
    raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    canonical_prediction = canonicalize_locomo_prediction(raw_prediction, question_spec)
    return raw_prediction, canonical_prediction


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return cast(
        str,
        apply_project_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
    )


def _stack_delta_states(delta_states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not delta_states:
        return {}
    keys = tuple(delta_states[0].keys())
    return {
        key: torch.cat([state[key] for state in delta_states], dim=0)
        for key in keys
    }


def _slice_delta_state(
    delta_state: dict[str, torch.Tensor] | None,
    start: int,
    end: int,
) -> dict[str, torch.Tensor] | None:
    if delta_state is None:
        return None
    return {
        key: value[start:end].contiguous()
        for key, value in delta_state.items()
    }


def _generate_prompt_chunk(
    *,
    model,
    tokenizer,
    device: str,
    message_batches: list[list[dict[str, str]]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_seed: int,
    delta_state: dict[str, torch.Tensor] | None,
) -> list[str]:
    rendered_prompts = [_render_chat_prompt(tokenizer, messages) for messages in message_batches]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        tokenized = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = tokenized.input_ids.to(device)
    attention_mask = getattr(tokenized, "attention_mask", None)
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)
    else:
        attention_mask = attention_mask.to(device)

    reset_delta_mem_states(model)
    if delta_state:
        load_delta_mem_online_state(model, delta_state)

    rng_devices = []
    if torch.cuda.is_available() and device.startswith("cuda"):
        rng_devices = [torch.device(device)]

    set_delta_mem_write_enabled(model, False)
    try:
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            with torch.inference_mode():
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "do_sample": do_sample,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                }
                if tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
                if do_sample:
                    generate_kwargs["top_k"] = OFFICIAL_TOP_K if top_k in (None, 0) else int(top_k)
                    generate_kwargs["top_p"] = OFFICIAL_TOP_P if top_p is None else float(top_p)
                    generate_kwargs["temperature"] = (
                        OFFICIAL_TEMPERATURE if temperature is None else float(temperature)
                    )
                outputs = model.generate(**generate_kwargs)
    finally:
        set_delta_mem_write_enabled(model, True)

    generated_ids = outputs[:, input_ids.shape[1] :]
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated_ids]


def _batched_generate_raw_predictions(
    *,
    model,
    tokenizer,
    device: str,
    message_batches: list[list[dict[str, str]]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_seed: int,
    delta_state: dict[str, torch.Tensor] | None,
) -> list[str]:
    if not message_batches:
        return []
    try:
        return _generate_prompt_chunk(
            model=model,
            tokenizer=tokenizer,
            device=device,
            message_batches=message_batches,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            batch_seed=batch_seed,
            delta_state=delta_state,
        )
    except torch.OutOfMemoryError:
        if len(message_batches) == 1:
            raise
        maybe_empty_cache(device)
        split = len(message_batches) // 2
        print(
            f"[locomo_delta] CUDA OOM at batch size {len(message_batches)}; retrying with {split} + {len(message_batches) - split}",
            flush=True,
        )
        first = _batched_generate_raw_predictions(
            model=model,
            tokenizer=tokenizer,
            device=device,
            message_batches=message_batches[:split],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            batch_seed=batch_seed,
            delta_state=_slice_delta_state(delta_state, 0, split),
        )
        second = _batched_generate_raw_predictions(
            model=model,
            tokenizer=tokenizer,
            device=device,
            message_batches=message_batches[split:],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            batch_seed=batch_seed + split,
            delta_state=_slice_delta_state(delta_state, split, len(message_batches)),
        )
        return first + second


def _build_sample_cache(
    *,
    model,
    tokenizer,
    device: str,
    sample: dict,
    condition_names: list[str],
    history_message_granularity: str,
    full_history_mode: str,
) -> dict[str, object]:
    cache: dict[str, object] = {"sample": sample}
    if full_history_mode == "history_replay" and "full_history_replay" in condition_names:
        history_messages = build_locomo_history_messages(
            sample,
            message_granularity=history_message_granularity,
        )
        cache["history_messages"] = history_messages
        cache["full_snapshot"] = build_teacher_forced_snapshot(
            model,
            tokenizer,
            device,
            history_messages,
        )

    return cache


def evaluate_locomo_conditions(
    *,
    model,
    tokenizer,
    device: str,
    samples: list[dict],
    question_tasks: list[tuple[int, int, int]],
    condition_names: list[str],
    max_new_tokens: int,
    seed: int,
    answer_reserve_tokens: int,
    full_history_mode: str,
    history_message_granularity: str,
    history_do_sample: bool,
    history_temperature: float,
    history_top_p: float,
    history_top_k: int,
    eval_batch_size: int,
    progress_bar: Any = None,
    progress_jsonl: Path | None = None,
) -> list[tuple[int, int, dict[str, object]]]:
    records: list[tuple[int, int, dict[str, object]]] = []
    sample_cache: dict[int, dict[str, object]] = {}
    started_at = time.perf_counter()
    total_tasks = len(question_tasks)
    effective_batch_size = max(1, int(eval_batch_size))
    max_context_tokens = None
    if full_history_mode == "official_prompt" and "full_history_replay" in condition_names:
        max_context_tokens = infer_model_context_window(model, tokenizer)

    for batch_start in range(0, total_tasks, effective_batch_size):
        batch_tasks = question_tasks[batch_start : batch_start + effective_batch_size]
        batch_items: list[dict[str, object]] = []
        for _, sample_idx, question_idx in batch_tasks:
            sample = samples[sample_idx]
            if sample_idx not in sample_cache:
                sample_cache[sample_idx] = _build_sample_cache(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    sample=sample,
                    condition_names=condition_names,
                    history_message_granularity=history_message_granularity,
                    full_history_mode=full_history_mode,
                )
            cache = sample_cache[sample_idx]
            qa = sample["qa"][question_idx]
            question_spec = prepare_locomo_question(
                qa,
                sample_id=str(sample["sample_id"]),
                question_index=question_idx,
                seed=seed,
            )
            batch_items.append(
                {
                    "sample_idx": sample_idx,
                    "question_idx": question_idx,
                    "sample": sample,
                    "qa": qa,
                    "cache": cache,
                    "question_spec": question_spec,
                    "question_prompt": build_official_question_prompt(question_spec),
                    "qa_record": {
                        "question": qa["question"],
                        "answer": qa.get("answer"),
                        "adversarial_answer": qa.get("adversarial_answer"),
                        "evidence": list(qa.get("evidence", [])),
                        "category": int(qa["category"]),
                        "conditions": {},
                    },
                }
            )

        for condition_offset, condition_name in enumerate(condition_names):
            if condition_name == "full_history_replay" and full_history_mode == "official_prompt":
                for item in batch_items:
                    qa = cast(dict, item["qa"])
                    question_spec = cast(Any, item["question_spec"])
                    question_idx = cast(int, item["question_idx"])
                    sample = cast(dict, item["sample"])
                    raw_prediction, prediction = generate_official_full_history_answer(
                        model,
                        tokenizer,
                        device,
                        sample,
                        qa,
                        question_index=question_idx,
                        seed=seed,
                        max_new_tokens=max_new_tokens,
                        answer_reserve_tokens=answer_reserve_tokens,
                        do_sample=history_do_sample,
                        temperature=history_temperature,
                        top_p=history_top_p,
                        top_k=history_top_k,
                    )
                    cast(dict[str, object], cast(dict[str, object], item["qa_record"])["conditions"])[condition_name] = {
                        "prediction": prediction,
                        "raw_prediction": raw_prediction,
                        "score": round(score_locomo_prediction(qa, prediction), 4),
                        "turn_stats": {
                            "official_aligned": True,
                            "answer_reserve_tokens": answer_reserve_tokens,
                            "condition_name": condition_name,
                            "full_history_mode": full_history_mode,
                            "history_message_granularity": history_message_granularity,
                        },
                    }
                continue

            message_batches: list[list[dict[str, str]]] = []
            delta_states: list[dict[str, torch.Tensor]] = []
            use_delta_state = False

            for item in batch_items:
                cache = cast(dict[str, object], item["cache"])
                question_prompt = cast(str, item["question_prompt"])
                if condition_name == "full_history_replay":
                    snapshot = cast(Any, cache.get("full_snapshot"))
                    history_messages = cast(list[dict[str, str]], cache["history_messages"])
                    prompt_messages = history_messages + [{"role": "user", "content": question_prompt}]
                    if snapshot is not None and getattr(snapshot, "delta_state", None):
                        use_delta_state = True
                        delta_states.append(cast(dict[str, torch.Tensor], snapshot.delta_state))
                else:
                    raise ValueError(f"Unsupported LoCoMo condition: {condition_name}")
                message_batches.append(prompt_messages)

            delta_state = _stack_delta_states(delta_states) if use_delta_state else None
            raw_predictions = _batched_generate_raw_predictions(
                model=model,
                tokenizer=tokenizer,
                device=device,
                message_batches=message_batches,
                max_new_tokens=max_new_tokens,
                do_sample=history_do_sample,
                temperature=history_temperature,
                top_p=history_top_p,
                top_k=history_top_k,
                batch_seed=seed + batch_start + (condition_offset * 100003),
                delta_state=delta_state,
            )

            for item, raw_prediction in zip(batch_items, raw_predictions):
                qa = cast(dict, item["qa"])
                question_spec = cast(Any, item["question_spec"])
                prediction = canonicalize_locomo_prediction(raw_prediction, question_spec)
                turn_stats: dict[str, object] = {
                    "batched_eval": True,
                    "eval_batch_size": len(batch_items),
                    "condition_name": condition_name,
                    "full_history_mode": full_history_mode,
                    "history_message_granularity": history_message_granularity,
                }
                cast(dict[str, object], cast(dict[str, object], item["qa_record"])["conditions"])[condition_name] = {
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "score": round(score_locomo_prediction(qa, prediction), 4),
                    "turn_stats": turn_stats,
                }

        for item in batch_items:
            sample_idx = cast(int, item["sample_idx"])
            question_idx = cast(int, item["question_idx"])
            sample = cast(dict, item["sample"])
            qa_record = cast(dict[str, object], item["qa_record"])
            records.append((sample_idx, question_idx, qa_record))
            if progress_jsonl is not None:
                append_jsonl(
                    progress_jsonl,
                    {
                        "sample_idx": sample_idx,
                        "question_idx": question_idx,
                        "sample_id": sample["sample_id"],
                        "question": qa_record["question"],
                        "category": qa_record["category"],
                        "conditions": qa_record["conditions"],
                    },
                )

        if progress_bar is not None:
            completed = batch_start + len(batch_items)
            elapsed = time.perf_counter() - started_at
            avg_seconds = elapsed / max(completed, 1)
            remaining = max(total_tasks - completed, 0)
            eta_seconds = avg_seconds * remaining
            progress_bar.set_postfix_str(
                f"avg={avg_seconds:.1f}s/q eta={_format_eta(eta_seconds)} batch={len(batch_items)}",
                refresh=True,
            )
            progress_bar.update(len(batch_items))
    return records


def main() -> None:
    args = parse_args()
    context = init_distributed(args.device)
    progress_bar = None
    try:
        set_all_seeds(args.seed)
        data_file = ensure_locomo_data_file(args.data_file, args.data_url)
        samples = load_locomo_samples(
            data_file,
            max_conversations=args.max_conversations,
            max_questions_per_conversation=args.max_questions_per_conversation,
            categories=args.categories,
        )
        validate_locomo_question_count(data_file, samples, categories=args.categories)
        question_tasks = local_question_tasks(samples, context)
        local_num_questions = len(question_tasks)
        num_eval_passes = int(not args.skip_base) + int(args.adapter_dir is not None)
        total_progress_steps = local_num_questions * max(num_eval_passes, 1)
        if context.rank == 0:
            progress_bar = tqdm(
                total=total_progress_steps,
                desc="locomo_eval",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        progress_jsonl = progress_jsonl_path(args.output_json, context)
        if progress_jsonl.exists():
            progress_jsonl.unlink()

        payload: dict[str, object] = {
            "model_path": args.model_path,
            "adapter_dir": None if args.adapter_dir is None else str(args.adapter_dir),
            "data_file": str(data_file),
            "data_url": args.data_url,
            "distributed": {
                "enabled": context.enabled,
                "world_size": context.world_size,
            },
            "num_conversations": len(samples),
            "num_questions": sum(len(sample["qa"]) for sample in samples),
            "categories": list(args.categories),
            "category_names": {str(category): CATEGORY_NAMES[category] for category in args.categories},
            "eval_batch_size": args.eval_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "answer_reserve_tokens": args.answer_reserve_tokens,
            "full_history_mode": args.full_history_mode,
            "history_message_granularity": args.history_message_granularity,
            "history_decode": {
                "do_sample": args.history_do_sample,
                "temperature": args.history_temperature,
                "top_p": args.history_top_p,
                "top_k": args.history_top_k,
            },
        }

        base_model = None
        base_tokenizer = None
        if not args.skip_base:
            if progress_bar is not None:
                progress_bar.set_description_str("locomo_base")
            base_model, base_tokenizer = load_base_model(
                model_path=args.model_path,
                device=context.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
            )
            base_condition_names = ["full_history_replay"]
            base_records_local = evaluate_locomo_conditions(
                model=base_model,
                tokenizer=base_tokenizer,
                device=context.device,
                samples=samples,
                question_tasks=question_tasks,
                condition_names=base_condition_names,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                answer_reserve_tokens=args.answer_reserve_tokens,
                full_history_mode=args.full_history_mode,
                history_message_granularity=args.history_message_granularity,
                history_do_sample=args.history_do_sample,
                history_temperature=args.history_temperature,
                history_top_p=args.history_top_p,
                history_top_k=args.history_top_k,
                eval_batch_size=args.eval_batch_size,
                progress_bar=progress_bar,
                progress_jsonl=progress_jsonl,
            )
            base_question_records = gather_question_records(base_records_local, context)
            if context.rank == 0:
                assert base_question_records is not None
                base_records = merge_question_records(samples, base_question_records)
                payload["base"] = {
                    "records": base_records,
                    "summary": summarize_locomo_records(base_records, condition_names=base_condition_names),
                }
            if args.adapter_dir is None:
                del base_model
                base_model = None
                base_tokenizer = None
                maybe_empty_cache(context.device)

        if context.enabled:
            if context.device.startswith("cuda"):
                dist.barrier(device_ids=[context.local_rank])
            else:
                dist.barrier()

        if args.adapter_dir is not None:
            if progress_bar is not None:
                progress_bar.set_description_str("locomo_delta")
            if base_model is not None and base_tokenizer is not None:
                delta_model = base_model
                delta_tokenizer = base_tokenizer
                delta_config_obj = attach_delta_adapter_in_place(
                    delta_model,
                    adapter_dir=args.adapter_dir,
                    rank=args.rank,
                    alpha=args.alpha,
                    beta_bias_init=args.beta_bias_init,
                    rankwise_gates=args.rankwise_gates,
                    output_init=args.output_init,
                    online_gain=args.online_gain,
                )
                delta_config = delta_config_obj.to_dict()
                base_model = None
                base_tokenizer = None
            else:
                delta_model, delta_tokenizer, delta_config_obj = load_delta_model(
                    model_path=args.model_path,
                    device=context.device,
                    dtype=args.dtype,
                    attn_implementation=args.attn_implementation,
                    adapter_dir=args.adapter_dir,
                    rank=args.rank,
                    alpha=args.alpha,
                    beta_bias_init=args.beta_bias_init,
                    rankwise_gates=args.rankwise_gates,
                    output_init=args.output_init,
                    online_gain=args.online_gain,
                )
                delta_config = delta_config_obj.to_dict()
            delta_records_local = evaluate_locomo_conditions(
                model=delta_model,
                tokenizer=delta_tokenizer,
                device=context.device,
                samples=samples,
                question_tasks=question_tasks,
                condition_names=args.delta_conditions,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                answer_reserve_tokens=args.answer_reserve_tokens,
                full_history_mode=args.full_history_mode,
                history_message_granularity=args.history_message_granularity,
                history_do_sample=args.history_do_sample,
                history_temperature=args.history_temperature,
                history_top_p=args.history_top_p,
                history_top_k=args.history_top_k,
                eval_batch_size=args.eval_batch_size,
                progress_bar=progress_bar,
                progress_jsonl=progress_jsonl,
            )
            delta_question_records = gather_question_records(delta_records_local, context)
            if context.rank == 0:
                assert delta_question_records is not None
                delta_records = merge_question_records(samples, delta_question_records)
                payload["delta"] = {
                    "config": delta_config,
                    "records": delta_records,
                    "summary": summarize_locomo_records(delta_records, condition_names=args.delta_conditions),
                }
            del delta_model
            maybe_empty_cache(context.device)

        if context.rank == 0:
            if progress_bar is not None:
                progress_bar.close()
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2))
            print(json.dumps(payload, indent=2))
    finally:
        if "progress_bar" in locals() and progress_bar is not None:
            progress_bar.close()
        finalize_distributed(context)


if __name__ == "__main__":
    main()
