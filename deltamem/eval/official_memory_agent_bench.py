from __future__ import annotations

import json
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

from deltamem.eval.common import finalize_distributed
from deltamem.eval.official_eval_utils import (
    DEFAULT_MEMORY_AGENT_BENCH_ROOT,
    append_jsonl_record,
    build_common_arg_parser,
    find_resume_record,
    gather_indexed_records,
    generate_from_message_batches,
    infer_model_context_window,
    init_distributed,
    load_dataset_cached,
    load_external_module,
    load_jsonl_records,
    load_model_for_eval,
    make_resume_prompt_prefix,
    resolve_hub_file,
    set_all_seeds,
    truncate_text_by_tokens,
)
from deltamem.eval.official_memory_agent_bench_templates import get_template


OFFICIAL_SOURCE_CONFIGS = {
    "eventqa_131072": {"context_max_length": 131072, "generation_max_length": 40, "max_test_samples": 5, "chunk_size": 4096},
    "eventqa_65536": {"context_max_length": 65536, "generation_max_length": 40, "max_test_samples": 5, "chunk_size": 4096},
    "eventqa_full": {"context_max_length": 800000, "generation_max_length": 40, "max_test_samples": 5, "chunk_size": 4096},
    "longmemeval_s_-1_500": {"context_max_length": 150000, "generation_max_length": 50, "max_test_samples": 500, "chunk_size": 4096},
    "longmemeval_s*": {"context_max_length": 400000, "generation_max_length": 50, "max_test_samples": 5, "chunk_size": 4096},
    "ruler_qa1_197K": {"context_max_length": 220000, "generation_max_length": 50, "max_test_samples": 100, "chunk_size": 4096},
    "ruler_qa2_421K": {"context_max_length": 524288, "generation_max_length": 50, "max_test_samples": 100, "chunk_size": 4096},
    "factconsolidation_mh_262k": {"context_max_length": 300000, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_mh_32k": {"context_max_length": 32768, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_mh_64k": {"context_max_length": 65536, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_mh_6k": {"context_max_length": 6000, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_sh_262k": {"context_max_length": 300000, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_sh_32k": {"context_max_length": 32768, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_sh_64k": {"context_max_length": 65536, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_sh_6k": {"context_max_length": 6000, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "detective_qa": {"context_max_length": 200000, "generation_max_length": 2000, "max_test_samples": 10, "chunk_size": 4096},
    "infbench_sum_eng_shots2": {"context_max_length": 1200000, "generation_max_length": 1200, "max_test_samples": 100, "chunk_size": 4096},
    "icl_banking77_5900shot_balance": {"context_max_length": 131072, "generation_max_length": 20, "max_test_samples": 100, "chunk_size": 4096},
    "icl_clinic150_7050shot_balance": {"context_max_length": 131072, "generation_max_length": 20, "max_test_samples": 100, "chunk_size": 4096},
    "icl_nlu_8296shot_balance": {"context_max_length": 131072, "generation_max_length": 20, "max_test_samples": 100, "chunk_size": 4096},
    "icl_trec_coarse_6600shot_balance": {"context_max_length": 131072, "generation_max_length": 20, "max_test_samples": 100, "chunk_size": 4096},
    "icl_trec_fine_6400shot_balance": {"context_max_length": 131072, "generation_max_length": 20, "max_test_samples": 100, "chunk_size": 4096},
    "recsys_redial_full": {"context_max_length": 1480000, "generation_max_length": 300, "max_test_samples": 1, "chunk_size": 4096},
}


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _install_editdistance_shim() -> None:
    if "editdistance" in sys.modules:
        return
    module = types.ModuleType("editdistance")
    setattr(module, "eval", _levenshtein_distance)
    sys.modules["editdistance"] = module


def load_mab_eval_utils(repo_root: Path):
    _install_editdistance_shim()
    return load_external_module(
        "official_memory_agent_bench_eval_utils",
        str(repo_root / "utils" / "eval_other_utils.py"),
    )


def parse_args():
    parser = build_common_arg_parser("Official-compatible MemoryAgentBench evaluation for Delta-Mem models")
    parser.add_argument("--split", required=True, choices=["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution"])
    parser.add_argument("--source", required=True, choices=sorted(OFFICIAL_SOURCE_CONFIGS))
    parser.add_argument("--context-max-length", type=int, default=0)
    parser.add_argument("--generation-max-length", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--buffer-length", type=int, default=4000)
    parser.add_argument("--input-length-limit", type=int, default=0)
    parser.add_argument("--external-memory-agent-bench-root", type=Path, default=DEFAULT_MEMORY_AGENT_BENCH_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_source_config(args) -> dict[str, int]:
    config = dict(OFFICIAL_SOURCE_CONFIGS[args.source])
    if args.context_max_length > 0:
        config["context_max_length"] = args.context_max_length
    if args.generation_max_length > 0:
        config["generation_max_length"] = args.generation_max_length
    if args.max_test_samples > 0:
        config["max_test_samples"] = args.max_test_samples
    if args.chunk_size > 0:
        config["chunk_size"] = args.chunk_size
    return config


def load_source_rows(args, source_config: dict[str, int]) -> list[dict]:
    parquet_path = resolve_hub_file(
        repo_id="ai-hyz/MemoryAgentBench",
        repo_type="dataset",
        filename=f"data/{args.split}-00000-of-00001.parquet",
        hub_cache_dir=args.hub_cache_dir,
        local_files_only=args.local_files_only,
    )
    dataset = load_dataset_cached(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        cache_dir=args.datasets_cache_dir,
        local_files_only=args.local_files_only,
    )
    rows: list[dict] = []
    for row in dataset:
        metadata = dict(row.get("metadata") or {})
        if metadata.get("source", "") != args.source:
            continue
        rows.append(
            {
                "context": str(row.get("context", "")),
                "questions": list(row.get("questions", []) or []),
                "answers": list(row.get("answers", []) or []),
                "metadata": metadata,
            }
        )
        if len(rows) >= source_config["max_test_samples"]:
            break
    return rows


def metadata_value(metadata: dict, key: str, question_index: int, default=None):
    values = metadata.get(key)
    if isinstance(values, list) and question_index < len(values):
        return values[question_index]
    return default


def build_dataset_config(args, source_config: dict[str, int]) -> dict[str, object]:
    return {
        "dataset": args.split,
        "sub_dataset": args.source,
        "chunk_size": source_config["chunk_size"],
        "debug": bool(args.debug),
        "seed": args.seed,
        "context_max_length": source_config["context_max_length"],
        "generation_max_length": source_config["generation_max_length"],
        "test_files": "",
        "demo_files": "",
        "use_chat_template": not args.no_chat_template,
        "max_test_samples": source_config["max_test_samples"],
        "shots": 0,
        "tag": None,
    }


def build_agent_config(args) -> dict[str, object]:
    return {
        "agent_name": "Long_context_agent_deltamem",
        "input_length_limit": args.input_length_limit,
        "buffer_length": args.buffer_length,
        "temperature": args.temperature,
    }


def _template_label_value(answer: object) -> str:
    if isinstance(answer, list):
        if not answer:
            return ""
        return str(answer[0]).strip()
    return str(answer).strip()


def build_query_answer_pairs(row: dict, *, source: str) -> list[tuple[str, object, object]]:
    query_template = get_template(source, "query", "Long_context_agent_deltamem")
    questions = row.get("questions") or []
    answers = row.get("answers") or []
    metadata = row.get("metadata") or {}
    pairs: list[tuple[str, object, object]] = []
    for question_index, question in enumerate(questions):
        answer = answers[question_index] if question_index < len(answers) else ""
        qa_metadata = {
            "question": question,
            "answer": answer,
            "label": _template_label_value(answer),
            "source": metadata.get("source", ""),
            "question_dates": metadata_value(metadata, "question_dates", question_index),
            "question_types": metadata_value(metadata, "question_types", question_index),
            "question_ids": metadata_value(metadata, "question_ids", question_index),
            "previous_events": metadata_value(metadata, "previous_events", question_index),
            "qa_pair_ids": metadata_value(metadata, "qa_pair_ids", question_index),
        }
        formatted_query = query_template.format(**qa_metadata)
        pairs.append((formatted_query, answer, qa_metadata.get("qa_pair_ids")))
    return pairs


def build_context_chunks(rows: list[dict], *, chunk_size: int, eval_utils_module) -> list[list[str]]:
    return [
        eval_utils_module.chunk_text_into_sentences(str(row.get("context", "")), chunk_size=chunk_size)
        for row in rows
    ]


def build_memorized_context(source: str, chunks: list[str]) -> str:
    memorize_template = get_template(source, "memorize", "Long_context_agent_deltamem")
    memorized = ""
    for chunk in chunks:
        format_kwargs = {"context": chunk}
        if "{time_stamp}" in memorize_template:
            format_kwargs["time_stamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        memorized += "\n" + memorize_template.format(**format_kwargs)
    return memorized.strip()


def truncate_memory_context(
    memory_context: str,
    *,
    tokenizer,
    context_max_length: int,
    raw_input_length_limit: int,
    buffer_length: int,
    generation_max_length: int,
) -> str:
    if raw_input_length_limit <= 0:
        return memory_context
    effective_input_length_limit = max(1, raw_input_length_limit - buffer_length - generation_max_length)
    if effective_input_length_limit > context_max_length + buffer_length:
        return memory_context
    truncated = memory_context
    if context_max_length > 0:
        truncated = truncate_text_by_tokens(
            truncated,
            tokenizer=tokenizer,
            max_tokens=context_max_length,
            keep="tail",
        )
    return truncate_text_by_tokens(
        truncated,
        tokenizer=tokenizer,
        max_tokens=effective_input_length_limit,
        keep="tail",
    )


def summarize_metrics(metrics: dict[str, list[object]]) -> dict[str, float]:
    averaged: dict[str, float] = {}
    for key, values in metrics.items():
        if not values:
            averaged[key] = 0.0
            continue
        numeric_values = [float(value) for value in values if isinstance(value, (int, float, bool))]
        if not numeric_values:
            averaged[key] = 0.0
            continue
        scale = 1.0 if ("_len" in key or "_time" in key) else 100.0
        averaged[key] = round(sum(numeric_values) / len(numeric_values) * scale, 4)
    return averaged


def main() -> None:
    args = parse_args()
    context = init_distributed(args.device)
    try:
        set_all_seeds(args.seed)
        source_config = resolve_source_config(args)
        dataset_config = build_dataset_config(args, source_config)
        agent_config = build_agent_config(args)
        eval_utils = load_mab_eval_utils(args.external_memory_agent_bench_root)

        rows = load_source_rows(args, source_config)
        context_chunks = build_context_chunks(rows, chunk_size=source_config["chunk_size"], eval_utils_module=eval_utils)
        query_answer_pairs = [build_query_answer_pairs(row, source=args.source) for row in rows]

        variant_name, model, tokenizer, _ = load_model_for_eval(args, context)
        inferred_context_window = infer_model_context_window(model, tokenizer)
        raw_input_length_limit = args.input_length_limit if args.input_length_limit > 0 else inferred_context_window
        system_message = get_template(args.source, "system", "Long_context_agent_deltamem")
        do_sample = bool(args.do_sample or args.temperature > 0)

        indexed_records: list[tuple[int, dict[str, object]]] = []
        local_metrics: dict[str, list[object]] = defaultdict(list)
        local_results: list[dict[str, object]] = []
        local_time_cost: list[float] = []
        start_time = time.time()
        existing_records = load_jsonl_records(args.output_jsonl) if args.output_jsonl is not None and (not context.enabled or context.rank == 0) else []
        global_record_index = 0

        for context_idx, (chunks, qa_pairs) in enumerate(zip(context_chunks, query_answer_pairs)):
            if context.enabled and context_idx % context.world_size != context.rank:
                global_record_index += len(qa_pairs)
                continue
            memorized_context = build_memorized_context(args.source, chunks)
            truncated_context = truncate_memory_context(
                memorized_context,
                tokenizer=tokenizer,
                context_max_length=source_config["context_max_length"],
                raw_input_length_limit=raw_input_length_limit,
                buffer_length=args.buffer_length,
                generation_max_length=source_config["generation_max_length"],
            )
            pending_items: list[tuple[int, int, str, object, object, str]] = []
            pending_batches: list[list[dict[str, str]]] = []
            for question_idx, (query, answer, qa_pair_id) in enumerate(qa_pairs):
                user_prompt = f"{truncated_context}\n{query}".strip()
                record_index = global_record_index
                global_record_index += 1
                resume_record = find_resume_record(
                    existing_records,
                    prompt_text=user_prompt,
                    question_text=query,
                )
                if resume_record is not None:
                    indexed_records.append((record_index, dict(resume_record)))
                    local_time_cost.append(0.0)
                    continue
                pending_items.append((record_index, question_idx, query, answer, qa_pair_id, user_prompt))
                pending_batches.append(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt},
                    ]
                )
            for start in range(0, len(pending_items), max(1, args.eval_batch_size)):
                chunk_items = pending_items[start : start + max(1, args.eval_batch_size)]
                chunk_batches = pending_batches[start : start + max(1, args.eval_batch_size)]
                outputs = generate_from_message_batches(
                    model,
                    tokenizer,
                    context.device,
                    chunk_batches,
                    max_new_tokens=source_config["generation_max_length"],
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    batch_size=max(1, args.eval_batch_size),
                    reset_delta_state=False,
                    use_chat_template=not args.no_chat_template,
                )
                for (record_index, question_idx, query, answer, qa_pair_id, user_prompt), output in zip(chunk_items, outputs):
                    metrics_before = len(local_results)
                    local_metrics, local_results = eval_utils.metrics_summarization(
                        dict(output),
                        query,
                        answer,
                        dataset_config,
                        local_metrics,
                        local_results,
                        query_id=metrics_before,
                        qa_pair_id=qa_pair_id,
                    )
                    result_record = local_results[-1]
                    result_record["record_index"] = record_index
                    result_record["context_id"] = context_idx
                    result_record["question_index"] = question_idx
                    result_record["split"] = args.split
                    result_record["source"] = args.source
                    result_record["query"] = query
                    result_record["resume_question"] = query
                    result_record["user_prompt"] = user_prompt
                    result_record["resume_prompt_prefix"] = make_resume_prompt_prefix(user_prompt)
                    indexed_records.append((record_index, dict(result_record)))
                    local_time_cost.append(time.time() - start_time)
                    if args.output_jsonl is not None and (not context.enabled or context.rank == 0):
                        append_jsonl_record(args.output_jsonl, dict(result_record))
                        existing_records.append(dict(result_record))

        merged_records = gather_indexed_records(indexed_records, context)
        if context.rank != 0:
            return
        assert merged_records is not None

        # Rebuild aggregate metrics from merged records to avoid rank-local bias.
        merged_metrics: dict[str, list[object]] = defaultdict(list)
        merged_results: list[dict[str, object]] = []
        for record in merged_records:
            for key, value in record.items():
                if key in {
                    "exact_match", "f1", "substring_exact_match", "rougeL_f1", "rougeL_recall",
                    "rougeLsum_f1", "rougeLsum_recall", "eventqa_recall", "input_len", "output_len",
                    "memory_construction_time", "query_time_len", "recsys_recall@1", "recsys_recall@5",
                    "recsys_recall@10", "parsed_output"
                }:
                    if isinstance(value, (int, float, bool)):
                        merged_metrics[key].append(value)
            merged_results.append(record)
        payload = {
            "agent_config": {
                **agent_config,
                "resolved_variant": variant_name,
                "resolved_input_length_limit": raw_input_length_limit,
            },
            "dataset_config": dataset_config,
            "data": merged_results,
            "metrics": dict(merged_metrics),
            "time_cost": local_time_cost,
            "averaged_metrics": summarize_metrics(merged_metrics),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    finally:
        finalize_distributed(context)


if __name__ == "__main__":
    main()
