from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from deltamem.eval.common import (
    load_base_model_and_tokenizer,
    load_delta_model_and_tokenizer,
    maybe_empty_cache,
    set_all_seeds,
)
from deltamem.eval.official_eval_utils import infer_model_context_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generation TPS and peak memory.")
    parser.add_argument("--model-kinds", nargs="+", choices=["base", "delta"], default=["base"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[4096, 16384, 32768])
    parser.add_argument("--decode-lengths", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--measurement-mode",
        choices=["decode_only", "full_generate"],
        default="full_generate",
        help="full_generate times prefill plus generation; decode_only times cached decode steps.",
    )
    return parser.parse_args()


def synchronize(device: str) -> None:
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize(device=device)


def choose_fill_token_ids(tokenizer) -> list[int]:
    for text in (
        "Memory benchmark sentence. Another factual sentence.\n",
        "This is a realistic benchmark prompt with punctuation and new lines.\n",
        "hello world\n",
    ):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if token_ids:
            return [int(token_id) for token_id in token_ids]
    for candidate in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ):
        if isinstance(candidate, int) and candidate >= 0:
            return [int(candidate)]
    return [0]


def load_model_for_benchmark(args: argparse.Namespace, model_kind: str):
    if model_kind == "base":
        return load_base_model_and_tokenizer(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
    if model_kind == "delta":
        if args.adapter_dir is None:
            raise ValueError("--adapter-dir is required when --model-kinds includes delta")
        model, tokenizer, _ = load_delta_model_and_tokenizer(
            model_path=args.model_path,
            adapter_dir=args.adapter_dir,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        return model, tokenizer
    raise ValueError(f"Unsupported model kind: {model_kind}")


def make_prompt_tensors(
    *,
    prompt_len: int,
    batch_size: int,
    fill_token_ids: list[int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fill = torch.tensor(fill_token_ids, dtype=torch.long, device=device)
    repeats = (prompt_len + fill.numel() - 1) // fill.numel()
    prompt_row = fill.repeat(repeats)[:prompt_len]
    input_ids = prompt_row.unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _supports_kwarg(model, name: str) -> bool:
    try:
        parameters = inspect.signature(model.forward).parameters.values()
    except (TypeError, ValueError):
        return False
    for parameter in parameters:
        if parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _prefill_standard(model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
    }
    if _supports_kwarg(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1
    with torch.inference_mode():
        return model(**kwargs)


def _decode_step_standard(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
):
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "use_cache": True,
    }
    if _supports_kwarg(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1
    with torch.inference_mode():
        return model(**kwargs)


def run_single_pass(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_len: int,
    device: str,
    measurement_mode: str,
    tokenizer,
) -> dict[str, float]:
    if decode_len <= 1:
        raise ValueError("decode_len must be greater than 1 for TPS benchmarking")
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("CUDA is required because this benchmark reports peak GPU memory.")

    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device=device)
    baseline_allocated_bytes = torch.cuda.memory_allocated(device=device)
    baseline_reserved_bytes = torch.cuda.memory_reserved(device=device)

    if measurement_mode == "full_generate":
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": decode_len,
            "use_cache": True,
            "do_sample": False,
        }
        if getattr(tokenizer, "pad_token_id", None) is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        started.record()
        with torch.inference_mode():
            outputs = model.generate(**generate_kwargs)
        ended.record()
        produced_tokens = float(max(0, outputs.shape[1] - input_ids.shape[1]) * input_ids.shape[0])
    else:
        outputs = _prefill_standard(model, input_ids=input_ids, attention_mask=attention_mask)
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        running_attention_mask = attention_mask

        started.record()
        for _ in range(decode_len - 1):
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones(
                        (running_attention_mask.shape[0], 1),
                        dtype=running_attention_mask.dtype,
                        device=running_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            outputs = _decode_step_standard(
                model,
                input_ids=next_tokens,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
            )
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
        ended.record()
        produced_tokens = float((decode_len - 1) * input_ids.shape[0])

    synchronize(device)

    elapsed_seconds = started.elapsed_time(ended) / 1000.0
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device=device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device=device)
    return {
        "elapsed_seconds": elapsed_seconds,
        "generated_tokens": produced_tokens,
        "peak_allocated_bytes": float(peak_allocated_bytes),
        "peak_reserved_bytes": float(peak_reserved_bytes),
        "baseline_allocated_bytes": float(baseline_allocated_bytes),
        "baseline_reserved_bytes": float(baseline_reserved_bytes),
    }


def aggregate_metrics(runs: list[dict[str, float]]) -> dict[str, float]:
    elapsed_seconds = sum(run["elapsed_seconds"] for run in runs) / len(runs)
    generated_tokens = sum(run["generated_tokens"] for run in runs) / len(runs)
    peak_allocated_bytes = max(run["peak_allocated_bytes"] for run in runs)
    peak_reserved_bytes = max(run["peak_reserved_bytes"] for run in runs)
    baseline_allocated_bytes = max(run["baseline_allocated_bytes"] for run in runs)
    baseline_reserved_bytes = max(run["baseline_reserved_bytes"] for run in runs)
    return {
        "tps": generated_tokens / elapsed_seconds,
        "peak_alloc_gb": peak_allocated_bytes / (1024**3),
        "peak_reserved_gb": peak_reserved_bytes / (1024**3),
        "baseline_alloc_gb": baseline_allocated_bytes / (1024**3),
        "baseline_reserved_gb": baseline_reserved_bytes / (1024**3),
        "elapsed_seconds": elapsed_seconds,
        "generated_tokens": generated_tokens,
    }


def format_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| model_kind | prompt_len | decode_len | tps | peak_alloc_gb | peak_reserved_gb |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        if row.get("status") == "skipped":
            lines.append(
                f"| {row['model_kind']} | {row['prompt_len']} | {row['decode_len']} | skipped | skipped | skipped |"
            )
            continue
        lines.append(
            "| {model_kind} | {prompt_len} | {decode_len} | {tps:.2f} | {peak_alloc_gb:.2f} | {peak_reserved_gb:.2f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("CUDA is required because this benchmark reports peak GPU memory.")

    set_all_seeds(args.seed)
    metadata = {
        "model_kinds": args.model_kinds,
        "model_path": args.model_path,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "batch_size": args.batch_size,
        "prompt_lengths": args.prompt_lengths,
        "decode_lengths": args.decode_lengths,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "measurement_mode": args.measurement_mode,
    }
    print(json.dumps(metadata, indent=2), flush=True)

    results: list[dict[str, Any]] = []
    contexts: dict[str, int] = {}
    for model_kind in args.model_kinds:
        model, tokenizer = load_model_for_benchmark(args, model_kind)
        model.eval()
        context_window = infer_model_context_window(model, tokenizer)
        contexts[model_kind] = context_window
        fill_token_ids = choose_fill_token_ids(tokenizer)
        print(f"[bench] loaded model_kind={model_kind} context_window={context_window}", flush=True)

        for prompt_len in args.prompt_lengths:
            for decode_len in args.decode_lengths:
                if prompt_len > context_window:
                    results.append(
                        {
                            "model_kind": model_kind,
                            "prompt_len": prompt_len,
                            "decode_len": decode_len,
                            "status": "skipped",
                            "reason": f"prompt_len exceeds inferred context window ({context_window})",
                        }
                    )
                    continue

                input_ids, attention_mask = make_prompt_tensors(
                    prompt_len=prompt_len,
                    batch_size=args.batch_size,
                    fill_token_ids=fill_token_ids,
                    device=args.device,
                )

                print(
                    f"[bench] model={model_kind} prompt={prompt_len} decode={decode_len} "
                    f"warmup={args.warmup_runs} measure={args.measure_runs}",
                    flush=True,
                )
                for _ in range(args.warmup_runs):
                    _ = run_single_pass(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decode_len=decode_len,
                        device=args.device,
                        measurement_mode=args.measurement_mode,
                        tokenizer=tokenizer,
                    )
                    maybe_empty_cache(args.device)

                measured_runs = []
                for _ in range(args.measure_runs):
                    measured_runs.append(
                        run_single_pass(
                            model,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decode_len=decode_len,
                            device=args.device,
                            measurement_mode=args.measurement_mode,
                            tokenizer=tokenizer,
                        )
                    )
                    maybe_empty_cache(args.device)

                row = {
                    "model_kind": model_kind,
                    "prompt_len": prompt_len,
                    "decode_len": decode_len,
                    **aggregate_metrics(measured_runs),
                }
                results.append(row)
                print(json.dumps(row, indent=2), flush=True)

        del model
        maybe_empty_cache(args.device)

    payload = {
        **metadata,
        "context_windows": contexts,
        "results": results,
    }

    print("\n" + format_table(results), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
