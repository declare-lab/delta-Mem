from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat demo for base, LoRA, or Delta-Mem models.")
    parser.add_argument("--mode", choices=["base", "delta", "lora"], default="delta")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--dump-state-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from deltamem.runtime.chat_base import load_qwen3_base_model
    from deltamem.runtime.chat_cli import run_chat_loop
    from deltamem.runtime.chat_lora import load_qwen3_lora_model
    from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model

    if args.mode == "base":
        model, tokenizer = load_qwen3_base_model(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        banner = "Base model chat ready."
        allow_snapshot_commands = False
    elif args.mode == "lora":
        if args.adapter_dir is None:
            raise ValueError("--adapter-dir is required for mode=lora")
        model, tokenizer = load_qwen3_lora_model(
            model_path=args.model_path,
            adapter_dir=args.adapter_dir,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        banner = "LoRA chat ready."
        allow_snapshot_commands = False
    else:
        if args.adapter_dir is None:
            raise ValueError("--adapter-dir is required for mode=delta")
        model, tokenizer = load_delta_mem_chat_model(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            adapter_dir=args.adapter_dir,
        )
        banner = "Delta-Mem online chat ready."
        allow_snapshot_commands = True
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=args.device)
    run_chat_loop(
        session,
        max_new_tokens=args.max_new_tokens,
        banner=banner,
        allow_snapshot_commands=allow_snapshot_commands,
        dump_state_stats=args.dump_state_stats and args.mode == "delta",
    )


if __name__ == "__main__":
    main()
