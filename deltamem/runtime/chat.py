from __future__ import annotations

import argparse
from pathlib import Path

from deltamem.runtime.chat_cli import run_chat_loop
from deltamem.runtime.session import (
    DeltaMemChatSession,
    load_delta_mem_chat_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat with Qwen3 plus online Delta-Mem session memory."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--dump-state-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_delta_mem_chat_model(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        adapter_dir=args.adapter_dir,
    )
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=args.device)
    run_chat_loop(
        session,
        max_new_tokens=args.max_new_tokens,
        banner="Delta-Mem online chat ready.",
        allow_snapshot_commands=True,
        dump_state_stats=args.dump_state_stats,
    )


if __name__ == "__main__":
    main()
