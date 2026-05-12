from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.runtime.chat_cli import run_chat_loop
from deltamem.runtime.session import DeltaMemChatSession, get_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat with a base Qwen3 model plus a LoRA adapter."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def load_qwen3_lora_model(
    *,
    model_path: str,
    adapter_dir: Path,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=get_dtype(dtype),
        device_map={"": device},
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).eval()
    model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    model, tokenizer = load_qwen3_lora_model(
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=args.device)
    run_chat_loop(
        session,
        max_new_tokens=args.max_new_tokens,
        banner="LoRA chat ready.",
    )


if __name__ == "__main__":
    main()
