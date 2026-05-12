from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_state_stats,
    collect_delta_mem_weight_stats,
    load_delta_mem_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Delta-Mem weights on top of a base HF model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=True,
    ).eval()
    config = HFDeltaMemConfig.from_pretrained(args.adapter_dir)
    replaced = attach_delta_mem(model, config)
    load_delta_mem_adapter(model, args.adapter_dir)
    result = {
        "adapter_dir": args.adapter_dir,
        "num_replaced_modules": len(replaced),
        "first_replaced_modules": replaced[:8],
        "weight_stats": collect_delta_mem_weight_stats(model),
        "state_stats": collect_delta_mem_state_stats(model),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
