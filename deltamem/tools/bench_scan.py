from __future__ import annotations

import argparse
import time

import torch

from deltamem.core.delta import DeltaMemAttention, HFDeltaMemConfig
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Torch vs Triton Delta-Mem scan kernels.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_qwen3_attention(layer_idx: int = 0) -> Qwen3Attention:
    config = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=max(layer_idx + 1, 1),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3Attention(config, layer_idx)


def make_module(rank: int, device: str, dtype: torch.dtype) -> DeltaMemAttention:
    base = make_qwen3_attention().to(device=device, dtype=dtype)
    module = DeltaMemAttention(
        base,
        HFDeltaMemConfig(rank=rank, output_init="random", rankwise_gates=True),
    )
    return module.to(device=device, dtype=dtype)


def random_inputs(
    batch_size: int,
    seq_len: int,
    rank: int,
    device: str,
    dtype: torch.dtype,
):
    return {
        "state": torch.randn(batch_size, rank, rank, device=device, dtype=dtype),
        "q": torch.randn(batch_size, seq_len, rank, device=device, dtype=dtype),
        "k": torch.randn(batch_size, seq_len, rank, device=device, dtype=dtype),
        "v": torch.randn(batch_size, seq_len, rank, device=device, dtype=dtype),
        "beta": torch.sigmoid(torch.randn(batch_size, seq_len, rank, 1, device=device, dtype=dtype)),
        "lam": torch.sigmoid(torch.randn(batch_size, seq_len, rank, 1, device=device, dtype=dtype)),
    }


def run_bench(module: DeltaMemAttention, impl: str, inputs: dict[str, torch.Tensor], warmup: int, iters: int) -> float:
    module.scan_impl = impl
    for _ in range(warmup):
        module._memory_affine_scan(
            inputs["state"],
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["beta"],
            inputs["lam"],
            token_mask=None,
        )
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    for _ in range(iters):
        module._memory_affine_scan(
            inputs["state"],
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["beta"],
            inputs["lam"],
            token_mask=None,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - started_at) * 1000.0 / iters


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the scan benchmark.")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    module = make_module(args.rank, args.device, dtype)
    inputs = random_inputs(args.batch_size, args.seq_len, args.rank, args.device, dtype)
    torch_ms = run_bench(module, "torch", inputs, args.warmup, args.iters)
    triton_ms = run_bench(module, "triton", inputs, args.warmup, args.iters)
    print(
        {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "rank": args.rank,
            "dtype": args.dtype,
            "torch_ms": round(torch_ms, 4),
            "triton_ms": round(triton_ms, 4),
            "speedup": round(torch_ms / triton_ms, 4),
        }
    )


if __name__ == "__main__":
    main()
