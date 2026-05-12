from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
)
from deltamem.eval.peft_compat import patch_peft_cache_compat
from deltamem.model_loading import resolve_attn_implementation
from deltamem.runtime.session import get_dtype


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: str


_CONTROL_GROUP = None


def _distributed_debug_enabled() -> bool:
    return os.environ.get("DELTALORA_DISTRIBUTED_DEBUG", "0").strip() == "1"


def _distributed_debug_log(context: DistributedContext, message: str) -> None:
    if not _distributed_debug_enabled():
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[dist-debug][{timestamp}][rank{context.rank}] {message}", flush=True)


def init_distributed(device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    resolved_device = device
    if enabled and device.startswith("cuda"):
        resolved_device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    if enabled and not dist.is_initialized():
        backend = "nccl" if resolved_device.startswith("cuda") else "gloo"
        timeout_minutes = int(os.environ.get("DIST_TIMEOUT_MINUTES", "180"))
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )
    context = DistributedContext(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=resolved_device,
    )
    # Create the CPU-side control group while all ranks are still in lockstep at startup.
    # Lazily creating it after hours of skewed evaluation can hit the Gloo store timeout.
    _ensure_control_group(context)
    return context


def _control_group_supported(context: DistributedContext) -> bool:
    if not context.enabled or not dist.is_initialized():
        return False
    if not context.device.startswith("cuda"):
        return False
    return dist.get_backend() != "gloo"


def _ensure_control_group(context: DistributedContext):
    global _CONTROL_GROUP
    if not _control_group_supported(context):
        return None
    if _CONTROL_GROUP is None:
        _distributed_debug_log(context, "creating CPU-side gloo control group")
        _CONTROL_GROUP = dist.new_group(backend="gloo")
    return _CONTROL_GROUP


def _get_control_group(context: DistributedContext):
    if not _control_group_supported(context):
        return None
    return _CONTROL_GROUP


def barrier_distributed(context: DistributedContext) -> None:
    if not context.enabled or not dist.is_initialized():
        return
    control_group = _get_control_group(context)
    if control_group is not None:
        dist.barrier(group=control_group)
    elif context.device.startswith("cuda"):
        dist.barrier(device_ids=[context.local_rank])
    else:
        dist.barrier()


def finalize_distributed(context: DistributedContext) -> None:
    global _CONTROL_GROUP
    if not context.enabled or not dist.is_initialized():
        return
    if not os.environ.get("DELTALORA_RECORD_GATHER_DIR"):
        try:
            barrier_distributed(context)
        except Exception:
            # Preserve the original failure if one rank already aborted before teardown.
            pass
    if _CONTROL_GROUP is not None:
        try:
            dist.destroy_process_group(_CONTROL_GROUP)
        finally:
            _CONTROL_GROUP = None
    dist.destroy_process_group()



def _gather_indexed_records_via_files(
    indexed_records: list[tuple[int, dict[str, object]]],
    context: DistributedContext,
    gather_dir: Path,
) -> list[dict[str, object]] | None:
    gather_dir.mkdir(parents=True, exist_ok=True)
    shard_path = gather_dir / f"rank_{context.rank:02d}.json"
    shard_tmp_path = gather_dir / f"rank_{context.rank:02d}.json.tmp"
    _distributed_debug_log(context, f"writing {len(indexed_records)} records to {shard_path}")
    shard_tmp_path.write_text(json.dumps(indexed_records, separators=(",", ":")))
    shard_tmp_path.replace(shard_path)
    _distributed_debug_log(context, f"published shard {shard_path.name}")
    if context.rank != 0:
        _distributed_debug_log(context, "published shard; non-zero rank returning without gather barrier")
        return None

    expected_paths = [gather_dir / f"rank_{rank:02d}.json" for rank in range(context.world_size)]
    timeout_seconds = max(1, int(os.environ.get("DELTALORA_RECORD_GATHER_TIMEOUT_SECONDS", "60")))
    deadline = time.time() + timeout_seconds
    missing_paths = [path for path in expected_paths if not path.exists()]
    _distributed_debug_log(context, f"rank0 waiting for {len(expected_paths)} shard files in {gather_dir}")
    while missing_paths and time.time() < deadline:
        time.sleep(1.0)
        missing_paths = [path for path in expected_paths if not path.exists()]
    if missing_paths:
        missing_list = ", ".join(path.name for path in missing_paths)
        raise TimeoutError(
            f"Timed out waiting for distributed record shards in {gather_dir}: {missing_list}"
        )

    merged: list[tuple[int, dict[str, object]]] = []
    for path in expected_paths:
        rank_records = json.loads(path.read_text())
        for item_index, record in rank_records:
            merged.append((int(item_index), record))
    merged.sort(key=lambda item: item[0])
    _distributed_debug_log(context, f"rank0 merged {len(merged)} gathered records from {len(expected_paths)} shards")
    return [record for _, record in merged]



def gather_indexed_records(
    indexed_records: list[tuple[int, dict[str, object]]],
    context: DistributedContext,
) -> list[dict[str, object]] | None:
    if not context.enabled:
        return [record for _, record in indexed_records]
    gather_dir = os.environ.get("DELTALORA_RECORD_GATHER_DIR")
    if gather_dir:
        return _gather_indexed_records_via_files(indexed_records, context, Path(gather_dir))
    control_group = _ensure_control_group(context)
    gathered: list[list[tuple[int, dict[str, object]]] | None] = [None] * context.world_size
    _distributed_debug_log(context, f"entering all_gather_object with {len(indexed_records)} records")
    if control_group is not None:
        dist.all_gather_object(gathered, indexed_records, group=control_group)
    else:
        dist.all_gather_object(gathered, indexed_records)
    if context.rank != 0:
        _distributed_debug_log(context, "completed all_gather_object; non-zero rank returning")
        return None
    merged = [item for rank_records in gathered if rank_records is not None for item in rank_records]
    merged.sort(key=lambda item: item[0])
    _distributed_debug_log(context, f"rank0 merged {len(merged)} gathered records from all_gather_object")
    return [record for _, record in merged]


def load_base_model_and_tokenizer(
    *,
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    resolved_attn_implementation = resolve_attn_implementation(
        model_path,
        attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=get_dtype(dtype),
        device_map={"": device},
        attn_implementation=resolved_attn_implementation,
        local_files_only=True,
    ).eval()
    return model, tokenizer


def attach_delta_adapter_in_place(model, adapter_dir: Path) -> dict[str, object]:
    config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    attach_delta_mem(model, config)
    load_delta_mem_adapter(model, adapter_dir)
    return config.to_dict()


def load_delta_model_and_tokenizer(
    *,
    model_path: str,
    adapter_dir: Path,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    model, tokenizer = load_base_model_and_tokenizer(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    config_dict = attach_delta_adapter_in_place(model, adapter_dir)
    return model, tokenizer, config_dict


def load_peft_model_and_tokenizer(
    *,
    model_path: str,
    adapter_dir: Path,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    patch_peft_cache_compat()
    model, tokenizer = load_base_model_and_tokenizer(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
    return model, tokenizer


def load_lora_model_and_tokenizer(
    *,
    model_path: str,
    adapter_dir: Path,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    return load_peft_model_and_tokenizer(
        model_path=model_path,
        adapter_dir=adapter_dir,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def maybe_empty_cache(device: str) -> None:
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
