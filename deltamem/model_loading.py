from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from transformers import AutoConfig


DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
DEFAULT_TRANSFORMERS_CACHE = Path(
    os.environ.get("HF_HUB_CACHE", os.environ.get("TRANSFORMERS_CACHE", str(DEFAULT_HF_HOME / "hub")))
)
DEFAULT_LOCAL_MODEL_PATH = os.environ.get("DELTALORA_MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
DEFAULT_DATASETS_CACHE_DIR = os.environ.get(
    "HF_DATASETS_CACHE",
    str(DEFAULT_HF_HOME / "datasets"),
)


@lru_cache(maxsize=32)
def load_local_config(model_path: str):
    return AutoConfig.from_pretrained(model_path, local_files_only=True)


def load_local_text_config(model_path: str):
    config = load_local_config(model_path)
    if hasattr(config, "get_text_config"):
        return config.get_text_config(decoder=True)
    return config


def resolve_attn_implementation(
    model_path: str,
    requested_attn_implementation: str | None,
) -> str | None:
    requested = (requested_attn_implementation or "auto").strip().lower()
    if requested in {"", "none", "auto"}:
        return "flash_attention_2"
    return requested_attn_implementation
