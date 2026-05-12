from __future__ import annotations

import os
from pathlib import Path

from deltamem.model_loading import DEFAULT_HF_HOME
from deltamem.train import delta_sft_experimental as _experimental
from deltamem.train.delta_sft_experimental import (
    DeltaMemTrainer,
    DialogueCausalLMCollator,
    EpisodeCausalLMCollator,
    _build_tokenized_dataset,
    build_episode_training_examples,
    get_dtype,
    load_examples,
    load_or_prepare_tokenized_dataset,
    parse_layer_indices,
    prepare_tokenized_dataset,
    tokenize_messages_for_sft,
)

SHARED_HF_DIR = Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME)))
SHARED_DATASET_DIR = Path(os.environ.get("HF_DATASETS_CACHE", str(SHARED_HF_DIR / "datasets")))
SHARED_MODEL_DIR = Path(os.environ.get("DELTALORA_MODEL_OUTPUT_ROOT", "models"))
TOKENIZED_DATASET_ROOT = Path(
    os.environ.get("DELTALORA_TOKENIZED_DATASET_ROOT", str(SHARED_DATASET_DIR / "deltamem_tokenized"))
)


def _validate_mainline_args(args) -> None:
    if args.memory_readout_mode != "delta":
        raise ValueError(
            "Mainline trainer only supports --memory-readout-mode delta. "
            "Use deltamem.train.delta_sft_experimental for experimental readouts."
        )


def parse_args():
    args = _experimental.parse_args()
    _validate_mainline_args(args)
    return args


def main() -> None:
    parse_args()
    _experimental.main()


__all__ = [
    "DeltaMemTrainer",
    "DialogueCausalLMCollator",
    "EpisodeCausalLMCollator",
    "SHARED_DATASET_DIR",
    "SHARED_HF_DIR",
    "SHARED_MODEL_DIR",
    "TOKENIZED_DATASET_ROOT",
    "_build_tokenized_dataset",
    "build_episode_training_examples",
    "get_dtype",
    "load_examples",
    "load_or_prepare_tokenized_dataset",
    "main",
    "parse_args",
    "parse_layer_indices",
    "prepare_tokenized_dataset",
    "tokenize_messages_for_sft",
]


if __name__ == "__main__":
    main()
