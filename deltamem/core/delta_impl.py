from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    eager_attention_forward as qwen3_eager_attention_forward,
)

from deltamem.core.backbone_compat import (
    HAS_SMOLLM3,
    SmolLM3Attention,
    ensure_attention_compat_views,
    smollm3_apply_rotary_pos_emb,
    smollm3_eager_attention_forward,
)
from deltamem.kernels.affine_scan import triton_affine_scan, triton_scan_support

SUPPORTED_BASE_ATTENTION_TYPES = (Qwen3Attention,)
if HAS_SMOLLM3:
    SUPPORTED_BASE_ATTENTION_TYPES += (SmolLM3Attention,)


VALID_DELTA_HEADS = ("q", "k", "v", "o")
VALID_STATE_UPDATE_MODES = ("standard", "lambda_outside", "no_lambda")
VALID_MEMORY_PARTITION_ROUTING = ("soft",)
VALID_MEMORY_PARTITION_BASIS = ("shared",)
VALID_MEMORY_READOUT_MODES = ("delta",)
VALID_MEMORY_WRITE_SOURCES = ("learned_hidden",)
VALID_MEMORY_WRITE_GRANULARITIES = (
    "token",
    "message_mean",
    "sentence_mean",
)
VALID_MEMORY_PARTITION_READ_MODES = ("softmax",)
VALID_GLOBAL_MEMORY_MODES = ("shared_rw",)
VALID_GLOBAL_MEMORY_MERGE_MODES = ("gated_residual",)
VALID_DELTA_SCALE_GRANULARITIES = ("layer", "head")
VALID_DELTA_SCALE_PARAMETERIZATIONS = ("alpha_over_rank")


def normalize_delta_heads(heads: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(heads, str):
        items = tuple(part.strip().lower() for part in heads.split(",") if part.strip())
    else:
        items = tuple(str(part).strip().lower() for part in heads if str(part).strip())
    if not items or items == ("none",):
        return ()
    invalid = [head for head in items if head not in VALID_DELTA_HEADS]
    if invalid:
        raise ValueError(f"Unsupported delta heads: {invalid}; expected subset of {VALID_DELTA_HEADS}")
    deduped: list[str] = []
    for head in items:
        if head not in deduped:
            deduped.append(head)
    return tuple(deduped)


def normalize_state_update_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_STATE_UPDATE_MODES:
        raise ValueError(
            f"Unsupported state update mode: {mode}; expected one of {VALID_STATE_UPDATE_MODES}"
        )
    return normalized


def normalize_memory_partition_routing(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_PARTITION_ROUTING:
        raise ValueError(
            "Unsupported memory partition routing mode: "
            f"{mode}; expected one of {VALID_MEMORY_PARTITION_ROUTING}"
        )
    return normalized


def normalize_memory_partition_basis(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_PARTITION_BASIS:
        raise ValueError(
            "Unsupported memory partition basis mode: "
            f"{mode}; expected one of {VALID_MEMORY_PARTITION_BASIS}"
        )
    return normalized


def normalize_memory_readout_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_READOUT_MODES:
        raise ValueError(
            "Only memory_readout_mode='delta' is still supported. "
            f"Got {mode!r}."
        )
    return normalized


def normalize_memory_write_source(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_WRITE_SOURCES:
        raise ValueError(
            "Unsupported memory write source: "
            f"{mode}; expected one of {VALID_MEMORY_WRITE_SOURCES}"
        )
    return normalized


def normalize_memory_write_granularity(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_WRITE_GRANULARITIES:
        raise ValueError(
            "Unsupported memory write granularity: "
            f"{mode}; expected one of {VALID_MEMORY_WRITE_GRANULARITIES}"
        )
    return normalized


def normalize_memory_partition_read_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_PARTITION_READ_MODES:
        raise ValueError(
            "Unsupported memory partition read mode: "
            f"{mode}; expected one of {VALID_MEMORY_PARTITION_READ_MODES}"
        )
    return normalized


def normalize_global_memory_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_GLOBAL_MEMORY_MODES:
        raise ValueError(
            "Unsupported global memory mode: "
            f"{mode}; expected one of {VALID_GLOBAL_MEMORY_MODES}"
        )
    return normalized


def normalize_global_memory_merge_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_GLOBAL_MEMORY_MERGE_MODES:
        raise ValueError(
            "Unsupported global memory merge mode: "
            f"{mode}; expected one of {VALID_GLOBAL_MEMORY_MERGE_MODES}"
        )
    return normalized


def normalize_delta_scale_granularity(granularity: str) -> str:
    normalized = str(granularity).strip().lower()
    if normalized not in VALID_DELTA_SCALE_GRANULARITIES:
        raise ValueError(
            "Unsupported delta scale granularity: "
            f"{granularity}; expected one of {VALID_DELTA_SCALE_GRANULARITIES}"
        )
    return normalized


def inverse_bounded_sigmoid(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        raise ValueError("max_value must be > 0")
    clipped = min(max(value / max_value, 1e-4), 1.0 - 1e-4)
    return math.log(clipped / (1.0 - clipped))


def normalize_delta_scale_parameterization(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_DELTA_SCALE_PARAMETERIZATIONS:
        raise ValueError(
            "Unsupported delta scale parameterization: "
            f"{mode}; expected one of {VALID_DELTA_SCALE_PARAMETERIZATIONS}"
        )
    return normalized


@dataclass(frozen=True)
class HFDeltaMemConfig:
    rank: int = 8
    alpha: float = 16.0
    beta_bias_init: float = -1.5
    normalize_qk: bool = True
    couple_lambda: bool = True
    state_update_mode: str = "standard"
    rankwise_gates: bool = True
    output_init: str = "zero"
    base_slice_ref_width: int = 8
    online_gain: float = 0.05
    num_state_heads: int = 1
    num_memory_partitions: int = 1
    num_global_memory_partitions: int = 0
    memory_partition_routing: str = "soft"
    memory_partition_basis: str = "shared"
    tie_memory_partition_read_write: bool = False
    memory_partition_read_mode: str = "softmax"
    memory_partition_sigmoid_gate_bias_init: float = -2.0
    slot_read_top_k: int = 0
    global_memory_mode: str = "shared_rw"
    global_memory_read_top_k: int = 0
    global_memory_merge_mode: str = "gated_residual"
    global_memory_gate_bias_init: float = -2.0
    global_memory_read_logit_bias: float = 0.0
    memory_reader_layers: tuple[int, ...] = ()
    memory_reader_hidden_size: int = 1024
    memory_reader_residual_scale: float = 0.1
    memory_reader_read_only: bool = True
    memory_readout_mode: str = "delta"
    memory_write_source: str = "learned_hidden"
    memory_write_granularity: str = "token"
    memory_write_proposals_per_message: int = 2
    synthetic_memory_slots: int = 1
    latent_memory_layers: tuple[int, ...] = ()
    latent_memory_hidden_size: int = 1024
    latent_memory_residual_scale: float = 0.1
    latent_memory_slots: int = 4
    latent_memory_init_std: float = 0.002
    latent_gate_init: float = 0.01
    target_modules: tuple[str, ...] = ("self_attn",)
    target_layers: tuple[int, ...] = ()
    delta_heads: tuple[str, ...] = VALID_DELTA_HEADS
    delta_o_rmsnorm: bool = False
    delta_o_rmsnorm_eps: float = 1e-6
    trainable_delta_scale: bool = False
    delta_scale_init: float = 1.0
    delta_scale_max: float = 2.0
    delta_scale_granularity: str = "layer"
    delta_scale_parameterization: str = "alpha_over_rank"

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_heads", normalize_delta_heads(self.delta_heads))
        object.__setattr__(
            self,
            "state_update_mode",
            normalize_state_update_mode(self.state_update_mode),
        )
        if int(self.num_state_heads) < 1:
            raise ValueError("num_state_heads must be >= 1")
        if int(self.num_memory_partitions) < 1:
            raise ValueError("num_memory_partitions must be >= 1")
        if int(self.num_global_memory_partitions) < 0:
            raise ValueError("num_global_memory_partitions must be >= 0")
        if int(self.num_global_memory_partitions) >= int(self.num_memory_partitions):
            raise ValueError(
                "num_global_memory_partitions must be smaller than num_memory_partitions"
            )
        if int(self.base_slice_ref_width) < 1:
            raise ValueError("base_slice_ref_width must be >= 1")
        if float(self.delta_o_rmsnorm_eps) <= 0.0:
            raise ValueError("delta_o_rmsnorm_eps must be > 0")
        if float(self.delta_scale_init) <= 0.0:
            raise ValueError("delta_scale_init must be > 0")
        if float(self.delta_scale_max) <= 0.0:
            raise ValueError("delta_scale_max must be > 0")
        if float(self.delta_scale_init) >= float(self.delta_scale_max):
            raise ValueError("delta_scale_init must be smaller than delta_scale_max")
        object.__setattr__(self, "num_state_heads", int(self.num_state_heads))
        object.__setattr__(self, "num_memory_partitions", int(self.num_memory_partitions))
        object.__setattr__(
            self,
            "num_global_memory_partitions",
            int(self.num_global_memory_partitions),
        )
        object.__setattr__(self, "base_slice_ref_width", int(self.base_slice_ref_width))
        object.__setattr__(self, "delta_o_rmsnorm", bool(self.delta_o_rmsnorm))
        object.__setattr__(self, "delta_o_rmsnorm_eps", float(self.delta_o_rmsnorm_eps))
        object.__setattr__(self, "trainable_delta_scale", bool(self.trainable_delta_scale))
        object.__setattr__(self, "delta_scale_init", float(self.delta_scale_init))
        object.__setattr__(self, "delta_scale_max", float(self.delta_scale_max))
        object.__setattr__(
            self,
            "delta_scale_granularity",
            normalize_delta_scale_granularity(self.delta_scale_granularity),
        )
        object.__setattr__(
            self,
            "delta_scale_parameterization",
            normalize_delta_scale_parameterization(self.delta_scale_parameterization),
        )
        object.__setattr__(
            self,
            "memory_partition_routing",
            normalize_memory_partition_routing(self.memory_partition_routing),
        )
        object.__setattr__(
            self,
            "memory_partition_basis",
            normalize_memory_partition_basis(self.memory_partition_basis),
        )
        object.__setattr__(
            self,
            "memory_partition_read_mode",
            normalize_memory_partition_read_mode(self.memory_partition_read_mode),
        )
        object.__setattr__(
            self,
            "global_memory_mode",
            normalize_global_memory_mode(self.global_memory_mode),
        )
        object.__setattr__(
            self,
            "global_memory_merge_mode",
            normalize_global_memory_merge_mode(self.global_memory_merge_mode),
        )
        if int(self.slot_read_top_k) < 0:
            raise ValueError("slot_read_top_k must be >= 0")
        if int(self.global_memory_read_top_k) < 0:
            raise ValueError("global_memory_read_top_k must be >= 0")
        if int(self.synthetic_memory_slots) < 1:
            raise ValueError("synthetic_memory_slots must be >= 1")
        if int(self.memory_write_proposals_per_message) < 1:
            raise ValueError("memory_write_proposals_per_message must be >= 1")
        if int(self.latent_memory_hidden_size) < 1:
            raise ValueError("latent_memory_hidden_size must be >= 1")
        if int(self.latent_memory_slots) < 1:
            raise ValueError("latent_memory_slots must be >= 1")
        if float(self.latent_memory_init_std) <= 0.0:
            raise ValueError("latent_memory_init_std must be > 0")
        if float(self.latent_gate_init) <= 0.0:
            raise ValueError("latent_gate_init must be > 0")
        object.__setattr__(self, "slot_read_top_k", int(self.slot_read_top_k))
        object.__setattr__(
            self,
            "global_memory_read_top_k",
            int(self.global_memory_read_top_k),
        )
        object.__setattr__(
            self,
            "global_memory_gate_bias_init",
            float(self.global_memory_gate_bias_init),
        )
        object.__setattr__(
            self,
            "memory_partition_sigmoid_gate_bias_init",
            float(self.memory_partition_sigmoid_gate_bias_init),
        )
        object.__setattr__(
            self,
            "global_memory_read_logit_bias",
            float(self.global_memory_read_logit_bias),
        )
        object.__setattr__(
            self,
            "synthetic_memory_slots",
            int(self.synthetic_memory_slots),
        )
        object.__setattr__(
            self,
            "memory_write_proposals_per_message",
            int(self.memory_write_proposals_per_message),
        )
        object.__setattr__(
            self,
            "latent_memory_hidden_size",
            int(self.latent_memory_hidden_size),
        )
        object.__setattr__(
            self,
            "latent_memory_slots",
            int(self.latent_memory_slots),
        )
        object.__setattr__(
            self,
            "latent_memory_init_std",
            float(self.latent_memory_init_std),
        )
        object.__setattr__(
            self,
            "latent_gate_init",
            float(self.latent_gate_init),
        )
        object.__setattr__(
            self,
            "memory_readout_mode",
            normalize_memory_readout_mode(self.memory_readout_mode),
        )
        object.__setattr__(
            self,
            "memory_write_source",
            normalize_memory_write_source(self.memory_write_source),
        )
        object.__setattr__(
            self,
            "memory_write_granularity",
            normalize_memory_write_granularity(self.memory_write_granularity),
        )
        if self.memory_reader_layers:
            raise ValueError(
                "memory_reader_layers is archived; active Delta-Mem only keeps TSW / MSW / SSW paths."
            )
        if self.num_memory_partitions != 1:
            raise ValueError(
                "num_memory_partitions is archived; active Delta-Mem only supports dense single-state memory (num_memory_partitions=1)."
            )
        if self.num_global_memory_partitions != 0:
            raise ValueError(
                "num_global_memory_partitions is archived; active Delta-Mem does not support global partitions."
            )
        if self.memory_partition_routing != "soft":
            raise ValueError(
                "memory_partition_routing is archived; active Delta-Mem only supports memory_partition_routing='soft'."
            )
        if self.memory_partition_basis != "shared":
            raise ValueError(
                "memory_partition_basis is archived; active Delta-Mem only supports memory_partition_basis='shared'."
            )
        if self.tie_memory_partition_read_write:
            raise ValueError(
                "tie_memory_partition_read_write is archived; active Delta-Mem only supports the dense single-state path."
            )
        if self.memory_partition_read_mode != "softmax":
            raise ValueError(
                "memory_partition_read_mode is archived; active Delta-Mem only supports memory_partition_read_mode='softmax'."
            )
        if self.slot_read_top_k != 0:
            raise ValueError(
                "slot_read_top_k is archived; active Delta-Mem only supports slot_read_top_k=0."
            )
        if self.global_memory_mode != "shared_rw":
            raise ValueError(
                "global_memory_mode is archived; active Delta-Mem only supports global_memory_mode='shared_rw'."
            )
        if self.global_memory_read_top_k != 0:
            raise ValueError(
                "global_memory_read_top_k is archived; active Delta-Mem only supports global_memory_read_top_k=0."
            )
        if self.global_memory_merge_mode != "gated_residual":
            raise ValueError(
                "global_memory_merge_mode is archived; active Delta-Mem only supports global_memory_merge_mode='gated_residual'."
            )
        if self.memory_write_source != "learned_hidden":
            raise ValueError(
                "memory_write_source is archived; active Delta-Mem only supports memory_write_source='learned_hidden'."
            )
        if self.memory_write_granularity == "message_proposals":
            raise ValueError(
                "message_proposals is archived; active Delta-Mem only supports token / message_mean / sentence_mean writes."
            )
        if self.memory_write_proposals_per_message != 2:
            raise ValueError(
                "memory_write_proposals_per_message is archived together with message_proposals writes."
            )
        if self.synthetic_memory_slots != 1:
            raise ValueError(
                "synthetic_memory_slots is archived together with synthetic_kv readout."
            )
        if self.latent_memory_layers:
            raise ValueError(
                "latent memory readouts are archived; active Delta-Mem only supports memory_readout_mode='delta'."
            )
        if self.num_state_heads > 1 and self.num_memory_partitions > 1:
            raise ValueError(
                "num_state_heads > 1 is currently only supported with num_memory_partitions == 1"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HFDeltaMemConfig":
        if "target_modules" in data and isinstance(data["target_modules"], list):
            data = dict(data)
            data["target_modules"] = tuple(data["target_modules"])
        if "memory_reader_layers" in data and isinstance(data["memory_reader_layers"], list):
            data = dict(data)
            data["memory_reader_layers"] = tuple(data["memory_reader_layers"])
        if "target_layers" in data and isinstance(data["target_layers"], list):
            data = dict(data)
            data["target_layers"] = tuple(data["target_layers"])
        if "latent_memory_layers" in data and isinstance(data["latent_memory_layers"], list):
            data = dict(data)
            data["latent_memory_layers"] = tuple(data["latent_memory_layers"])
        if "delta_heads" in data and isinstance(data["delta_heads"], list):
            data = dict(data)
            data["delta_heads"] = tuple(data["delta_heads"])
        return cls(**data)

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "delta_mem_config.json").write_text(
            json.dumps(self.to_dict(), indent=2)
        )

    @classmethod
    def from_pretrained(cls, input_dir: str | Path) -> "HFDeltaMemConfig":
        input_path = Path(input_dir)
        return cls.from_dict(
            json.loads((input_path / "delta_mem_config.json").read_text())
        )


class DeltaMemAttention(nn.Module):
    def __init__(self, base: Qwen3Attention | SmolLM3Attention, config: HFDeltaMemConfig) -> None:
        super().__init__()
        self.base = ensure_attention_compat_views(base)
        base = self.base
        self.config = base.config
        self.delta_config = config
        self.layer_idx = base.layer_idx
        self.head_dim = base.head_dim
        self.num_key_value_groups = base.num_key_value_groups
        self.scaling = base.scaling
        self.attention_dropout = base.attention_dropout
        self.is_causal = base.is_causal
        self.sliding_window = getattr(base, "sliding_window", getattr(base.config, "sliding_window", None))
        self.is_smollm3_attention = isinstance(base, SmolLM3Attention)
        if self.is_smollm3_attention:
            self.eager_attention_forward = smollm3_eager_attention_forward
        else:
            self.eager_attention_forward = qwen3_eager_attention_forward
        self.layer_type = getattr(base, "layer_type", None)
        self.is_sliding = getattr(base, "is_sliding", False)
        self.is_kv_shared_layer = getattr(base, "is_kv_shared_layer", False)
        self.kv_shared_layer_index = getattr(base, "kv_shared_layer_index", None)
        self.store_full_length_kv = getattr(base, "store_full_length_kv", False)
        self.has_packed_qkv_proj = hasattr(base, "qkv_proj") and getattr(base, "qkv_proj", None) is not None

        self.rank = config.rank
        self.num_state_heads = config.num_state_heads
        self.state_read_dim = self.rank * self.num_state_heads
        self.multi_head_state = self.num_state_heads > 1
        self.delta_scaling = config.alpha / config.rank
        self.trainable_delta_scale = config.trainable_delta_scale
        self.delta_scale_max = config.delta_scale_max
        self.delta_scale_granularity = config.delta_scale_granularity
        self.normalize_qk = config.normalize_qk
        self.couple_lambda = config.couple_lambda
        self.state_update_mode = config.state_update_mode
        self.rankwise_gates = config.rankwise_gates
        self.output_init = config.output_init
        self.base_slice_ref_width = config.base_slice_ref_width
        self.online_gain = config.online_gain
        self.num_memory_partitions = config.num_memory_partitions
        self.num_global_memory_partitions = config.num_global_memory_partitions
        self.memory_partition_routing = config.memory_partition_routing
        self.memory_partition_basis = config.memory_partition_basis
        self.tie_memory_partition_read_write = config.tie_memory_partition_read_write
        self.memory_partition_read_mode = config.memory_partition_read_mode
        self.memory_partition_sigmoid_gate_bias_init = config.memory_partition_sigmoid_gate_bias_init
        self.slot_read_top_k = config.slot_read_top_k
        self.global_memory_mode = config.global_memory_mode
        self.global_memory_read_top_k = config.global_memory_read_top_k
        self.global_memory_merge_mode = config.global_memory_merge_mode
        self.global_memory_gate_bias_init = config.global_memory_gate_bias_init
        self.global_memory_read_logit_bias = config.global_memory_read_logit_bias
        self.memory_readout_mode = config.memory_readout_mode
        self.synthetic_memory_slots = config.synthetic_memory_slots
        self.latent_memory_hidden_size = config.latent_memory_hidden_size
        self.latent_memory_residual_scale = config.latent_memory_residual_scale
        self.latent_memory_slots = config.latent_memory_slots
        self.latent_memory_init_std = config.latent_memory_init_std
        self.latent_gate_init = config.latent_gate_init
        self.latent_memory_enabled = (
            self.memory_readout_mode in {"latent_context", "memory_branch"}
            and self.layer_idx in config.latent_memory_layers
        )
        self.gate_dim_per_head = config.rank if config.rankwise_gates else 1
        self.gate_dim = self.gate_dim_per_head * self.num_state_heads
        self.active_delta_heads = frozenset(config.delta_heads)
        if self.trainable_delta_scale:
            scale_shape = (len(VALID_DELTA_HEADS),) if self.delta_scale_granularity == "head" else (1,)
            init_raw = inverse_bounded_sigmoid(config.delta_scale_init, self.delta_scale_max)
            self.delta_scale_raw = nn.Parameter(torch.full(scale_shape, init_raw))
        self.delta_o_rmsnorm = config.delta_o_rmsnorm
        self.delta_o_rmsnorm_eps = config.delta_o_rmsnorm_eps

        hidden_size = base.q_proj.in_features
        self.hidden_size = hidden_size
        self.query_out_features = base.q_proj.out_features
        self.key_out_features = base.k_proj.out_features
        self.base_v_out_features = (
            base.v_proj.out_features if base.v_proj is not None else base.k_proj.out_features
        )
        self.num_key_value_heads = base.k_proj.out_features // self.head_dim
        self.synthetic_state_dim = config.rank * config.rank * config.num_state_heads
        self.partition_state_dim = config.rank * config.rank
        self.memory_write_source = config.memory_write_source
        self.memory_write_granularity = config.memory_write_granularity
        self.memory_write_proposals_per_message = config.memory_write_proposals_per_message
        self.split_global_memory_enabled = False

        self.memory_q_proj = nn.Parameter(torch.empty(self.state_read_dim, hidden_size))
        self.memory_k_proj = nn.Parameter(torch.empty(self.state_read_dim, hidden_size))
        self.memory_v_proj = nn.Parameter(torch.empty(self.state_read_dim, hidden_size))

        self.delta_q_proj = nn.Parameter(torch.empty(base.q_proj.out_features, self.state_read_dim))
        self.delta_k_proj = nn.Parameter(torch.empty(base.k_proj.out_features, self.state_read_dim))
        self.delta_v_proj = nn.Parameter(torch.empty(self.base_v_out_features, self.state_read_dim))
        self.delta_o_proj = nn.Parameter(torch.empty(base.o_proj.out_features, self.state_read_dim))
        if self.delta_o_rmsnorm:
            self.delta_o_rmsnorm_weight = nn.Parameter(torch.ones(base.o_proj.out_features))

        self.beta_proj = nn.Parameter(torch.empty(self.gate_dim, hidden_size))
        self.beta_bias = nn.Parameter(torch.full((self.gate_dim,), config.beta_bias_init))
        if not config.couple_lambda:
            self.lambda_proj = nn.Parameter(torch.empty(self.gate_dim, hidden_size))
            self.lambda_bias = nn.Parameter(
                torch.full((self.gate_dim,), -config.beta_bias_init)
            )

        self.memory_reader_enabled = False
        self.memory_reader_hidden_size = config.memory_reader_hidden_size
        self.memory_reader_residual_scale = config.memory_reader_residual_scale
        self.memory_reader_read_only = config.memory_reader_read_only
        self.reset_parameters()
        self.delta_state: torch.Tensor | None = None
        self.cached_latent_slots: torch.Tensor | None = None
        self.cached_latent_slot_valid: torch.Tensor | None = None
        self.read_context_mask: torch.Tensor | None = None
        self.last_beta_mean: torch.Tensor | None = None
        self.last_lambda_mean: torch.Tensor | None = None
        self.write_enabled = True
        self.last_memory_reader_output: torch.Tensor | None = None
        self.last_memory_reader_norm: torch.Tensor | None = None
        self.last_latent_slot_valid_ratio: torch.Tensor | None = None
        self.last_latent_slot_norm: torch.Tensor | None = None
        self.last_latent_gate_value: torch.Tensor | None = None
        self.last_write_routes: torch.Tensor | None = None
        self.last_read_routes: torch.Tensor | None = None
        self.last_base_o_norm: torch.Tensor | None = None
        self.last_delta_o_norm: torch.Tensor | None = None
        self.last_delta_o_ratio: torch.Tensor | None = None
        self.write_message_ids: torch.Tensor | None = None
        self.write_sentence_ids: torch.Tensor | None = None
        self.scan_impl = os.environ.get("DELTA_MEM_SCAN_IMPL", "auto")

    def _normalize_query_states(self, states: torch.Tensor) -> torch.Tensor:
        q_norm = getattr(self.base, "q_norm", None)
        if q_norm is None:
            return states
        return q_norm(states)

    def _normalize_key_states(self, states: torch.Tensor) -> torch.Tensor:
        k_norm = getattr(self.base, "k_norm", None)
        if k_norm is None:
            return states
        return k_norm(states)

    def _normalize_value_states(self, states: torch.Tensor) -> torch.Tensor:
        v_norm = getattr(self.base, "v_norm", None)
        if v_norm is None:
            return states
        return v_norm(states)

    def _apply_standard_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_smollm3_attention:
            if not bool(getattr(self.base, "use_rope", True)):
                return query_states, key_states
            return smollm3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    def _init_delta_head(self, head: nn.Parameter, base_weight: torch.Tensor) -> None:
        if self.output_init == "zero":
            nn.init.zeros_(head)
            return
        if self.output_init == "random":
            nn.init.kaiming_uniform_(head, a=math.sqrt(5))
            with torch.no_grad():
                head.mul_(self.online_gain)
            return
        if self.output_init not in {"base_slice", "base_slice_fixed"}:
            raise ValueError(f"Unsupported output_init: {self.output_init}")
        with torch.no_grad():
            if self.output_init == "base_slice":
                slice_width = min(self.rank, base_weight.shape[1])
            else:
                slice_width = min(self.base_slice_ref_width, self.rank, base_weight.shape[1])
            head.zero_()
            if slice_width == 0:
                return
            base_slice = base_weight[:, :slice_width].detach().clone().float()
            base_slice = F.normalize(base_slice, dim=0, eps=1e-6)
            head[:, :slice_width].copy_((base_slice * self.online_gain).to(head.dtype))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.memory_q_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_k_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_v_proj, a=math.sqrt(5))
        self._init_delta_head(self.delta_q_proj, self.base.q_proj.weight)
        self._init_delta_head(self.delta_k_proj, self.base.k_proj.weight)
        self._init_delta_head(
            self.delta_v_proj,
            self.base.v_proj.weight if self.base.v_proj is not None else self.base.k_proj.weight,
        )
        self._init_delta_head(self.delta_o_proj, self.base.o_proj.weight)
        for head_name, param in (
            ("q", self.delta_q_proj),
            ("k", self.delta_k_proj),
            ("v", self.delta_v_proj),
            ("o", self.delta_o_proj),
        ):
            if head_name not in self.active_delta_heads:
                nn.init.zeros_(param)
        if self.delta_o_rmsnorm:
            nn.init.ones_(self.delta_o_rmsnorm_weight)
        nn.init.zeros_(self.beta_proj)
        if not self.couple_lambda:
            nn.init.zeros_(self.lambda_proj)

    def reset_state(self) -> None:
        self.delta_state = None
        self.read_context_mask = None
        self.last_beta_mean = None
        self.last_lambda_mean = None
        self.last_memory_reader_output = None
        self.last_memory_reader_norm = None
        self.last_latent_slot_valid_ratio = None
        self.last_latent_slot_norm = None
        self.last_latent_gate_value = None
        self.last_write_routes = None
        self.last_read_routes = None
        self.last_base_o_norm = None
        self.last_delta_o_norm = None
        self.last_delta_o_ratio = None
        self.last_global_merge_gate = None
        self.write_message_ids = None
        self.write_sentence_ids = None

    def set_write_enabled(self, enabled: bool) -> None:
        if enabled:
            self.read_context_mask = None
        else:
            self.write_message_ids = None
            self.write_sentence_ids = None
        self.write_enabled = enabled

    def set_write_message_ids(self, message_ids: torch.Tensor | None) -> None:
        self.write_message_ids = message_ids

    def set_write_sentence_ids(self, sentence_ids: torch.Tensor | None) -> None:
        self.write_sentence_ids = sentence_ids

    def is_trainable_parameter(self, sub_name: str) -> bool:
        if sub_name in {"memory_q_proj", "memory_k_proj", "memory_v_proj"}:
            return True
        if sub_name == "delta_q_proj":
            return "q" in self.active_delta_heads
        if sub_name == "delta_k_proj":
            return "k" in self.active_delta_heads
        if sub_name == "delta_v_proj":
            return "v" in self.active_delta_heads
        if sub_name == "delta_o_proj":
            return "o" in self.active_delta_heads
        if sub_name == "delta_o_rmsnorm_weight":
            return self.delta_o_rmsnorm and "o" in self.active_delta_heads
        if sub_name == "delta_scale_raw":
            return self.trainable_delta_scale
        return True

    def _ensure_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self.delta_state is None
            or self.delta_state.size(0) != batch_size
            or self.delta_state.device != device
            or self.delta_state.dtype != dtype
        ):
            if self.multi_head_state:
                self.delta_state = torch.zeros(
                    batch_size,
                    self.num_state_heads,
                    self.rank,
                    self.rank,
                    device=device,
                    dtype=dtype,
                )
            else:
                self.delta_state = torch.zeros(
                    batch_size,
                    self.rank,
                    self.rank,
                    device=device,
                    dtype=dtype,
                )
        return self.delta_state

    def _reshape_state_heads(self, projected: torch.Tensor) -> torch.Tensor:
        if not self.multi_head_state:
            return projected
        return projected.view(*projected.shape[:-1], self.num_state_heads, self.rank)

    def _flatten_state_heads(self, projected: torch.Tensor) -> torch.Tensor:
        if not self.multi_head_state:
            return projected
        return projected.reshape(*projected.shape[:-2], self.state_read_dim)

    def _normalize_memory_projection(self, projected: torch.Tensor) -> torch.Tensor:
        if self.normalize_qk:
            if self.multi_head_state and projected.size(-1) == self.state_read_dim:
                projected = self._reshape_state_heads(projected)
                projected = torch.tanh(projected)
                projected = F.normalize(projected, dim=-1, eps=1e-6)
                projected = self._flatten_state_heads(projected)
            else:
                projected = torch.tanh(projected)
                projected = F.normalize(projected, dim=-1, eps=1e-6)
        return projected

    def _split_packed_qkv(self, packed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_end = self.query_out_features
        key_end = query_end + self.key_out_features
        query_states = packed_qkv[..., :query_end]
        key_states = packed_qkv[..., query_end:key_end]
        value_states = packed_qkv[..., key_end:key_end + self.base_v_out_features]
        return query_states, key_states, value_states

    def _base_query_projection(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_packed_qkv_proj:
            packed_qkv = self.base.qkv_proj(hidden_states)
            query_states, _, _ = self._split_packed_qkv(packed_qkv)
            return query_states
        return self.base.q_proj(hidden_states)

    def _base_qkv_projections(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_packed_qkv_proj:
            return self._split_packed_qkv(self.base.qkv_proj(hidden_states))
        query_states = self.base.q_proj(hidden_states)
        key_states = self.base.k_proj(hidden_states)
        value_states = self.base.v_proj(hidden_states) if self.base.v_proj is not None else key_states
        return query_states, key_states, value_states

    def _compute_delta_qkv_from_reads(
        self,
        reads: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        delta_q = self._project_delta_head(reads, self.delta_q_proj, "q")
        delta_k = self._project_delta_head(reads, self.delta_k_proj, "k")
        delta_v = self._project_delta_head(reads, self.delta_v_proj, "v")
        return delta_q, delta_k, delta_v

    def _apply_delta_qkv(
        self,
        hidden_states: torch.Tensor,
        delta_q: torch.Tensor | None,
        delta_k: torch.Tensor | None,
        delta_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_packed_qkv_proj:
            packed_qkv = self.base.qkv_proj(hidden_states)
            packed_delta_parts = []
            for delta_part, width in (
                (delta_q, self.query_out_features),
                (delta_k, self.key_out_features),
                (delta_v, self.base_v_out_features),
            ):
                if delta_part is None:
                    packed_delta_parts.append(packed_qkv.new_zeros(*packed_qkv.shape[:-1], width))
                else:
                    packed_delta_parts.append(delta_part.to(hidden_states.dtype))
            packed_qkv = packed_qkv + torch.cat(packed_delta_parts, dim=-1)
            return self._split_packed_qkv(packed_qkv)
        query_states = self.base.q_proj(hidden_states)
        if delta_q is not None:
            query_states = query_states + delta_q.to(hidden_states.dtype)
        key_states = self.base.k_proj(hidden_states)
        if delta_k is not None:
            key_states = key_states + delta_k.to(hidden_states.dtype)
        value_states = self.base.v_proj(hidden_states) if self.base.v_proj is not None else key_states
        if delta_v is not None:
            value_states = value_states + delta_v.to(hidden_states.dtype)
        return query_states, key_states, value_states

    def _memory_sequence_projections(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_weights = [self.beta_proj]
        split_sizes = [self.gate_dim]
        if not self.couple_lambda:
            gate_weights.append(self.lambda_proj)
            split_sizes.append(self.gate_dim)

        packed_gate_weight = torch.cat(gate_weights, dim=0)
        packed_gates = F.linear(hidden_states, packed_gate_weight)
        gate_splits = torch.split(packed_gates, split_sizes, dim=-1)

        packed_memory_weight = torch.cat(
            [self.memory_q_proj, self.memory_k_proj, self.memory_v_proj],
            dim=0,
        )
        packed_memory = F.linear(hidden_states, packed_memory_weight)
        memory_q, memory_k, memory_v = torch.split(
            packed_memory,
            [self.state_read_dim, self.state_read_dim, self.state_read_dim],
            dim=-1,
        )
        memory_q = self._normalize_memory_projection(memory_q)
        memory_k = self._normalize_memory_projection(memory_k)

        beta = torch.sigmoid(
            gate_splits[0]
            + self.beta_bias.view(*([1] * (hidden_states.dim() - 1)), self.gate_dim)
        ).unsqueeze(-1)
        if self.state_update_mode == "no_lambda":
            lam = torch.ones_like(beta)
        elif self.couple_lambda:
            lam = 1.0 - beta
        else:
            lam = torch.sigmoid(
                gate_splits[1]
                + self.lambda_bias.view(
                    *([1] * (hidden_states.dim() - 1)),
                    self.gate_dim,
                )
            ).unsqueeze(-1)
        return memory_q, memory_k, memory_v, beta, lam

    def _partition_memory_projections(
        self,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return memory_q_seq, memory_k_seq, memory_v_seq

    def _memory_partition_routes(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        message_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ones = hidden_states.new_ones(*hidden_states.shape[:-1], 1)
        return ones, ones

    def _memory_update_coefficients(
        self,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        *,
        write_route_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta_rows = beta_seq.squeeze(-1) if beta_seq.ndim == 4 else beta_seq
        lambda_rows = lambda_seq.squeeze(-1) if lambda_seq.ndim == 4 else lambda_seq
        if self.multi_head_state:
            beta_rows = beta_rows.view(
                beta_rows.size(0),
                beta_rows.size(1),
                self.num_state_heads,
                self.gate_dim_per_head,
            )
            lambda_rows = lambda_rows.view(
                lambda_rows.size(0),
                lambda_rows.size(1),
                self.num_state_heads,
                self.gate_dim_per_head,
            )
            if self.gate_dim_per_head == 1:
                beta_rows = beta_rows.expand(-1, -1, -1, self.rank)
                lambda_rows = lambda_rows.expand(-1, -1, -1, self.rank)
        else:
            if beta_rows.size(-1) == 1:
                beta_rows = beta_rows.expand(beta_rows.size(0), beta_rows.size(1), self.rank)
            if lambda_rows.size(-1) == 1:
                lambda_rows = lambda_rows.expand(lambda_rows.size(0), lambda_rows.size(1), self.rank)

        if self.state_update_mode == "standard":
            keep_seq = lambda_rows
            erase_seq = beta_rows
            write_seq = beta_rows
        elif self.state_update_mode == "lambda_outside":
            keep_seq = lambda_rows
            erase_seq = lambda_rows * beta_rows
            write_seq = beta_rows
        elif self.state_update_mode == "no_lambda":
            keep_seq = torch.ones_like(beta_rows)
            erase_seq = beta_rows
            write_seq = beta_rows
        else:  # pragma: no cover
            raise ValueError(f"Unsupported state update mode: {self.state_update_mode}")

        if write_route_seq is None:
            return keep_seq, erase_seq, write_seq

        route = write_route_seq.permute(0, 2, 1).unsqueeze(-1)
        keep_seq = 1.0 - route + route * keep_seq.unsqueeze(1)
        erase_seq = route * erase_seq.unsqueeze(1)
        write_seq = route * write_seq.unsqueeze(1)
        return keep_seq, erase_seq, write_seq

    def _token_validity_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask[:, -seq_len:].to(device=device).ne(0)
        if attention_mask.dim() == 4:
            if attention_mask.size(0) != batch_size:
                raise ValueError(
                    "attention_mask batch dimension does not match hidden_states batch size"
                )
            if attention_mask.size(-2) < seq_len or attention_mask.size(-1) < seq_len:
                raise ValueError("attention_mask is shorter than the current sequence length")
            query_mask = attention_mask[:, 0, -seq_len:, -seq_len:]
            diagonal = query_mask.diagonal(dim1=-2, dim2=-1)
            return diagonal.eq(0)
        raise ValueError(
            f"Unsupported attention_mask shape for Delta-Mem state updates: {tuple(attention_mask.shape)}"
        )

    def _masked_gate_mean(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_mask is None:
            return values.mean()
        expanded_mask = token_mask.unsqueeze(-1).unsqueeze(-1)
        masked_values = values * expanded_mask.to(dtype=values.dtype)
        denom = expanded_mask.sum().clamp_min(1).to(dtype=values.dtype)
        return masked_values.sum() / denom

    def _masked_hidden_norm(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        norms = values.float().norm(dim=-1)
        if token_mask is None:
            return norms.mean()
        if not token_mask.any():
            return norms.new_zeros(())
        return norms.masked_select(token_mask).mean()

    def _masked_hidden_mean(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_mask is None:
            return values.mean(dim=1)
        weights = token_mask.unsqueeze(-1).to(dtype=values.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denom

    def _masked_ratio_mean(
        self,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        ratios = numerator.float() / denominator.float().clamp_min(eps)
        if token_mask is None:
            return ratios.mean()
        if not token_mask.any():
            return ratios.new_zeros(())
        return ratios.masked_select(token_mask).mean()

    def _apply_delta_o_rmsnorm(self, delta_o: torch.Tensor) -> torch.Tensor:
        if not self.delta_o_rmsnorm:
            return delta_o
        normalized = F.rms_norm(
            delta_o.float(),
            (delta_o.shape[-1],),
            weight=self.delta_o_rmsnorm_weight.float(),
            eps=self.delta_o_rmsnorm_eps,
        )
        return normalized.to(dtype=delta_o.dtype)

    def _delta_scale_multiplier(self, head_name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.trainable_delta_scale:
            return torch.ones((), dtype=dtype, device=device)
        if self.delta_scale_granularity == "head":
            head_index = VALID_DELTA_HEADS.index(head_name)
            raw = self.delta_scale_raw[head_index]
        else:
            raw = self.delta_scale_raw[0]
        return (torch.sigmoid(raw) * self.delta_scale_max).to(device=device, dtype=dtype)

    def _project_delta_head(
        self,
        reads: torch.Tensor,
        weight: torch.Tensor,
        head_name: str,
    ) -> torch.Tensor | None:
        if head_name not in self.active_delta_heads:
            return None
        projected = F.linear(reads, weight)
        scale = self._delta_scale_multiplier(head_name, projected.dtype, projected.device)
        return projected * self.delta_scaling * scale

    def _resolve_read_context_mask(
        self,
        token_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.read_context_mask is None:
            return token_mask
        if self.read_context_mask.size(0) != batch_size or self.read_context_mask.size(1) != seq_len:
            return token_mask
        return self.read_context_mask.to(device=device)

    def _global_partition_logit_bias(
        self,
        partition_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if (
            self.num_global_memory_partitions <= 0
            or self.global_memory_read_logit_bias == 0.0
            or partition_count < self.num_global_memory_partitions
        ):
            return None
        bias = torch.zeros(partition_count, device=device, dtype=dtype)
        bias[: self.num_global_memory_partitions] = self.global_memory_read_logit_bias
        return bias

    def _partition_query_scores(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        *,
        partition_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_q_seq.ndim == 4:
            partition_queries = memory_q_seq.permute(0, 2, 1, 3)
        else:
            partition_queries = memory_q_seq.unsqueeze(2).expand(
                -1,
                -1,
                partition_reads.size(2),
                -1,
            )
        scores = (partition_reads * partition_queries).sum(dim=-1) / math.sqrt(float(self.rank))
        if partition_logit_bias is not None:
            scores = scores + partition_logit_bias.view(1, 1, -1)
        return scores

    def _mask_partition_top_k(
        self,
        scores: torch.Tensor,
        *,
        top_k: int,
    ) -> torch.Tensor:
        if 0 < top_k < scores.size(-1):
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
            masked_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
            masked_scores.scatter_(-1, top_indices, top_scores)
            return masked_scores
        return scores

    def _partition_query_softmax_weights(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        top_k: int,
        partition_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self._partition_query_scores(
            partition_reads,
            memory_q_seq,
            partition_logit_bias=partition_logit_bias,
        )
        scores = self._mask_partition_top_k(scores, top_k=top_k)
        weights = F.softmax(scores, dim=-1)
        if token_mask is not None:
            weights = weights * token_mask.unsqueeze(-1).to(dtype=weights.dtype)
        return weights

    def _partition_sigmoid_weights(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        top_k: int,
        partition_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self._partition_query_scores(
            partition_reads,
            memory_q_seq,
            partition_logit_bias=partition_logit_bias,
        )
        scores = self._mask_partition_top_k(scores, top_k=top_k)
        if hasattr(self, "partition_sigmoid_gate_bias"):
            scores = scores + self.partition_sigmoid_gate_bias.view(1, 1, -1)
        weights = torch.sigmoid(scores)
        if token_mask is not None:
            weights = weights * token_mask.unsqueeze(-1).to(dtype=weights.dtype)
        return weights

    def _partition_cross_attention_reads(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        top_k: int,
        use_sigmoid_gates: bool,
    ) -> torch.Tensor:
        if memory_q_seq.ndim == 4:
            query = memory_q_seq.permute(0, 2, 1, 3)
        else:
            query = memory_q_seq.unsqueeze(2).expand(
                -1,
                -1,
                partition_reads.size(2),
                -1,
            )
        key_states = F.linear(partition_reads, self.partition_cross_attn_key_proj)
        value_states = F.linear(partition_reads, self.partition_cross_attn_value_proj)
        scores = (query * key_states).sum(dim=-1) / math.sqrt(float(self.rank))
        scores = self._mask_partition_top_k(scores, top_k=top_k)
        attn = F.softmax(scores, dim=-1)
        effective_weights = attn
        if use_sigmoid_gates:
            gate_scores = scores
            if hasattr(self, "partition_sigmoid_gate_bias"):
                gate_scores = gate_scores + self.partition_sigmoid_gate_bias.view(1, 1, -1)
            effective_weights = attn * torch.sigmoid(gate_scores)
        if token_mask is not None:
            effective_weights = effective_weights * token_mask.unsqueeze(-1).to(dtype=effective_weights.dtype)
        self.last_read_routes = effective_weights
        return torch.einsum("btp,btpi->bti", effective_weights, value_states)

    def _slot_query_softmax_weights(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._partition_query_softmax_weights(
            partition_reads,
            memory_q_seq,
            token_mask,
            top_k=self.slot_read_top_k,
            partition_logit_bias=self._global_partition_logit_bias(
                partition_reads.size(2),
                device=partition_reads.device,
                dtype=partition_reads.dtype,
            ),
        )

    def _split_global_partition_queries(
        self,
        memory_q_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if memory_q_seq.ndim == 4:
            return (
                memory_q_seq[:, : self.num_global_memory_partitions],
                memory_q_seq[:, self.num_global_memory_partitions :],
            )
        return memory_q_seq, memory_q_seq

    def _merge_split_partition_reads(
        self,
        local_reads: torch.Tensor,
        global_reads: torch.Tensor,
        local_routes: torch.Tensor,
        global_routes: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = local_reads.new_ones(*local_reads.shape[:-1], 1)
        if self.global_memory_merge_mode == "gated_residual":
            gate_input = torch.cat([local_reads, global_reads], dim=-1)
            gate = torch.sigmoid(self.global_memory_gate_proj(gate_input))
            if token_mask is not None:
                gate = gate * token_mask.unsqueeze(-1).to(dtype=gate.dtype)
            reads = local_reads + gate * global_reads
        else:
            reads = local_reads + global_reads
        effective_routes = torch.cat([global_routes * gate, local_routes], dim=-1)
        denom = effective_routes.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        effective_routes = effective_routes / denom
        self.last_global_merge_gate = gate.squeeze(-1).detach().float()
        return reads, effective_routes

    def _aggregate_split_partition_reads(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.global_memory_merge_mode == "joint_softmax":
            read_routes = self._partition_query_softmax_weights(
                partition_reads,
                memory_q_seq,
                token_mask,
                top_k=self.slot_read_top_k,
                partition_logit_bias=self._global_partition_logit_bias(
                    partition_reads.size(2),
                    device=partition_reads.device,
                    dtype=partition_reads.dtype,
                ),
            )
            self.last_read_routes = read_routes
            self.last_global_merge_gate = None
            return torch.einsum("btp,btpi->bti", read_routes, partition_reads)

        global_partition_reads = partition_reads[:, :, : self.num_global_memory_partitions, :]
        local_partition_reads = partition_reads[:, :, self.num_global_memory_partitions :, :]
        global_memory_q_seq, local_memory_q_seq = self._split_global_partition_queries(memory_q_seq)
        global_routes = self._partition_query_softmax_weights(
            global_partition_reads,
            global_memory_q_seq,
            token_mask,
            top_k=self.global_memory_read_top_k,
        )
        local_routes = self._partition_query_softmax_weights(
            local_partition_reads,
            local_memory_q_seq,
            token_mask,
            top_k=self.slot_read_top_k,
        )
        global_reads = torch.einsum("btp,btpi->bti", global_routes, global_partition_reads)
        local_reads = torch.einsum("btp,btpi->bti", local_routes, local_partition_reads)
        if token_mask is not None:
            mask = token_mask.unsqueeze(-1).to(dtype=global_reads.dtype)
            global_reads = global_reads * mask
            local_reads = local_reads * mask
        reads, effective_routes = self._merge_split_partition_reads(
            local_reads,
            global_reads,
            local_routes,
            global_routes,
            token_mask,
        )
        self.last_read_routes = effective_routes
        return reads

    def _aggregate_partition_reads(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        read_route_seq: torch.Tensor | None,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if read_route_seq is None:
            read_route_seq = self._slot_query_softmax_weights(
                partition_reads,
                memory_q_seq,
                token_mask,
            )
        self.last_read_routes = read_route_seq
        return torch.einsum("btp,btpi->bti", read_route_seq, partition_reads)

    def _token_state_reads(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        read_route_seq: torch.Tensor | None,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.multi_head_state:
            head_q = memory_q_seq.view(
                memory_q_seq.size(0),
                memory_q_seq.size(1),
                self.num_state_heads,
                self.rank,
            )
            reads = torch.einsum("bhij,bthj->bthi", state, head_q)
            reads = reads.reshape(memory_q_seq.size(0), memory_q_seq.size(1), self.state_read_dim)
        else:
            reads = torch.einsum("bij,btj->bti", state, memory_q_seq)
        if token_mask is not None:
            reads = reads * token_mask.unsqueeze(-1).to(dtype=reads.dtype)
        return reads

    def _message_write_inputs(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.write_enabled or self.write_message_ids is None:
            return None
        message_ids = self.write_message_ids
        if message_ids.dim() != 2:
            return None
        if message_ids.size(0) != hidden_states.size(0) or message_ids.size(1) != hidden_states.size(1):
            return None
        message_ids = message_ids.to(device=hidden_states.device)
        active_mask = message_ids.ge(0)
        if token_mask is not None:
            active_mask = active_mask & token_mask
        if not active_mask.any():
            return None
        return message_ids, active_mask

    def _build_message_write_means(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        message_inputs = self._message_write_inputs(hidden_states, token_mask)
        if not self.write_enabled or self.memory_write_granularity != "message_mean" or message_inputs is None:
            return None, None, None
        message_ids, active_mask = message_inputs
        max_message_id = int(message_ids.masked_select(active_mask).max().item())
        num_messages_max = max_message_id + 1
        message_hidden = hidden_states.new_zeros(
            hidden_states.size(0),
            num_messages_max,
            hidden_states.size(-1),
        )
        message_mask = torch.zeros(
            hidden_states.size(0),
            num_messages_max,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        summary_message_ids = torch.full(
            (hidden_states.size(0), num_messages_max),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for batch_idx in range(hidden_states.size(0)):
            sample_message_ids = message_ids[batch_idx]
            sample_active_mask = active_mask[batch_idx]
            if not sample_active_mask.any():
                continue
            for message_id in sample_message_ids.masked_select(sample_active_mask).unique(sorted=True).tolist():
                current_message_id = int(message_id)
                token_selector = sample_active_mask & sample_message_ids.eq(current_message_id)
                message_hidden[batch_idx, current_message_id] = hidden_states[batch_idx, token_selector].mean(dim=0)
                message_mask[batch_idx, current_message_id] = True
                summary_message_ids[batch_idx, current_message_id] = current_message_id
        return message_hidden, message_mask, summary_message_ids

    def _sentence_write_inputs(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.write_enabled or self.write_message_ids is None or self.write_sentence_ids is None:
            return None
        message_ids = self.write_message_ids
        sentence_ids = self.write_sentence_ids
        if message_ids.dim() != 2 or sentence_ids.dim() != 2:
            return None
        if (
            message_ids.size(0) != hidden_states.size(0)
            or message_ids.size(1) != hidden_states.size(1)
            or sentence_ids.size(0) != hidden_states.size(0)
            or sentence_ids.size(1) != hidden_states.size(1)
        ):
            return None
        message_ids = message_ids.to(device=hidden_states.device)
        sentence_ids = sentence_ids.to(device=hidden_states.device)
        active_mask = message_ids.ge(0) & sentence_ids.ge(0)
        if token_mask is not None:
            active_mask = active_mask & token_mask
        if not active_mask.any():
            return None
        return message_ids, sentence_ids, active_mask

    def _build_sentence_write_means(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        sentence_inputs = self._sentence_write_inputs(hidden_states, token_mask)
        if not self.write_enabled or self.memory_write_granularity != "sentence_mean" or sentence_inputs is None:
            return None, None, None
        message_ids, sentence_ids, active_mask = sentence_inputs
        max_sentence_id = int(sentence_ids.masked_select(active_mask).max().item())
        num_sentences_max = max_sentence_id + 1
        sentence_hidden = hidden_states.new_zeros(
            hidden_states.size(0),
            num_sentences_max,
            hidden_states.size(-1),
        )
        sentence_mask = torch.zeros(
            hidden_states.size(0),
            num_sentences_max,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        sentence_message_ids = torch.full(
            (hidden_states.size(0), num_sentences_max),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for batch_idx in range(hidden_states.size(0)):
            sample_sentence_ids = sentence_ids[batch_idx]
            sample_message_ids = message_ids[batch_idx]
            sample_active_mask = active_mask[batch_idx]
            if not sample_active_mask.any():
                continue
            for sentence_id in sample_sentence_ids.masked_select(sample_active_mask).unique(sorted=True).tolist():
                current_sentence_id = int(sentence_id)
                token_selector = sample_active_mask & sample_sentence_ids.eq(current_sentence_id)
                sentence_hidden[batch_idx, current_sentence_id] = hidden_states[batch_idx, token_selector].mean(dim=0)
                sentence_mask[batch_idx, current_sentence_id] = True
                sentence_message_ids[batch_idx, current_sentence_id] = int(
                    sample_message_ids.masked_select(token_selector)[0].item()
                )
        return sentence_hidden, sentence_mask, sentence_message_ids

    def _build_message_write_proposals(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        token_k_seq: torch.Tensor,
        token_v_seq: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        message_inputs = self._message_write_inputs(hidden_states, token_mask)
        if (
            not self.write_enabled
            or self.memory_write_granularity != "message_proposals"
            or message_inputs is None
        ):
            return None, None, None

        message_ids, active_mask = message_inputs
        state_summary = self.message_proposal_state_proj(
            state.reshape(hidden_states.size(0), -1).float().to(hidden_states.dtype)
        )
        token_features = torch.tanh(self.message_proposal_token_proj(hidden_states))
        novelty = hidden_states.new_zeros(hidden_states.size(0), hidden_states.size(1))
        if state.ndim == 3 and token_k_seq.ndim == 3 and token_v_seq.ndim == 3:
            predicted_values = torch.einsum("bij,btj->bti", state, token_k_seq)
            novelty = (token_v_seq - predicted_values).float().norm(dim=-1).to(hidden_states.dtype)
            novelty = novelty * active_mask.to(dtype=novelty.dtype)

        max_message_id = int(message_ids.masked_select(active_mask).max().item())
        num_messages_max = max_message_id + 1
        num_proposals_max = num_messages_max * self.memory_write_proposals_per_message
        proposal_hidden = hidden_states.new_zeros(
            hidden_states.size(0),
            num_proposals_max,
            hidden_states.size(-1),
        )
        proposal_mask = torch.zeros(
            hidden_states.size(0),
            num_proposals_max,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        proposal_message_ids = torch.full(
            (hidden_states.size(0), num_proposals_max),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )

        for batch_idx in range(hidden_states.size(0)):
            sample_message_ids = message_ids[batch_idx]
            sample_active_mask = active_mask[batch_idx]
            if not sample_active_mask.any():
                continue
            sample_state_summary = state_summary[batch_idx]
            proposal_index = 0
            for message_id in sample_message_ids.masked_select(sample_active_mask).unique(sorted=True).tolist():
                current_message_id = int(message_id)
                message_mask = sample_active_mask & sample_message_ids.eq(current_message_id)
                message_hidden_slice = hidden_states[batch_idx, message_mask]
                message_features = token_features[batch_idx, message_mask]
                message_novelty = novelty[batch_idx, message_mask]
                message_summary = message_hidden_slice.mean(dim=0)
                query_base = sample_state_summary + self.message_proposal_message_proj(message_summary)
                coverage = message_novelty.new_zeros(message_hidden_slice.size(0))
                for slot_idx in range(self.memory_write_proposals_per_message):
                    slot_query = query_base + self.message_proposal_slot_queries[slot_idx].to(
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    logits = torch.matmul(message_features, slot_query)
                    logits = logits + self.message_proposal_novelty_scale.to(hidden_states.dtype) * message_novelty
                    if slot_idx > 0:
                        logits = logits - self.message_proposal_coverage_scale.to(hidden_states.dtype) * coverage
                    attention = torch.softmax(logits, dim=0)
                    proposal_hidden[batch_idx, proposal_index] = torch.matmul(attention, message_hidden_slice)
                    proposal_mask[batch_idx, proposal_index] = True
                    proposal_message_ids[batch_idx, proposal_index] = current_message_id
                    coverage = coverage + attention
                    proposal_index += 1
        return proposal_hidden, proposal_mask, proposal_message_ids

    def _latent_memory_active(self) -> bool:
        return self.latent_memory_enabled and not self.write_enabled

    def _compute_latent_memory_slots(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_mask = self._resolve_read_context_mask(
            token_mask,
            batch_size=hidden_states.size(0),
            seq_len=hidden_states.size(1),
            device=hidden_states.device,
        )
        question_summary = self._masked_hidden_mean(hidden_states, context_mask)
        latent_query = self._normalize_memory_projection(
            F.linear(question_summary, self.latent_query_proj)
        )
        if self.num_memory_partitions > 1:
            if self.tie_memory_partition_read_write:
                route_logits = F.linear(question_summary, self.write_router_proj, self.write_router_bias)
            else:
                route_logits = F.linear(question_summary, self.read_router_proj, self.read_router_bias)
            read_route = self._routing_distribution(route_logits)
            partition_reads = torch.einsum("bpij,bj->bpi", state, latent_query)
            retrieved = torch.einsum("bp,bpr->br", read_route, partition_reads)
        else:
            retrieved = torch.einsum("bij,bj->bi", state, latent_query)
        # For latent-context readout, the prefix content should come from state retrieval
        # rather than directly rewriting the current read chunk into synthetic KV.
        if self.memory_readout_mode == "latent_context":
            slot_source = torch.zeros_like(question_summary)
        else:
            slot_source = question_summary
        weaver_input = torch.cat([slot_source, retrieved.to(dtype=question_summary.dtype)], dim=-1)
        latent_hidden = F.silu(self.latent_weaver_up(weaver_input))
        latent_slots = self.latent_weaver_down(latent_hidden).view(
            hidden_states.size(0),
            self.latent_memory_slots,
            self.hidden_size,
        )
        has_state = state.float().flatten(1).pow(2).mean(dim=-1) > 1e-12
        slot_valid = has_state.unsqueeze(-1).expand(-1, self.latent_memory_slots)
        latent_slots = latent_slots * slot_valid.unsqueeze(-1).to(dtype=latent_slots.dtype)
        return latent_slots, slot_valid

    def _get_or_build_latent_memory_slots(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        past_key_values,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._latent_memory_active():
            return None, None
        if (
            past_key_values is not None
            and self.cached_latent_slots is not None
            and self.cached_latent_slot_valid is not None
            and self.cached_latent_slots.size(0) == hidden_states.size(0)
        ):
            cached_slots = self.cached_latent_slots.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            cached_valid = self.cached_latent_slot_valid.to(device=hidden_states.device)
            return cached_slots, cached_valid
        latent_slots, slot_valid = self._compute_latent_memory_slots(hidden_states, state, token_mask)
        self.cached_latent_slots = latent_slots.detach()
        self.cached_latent_slot_valid = slot_valid.detach()
        return latent_slots, slot_valid

    def _latent_gate_value(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.latent_memory_enabled:
            return torch.tensor(1.0, device=device, dtype=dtype)
        return self.latent_gate.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    def _latent_slots_to_kv(
        self,
        latent_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent_slots.size(0)
        latent_k = F.linear(latent_slots, self.latent_k_proj)
        latent_v = F.linear(latent_slots, self.latent_v_proj)
        latent_k = self._normalize_key_states(
            latent_k.view(
                batch_size,
                self.latent_memory_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        latent_v = latent_v.view(
            batch_size,
            self.latent_memory_slots,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        return latent_k, latent_v

    def _prepend_latent_context(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        latent_slots: torch.Tensor,
        slot_valid: torch.Tensor,
        query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        latent_k, latent_v = self._latent_slots_to_kv(latent_slots)
        latent_gate = self._latent_gate_value(device=key_states.device, dtype=key_states.dtype)
        latent_k = latent_k.to(dtype=key_states.dtype) * latent_gate
        latent_v = latent_v.to(dtype=value_states.dtype) * latent_gate.to(dtype=value_states.dtype)
        key_states = torch.cat([latent_k, key_states], dim=-2)
        value_states = torch.cat([latent_v, value_states], dim=-2)
        if attention_mask is None:
            if bool(slot_valid.all()):
                return key_states, value_states, None
            full_mask = key_states.new_zeros(
                (key_states.size(0), 1, query_len, key_states.size(-2)),
                dtype=torch.float32,
            )
            full_mask[:, :, :, : self.latent_memory_slots] = full_mask[:, :, :, : self.latent_memory_slots].masked_fill(
                (~slot_valid).view(key_states.size(0), 1, 1, self.latent_memory_slots),
                torch.finfo(full_mask.dtype).min,
            )
            return key_states, value_states, full_mask.to(dtype=key_states.dtype)
        if attention_mask.dim() == 2:
            memory_mask = slot_valid.to(device=attention_mask.device, dtype=attention_mask.dtype)
            return (
                key_states,
                value_states,
                torch.cat([memory_mask, attention_mask], dim=-1),
            )
        if attention_mask.dim() == 4:
            memory_bias = attention_mask.new_zeros(
                attention_mask.size(0),
                attention_mask.size(1),
                attention_mask.size(2),
                self.latent_memory_slots,
            )
            min_value = torch.finfo(memory_bias.dtype).min
            memory_bias = memory_bias.masked_fill(
                (~slot_valid).view(attention_mask.size(0), 1, 1, self.latent_memory_slots),
                min_value,
            )
            return (
                key_states,
                value_states,
                torch.cat([memory_bias, attention_mask], dim=-1),
            )
        raise ValueError(
            f"Unsupported attention_mask shape for latent context readout: {tuple(attention_mask.shape)}"
        )

    def _memory_branch_attention(
        self,
        hidden_states: torch.Tensor,
        latent_slots: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(slot_valid.any()):
            return hidden_states.new_zeros(hidden_states.shape)
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self._normalize_query_states(
            self._base_query_projection(hidden_states).view(batch_size, seq_len, -1, self.head_dim)
        ).transpose(1, 2)
        key_states, value_states = self._latent_slots_to_kv(latent_slots)
        if key_states.size(1) != query_states.size(1):
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_scores = torch.matmul(
            query_states.float(),
            key_states.float().transpose(-2, -1),
        ) * self.scaling
        attn_scores = attn_scores.masked_fill(
            (~slot_valid).view(batch_size, 1, 1, self.latent_memory_slots),
            torch.finfo(attn_scores.dtype).min,
        )
        attn_probs = torch.softmax(attn_scores, dim=-1).to(dtype=value_states.dtype)
        branch_output = torch.matmul(attn_probs, value_states)
        return branch_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()

    def _synthetic_kv_active(self) -> bool:
        return self.memory_readout_mode == "synthetic_kv" and not self.write_enabled

    def _project_state_to_synthetic_kv(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        state_flat = state.reshape(batch_size, -1)
        synthetic_k = F.linear(state_flat, self.synthetic_k_proj)
        synthetic_v = F.linear(state_flat, self.synthetic_v_proj)
        synthetic_k = synthetic_k.view(
            batch_size,
            self.synthetic_memory_slots,
            self.base.k_proj.out_features,
        )
        synthetic_v = synthetic_v.view(
            batch_size,
            self.synthetic_memory_slots,
            self.base_v_out_features,
        )
        slot_valid = (
            synthetic_k.float().pow(2).mean(dim=-1)
            + synthetic_v.float().pow(2).mean(dim=-1)
        ) > 1e-12
        synthetic_k = self._normalize_key_states(
            synthetic_k.view(
                batch_size,
                self.synthetic_memory_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        synthetic_v = synthetic_v.view(
            batch_size,
            self.synthetic_memory_slots,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        return synthetic_k, synthetic_v, slot_valid

    def _prepend_synthetic_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        state: torch.Tensor,
        query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        synthetic_k, synthetic_v, slot_valid = self._project_state_to_synthetic_kv(state)
        key_states = torch.cat([synthetic_k.to(dtype=key_states.dtype), key_states], dim=-2)
        value_states = torch.cat([synthetic_v.to(dtype=value_states.dtype), value_states], dim=-2)

        if attention_mask is None:
            if bool(slot_valid.all()):
                return key_states, value_states, None
            full_mask = key_states.new_zeros(
                (key_states.size(0), 1, query_len, key_states.size(-2)),
                dtype=torch.float32,
            )
            full_mask[:, :, :, : self.synthetic_memory_slots] = full_mask[:, :, :, : self.synthetic_memory_slots].masked_fill(
                (~slot_valid).view(key_states.size(0), 1, 1, self.synthetic_memory_slots),
                torch.finfo(full_mask.dtype).min,
            )
            return key_states, value_states, full_mask.to(dtype=key_states.dtype)

        if attention_mask.dim() == 2:
            memory_mask = slot_valid.to(device=attention_mask.device, dtype=attention_mask.dtype)
            return (
                key_states,
                value_states,
                torch.cat([memory_mask, attention_mask], dim=-1),
            )
        if attention_mask.dim() == 4:
            memory_bias = attention_mask.new_zeros(
                attention_mask.size(0),
                attention_mask.size(1),
                attention_mask.size(2),
                self.synthetic_memory_slots,
            )
            min_value = torch.finfo(memory_bias.dtype).min
            memory_bias = memory_bias.masked_fill(
                (~slot_valid).view(attention_mask.size(0), 1, 1, self.synthetic_memory_slots),
                min_value,
            )
            return (
                key_states,
                value_states,
                torch.cat([memory_bias, attention_mask], dim=-1),
            )
        raise ValueError(
            f"Unsupported attention_mask shape for synthetic KV readout: {tuple(attention_mask.shape)}"
        )

    def _memory_affine_scan_torch(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        keep_seq: torch.Tensor,
        erase_seq: torch.Tensor,
        write_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = memory_q_seq.shape
        current_state = state
        read_steps: list[torch.Tensor] = []

        for token_idx in range(seq_len):
            q_t = memory_q_seq[:, token_idx, :]
            k_t = memory_k_seq[:, token_idx, :]
            v_t = memory_v_seq[:, token_idx, :]
            keep_t = keep_seq[:, token_idx, :].unsqueeze(-1)
            erase_t = erase_seq[:, token_idx, :].unsqueeze(-1)
            write_t = write_seq[:, token_idx, :].unsqueeze(-1)

            read_t = torch.einsum("bij,bj->bi", current_state, q_t)

            if token_mask is not None:
                valid = token_mask[:, token_idx].view(batch_size, 1)
                read_t = read_t * valid.to(dtype=read_t.dtype)

            pred_t = torch.einsum("bij,bj->bi", current_state, k_t)
            write_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(1)
            pred_outer = pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
            next_state = keep_t * current_state - erase_t * pred_outer + write_t * write_outer

            if token_mask is not None:
                valid_state = token_mask[:, token_idx].view(batch_size, 1, 1).to(dtype=next_state.dtype)
                current_state = next_state * valid_state + current_state * (1.0 - valid_state)
            else:
                current_state = next_state

            read_steps.append(read_t)

        reads = torch.stack(read_steps, dim=1)
        return current_state, reads

    def _memory_affine_scan(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        write_route_seq: torch.Tensor | None = None,
        read_route_seq: torch.Tensor | None = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep_seq, erase_seq, write_seq = self._memory_update_coefficients(
            beta_seq,
            lambda_seq,
            write_route_seq=write_route_seq,
        )
        if self.multi_head_state:
            batch_size, seq_len, _ = memory_q_seq.shape
            q_for_scan = memory_q_seq.view(batch_size, seq_len, self.num_state_heads, self.rank)
            k_for_scan = memory_k_seq.view(batch_size, seq_len, self.num_state_heads, self.rank)
            v_for_scan = memory_v_seq.view(batch_size, seq_len, self.num_state_heads, self.rank)
            state_for_scan = state.reshape(batch_size * self.num_state_heads, self.rank, self.rank)
            q_for_scan = q_for_scan.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            k_for_scan = k_for_scan.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            v_for_scan = v_for_scan.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            keep_for_scan = keep_seq.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            erase_for_scan = erase_seq.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            write_for_scan = write_seq.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            token_mask_for_scan = None
            if token_mask is not None:
                token_mask_for_scan = (
                    token_mask.unsqueeze(1)
                    .expand(batch_size, self.num_state_heads, seq_len)
                    .reshape(batch_size * self.num_state_heads, seq_len)
                )
        else:
            single_partition = state.ndim == 3
            if single_partition:
                state_for_scan = state
                q_for_scan = memory_q_seq
                k_for_scan = memory_k_seq
                v_for_scan = memory_v_seq
                keep_for_scan = keep_seq
                erase_for_scan = erase_seq
                write_for_scan = write_seq
                token_mask_for_scan = token_mask
            else:
                batch_size, num_partitions, rank, _ = state.shape
                seq_len = memory_q_seq.size(-2)
                state_for_scan = state.reshape(batch_size * num_partitions, rank, rank)
                if memory_q_seq.ndim == 4:
                    q_for_scan = memory_q_seq.reshape(batch_size * num_partitions, seq_len, rank)
                    k_for_scan = memory_k_seq.reshape(batch_size * num_partitions, seq_len, rank)
                    v_for_scan = memory_v_seq.reshape(batch_size * num_partitions, seq_len, rank)
                else:
                    q_for_scan = (
                        memory_q_seq.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len, rank)
                        .reshape(batch_size * num_partitions, seq_len, rank)
                    )
                    k_for_scan = (
                        memory_k_seq.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len, rank)
                        .reshape(batch_size * num_partitions, seq_len, rank)
                    )
                    v_for_scan = (
                        memory_v_seq.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len, rank)
                        .reshape(batch_size * num_partitions, seq_len, rank)
                    )
                keep_for_scan = keep_seq.reshape(batch_size * num_partitions, seq_len, rank)
                erase_for_scan = erase_seq.reshape(batch_size * num_partitions, seq_len, rank)
                write_for_scan = write_seq.reshape(batch_size * num_partitions, seq_len, rank)
                token_mask_for_scan = None
                if token_mask is not None:
                    token_mask_for_scan = (
                        token_mask.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len)
                        .reshape(batch_size * num_partitions, seq_len)
                    )

        use_triton = self.scan_impl != "torch"
        if use_triton:
            support = triton_scan_support(
                state_for_scan,
                q_for_scan,
                k_for_scan,
                v_for_scan,
                keep_for_scan,
                erase_for_scan,
                write_for_scan,
            )
            use_triton = support.supported and self.scan_impl in {"auto", "triton"}
            if self.scan_impl == "triton" and not support.supported:
                raise RuntimeError(f"Triton scan requested but unavailable: {support.reason}")
        if use_triton:
            final_state, reads = triton_affine_scan(
                state_for_scan,
                q_for_scan,
                k_for_scan,
                v_for_scan,
                keep_for_scan,
                erase_for_scan,
                write_for_scan,
                token_mask=token_mask_for_scan,
            )
        else:
            final_state, reads = self._memory_affine_scan_torch(
                state_for_scan,
                q_for_scan,
                k_for_scan,
                v_for_scan,
                keep_for_scan,
                erase_for_scan,
                write_for_scan,
                token_mask=token_mask_for_scan,
            )

        if self.multi_head_state:
            batch_size, seq_len, _ = memory_q_seq.shape
            final_state = final_state.reshape(batch_size, self.num_state_heads, self.rank, self.rank)
            reads = reads.reshape(batch_size, self.num_state_heads, seq_len, self.rank)
            reads = reads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.state_read_dim)
            return final_state, reads

        if state.ndim == 3:
            return final_state, reads

        batch_size, num_partitions, rank, _ = state.shape
        seq_len = memory_q_seq.size(-2)
        final_state = final_state.reshape(batch_size, num_partitions, rank, rank)
        partition_reads = reads.reshape(batch_size, num_partitions, seq_len, rank).permute(0, 2, 1, 3)
        aggregated_reads = self._aggregate_partition_reads(
            partition_reads,
            memory_q_seq,
            read_route_seq,
            token_mask,
        )
        if token_mask is not None:
            aggregated_reads = aggregated_reads * token_mask.unsqueeze(-1).to(dtype=aggregated_reads.dtype)
        return final_state, aggregated_reads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"DeltaMemAttention expects [batch, seq, hidden], got {tuple(hidden_states.shape)}"
            )

        batch_size, seq_len, _ = hidden_states.shape
        state = self._ensure_state(batch_size, hidden_states.device, hidden_states.dtype)
        token_mask = self._token_validity_mask(
            attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
        )
        token_memory_q_seq, token_memory_k_seq, token_memory_v_seq, beta_seq, lambda_seq = (
            self._memory_sequence_projections(hidden_states)
        )
        token_memory_q_seq, token_memory_k_seq, token_memory_v_seq = self._partition_memory_projections(
            token_memory_q_seq,
            token_memory_k_seq,
            token_memory_v_seq,
        )
        write_route_seq, read_route_seq = self._memory_partition_routes(hidden_states, token_mask)
        stats_beta = beta_seq
        stats_lambda = lambda_seq
        stats_mask = token_mask
        if self.num_memory_partitions > 1:
            self.last_write_routes = write_route_seq
            self.last_read_routes = read_route_seq
        else:
            self.last_write_routes = None
            self.last_read_routes = None
        if self.write_enabled:
            state_before_write = state
            write_hidden = None
            write_mask = None
            write_message_ids = None
            if self.memory_write_granularity == "message_mean":
                write_hidden, write_mask, write_message_ids = self._build_message_write_means(
                    hidden_states,
                    token_mask,
                )
            elif self.memory_write_granularity == "sentence_mean":
                write_hidden, write_mask, write_message_ids = self._build_sentence_write_means(
                    hidden_states,
                    token_mask,
                )
            if write_hidden is not None and write_mask is not None:
                write_memory_q_seq, write_memory_k_seq, write_memory_v_seq, stats_beta, stats_lambda = (
                    self._memory_sequence_projections(write_hidden)
                )
                write_memory_q_seq, write_memory_k_seq, write_memory_v_seq = self._partition_memory_projections(
                    write_memory_q_seq,
                    write_memory_k_seq,
                    write_memory_v_seq,
                )
                proposal_write_route_seq, proposal_read_route_seq = self._memory_partition_routes(
                    write_hidden,
                    write_mask,
                    message_ids=write_message_ids,
                )
                if self.num_memory_partitions > 1:
                    self.last_write_routes = proposal_write_route_seq
                    self.last_read_routes = proposal_read_route_seq
                state, _ = self._memory_affine_scan(
                    state,
                    write_memory_q_seq,
                    write_memory_k_seq,
                    write_memory_v_seq,
                    stats_beta,
                    stats_lambda,
                    write_route_seq=proposal_write_route_seq if self.num_memory_partitions > 1 else None,
                    read_route_seq=proposal_read_route_seq if self.num_memory_partitions > 1 else None,
                    token_mask=write_mask,
                )
                reads = self._token_state_reads(
                    state_before_write,
                    token_memory_q_seq,
                    read_route_seq,
                    token_mask,
                )
                stats_mask = write_mask
            else:
                state, reads = self._memory_affine_scan(
                    state,
                    token_memory_q_seq,
                    token_memory_k_seq,
                    token_memory_v_seq,
                    beta_seq,
                    lambda_seq,
                    write_route_seq=write_route_seq if self.num_memory_partitions > 1 else None,
                    read_route_seq=read_route_seq if self.num_memory_partitions > 1 else None,
                    token_mask=token_mask,
                )
        else:
            reads = self._token_state_reads(
                state,
                token_memory_q_seq,
                read_route_seq,
                token_mask,
            )
        self.delta_state = state
        self.last_beta_mean = self._masked_gate_mean(stats_beta, stats_mask)
        self.last_lambda_mean = self._masked_gate_mean(stats_lambda, stats_mask)
        latent_slots, latent_slot_valid = None, None
        self.last_latent_slot_valid_ratio = hidden_states.new_zeros(())
        self.last_latent_slot_norm = hidden_states.new_zeros(())

        delta_q, delta_k, delta_v = self._compute_delta_qkv_from_reads(reads)
        delta_o = self._project_delta_head(reads, self.delta_o_proj, "o")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base.head_dim)

        query_states, key_states, value_states = self._apply_delta_qkv(
            hidden_states,
            delta_q,
            delta_k,
            delta_v,
        )

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        cos, sin = position_embeddings
        query_states = self._normalize_query_states(query_states).transpose(1, 2)
        key_states = self._normalize_key_states(key_states).transpose(1, 2)
        value_states = self._normalize_value_states(value_states).transpose(1, 2)
        query_states, key_states = self._apply_standard_rotary(
            query_states,
            key_states,
            cos,
            sin,
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.base.layer_idx,
                cache_kwargs,
            )

        attention_mask_for_attn = attention_mask
        if self._synthetic_kv_active():
            key_states, value_states, attention_mask_for_attn = self._prepend_synthetic_kv(
                key_states,
                value_states,
                attention_mask,
                state=state,
                query_len=query_states.size(-2),
            )
        elif self.memory_readout_mode == "latent_context" and latent_slots is not None and latent_slot_valid is not None:
            key_states, value_states, attention_mask_for_attn = self._prepend_latent_context(
                key_states,
                value_states,
                attention_mask,
                latent_slots=latent_slots,
                slot_valid=latent_slot_valid,
                query_len=query_states.size(-2),
            )

        attention_interface = self.eager_attention_forward
        use_prefixed_memory = self._synthetic_kv_active() or (
            self.memory_readout_mode == "latent_context"
            and latent_slots is not None
            and latent_slot_valid is not None
        )
        if not use_prefixed_memory and self.base.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.base.config._attn_implementation
            ]

        attn_kwargs = dict(kwargs)
        if self.sliding_window is not None:
            attn_kwargs["sliding_window"] = self.sliding_window
        attn_output, attn_weights = attention_interface(
            self.base,
            query_states,
            key_states,
            value_states,
            attention_mask_for_attn,
            dropout=0.0 if not self.training else self.base.attention_dropout,
            scaling=self.base.scaling,
            **attn_kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        base_o_output = self.base.o_proj(attn_output)
        self.last_base_o_norm = self._masked_hidden_norm(base_o_output, token_mask)
        attn_output = base_o_output
        self.last_delta_o_norm = None
        self.last_delta_o_ratio = None
        if delta_o is not None:
            delta_o_typed = self._apply_delta_o_rmsnorm(delta_o.to(hidden_states.dtype))
            self.last_delta_o_norm = self._masked_hidden_norm(delta_o_typed, token_mask)
            self.last_delta_o_ratio = self._masked_ratio_mean(
                delta_o_typed.norm(dim=-1),
                base_o_output.norm(dim=-1),
                token_mask,
            )
            attn_output = attn_output + delta_o_typed
        if self.last_latent_gate_value is None:
            self.last_latent_gate_value = attn_output.new_zeros(()).detach().float()
        self.last_memory_reader_output = None
        self.last_memory_reader_norm = None
        return attn_output, attn_weights


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def attach_delta_mem(model: nn.Module, config: HFDeltaMemConfig) -> list[str]:
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, SUPPORTED_BASE_ATTENTION_TYPES):
            continue
        if name.split(".")[-1] not in config.target_modules:
            continue
        if config.target_layers and module.layer_idx not in config.target_layers:
            continue
        module = ensure_attention_compat_views(module)
        parent, attr = _get_parent_module(model, name)
        wrapped = DeltaMemAttention(module, config).to(
            device=module.q_proj.weight.device,
            dtype=module.q_proj.weight.dtype,
        )
        setattr(parent, attr, wrapped)
        replaced.append(name)
    if not replaced:
        raise RuntimeError("No target modules were replaced")
    return replaced


def reset_delta_mem_states(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, DeltaMemAttention):
            module.reset_state()


def iter_delta_mem_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, DeltaMemAttention):
            yield name, module


def set_delta_mem_write_enabled(model: nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_write_enabled(enabled)


def set_delta_mem_write_message_ids(
    model: nn.Module,
    message_ids: torch.Tensor | None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_write_message_ids(message_ids)


def set_delta_mem_write_sentence_ids(
    model: nn.Module,
    sentence_ids: torch.Tensor | None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_write_sentence_ids(sentence_ids)


def set_delta_mem_read_context_mask(
    model: nn.Module,
    token_mask: torch.Tensor | None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.read_context_mask = token_mask


def get_delta_mem_write_regularization(
    model: nn.Module,
    *,
    target: float = 0.0,
) -> torch.Tensor:
    penalties = []
    for _, module in iter_delta_mem_modules(model):
        if module.last_beta_mean is None:
            continue
        penalties.append((module.last_beta_mean - target).pow(2))
    if penalties:
        return torch.stack(penalties).mean()
    try:
        reference = next(model.parameters())
        return reference.new_zeros(())
    except StopIteration:
        return torch.zeros(())


def collect_delta_mem_gate_stats(model: nn.Module) -> dict[str, float]:
    stats = {
        "num_modules": 0,
        "beta_mean": 0.0,
        "lambda_mean": 0.0,
        "rankwise_gate_modules": 0,
    }
    for _, module in iter_delta_mem_modules(model):
        stats["num_modules"] += 1
        if module.rankwise_gates:
            stats["rankwise_gate_modules"] += 1
        if module.last_beta_mean is not None:
            stats["beta_mean"] += float(module.last_beta_mean.detach().float().item())
        if module.last_lambda_mean is not None:
            stats["lambda_mean"] += float(module.last_lambda_mean.detach().float().item())
    if stats["num_modules"] > 0:
        stats["beta_mean"] /= stats["num_modules"]
        stats["lambda_mean"] /= stats["num_modules"]
    return stats


def collect_delta_mem_weight_stats(model: nn.Module) -> dict[str, float]:
    stats: dict[str, float] = {
        "num_modules": 0,
        "memory_reader_modules": 0,
        "memory_q_proj_norm_sum": 0.0,
        "memory_k_proj_norm_sum": 0.0,
        "memory_v_proj_norm_sum": 0.0,
        "delta_q_proj_norm_sum": 0.0,
        "delta_k_proj_norm_sum": 0.0,
        "delta_v_proj_norm_sum": 0.0,
        "delta_o_proj_norm_sum": 0.0,
        "delta_scale_mean_sum": 0.0,
        "trainable_delta_scale_modules": 0,
        "beta_proj_norm_sum": 0.0,
        "beta_bias_mean_sum": 0.0,
        "memory_reader_up_norm_sum": 0.0,
        "memory_reader_down_norm_sum": 0.0,
    }
    for _, module in iter_delta_mem_modules(model):
        stats["num_modules"] += 1
        stats["memory_q_proj_norm_sum"] += module.memory_q_proj.float().norm().item()
        stats["memory_k_proj_norm_sum"] += module.memory_k_proj.float().norm().item()
        stats["memory_v_proj_norm_sum"] += module.memory_v_proj.float().norm().item()
        stats["delta_q_proj_norm_sum"] += module.delta_q_proj.float().norm().item()
        stats["delta_k_proj_norm_sum"] += module.delta_k_proj.float().norm().item()
        stats["delta_v_proj_norm_sum"] += module.delta_v_proj.float().norm().item()
        stats["delta_o_proj_norm_sum"] += module.delta_o_proj.float().norm().item()
        if module.trainable_delta_scale:
            stats["trainable_delta_scale_modules"] += 1
            stats["delta_scale_mean_sum"] += (
                torch.sigmoid(module.delta_scale_raw.float()).mean().item() * module.delta_scale_max
            )
        stats["beta_proj_norm_sum"] += module.beta_proj.float().norm().item()
        stats["beta_bias_mean_sum"] += module.beta_bias.float().mean().item()
        if module.memory_reader_enabled:
            stats["memory_reader_modules"] += 1
            stats["memory_reader_up_norm_sum"] += (
                module.memory_reader_up.weight.float().norm().item()
            )
            stats["memory_reader_down_norm_sum"] += (
                module.memory_reader_down.weight.float().norm().item()
            )
    return stats


def snapshot_delta_mem_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        snapshot[f"{name}.memory_q_proj"] = module.memory_q_proj.detach().float().cpu().clone()
        snapshot[f"{name}.memory_k_proj"] = module.memory_k_proj.detach().float().cpu().clone()
        snapshot[f"{name}.memory_v_proj"] = module.memory_v_proj.detach().float().cpu().clone()
        snapshot[f"{name}.delta_q_proj"] = module.delta_q_proj.detach().float().cpu().clone()
        snapshot[f"{name}.delta_k_proj"] = module.delta_k_proj.detach().float().cpu().clone()
        snapshot[f"{name}.delta_v_proj"] = module.delta_v_proj.detach().float().cpu().clone()
        snapshot[f"{name}.delta_o_proj"] = module.delta_o_proj.detach().float().cpu().clone()
        snapshot[f"{name}.beta_proj"] = module.beta_proj.detach().float().cpu().clone()
        snapshot[f"{name}.beta_bias"] = module.beta_bias.detach().float().cpu().clone()
        if not module.couple_lambda:
            snapshot[f"{name}.lambda_proj"] = module.lambda_proj.detach().float().cpu().clone()
            snapshot[f"{name}.lambda_bias"] = module.lambda_bias.detach().float().cpu().clone()
        if module.memory_reader_enabled:
            snapshot[f"{name}.memory_reader_up.weight"] = (
                module.memory_reader_up.weight.detach().float().cpu().clone()
            )
            snapshot[f"{name}.memory_reader_up.bias"] = (
                module.memory_reader_up.bias.detach().float().cpu().clone()
            )
            snapshot[f"{name}.memory_reader_down.weight"] = (
                module.memory_reader_down.weight.detach().float().cpu().clone()
            )
            snapshot[f"{name}.memory_reader_down.bias"] = (
                module.memory_reader_down.bias.detach().float().cpu().clone()
            )
    return snapshot


def diff_delta_mem_snapshots(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, float]:
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    for key, before_tensor in before.items():
        diff = (after[key] - before_tensor).abs()
        max_abs_diff = max(max_abs_diff, diff.max().item())
        total_abs_diff += diff.sum().item()
    return {
        "max_abs_diff": max_abs_diff,
        "total_abs_diff": total_abs_diff,
    }


def collect_delta_mem_state_stats(model: nn.Module) -> dict[str, float]:
    num_modules = 0
    nonzero_modules = 0
    max_state_norm = 0.0
    mean_state_norm = 0.0
    max_state_abs = 0.0
    for _, module in iter_delta_mem_modules(model):
        num_modules += 1
        if module.delta_state is None:
            continue
        state = module.delta_state.float()
        state_norm = state.norm().item()
        mean_state_norm += state_norm
        max_state_norm = max(max_state_norm, state_norm)
        max_state_abs = max(max_state_abs, state.abs().max().item())
        if state.abs().max().item() > 0:
            nonzero_modules += 1
    if num_modules > 0:
        mean_state_norm /= num_modules
    return {
        "num_modules": num_modules,
        "nonzero_modules": nonzero_modules,
        "max_state_norm": max_state_norm,
        "mean_state_norm": mean_state_norm,
        "max_state_abs": max_state_abs,
    }


def collect_delta_mem_memory_reader_stats(model: nn.Module) -> dict[str, float]:
    enabled_modules = 0
    active_modules = 0
    mean_output_norm = 0.0
    max_output_norm = 0.0
    for _, module in iter_delta_mem_modules(model):
        if not module.memory_reader_enabled:
            continue
        enabled_modules += 1
        if module.last_memory_reader_norm is None:
            continue
        active_modules += 1
        norm = float(module.last_memory_reader_norm.detach().float().item())
        mean_output_norm += norm
        max_output_norm = max(max_output_norm, norm)
    if active_modules > 0:
        mean_output_norm /= active_modules
    return {
        "enabled_modules": enabled_modules,
        "active_modules": active_modules,
        "mean_output_norm": mean_output_norm,
        "max_output_norm": max_output_norm,
    }


def collect_delta_mem_memory_reader_outputs(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    outputs: list[tuple[str, torch.Tensor]] = []
    for name, module in iter_delta_mem_modules(model):
        if not module.memory_reader_enabled or module.last_memory_reader_output is None:
            continue
        outputs.append((name, module.last_memory_reader_output))
    return outputs


def collect_delta_mem_output_ratio_stats(model: nn.Module) -> dict[str, float]:
    num_modules = 0
    modules_with_delta_o = 0
    mean_base_o_norm = 0.0
    mean_delta_o_norm = 0.0
    mean_delta_o_ratio = 0.0
    max_delta_o_ratio = 0.0
    for _, module in iter_delta_mem_modules(model):
        num_modules += 1
        if module.last_base_o_norm is not None:
            mean_base_o_norm += float(module.last_base_o_norm.detach().float().item())
        if module.last_delta_o_norm is not None:
            modules_with_delta_o += 1
            mean_delta_o_norm += float(module.last_delta_o_norm.detach().float().item())
        if module.last_delta_o_ratio is not None:
            ratio = float(module.last_delta_o_ratio.detach().float().item())
            mean_delta_o_ratio += ratio
            max_delta_o_ratio = max(max_delta_o_ratio, ratio)
    if num_modules > 0:
        mean_base_o_norm /= num_modules
    if modules_with_delta_o > 0:
        mean_delta_o_norm /= modules_with_delta_o
        mean_delta_o_ratio /= modules_with_delta_o
    return {
        "num_modules": num_modules,
        "modules_with_delta_o": modules_with_delta_o,
        "mean_base_o_norm": mean_base_o_norm,
        "mean_delta_o_norm": mean_delta_o_norm,
        "mean_delta_o_ratio": mean_delta_o_ratio,
        "max_delta_o_ratio": max_delta_o_ratio,
    }


def collect_delta_mem_latent_stats(model: nn.Module) -> dict[str, float]:
    enabled_modules = 0
    active_modules = 0
    valid_ratio = 0.0
    mean_slot_norm = 0.0
    gate_mean = 0.0
    for _, module in iter_delta_mem_modules(model):
        if not module.latent_memory_enabled:
            continue
        enabled_modules += 1
        if module.last_latent_slot_valid_ratio is not None:
            active_modules += 1
            valid_ratio += float(module.last_latent_slot_valid_ratio.detach().float().item())
        if module.last_latent_slot_norm is not None:
            mean_slot_norm += float(module.last_latent_slot_norm.detach().float().item())
        if module.last_latent_gate_value is not None:
            gate_mean += float(module.last_latent_gate_value.detach().float().item())
    if active_modules > 0:
        valid_ratio /= active_modules
        mean_slot_norm /= active_modules
        gate_mean /= active_modules
    return {
        "enabled_modules": enabled_modules,
        "active_modules": active_modules,
        "valid_ratio": valid_ratio,
        "mean_slot_norm": mean_slot_norm,
        "gate_mean": gate_mean,
    }


def collect_delta_mem_partition_route_stats(model: nn.Module) -> dict[str, float]:
    stats = {
        "enabled_modules": 0,
        "tied_read_write_modules": 0,
        "active_modules": 0,
        "write_route_entropy": 0.0,
        "read_route_entropy": 0.0,
        "route_alignment_mse": 0.0,
        "route_overlap": 0.0,
        "write_route_max": 0.0,
        "read_route_max": 0.0,
        "write_route_balance_l2": 0.0,
        "read_route_balance_l2": 0.0,
    }
    for _, module in iter_delta_mem_modules(model):
        if module.num_memory_partitions <= 1:
            continue
        stats["enabled_modules"] += 1
        if module.tie_memory_partition_read_write:
            stats["tied_read_write_modules"] += 1
        if module.last_write_routes is None or module.last_read_routes is None:
            continue
        stats["active_modules"] += 1
        write_routes = module.last_write_routes.detach().float()
        read_routes = module.last_read_routes.detach().float()
        write_entropy = -(write_routes * write_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        read_entropy = -(read_routes * read_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        uniform = write_routes.new_full(
            (module.num_memory_partitions,),
            1.0 / module.num_memory_partitions,
        )
        write_usage = write_routes.mean(dim=(0, 1))
        read_usage = read_routes.mean(dim=(0, 1))
        stats["write_route_entropy"] += float(write_entropy.item())
        stats["read_route_entropy"] += float(read_entropy.item())
        stats["route_alignment_mse"] += float((write_routes - read_routes).pow(2).mean().item())
        stats["route_overlap"] += float((write_routes * read_routes).sum(dim=-1).mean().item())
        stats["write_route_max"] += float(write_routes.max(dim=-1).values.mean().item())
        stats["read_route_max"] += float(read_routes.max(dim=-1).values.mean().item())
        stats["write_route_balance_l2"] += float(((write_usage - uniform).pow(2)).mean().item())
        stats["read_route_balance_l2"] += float(((read_usage - uniform).pow(2)).mean().item())
    if stats["active_modules"] > 0:
        for key in (
            "write_route_entropy",
            "read_route_entropy",
            "route_alignment_mse",
            "route_overlap",
            "write_route_max",
            "read_route_max",
            "write_route_balance_l2",
            "read_route_balance_l2",
        ):
            stats[key] /= stats["active_modules"]
    return stats


def get_delta_mem_partition_regularization(model: nn.Module) -> dict[str, torch.Tensor]:
    alignment_losses: list[torch.Tensor] = []
    entropy_losses: list[torch.Tensor] = []
    balance_losses: list[torch.Tensor] = []
    reference = None
    for _, module in iter_delta_mem_modules(model):
        if module.num_memory_partitions <= 1:
            continue
        if module.last_write_routes is None or module.last_read_routes is None:
            continue
        if reference is None:
            reference = module.memory_q_proj
        write_routes = module.last_write_routes
        read_routes = module.last_read_routes
        alignment_losses.append((write_routes - read_routes).pow(2).mean())
        write_entropy = -(write_routes * write_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        read_entropy = -(read_routes * read_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        entropy_losses.append((write_entropy + read_entropy) * 0.5)
        uniform = write_routes.new_full(
            (module.num_memory_partitions,),
            1.0 / module.num_memory_partitions,
        )
        write_usage = write_routes.mean(dim=(0, 1))
        read_usage = read_routes.mean(dim=(0, 1))
        balance_losses.append(
            (((write_usage - uniform).pow(2)).mean() + ((read_usage - uniform).pow(2)).mean()) * 0.5
        )
    if reference is None:
        try:
            reference = next(model.parameters())
        except StopIteration:
            zero = torch.zeros(())
            return {"alignment": zero, "entropy": zero, "balance": zero}
    zero = reference.new_zeros(())
    return {
        "alignment": torch.stack(alignment_losses).mean() if alignment_losses else zero,
        "entropy": torch.stack(entropy_losses).mean() if entropy_losses else zero,
        "balance": torch.stack(balance_losses).mean() if balance_losses else zero,
    }


def get_delta_mem_online_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        if module.delta_state is None:
            continue
        state[name] = module.delta_state.detach().cpu().clone()
    return state


def load_delta_mem_online_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    for name, tensor in state.items():
        module = module_map[name]
        if not isinstance(module, DeltaMemAttention):
            raise TypeError(f"{name} is not a DeltaMemAttention")
        module.delta_state = tensor.to(
            device=module.base.q_proj.weight.device,
            dtype=module.base.q_proj.weight.dtype,
        )


def freeze_non_delta_mem_params(model: nn.Module) -> list[str]:
    trainable = []
    for param in model.parameters():
        param.requires_grad = False
    for name, module in iter_delta_mem_modules(model):
        for sub_name, param in module.named_parameters():
            if sub_name.startswith("base."):
                param.requires_grad = False
                continue
            trainable_flag = module.is_trainable_parameter(sub_name)
            param.requires_grad = trainable_flag
            if trainable_flag:
                trainable.append(f"{name}.{sub_name}")
    return trainable


def get_delta_mem_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        for sub_name, param in module.named_parameters():
            if sub_name.startswith("base."):
                continue
            state_dict[f"{name}.{sub_name}"] = param.detach().cpu().clone()
    return state_dict


def load_delta_mem_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    for full_name, tensor in state_dict.items():
        module_name, param_name = full_name.rsplit(".", 1)
        module = module_map[module_name]
        param = getattr(module, param_name)
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def save_delta_mem_adapter(
    model: nn.Module,
    output_dir: str | Path,
    config: HFDeltaMemConfig,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_path)
    torch.save(get_delta_mem_state_dict(model), output_path / "delta_mem_adapter.pt")


def load_delta_mem_adapter(model: nn.Module, input_dir: str | Path) -> HFDeltaMemConfig:
    input_path = Path(input_dir)
    config = HFDeltaMemConfig.from_pretrained(input_path)
    adapter_state = torch.load(
        input_path / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    load_delta_mem_state_dict(model, adapter_state)
    return config
