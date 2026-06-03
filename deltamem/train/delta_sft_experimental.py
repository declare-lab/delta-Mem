from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_gate_stats,
    collect_delta_mem_partition_route_stats,
    collect_delta_mem_state_stats,
    collect_delta_mem_weight_stats,
    freeze_non_delta_mem_params,
    get_delta_mem_online_state,
    get_delta_mem_partition_regularization,
    get_delta_mem_write_regularization,
    iter_delta_mem_modules,
    load_delta_mem_online_state,
    normalize_delta_heads,
    normalize_memory_readout_mode,
    normalize_state_update_mode,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
from deltamem.chat_templates import apply_chat_template as apply_project_chat_template
from deltamem.model_loading import resolve_attn_implementation
from deltamem.core.write_segmentation import split_text_into_sentence_token_chunks

class _AccelerateKernelWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Detected kernel version" not in record.getMessage()


def suppress_non_actionable_accelerate_warnings() -> None:
    logger = logging.getLogger("accelerate.utils.other")
    if not any(isinstance(item, _AccelerateKernelWarningFilter) for item in logger.filters):
        logger.addFilter(_AccelerateKernelWarningFilter())


def compute_warmup_steps(
    *,
    train_samples: int,
    per_device_train_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    max_steps: int,
    warmup_ratio: float,
) -> int:
    if warmup_ratio <= 0.0:
        return 0
    if max_steps > 0:
        total_steps = max_steps
    else:
        global_micro_batch = max(1, per_device_train_batch_size) * max(1, world_size)
        num_batches = max(1, math.ceil(train_samples / global_micro_batch))
        steps_per_epoch = max(1, math.ceil(num_batches / max(1, gradient_accumulation_steps)))
        total_steps = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    return max(1, math.ceil(total_steps * warmup_ratio))


@dataclass
class SFTExample:
    messages: list[dict[str, str]]


class DeltaMemTrainer(Trainer):
    def __init__(
        self,
        *args,
        delta_config: HFDeltaMemConfig | None = None,
        write_sparsity_weight: float = 0.0,
        write_sparsity_target: float = 0.0,
        memory_loss_mode: str = "context_dropout_ce",
        memory_contrast_weight: float = 0.1,
        memory_kl_weight: float = 0.1,
        memory_margin: float = 0.1,
        memory_causal_weight: float = 1.0,
        memory_anchor_weight: float = 1.0,
        memory_anchor_margin: float = 0.005,
        memory_full_ce_weight: float = 0.0,
        memory_full_ce_max_length: int = 2048,
        memory_recover_weight: float = 0.25,
        memory_need_floor: float = 0.15,
        memory_probe_weight: float = 0.0,
        memory_probe_alpha: float = 0.4,
        memory_probe_margin: float = 0.01,
        memory_partition_alignment_weight: float = 0.0,
        memory_partition_entropy_weight: float = 0.0,
        memory_partition_balance_weight: float = 0.0,
        memory_dropout_no_memory_prob: float = 0.0,
        memory_dropout_state_only_prob: float = 0.0,
        context_ablation_mode: str = "mixed",
        context_ablation_no_state_prob: float = 0.2,
        context_ablation_state_only_prob: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.delta_config = delta_config
        self.write_sparsity_weight = write_sparsity_weight
        self.write_sparsity_target = write_sparsity_target
        self.memory_loss_mode = memory_loss_mode
        self.memory_contrast_weight = memory_contrast_weight
        self.memory_kl_weight = memory_kl_weight
        self.memory_margin = memory_margin
        self.memory_causal_weight = memory_causal_weight
        self.memory_anchor_weight = memory_anchor_weight
        self.memory_anchor_margin = memory_anchor_margin
        self.memory_full_ce_weight = memory_full_ce_weight
        self.memory_full_ce_max_length = memory_full_ce_max_length
        self.memory_recover_weight = memory_recover_weight
        self.memory_need_floor = memory_need_floor
        self.memory_probe_weight = memory_probe_weight
        if self.memory_probe_weight > 0.0:
            raise ValueError("memory_probe was removed with archived memory_reader support")
        self.memory_probe_alpha = memory_probe_alpha
        self.memory_probe_margin = memory_probe_margin
        self.memory_partition_alignment_weight = memory_partition_alignment_weight
        self.memory_partition_entropy_weight = memory_partition_entropy_weight
        self.memory_partition_balance_weight = memory_partition_balance_weight
        self.memory_dropout_no_memory_prob = memory_dropout_no_memory_prob
        self.memory_dropout_state_only_prob = memory_dropout_state_only_prob
        self.context_ablation_mode = context_ablation_mode
        self.context_ablation_no_state_prob = context_ablation_no_state_prob
        self.context_ablation_state_only_prob = context_ablation_state_only_prob
        self._last_write_sparsity_loss = 0.0
        self._last_memory_keep_loss = 0.0
        self._last_memory_reset_loss = 0.0
        self._last_memory_corrupt_loss = 0.0
        self._last_memory_margin_loss = 0.0
        self._last_memory_causal_loss = 0.0
        self._last_memory_anchor_loss = 0.0
        self._last_memory_full_ce_loss = 0.0
        self._last_memory_kl_loss = 0.0
        self._last_memory_reset_kl_loss = 0.0
        self._last_memory_margin_gap = 0.0
        self._last_memory_teacher_loss = 0.0
        self._last_memory_wmem = 0.0
        self._last_memory_probe_keep_loss = 0.0
        self._last_memory_probe_reset_loss = 0.0
        self._last_memory_probe_margin_loss = 0.0
        self._last_memory_probe_gap = 0.0
        self._last_memory_probe_kl_loss = 0.0
        self._last_memory_probe_ce_loss = 0.0
        self._last_memory_partition_alignment_loss = 0.0
        self._last_memory_partition_entropy_loss = 0.0
        self._last_memory_partition_balance_loss = 0.0
        self._last_partition_enabled_modules = 0.0
        self._last_partition_tied_read_write_modules = 0.0
        self._last_partition_active_modules = 0.0
        self._last_partition_write_route_entropy = 0.0
        self._last_partition_read_route_entropy = 0.0
        self._last_partition_route_alignment_mse = 0.0
        self._last_partition_route_overlap = 0.0
        self._last_partition_write_route_max = 0.0
        self._last_partition_read_route_max = 0.0
        self._last_partition_write_route_balance_l2 = 0.0
        self._last_partition_read_route_balance_l2 = 0.0
        self._ddp_static_graph_initialized = False

    def _maybe_enable_static_graph(self, model) -> None:
        if self._ddp_static_graph_initialized:
            return
        # The active trainer only keeps delta readout, so the legacy stacked-read
        # branches no longer need special DDP static-graph handling.
        self._ddp_static_graph_initialized = True

    def _reset_online_state(self, model) -> None:
        reset_delta_mem_states(model)
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_message_ids(model, None)
        set_delta_mem_write_sentence_ids(model, None)
        set_delta_mem_write_enabled(model, True)

    def _build_read_context_mask(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        labels = model_inputs.get("labels")
        attention_mask = model_inputs.get("attention_mask")
        if labels is None or attention_mask is None:
            return None
        return labels.eq(-100) & attention_mask.ne(0)

    def _unwrap_base_model(self, model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def _scatter_episode_state(
        self,
        model,
        active_rows: torch.Tensor,
        batch_size: int,
    ) -> None:
        for _, module in iter_delta_mem_modules(model):
            if module.delta_state is None:
                continue
            active_state = module.delta_state
            full_state = active_state.new_zeros((batch_size, *active_state.shape[1:]))
            full_state[active_rows.to(device=active_state.device)] = active_state
            module.delta_state = full_state

    def _corrupt_online_state(
        self,
        online_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        corrupted: dict[str, torch.Tensor] = {}
        for name, tensor in online_state.items():
            corrupt = tensor.clone()
            if corrupt.ndim == 3:
                size = corrupt.size(-1)
                row_perm = torch.roll(torch.arange(size, device=corrupt.device), shifts=1)
                col_perm = torch.arange(size - 1, -1, -1, device=corrupt.device)
                corrupt = corrupt.index_select(-2, row_perm).index_select(-1, col_perm)
            elif corrupt.ndim == 4:
                num_partitions = corrupt.size(1)
                size = corrupt.size(-1)
                part_perm = torch.roll(torch.arange(num_partitions, device=corrupt.device), shifts=1)
                row_perm = torch.roll(torch.arange(size, device=corrupt.device), shifts=1)
                col_perm = torch.arange(size - 1, -1, -1, device=corrupt.device)
                corrupt = (
                    corrupt.index_select(1, part_perm)
                    .index_select(-2, row_perm)
                    .index_select(-1, col_perm)
                )
            corrupted[name] = corrupt
        return corrupted

    def _prime_episode_state(
        self,
        model,
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        batch_size: int,
        write_message_ids: torch.Tensor | None = None,
        write_sentence_ids: torch.Tensor | None = None,
    ) -> None:
        if write_input_ids is None:
            set_delta_mem_write_message_ids(model, None)
            set_delta_mem_write_sentence_ids(model, None)
            return
        if write_attention_mask is None:
            raise ValueError("Episode batches require write_attention_mask")
        active_rows = write_attention_mask.any(dim=1)
        if not active_rows.any():
            set_delta_mem_write_message_ids(model, None)
            set_delta_mem_write_sentence_ids(model, None)
            return
        active_message_ids = None
        if write_message_ids is not None:
            active_message_ids = write_message_ids[active_rows]
        active_sentence_ids = None
        if write_sentence_ids is not None:
            active_sentence_ids = write_sentence_ids[active_rows]
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_message_ids(model, active_message_ids)
        set_delta_mem_write_sentence_ids(model, active_sentence_ids)
        set_delta_mem_write_enabled(model, True)
        model(
            input_ids=write_input_ids[active_rows],
            attention_mask=write_attention_mask[active_rows],
            use_cache=False,
            return_dict=True,
        )
        set_delta_mem_write_message_ids(model, None)
        set_delta_mem_write_sentence_ids(model, None)
        self._scatter_episode_state(model, active_rows, batch_size)

    def _gather_teacher_read_logits(
        self,
        teacher_logits: torch.Tensor,
        write_lengths: torch.Tensor,
        read_lengths: torch.Tensor,
        read_width: int,
    ) -> torch.Tensor:
        gathered = teacher_logits.new_zeros(
            (teacher_logits.size(0), read_width, teacher_logits.size(-1))
        )
        for row_idx in range(teacher_logits.size(0)):
            write_len = int(write_lengths[row_idx].item())
            read_len = int(read_lengths[row_idx].item())
            gathered[row_idx, :read_len] = teacher_logits[
                row_idx,
                write_len : write_len + read_len,
            ]
        return gathered

    def _masked_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not token_mask.any():
            return student_logits.new_zeros(())
        log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl = F.kl_div(log_probs, teacher_probs, reduction="none").sum(dim=-1)
        return kl.masked_select(token_mask).mean()

    def _margin_objective(self, gap: torch.Tensor, margin: float) -> torch.Tensor:
        scaled_gap = (margin - gap) / max(margin, 1e-6)
        return F.softplus(scaled_gap)

    def _masked_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_mask = token_mask[:, 1:]
        if not shift_mask.any():
            return logits.new_zeros(())
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        return ce.masked_select(shift_mask).mean()

    def _capture_live_online_state(
        self,
        model,
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, module in iter_delta_mem_modules(model):
            if module.delta_state is None:
                continue
            state[name] = module.delta_state
        return state

    def _stack_batch_tensor(self, tensor: torch.Tensor, repeats: int) -> torch.Tensor:
        return torch.cat([tensor] * repeats, dim=0)

    def _split_stacked_tensor(
        self,
        tensor: torch.Tensor | None,
        batch_size: int,
        num_variants: int,
    ) -> list[torch.Tensor | None]:
        if tensor is None:
            return [None] * num_variants
        return list(torch.split(tensor, batch_size, dim=0))

    def _memory_branch_uses_stacked_variants(self) -> bool:
        return False

    def _compute_memory_branch_loss_stacked(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_labels: torch.Tensor,
        write_lengths: torch.Tensor,
        read_lengths: torch.Tensor,
        loss_kwargs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        token_mask = model_inputs["labels"].ne(-100) & model_inputs["attention_mask"].ne(0)
        read_context_mask = self._build_read_context_mask(model_inputs)
        keep_online_state = self._capture_live_online_state(model)
        if not keep_online_state:
            raise RuntimeError("memory_branch stacked read requires primed online state")

        variant_names = ["keep", "reset"]
        if self.memory_loss_mode in {"state_causal_anchor"}:
            variant_names.append("corrupt")
        num_variants = len(variant_names)

        detached_state = {name: tensor.detach().clone() for name, tensor in keep_online_state.items()}
        reset_state = {name: torch.zeros_like(tensor) for name, tensor in detached_state.items()}
        corrupt_state = self._corrupt_online_state(detached_state)
        stacked_state: dict[str, torch.Tensor] = {}
        for name, keep_tensor in keep_online_state.items():
            variant_tensors = [keep_tensor, reset_state[name]]
            if "corrupt" in variant_names:
                variant_tensors.append(corrupt_state[name])
            stacked_state[name] = torch.cat(variant_tensors, dim=0)

        load_delta_mem_online_state(model, stacked_state)
        stacked_model_inputs = {
            key: self._stack_batch_tensor(value, num_variants)
            for key, value in model_inputs.items()
            if key != "labels"
        }
        stacked_read_context_mask = None
        if read_context_mask is not None:
            stacked_read_context_mask = self._stack_batch_tensor(read_context_mask, num_variants)
        set_delta_mem_read_context_mask(model, stacked_read_context_mask)
        set_delta_mem_write_enabled(model, False)
        stacked_outputs = model(**stacked_model_inputs)
        if not isinstance(stacked_outputs, dict):
            stacked_outputs = {
                "logits": stacked_outputs.logits,
            }
        stacked_logits = stacked_outputs["logits"]
        batch_size = int(model_inputs["input_ids"].size(0))
        split_logits = self._split_stacked_tensor(stacked_logits, batch_size, num_variants)
        keep_logits = split_logits[0]
        reset_logits = split_logits[1]
        corrupt_logits = split_logits[2] if len(split_logits) > 2 else None
        assert keep_logits is not None and reset_logits is not None

        keep_loss = self._masked_lm_loss(keep_logits, model_inputs["labels"], token_mask)
        reset_loss = self._masked_lm_loss(reset_logits, model_inputs["labels"], token_mask)
        corrupt_loss = keep_loss.new_zeros(())
        if corrupt_logits is not None:
            corrupt_loss = self._masked_lm_loss(corrupt_logits, model_inputs["labels"], token_mask)

        keep_outputs = {
            "loss": keep_loss,
            "logits": keep_logits,
        }
        reset_outputs = {
            "loss": reset_loss,
            "logits": reset_logits,
        }

        teacher_loss = keep_loss.new_zeros(())
        full_ce_loss = keep_loss.new_zeros(())
        teacher_read_logits = None
        keep_kl_loss = keep_loss.new_zeros(())
        reset_kl_loss = keep_loss.new_zeros(())
        if True:
            with torch.no_grad():
                self._reset_online_state(model)
                set_delta_mem_read_context_mask(model, None)
                set_delta_mem_write_enabled(model, True)
                teacher_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=full_labels,
                    **loss_kwargs,
                )
                teacher_logits = (
                    teacher_outputs["logits"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs.logits
                )
                teacher_loss = (
                    teacher_outputs["loss"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs[0]
                )
                if teacher_loss.ndim > 0:
                    teacher_loss = teacher_loss.mean()
                teacher_read_logits = self._gather_teacher_read_logits(
                    teacher_logits,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    read_width=model_inputs["input_ids"].size(1),
                )

            keep_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )
            reset_kl_loss = self._masked_kl_loss(
                reset_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )

        if self.memory_full_ce_weight > 0.0:
            aux_input_ids = full_input_ids
            aux_attention_mask = full_attention_mask
            aux_labels = full_labels
            if self.memory_full_ce_max_length > 0 and aux_input_ids.size(1) > self.memory_full_ce_max_length:
                aux_input_ids = aux_input_ids[:, -self.memory_full_ce_max_length :]
                aux_attention_mask = aux_attention_mask[:, -self.memory_full_ce_max_length :]
                aux_labels = aux_labels[:, -self.memory_full_ce_max_length :]
            self._reset_online_state(model)
            set_delta_mem_read_context_mask(model, None)
            set_delta_mem_write_enabled(model, True)
            full_ce_outputs = model(
                input_ids=aux_input_ids,
                attention_mask=aux_attention_mask,
                labels=aux_labels,
                **loss_kwargs,
            )
            full_ce_loss = (
                full_ce_outputs["loss"] if isinstance(full_ce_outputs, dict) else full_ce_outputs[0]
            )
            if full_ce_loss.ndim > 0:
                full_ce_loss = full_ce_loss.mean()

        margin_gap = reset_loss - keep_loss
        wmem = keep_loss.new_zeros(())
        causal_loss = keep_loss.new_zeros(())
        anchor_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode == "teacher_gap_kl":
            margin_gap = reset_kl_loss - keep_kl_loss
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        elif self.memory_loss_mode == "state_causal_anchor":
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            causal_gap = corrupt_loss - keep_loss
            causal_loss = self._margin_objective(causal_gap, self.memory_margin)
            anchor_gap = keep_loss - teacher_loss
            scaled_anchor = (anchor_gap - self.memory_anchor_margin) / max(
                self.memory_anchor_margin,
                1e-6,
            )
            anchor_loss = F.softplus(scaled_anchor)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_causal_weight * causal_loss
                + self.memory_anchor_weight * anchor_loss
            )
        elif self.memory_loss_mode in {"teacher_kl_only", "teacher_kl_wmem1"}:
            margin_loss = keep_loss.new_zeros(())
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = keep_loss.new_tensor(1.0)
            weighted = self.memory_kl_weight * keep_kl_loss
        elif self.memory_loss_mode == "teacher_kl_wmem":
            margin_loss = keep_loss.new_zeros(())
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = margin_gap.clamp_(min=0.0, max=1.0)
            weighted = self.memory_kl_weight * wmem * keep_kl_loss
        elif self.memory_loss_mode in {"state_margin_kl", "latent_prefix_margin"}:
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        else:
            raise ValueError(f"Unsupported memory_loss_mode: {self.memory_loss_mode}")
        weighted = weighted + self.memory_full_ce_weight * full_ce_loss
        probe_stats = {
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

        memory_loss = weighted
        total_loss = keep_loss + memory_loss
        outputs = dict(keep_outputs)
        outputs["memory_loss"] = memory_loss.detach()
        outputs["memory_full_ce_loss"] = total_loss.new_tensor(float(full_ce_loss.detach().float().item())).detach()
        outputs["memory_keep_loss"] = total_loss.detach()
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        return total_loss, outputs, {
            "keep_loss": float(keep_loss.detach().float().item()),
            "reset_loss": float(reset_loss.detach().float().item()),
            "corrupt_loss": float(corrupt_loss.detach().float().item()),
            "teacher_loss": float(teacher_loss.detach().float().item()),
            "margin_loss": float(margin_loss.detach().float().item()),
            "causal_loss": float(causal_loss.detach().float().item()),
            "anchor_loss": float(anchor_loss.detach().float().item()),
            "full_ce_loss": float(full_ce_loss.detach().float().item()),
            "kl_loss": float(keep_kl_loss.detach().float().item()),
            "reset_kl_loss": float(reset_kl_loss.detach().float().item()),
            "margin_gap": float(margin_gap.detach().float().item()),
            "wmem": float(wmem.detach().float().item()),
            **probe_stats,
        }

    def _build_full_sequence_inputs(
        self,
        full_input_ids: torch.Tensor | None,
        full_attention_mask: torch.Tensor | None,
        full_labels: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if full_input_ids is None or full_attention_mask is None or full_labels is None:
            raise ValueError("Full-sequence context ablations require full episode tensors")
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": full_labels,
        }

    def _sample_context_ablation_mode(self) -> str:
        mode = self.context_ablation_mode
        if mode != "mixed":
            return mode
        no_state_prob = self.context_ablation_no_state_prob
        state_only_prob = self.context_ablation_state_only_prob
        if (
            no_state_prob < 0.0
            or state_only_prob < 0.0
            or no_state_prob + state_only_prob > 1.0
        ):
            raise ValueError(
                "context ablation probabilities must satisfy p >= 0, q >= 0, p + q <= 1"
            )
        mode_sample = float(torch.rand(()).item())
        if mode_sample < no_state_prob:
            return "full_context_no_state"
        if mode_sample < no_state_prob + state_only_prob:
            return "state_only"
        return "full_context_plus_state"

    def _compute_context_dropout_ce(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        state_only_input_ids: torch.Tensor | None,
        state_only_attention_mask: torch.Tensor | None,
        state_only_labels: torch.Tensor | None,
        state_only_write_input_ids: torch.Tensor | None,
        state_only_write_attention_mask: torch.Tensor | None,
        state_only_write_message_ids: torch.Tensor | None,
        state_only_write_sentence_ids: torch.Tensor | None,
    ):
        no_memory_prob = self.memory_dropout_no_memory_prob
        state_only_prob = self.memory_dropout_state_only_prob
        if no_memory_prob < 0.0 or state_only_prob < 0.0 or no_memory_prob + state_only_prob > 1.0:
            raise ValueError("memory dropout probabilities must satisfy p >= 0, q >= 0, p + q <= 1")

        mode_sample = float(torch.rand((), device=model_inputs["input_ids"].device).item())
        if mode_sample < no_memory_prob:
            mode = "no_memory"
        elif mode_sample < no_memory_prob + state_only_prob:
            mode = "state_only"
        else:
            mode = "both"

        if mode == "state_only":
            if (
                state_only_input_ids is None
                or state_only_attention_mask is None
                or state_only_labels is None
            ):
                raise ValueError("context_dropout_ce requires state_only episode tensors")
            active_inputs = {
                "input_ids": state_only_input_ids,
                "attention_mask": state_only_attention_mask,
                "labels": state_only_labels,
            }
            batch_size = int(state_only_input_ids.size(0))
            self._reset_online_state(model)
            self._prime_episode_state(
                model,
                write_input_ids=state_only_write_input_ids,
                write_attention_mask=state_only_write_attention_mask,
                batch_size=batch_size,
                write_message_ids=state_only_write_message_ids,
                write_sentence_ids=state_only_write_sentence_ids,
            )
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 1.0
        elif mode == "no_memory":
            active_inputs = model_inputs
            self._reset_online_state(model)
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 0.0
        else:
            active_inputs = model_inputs
            batch_size = int(model_inputs["input_ids"].size(0))
            self._reset_online_state(model)
            self._prime_episode_state(
                model,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                batch_size=batch_size,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
            )
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 1.0

        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                "logits": outputs.logits,
            }
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if loss.ndim > 0:
            loss = loss.mean()
        return loss, outputs, {
            "keep_loss": float(loss.detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": 0.0,
            "anchor_loss": 0.0,
            "full_ce_loss": 0.0,
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": 0.0,
            "wmem": wmem,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

    def _compute_context_ablation_ce(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        state_only_input_ids: torch.Tensor | None,
        state_only_attention_mask: torch.Tensor | None,
        state_only_labels: torch.Tensor | None,
        state_only_write_input_ids: torch.Tensor | None,
        state_only_write_attention_mask: torch.Tensor | None,
        state_only_write_message_ids: torch.Tensor | None,
        state_only_write_sentence_ids: torch.Tensor | None,
        full_input_ids: torch.Tensor | None,
        full_attention_mask: torch.Tensor | None,
        full_labels: torch.Tensor | None,
    ):
        mode = self._sample_context_ablation_mode()
        if mode == "full_context_no_state":
            active_inputs = self._build_full_sequence_inputs(
                full_input_ids,
                full_attention_mask,
                full_labels,
            )
            self._reset_online_state(model)
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 0.0
        elif mode == "state_only":
            if (
                state_only_input_ids is None
                or state_only_attention_mask is None
                or state_only_labels is None
            ):
                raise ValueError("context_ablation_ce requires state_only episode tensors")
            active_inputs = {
                "input_ids": state_only_input_ids,
                "attention_mask": state_only_attention_mask,
                "labels": state_only_labels,
            }
            batch_size = int(state_only_input_ids.size(0))
            self._reset_online_state(model)
            self._prime_episode_state(
                model,
                write_input_ids=state_only_write_input_ids,
                write_attention_mask=state_only_write_attention_mask,
                batch_size=batch_size,
                write_message_ids=state_only_write_message_ids,
                write_sentence_ids=state_only_write_sentence_ids,
            )
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 1.0
        elif mode == "full_context_plus_state":
            active_inputs = self._build_full_sequence_inputs(
                full_input_ids,
                full_attention_mask,
                full_labels,
            )
            batch_size = int(model_inputs["input_ids"].size(0))
            self._reset_online_state(model)
            self._prime_episode_state(
                model,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                batch_size=batch_size,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
            )
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 1.0
        else:
            raise ValueError(f"Unsupported context ablation mode: {mode}")

        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                "logits": outputs.logits,
            }
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if loss.ndim > 0:
            loss = loss.mean()
        return loss, outputs, {
            "keep_loss": float(loss.detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": 0.0,
            "anchor_loss": 0.0,
            "full_ce_loss": 0.0,
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": 0.0,
            "wmem": wmem,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

    def _compute_memory_objective(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        keep_outputs,
        keep_loss: torch.Tensor,
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_labels: torch.Tensor,
        write_lengths: torch.Tensor,
        read_lengths: torch.Tensor,
        keep_online_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.memory_loss_mode == "none":
            zero = keep_loss.new_zeros(())
            return zero, {
                "keep_loss": float(keep_loss.detach().float().item()),
                "reset_loss": 0.0,
                "corrupt_loss": 0.0,
                "teacher_loss": 0.0,
                "margin_loss": 0.0,
                "causal_loss": 0.0,
                "anchor_loss": 0.0,
                "full_ce_loss": 0.0,
                "kl_loss": 0.0,
                "reset_kl_loss": 0.0,
                "margin_gap": 0.0,
                "wmem": 0.0,
                "probe_keep_loss": 0.0,
                "probe_reset_loss": 0.0,
                "probe_margin_loss": 0.0,
                "probe_gap": 0.0,
                "probe_kl": 0.0,
                "probe_ce": 0.0,
            }
        if self.memory_loss_mode not in {
            "state_margin_kl",
            "latent_prefix_margin",
            "state_causal_anchor",
            "teacher_gap_kl",
            "teacher_kl_only",
            "teacher_kl_wmem1",
            "teacher_kl_wmem",
            "keep_only",
            "keep_full_kl",
            "keep_fullstate_kl",
            "keep_dual_kl",
        }:
            raise ValueError(f"Unsupported memory_loss_mode: {self.memory_loss_mode}")

        token_mask = model_inputs["labels"].ne(-100) & model_inputs["attention_mask"].ne(0)
        read_context_mask = self._build_read_context_mask(model_inputs)
        self._reset_online_state(model)
        set_delta_mem_read_context_mask(model, read_context_mask)
        set_delta_mem_write_enabled(model, False)
        reset_outputs = model(**model_inputs, **loss_kwargs)
        reset_loss = (
            reset_outputs["loss"] if isinstance(reset_outputs, dict) else reset_outputs[0]
        )
        if reset_loss.ndim > 0:
            reset_loss = reset_loss.mean()
        corrupt_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode in {"state_causal_anchor"}:
            self._reset_online_state(model)
            load_delta_mem_online_state(
                model,
                self._corrupt_online_state(keep_online_state),
            )
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            corrupt_outputs = model(**model_inputs, **loss_kwargs)
            corrupt_loss = (
                corrupt_outputs["loss"] if isinstance(corrupt_outputs, dict) else corrupt_outputs[0]
            )
            if corrupt_loss.ndim > 0:
                corrupt_loss = corrupt_loss.mean()

        teacher_loss = keep_loss.new_zeros(())
        full_ce_loss = keep_loss.new_zeros(())
        teacher_read_logits = None
        fullstate_teacher_read_logits = None
        keep_kl_loss = keep_loss.new_zeros(())
        reset_kl_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode in {
            "state_margin_kl",
            "latent_prefix_margin",
            "state_causal_anchor",
            "teacher_gap_kl",
            "teacher_kl_only",
            "teacher_kl_wmem1",
            "teacher_kl_wmem",
            "keep_full_kl",
            "keep_dual_kl",
        }:
            with torch.no_grad():
                self._reset_online_state(model)
                set_delta_mem_read_context_mask(model, None)
                set_delta_mem_write_enabled(model, True)
                teacher_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=full_labels,
                    **loss_kwargs,
                )
                teacher_logits = (
                    teacher_outputs["logits"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs.logits
                )
                teacher_loss = (
                    teacher_outputs["loss"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs[0]
                )
                if teacher_loss.ndim > 0:
                    teacher_loss = teacher_loss.mean()
                teacher_read_logits = self._gather_teacher_read_logits(
                    teacher_logits,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    read_width=model_inputs["input_ids"].size(1),
                )

            keep_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )
            reset_kl_loss = self._masked_kl_loss(
                reset_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )

        if self.memory_loss_mode in {"keep_fullstate_kl", "keep_dual_kl"}:
            with torch.no_grad():
                self._reset_online_state(model)
                if keep_online_state:
                    load_delta_mem_online_state(
                        model,
                        {name: tensor.detach().clone() for name, tensor in keep_online_state.items()},
                    )
                set_delta_mem_read_context_mask(model, None)
                set_delta_mem_write_enabled(model, True)
                fullstate_teacher_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=full_labels,
                    **loss_kwargs,
                )
                fullstate_teacher_logits = (
                    fullstate_teacher_outputs["logits"]
                    if isinstance(fullstate_teacher_outputs, dict)
                    else fullstate_teacher_outputs.logits
                )
                fullstate_teacher_read_logits = self._gather_teacher_read_logits(
                    fullstate_teacher_logits,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    read_width=model_inputs["input_ids"].size(1),
                )

        if self.memory_full_ce_weight > 0.0:
            aux_input_ids = full_input_ids
            aux_attention_mask = full_attention_mask
            aux_labels = full_labels
            if self.memory_full_ce_max_length > 0 and aux_input_ids.size(1) > self.memory_full_ce_max_length:
                aux_input_ids = aux_input_ids[:, -self.memory_full_ce_max_length :]
                aux_attention_mask = aux_attention_mask[:, -self.memory_full_ce_max_length :]
                aux_labels = aux_labels[:, -self.memory_full_ce_max_length :]
            self._reset_online_state(model)
            set_delta_mem_read_context_mask(model, None)
            set_delta_mem_write_enabled(model, True)
            full_ce_outputs = model(
                input_ids=aux_input_ids,
                attention_mask=aux_attention_mask,
                labels=aux_labels,
                **loss_kwargs,
            )
            full_ce_loss = (
                full_ce_outputs["loss"] if isinstance(full_ce_outputs, dict) else full_ce_outputs[0]
            )
            if full_ce_loss.ndim > 0:
                full_ce_loss = full_ce_loss.mean()

        margin_gap = reset_loss - keep_loss
        wmem = keep_loss.new_zeros(())
        causal_loss = keep_loss.new_zeros(())
        anchor_loss = keep_loss.new_zeros(())
        margin_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode == "keep_only":
            weighted = keep_loss.new_zeros(())
        elif self.memory_loss_mode == "keep_full_kl":
            weighted = self.memory_kl_weight * keep_kl_loss
            wmem = keep_loss.new_tensor(1.0)
        elif self.memory_loss_mode == "keep_fullstate_kl":
            if fullstate_teacher_read_logits is None:
                raise ValueError("keep_fullstate_kl requires fullstate teacher logits")
            reset_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                fullstate_teacher_read_logits,
                token_mask,
            )
            weighted = self.memory_kl_weight * reset_kl_loss
            wmem = keep_loss.new_tensor(1.0)
        elif self.memory_loss_mode == "keep_dual_kl":
            if fullstate_teacher_read_logits is None:
                raise ValueError("keep_dual_kl requires fullstate teacher logits")
            reset_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                fullstate_teacher_read_logits,
                token_mask,
            )
            weighted = self.memory_kl_weight * (keep_kl_loss + reset_kl_loss)
            wmem = keep_loss.new_tensor(1.0)
        elif self.memory_loss_mode == "teacher_gap_kl":
            margin_gap = reset_kl_loss - keep_kl_loss
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        elif self.memory_loss_mode == "state_causal_anchor":
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            causal_gap = corrupt_loss - keep_loss
            causal_loss = self._margin_objective(causal_gap, self.memory_margin)
            anchor_gap = keep_loss - teacher_loss
            scaled_anchor = (anchor_gap - self.memory_anchor_margin) / max(
                self.memory_anchor_margin,
                1e-6,
            )
            anchor_loss = F.softplus(scaled_anchor)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_causal_weight * causal_loss
                + self.memory_anchor_weight * anchor_loss
            )
        elif self.memory_loss_mode in {"teacher_kl_only", "teacher_kl_wmem1"}:
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = keep_loss.new_tensor(1.0)
            weighted = self.memory_kl_weight * keep_kl_loss
        elif self.memory_loss_mode == "teacher_kl_wmem":
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = margin_gap.clamp_(min=0.0, max=1.0)
            weighted = self.memory_kl_weight * wmem * keep_kl_loss
        elif self.memory_loss_mode in {"state_margin_kl", "latent_prefix_margin"}:
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        else:
            raise ValueError(f"Unsupported memory_loss_mode: {self.memory_loss_mode}")
        weighted = weighted + self.memory_full_ce_weight * full_ce_loss
        probe_stats = {
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }
        return weighted, {
            "keep_loss": float(keep_loss.detach().float().item()),
            "reset_loss": float(reset_loss.detach().float().item()),
            "corrupt_loss": float(corrupt_loss.detach().float().item()),
            "teacher_loss": float(teacher_loss.detach().float().item()),
            "margin_loss": float(margin_loss.detach().float().item()),
            "causal_loss": float(causal_loss.detach().float().item()),
            "anchor_loss": float(anchor_loss.detach().float().item()),
            "full_ce_loss": float(full_ce_loss.detach().float().item()),
            "kl_loss": float(keep_kl_loss.detach().float().item()),
            "reset_kl_loss": float(reset_kl_loss.detach().float().item()),
            "margin_gap": float(margin_gap.detach().float().item()),
            "wmem": float(wmem.detach().float().item()),
            **probe_stats,
        }

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        loss_kwargs = {}
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch

        model_inputs = dict(inputs)
        write_input_ids = model_inputs.pop("write_input_ids", None)
        write_attention_mask = model_inputs.pop("write_attention_mask", None)
        write_message_ids = model_inputs.pop("write_message_ids", None)
        write_sentence_ids = model_inputs.pop("write_sentence_ids", None)
        state_only_write_input_ids = model_inputs.pop("state_only_write_input_ids", None)
        state_only_write_attention_mask = model_inputs.pop("state_only_write_attention_mask", None)
        state_only_write_message_ids = model_inputs.pop("state_only_write_message_ids", None)
        state_only_write_sentence_ids = model_inputs.pop("state_only_write_sentence_ids", None)
        state_only_input_ids = model_inputs.pop("state_only_input_ids", None)
        state_only_attention_mask = model_inputs.pop("state_only_attention_mask", None)
        state_only_labels = model_inputs.pop("state_only_labels", None)
        full_input_ids = model_inputs.pop("full_input_ids", None)
        full_attention_mask = model_inputs.pop("full_attention_mask", None)
        full_labels = model_inputs.pop("full_labels", None)
        write_lengths = model_inputs.pop("write_lengths", None)
        read_lengths = model_inputs.pop("read_lengths", None)

        batch_size = int(model_inputs["input_ids"].size(0))

        has_episode_memory_inputs = (
            write_input_ids is not None
            and full_input_ids is not None
            and full_attention_mask is not None
            and full_labels is not None
            and write_lengths is not None
            and read_lengths is not None
        )
        memory_stats = {
            "keep_loss": 0.0,
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": 0.0,
            "anchor_loss": 0.0,
            "full_ce_loss": 0.0,
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": 0.0,
            "wmem": 0.0,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }
        if self.memory_loss_mode == "context_dropout_ce" and has_episode_memory_inputs:
            loss, outputs, memory_stats = self._compute_context_dropout_ce(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                state_only_input_ids=state_only_input_ids,
                state_only_attention_mask=state_only_attention_mask,
                state_only_labels=state_only_labels,
                state_only_write_input_ids=state_only_write_input_ids,
                state_only_write_attention_mask=state_only_write_attention_mask,
                state_only_write_message_ids=state_only_write_message_ids,
                state_only_write_sentence_ids=state_only_write_sentence_ids,
            )
        elif self.memory_loss_mode == "context_ablation_ce" and has_episode_memory_inputs:
            loss, outputs, memory_stats = self._compute_context_ablation_ce(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                state_only_input_ids=state_only_input_ids,
                state_only_attention_mask=state_only_attention_mask,
                state_only_labels=state_only_labels,
                state_only_write_input_ids=state_only_write_input_ids,
                state_only_write_attention_mask=state_only_write_attention_mask,
                state_only_write_message_ids=state_only_write_message_ids,
                state_only_write_sentence_ids=state_only_write_sentence_ids,
                full_input_ids=full_input_ids,
                full_attention_mask=full_attention_mask,
                full_labels=full_labels,
            )
        else:
            self._prime_episode_state(
                model,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                batch_size=batch_size,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
            )
            if has_episode_memory_inputs and self._memory_branch_uses_stacked_variants():
                loss, outputs, memory_stats = self._compute_memory_branch_loss_stacked(
                    model,
                    model_inputs,
                    full_input_ids=full_input_ids,
                    full_attention_mask=full_attention_mask,
                    full_labels=full_labels,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    loss_kwargs=loss_kwargs,
                )
            else:
                read_context_mask = self._build_read_context_mask(model_inputs)
                set_delta_mem_read_context_mask(model, read_context_mask)
                set_delta_mem_write_enabled(model, False)
                outputs = model(**model_inputs, **loss_kwargs)
                if not isinstance(outputs, dict):
                    outputs = {
                        "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                        "logits": outputs.logits,
                    }
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                if loss.ndim > 0:
                    loss = loss.mean()
                memory_stats["keep_loss"] = float(loss.detach().float().item())
                if has_episode_memory_inputs:
                    keep_online_state = get_delta_mem_online_state(model)
                    memory_loss, memory_stats = self._compute_memory_objective(
                        model,
                        model_inputs,
                        outputs,
                        loss,
                        loss_kwargs=loss_kwargs,
                        write_input_ids=write_input_ids,
                        write_attention_mask=write_attention_mask,
                        full_input_ids=full_input_ids,
                        full_attention_mask=full_attention_mask,
                        full_labels=full_labels,
                        write_lengths=write_lengths,
                        read_lengths=read_lengths,
                        keep_online_state=keep_online_state,
                    )
                    loss = loss + memory_loss
                    outputs["memory_loss"] = memory_loss.detach()
                    outputs["memory_full_ce_loss"] = loss.new_tensor(memory_stats["full_ce_loss"]).detach()
                    outputs["memory_keep_loss"] = loss.detach()

        partition_regularization = None
        if (
            self.memory_partition_alignment_weight > 0.0
            or self.memory_partition_entropy_weight > 0.0
            or self.memory_partition_balance_weight > 0.0
        ):
            partition_regularization = get_delta_mem_partition_regularization(model)
            self._last_memory_partition_alignment_loss = float(
                partition_regularization["alignment"].detach().float().cpu().item()
            )
            self._last_memory_partition_entropy_loss = float(
                partition_regularization["entropy"].detach().float().cpu().item()
            )
            self._last_memory_partition_balance_loss = float(
                partition_regularization["balance"].detach().float().cpu().item()
            )
        else:
            self._last_memory_partition_alignment_loss = 0.0
            self._last_memory_partition_entropy_loss = 0.0
            self._last_memory_partition_balance_loss = 0.0
        partition_route_stats = collect_delta_mem_partition_route_stats(model)
        self._last_partition_enabled_modules = partition_route_stats["enabled_modules"]
        self._last_partition_tied_read_write_modules = partition_route_stats["tied_read_write_modules"]
        self._last_partition_active_modules = partition_route_stats["active_modules"]
        self._last_partition_write_route_entropy = partition_route_stats["write_route_entropy"]
        self._last_partition_read_route_entropy = partition_route_stats["read_route_entropy"]
        self._last_partition_route_alignment_mse = partition_route_stats["route_alignment_mse"]
        self._last_partition_route_overlap = partition_route_stats["route_overlap"]
        self._last_partition_write_route_max = partition_route_stats["write_route_max"]
        self._last_partition_read_route_max = partition_route_stats["read_route_max"]
        self._last_partition_write_route_balance_l2 = partition_route_stats["write_route_balance_l2"]
        self._last_partition_read_route_balance_l2 = partition_route_stats["read_route_balance_l2"]
        self._last_memory_keep_loss = memory_stats["keep_loss"]
        self._last_memory_reset_loss = memory_stats["reset_loss"]
        self._last_memory_corrupt_loss = memory_stats["corrupt_loss"]
        self._last_memory_teacher_loss = memory_stats["teacher_loss"]
        self._last_memory_margin_loss = memory_stats["margin_loss"]
        self._last_memory_causal_loss = memory_stats["causal_loss"]
        self._last_memory_anchor_loss = memory_stats["anchor_loss"]
        self._last_memory_full_ce_loss = memory_stats["full_ce_loss"]
        self._last_memory_kl_loss = memory_stats["kl_loss"]
        self._last_memory_reset_kl_loss = memory_stats["reset_kl_loss"]
        self._last_memory_margin_gap = memory_stats["margin_gap"]
        self._last_memory_wmem = memory_stats["wmem"]
        self._last_memory_probe_keep_loss = memory_stats["probe_keep_loss"]
        self._last_memory_probe_reset_loss = memory_stats["probe_reset_loss"]
        self._last_memory_probe_margin_loss = memory_stats["probe_margin_loss"]
        self._last_memory_probe_gap = memory_stats["probe_gap"]
        self._last_memory_probe_kl_loss = memory_stats["probe_kl"]
        self._last_memory_probe_ce_loss = memory_stats["probe_ce"]
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        if partition_regularization is not None:
            loss = loss + (
                self.memory_partition_alignment_weight * partition_regularization["alignment"]
                + self.memory_partition_entropy_weight * partition_regularization["entropy"]
                + self.memory_partition_balance_weight * partition_regularization["balance"]
            )
        if self.write_sparsity_weight > 0:
            write_sparsity_loss = get_delta_mem_write_regularization(
                model,
                target=self.write_sparsity_target,
            )
            self._last_write_sparsity_loss = float(write_sparsity_loss.detach().float().cpu().item())
            loss = loss + self.write_sparsity_weight * write_sparsity_loss
            if isinstance(outputs, dict):
                outputs = dict(outputs)
                outputs["write_sparsity_loss"] = write_sparsity_loss.detach()
                if partition_regularization is not None:
                    outputs["partition_alignment_loss"] = partition_regularization["alignment"].detach()
                    outputs["partition_entropy_loss"] = partition_regularization["entropy"].detach()
                    outputs["partition_balance_loss"] = partition_regularization["balance"].detach()
        else:
            self._last_write_sparsity_loss = 0.0
            if isinstance(outputs, dict) and partition_regularization is not None:
                outputs = dict(outputs)
                outputs["partition_alignment_loss"] = partition_regularization["alignment"].detach()
                outputs["partition_entropy_loss"] = partition_regularization["entropy"].detach()
                outputs["partition_balance_loss"] = partition_regularization["balance"].detach()
        if loss.ndim > 0:
            loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        enriched_logs = dict(logs)
        if self.model is not None and getattr(self, "log_delta_debug_stats", False):
            gate_stats = collect_delta_mem_gate_stats(self.model)
            state_stats = collect_delta_mem_state_stats(self.model)
            weight_stats = collect_delta_mem_weight_stats(self.model)
            enriched_logs.update(
                {
                    "delta/beta_mean": gate_stats["beta_mean"],
                    "delta/lambda_mean": gate_stats["lambda_mean"],
                    "delta/rankwise_gate_modules": gate_stats["rankwise_gate_modules"],
                    "delta/partition_enabled_modules": self._last_partition_enabled_modules,
                    "delta/partition_tied_read_write_modules": self._last_partition_tied_read_write_modules,
                    "delta/partition_active_modules": self._last_partition_active_modules,
                    "delta/partition_write_route_entropy": self._last_partition_write_route_entropy,
                    "delta/partition_read_route_entropy": self._last_partition_read_route_entropy,
                    "delta/partition_route_alignment_mse": self._last_partition_route_alignment_mse,
                    "delta/partition_route_overlap": self._last_partition_route_overlap,
                    "delta/partition_write_route_max": self._last_partition_write_route_max,
                    "delta/partition_read_route_max": self._last_partition_read_route_max,
                    "delta/partition_write_route_balance_l2": self._last_partition_write_route_balance_l2,
                    "delta/partition_read_route_balance_l2": self._last_partition_read_route_balance_l2,
                    "delta/nonzero_state_modules": state_stats["nonzero_modules"],
                    "delta/max_state_norm": state_stats["max_state_norm"],
                    "delta/mean_state_norm": state_stats["mean_state_norm"],
                    "delta/max_state_abs": state_stats["max_state_abs"],
                    "delta/delta_o_proj_norm_sum": weight_stats["delta_o_proj_norm_sum"],
                    "delta/delta_scale_mean": (
                        weight_stats["delta_scale_mean_sum"]
                        / max(weight_stats["trainable_delta_scale_modules"], 1)
                    ),
                    "delta/write_sparsity_loss": self._last_write_sparsity_loss,
                    "delta/memory_keep_loss": self._last_memory_keep_loss,
                    "delta/memory_reset_loss": self._last_memory_reset_loss,
                    "delta/memory_corrupt_loss": self._last_memory_corrupt_loss,
                    "delta/memory_teacher_loss": self._last_memory_teacher_loss,
                    "delta/memory_margin_loss": self._last_memory_margin_loss,
                    "delta/memory_causal_loss": self._last_memory_causal_loss,
                    "delta/memory_anchor_loss": self._last_memory_anchor_loss,
                    "delta/memory_full_ce_loss": self._last_memory_full_ce_loss,
                    "delta/memory_kl_loss": self._last_memory_kl_loss,
                    "delta/memory_reset_kl_loss": self._last_memory_reset_kl_loss,
                    "delta/memory_margin_gap": self._last_memory_margin_gap,
                    "delta/memory_wmem": self._last_memory_wmem,
                    "delta/memory_probe_keep_loss": self._last_memory_probe_keep_loss,
                    "delta/memory_probe_reset_loss": self._last_memory_probe_reset_loss,
                    "delta/memory_probe_margin_loss": self._last_memory_probe_margin_loss,
                    "delta/memory_probe_gap": self._last_memory_probe_gap,
                    "delta/memory_probe_kl_loss": self._last_memory_probe_kl_loss,
                    "delta/memory_probe_ce_loss": self._last_memory_probe_ce_loss,
                    "delta/memory_partition_alignment_loss": self._last_memory_partition_alignment_loss,
                    "delta/memory_partition_entropy_loss": self._last_memory_partition_entropy_loss,
                    "delta/memory_partition_balance_loss": self._last_memory_partition_balance_loss,
                }
            )
        super().log(enriched_logs, start_time=start_time)

    def training_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._maybe_enable_static_graph(model)
        self._reset_online_state(model)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def prediction_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        self._reset_online_state(model)
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.is_world_process_zero():
            return
        if self.delta_config is None:
            raise ValueError("DeltaMemTrainer.save_model requires delta_config")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        save_delta_mem_adapter(model, output_path, self.delta_config)


def parse_args() -> argparse.Namespace:
    default_optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    parser = argparse.ArgumentParser(description="Train Delta-Mem on a Hugging Face causal LM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--tokenized-dataset-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--tokenized-dataset-root", type=Path, default=None)
    parser.add_argument(
        "--tokenized-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--ddp-backend", default="nccl")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--num-state-heads", type=int, default=1)
    parser.add_argument("--beta-bias-init", type=float, default=-1.5)
    parser.add_argument(
        "--couple-lambda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--state-update-mode",
        default="standard",
        choices=["standard", "lambda_outside", "no_lambda"],
    )
    parser.add_argument(
        "--output-init",
        default="base_slice",
        choices=["zero", "base_slice", "base_slice_fixed", "random"],
    )
    parser.add_argument("--base-slice-ref-width", type=int, default=8)
    parser.add_argument(
        "--delta-heads",
        default="q,k,v,o",
        help="Comma-separated subset of Delta attention heads to enable, e.g. q,k,o",
    )
    parser.add_argument(
        "--delta-o-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply RMSNorm to delta_o before adding it to the base attention output.",
    )
    parser.add_argument("--delta-o-rmsnorm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--trainable-delta-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn an extra bounded multiplier on top of the fixed alpha/rank scaling.",
    )
    parser.add_argument("--delta-scale-init", type=float, default=1.0)
    parser.add_argument("--delta-scale-max", type=float, default=2.0)
    parser.add_argument(
        "--delta-scale-granularity",
        default="layer",
        choices=["layer", "head"],
        help="Whether the learned delta scale is shared per layer or split per delta head.",
    )
    parser.add_argument(
        "--delta-scale-parameterization",
        default="alpha_over_rank",
        choices=["alpha_over_rank", "rank_over_alpha"],
        help="Base scaling formula used before any learned delta scale multiplier is applied.",
    )
    parser.add_argument("--online-gain", type=float, default=0.05)
    parser.add_argument(
        "--target-layers",
        default="off",
        help="Comma-separated attention layer indices to wrap with Delta-Mem. 'off' means all layers.",
    )
    parser.add_argument(
        "--memory-readout-mode",
        default="delta",
        choices=["delta"],
    )
    parser.add_argument(
        "--memory-write-source",
        default="learned_hidden",
        choices=["learned_hidden", "base_qkv"],
    )
    parser.add_argument(
        "--memory-write-granularity",
        default="token",
        choices=["token", "message_mean", "sentence_mean"],
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--training-mode",
        default="episode",
        choices=["dialogue", "episode"],
    )
    parser.add_argument(
        "--episode-recent-messages",
        type=int,
        default=4,
        help="Number of trailing non-system messages to keep visible during episode training.",
    )
    parser.add_argument(
        "--max-write-length",
        type=int,
        default=1024,
        help="Maximum number of write-history tokens kept in episode training.",
    )
    parser.add_argument("--memory-contrast-weight", type=float, default=0.1)
    parser.add_argument("--memory-kl-weight", type=float, default=0.1)
    parser.add_argument("--memory-margin", type=float, default=0.1)
    parser.add_argument("--memory-causal-weight", type=float, default=1.0)
    parser.add_argument("--memory-anchor-weight", type=float, default=1.0)
    parser.add_argument("--memory-anchor-margin", type=float, default=0.005)
    parser.add_argument("--memory-recover-weight", type=float, default=0.25)
    parser.add_argument("--memory-need-floor", type=float, default=0.15)
    parser.add_argument("--memory-dropout-state-only-prob", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optim", default=default_optim)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--deepspeed-config", type=Path, default=None)
    parser.add_argument("--write-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--write-sparsity-target", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="delta-mem-qwen3-sft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--log-delta-debug-stats", action="store_true")
    parser.add_argument(
        "--assistant-loss-mode",
        default="all_assistant_turns",
        choices=["all_assistant_turns", "final_assistant_only"],
    )
    parser.add_argument("--rankwise-gates", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.memory_loss_mode = "context_dropout_ce"
    args.num_memory_partitions = 1
    args.num_global_memory_partitions = 0
    args.memory_partition_routing = "soft"
    args.memory_partition_basis = "shared"
    args.tie_memory_partition_read_write = False
    args.memory_partition_read_mode = "softmax"
    args.memory_partition_sigmoid_gate_bias_init = -2.0
    args.slot_read_top_k = 0
    args.global_memory_mode = "shared_rw"
    args.global_memory_read_top_k = 0
    args.global_memory_merge_mode = "gated_residual"
    args.global_memory_gate_bias_init = -2.0
    args.global_memory_read_logit_bias = 0.0
    args.memory_write_proposals_per_message = 2
    args.context_ablation_mode = "mixed"
    args.context_ablation_no_state_prob = 0.2
    args.context_ablation_state_only_prob = 0.2
    args.memory_full_ce_weight = 0.0
    args.memory_full_ce_max_length = 2048
    args.memory_probe_weight = 0.0
    args.memory_probe_alpha = 0.4
    args.memory_probe_margin = 0.01
    args.memory_partition_alignment_weight = 0.0
    args.memory_partition_entropy_weight = 0.0
    args.memory_partition_balance_weight = 0.0
    args.memory_dropout_no_memory_prob = 0.0
    if args.memory_dropout_state_only_prob < 0.0 or args.memory_dropout_state_only_prob > 1.0:
        raise ValueError("memory-dropout-state-only-prob must satisfy 0 <= p <= 1")
    return args


def get_dtype(name: str) -> torch.dtype:
    table = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return table[name]


def parse_layer_indices(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if not raw or raw.lower() in {"none", "off"}:
        return ()
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def parse_delta_heads(raw: str) -> tuple[str, ...]:
    return normalize_delta_heads(raw)


def load_examples(args: argparse.Namespace) -> Dataset:
    cache_dir = str(args.hf_cache_dir) if args.hf_cache_dir is not None else None
    if args.train_file is not None:
        suffix = args.train_file.suffix.lower()
        if suffix == ".jsonl":
            dataset = load_dataset(
                "json",
                data_files=str(args.train_file),
                split="train",
                cache_dir=cache_dir,
            )
        elif suffix == ".json":
            loaded = json.loads(args.train_file.read_text())
            if isinstance(loaded, list):
                dataset = Dataset.from_list(loaded)
            elif isinstance(loaded, dict):
                dataset = Dataset.from_list([loaded])
            else:
                raise ValueError("Unsupported JSON format")
        else:
            raise ValueError(f"Unsupported train file: {args.train_file}")
        return dataset
    if args.dataset_name is not None:
        loaded = load_dataset(args.dataset_name, cache_dir=cache_dir)
        if isinstance(loaded, DatasetDict):
            return loaded[args.dataset_split]
        return loaded
    raise ValueError("Provide either --train-file or --dataset-name")


def normalize_example(example: dict) -> SFTExample:
    if "messages" in example:
        messages = example["messages"]
        if not messages:
            raise ValueError("messages examples must not be empty")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("messages examples must contain at least one assistant turn")
        return SFTExample(messages=[dict(message) for message in messages])
    if "prompt" in example and "response" in example:
        return SFTExample(
            messages=[
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ],
        )
    raise ValueError("Each example must have either `messages` or `prompt`/`response`.")


def _tokenize_chat_messages(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    tokenized = apply_project_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    return tokenized.squeeze(0).tolist()


def _tokenize_text_no_special_tokens(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _find_subsequence_start(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return 0
    max_start = len(haystack) - len(needle)
    for start in range(max_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


_MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS = 16


def _longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _chat_template_delta(
    previous_ids: list[int],
    current_ids: list[int],
    *,
    error_message: str,
) -> tuple[int, list[int]]:
    prefix_len = _longest_common_prefix_length(previous_ids, current_ids)
    rollback_tokens = len(previous_ids) - prefix_len
    if rollback_tokens > _MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS:
        raise ValueError(error_message)
    return prefix_len, current_ids[prefix_len:]


def _sentence_ids_for_message_delta(
    tokenizer,
    message_content: str,
    delta_ids: list[int],
    next_sentence_id: int,
) -> tuple[list[int], int]:
    sentence_ids = [-1] * len(delta_ids)
    content_ids = _tokenize_text_no_special_tokens(tokenizer, message_content)
    if not content_ids:
        return sentence_ids, next_sentence_id
    content_start = _find_subsequence_start(delta_ids, content_ids)
    if content_start is None:
        return sentence_ids, next_sentence_id
    sentence_chunks = split_text_into_sentence_token_chunks(message_content)
    sentence_chunk_ids = [
        _tokenize_text_no_special_tokens(tokenizer, sentence_chunk)
        for sentence_chunk in sentence_chunks
    ]
    flat_sentence_ids = [token_id for chunk_ids in sentence_chunk_ids for token_id in chunk_ids]
    if flat_sentence_ids != content_ids:
        sentence_ids[content_start : content_start + len(content_ids)] = [next_sentence_id] * len(content_ids)
        return sentence_ids, next_sentence_id + 1
    position = content_start
    for chunk_ids in sentence_chunk_ids:
        if not chunk_ids:
            continue
        sentence_ids[position : position + len(chunk_ids)] = [next_sentence_id] * len(chunk_ids)
        position += len(chunk_ids)
        next_sentence_id += 1
    return sentence_ids, next_sentence_id


def _tokenize_chat_messages_with_write_span_ids(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    include_sentence_ids: bool,
) -> tuple[list[int], list[int], list[int]]:
    input_ids: list[int] = []
    message_ids: list[int] = []
    sentence_ids: list[int] = []
    previous_ids: list[int] = []
    next_message_id = 0
    next_sentence_id = 0
    for index, message in enumerate(messages):
        current_ids = _tokenize_chat_messages(tokenizer, messages[: index + 1])
        prefix_len, delta_ids = _chat_template_delta(
            previous_ids,
            current_ids,
            error_message="Chat template tokenization is not prefix-stable; cannot recover write message spans safely.",
        )
        if prefix_len < len(input_ids):
            del input_ids[prefix_len:]
            del message_ids[prefix_len:]
            del sentence_ids[prefix_len:]
        input_ids.extend(delta_ids)
        if message["role"] == "system":
            message_ids.extend([-1] * len(delta_ids))
            sentence_ids.extend([-1] * len(delta_ids))
            previous_ids = current_ids
            continue

        message_id = next_message_id
        next_message_id += 1
        message_ids.extend([message_id] * len(delta_ids))
        if include_sentence_ids:
            message_sentence_ids, next_sentence_id = _sentence_ids_for_message_delta(
                tokenizer,
                message["content"],
                delta_ids,
                next_sentence_id,
            )
            sentence_ids.extend(message_sentence_ids)
        else:
            sentence_ids.extend([-1] * len(delta_ids))
        previous_ids = current_ids
    return input_ids, message_ids, sentence_ids


def _tokenize_chat_messages_with_message_ids(
    tokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    input_ids, message_ids, _ = _tokenize_chat_messages_with_write_span_ids(
        tokenizer,
        messages,
        include_sentence_ids=False,
    )
    return input_ids, message_ids


def _truncate_sft_sequence(
    input_ids: list[int],
    labels: list[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if len(input_ids) <= max_length:
        return input_ids, labels
    start = len(input_ids) - max_length
    return input_ids[start:], labels[start:]


def _select_supervised_assistant_indices(
    messages: list[dict[str, str]],
    assistant_loss_mode: str,
) -> list[int]:
    assistant_indices = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("No assistant turns found for supervision")
    if assistant_loss_mode == "all_assistant_turns":
        return assistant_indices
    if assistant_loss_mode == "final_assistant_only":
        return [assistant_indices[-1]]
    raise ValueError(f"Unsupported assistant_loss_mode: {assistant_loss_mode}")


def _split_system_prefix(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    system_prefix: list[dict[str, str]] = []
    for message in messages:
        if message["role"] != "system":
            break
        system_prefix.append(dict(message))
    return system_prefix, [dict(message) for message in messages[len(system_prefix) :]]


def tokenize_messages_for_sft(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    *,
    assistant_loss_mode: str,
) -> dict:
    supervised_assistant_indices = set(
        _select_supervised_assistant_indices(messages, assistant_loss_mode)
    )

    input_ids: list[int] = []
    labels: list[int] = []
    previous_ids: list[int] = []
    for index in range(len(messages)):
        current_ids = _tokenize_chat_messages(tokenizer, messages[: index + 1])
        prefix_len, delta_ids = _chat_template_delta(
            previous_ids,
            current_ids,
            error_message="Chat template tokenization is not prefix-stable; cannot build assistant-span labels safely.",
        )
        if prefix_len < len(input_ids):
            del input_ids[prefix_len:]
            del labels[prefix_len:]
        input_ids.extend(delta_ids)
        if index in supervised_assistant_indices:
            labels.extend(delta_ids)
        else:
            labels.extend([-100] * len(delta_ids))
        previous_ids = current_ids

    input_ids, labels = _truncate_sft_sequence(input_ids, labels, max_length)
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_episode_training_examples(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    *,
    assistant_loss_mode: str,
    episode_recent_messages: int,
    max_write_length: int,
    include_sentence_ids: bool,
) -> list[dict]:
    if episode_recent_messages < 0:
        raise ValueError("episode_recent_messages must be >= 0")
    if max_write_length <= 0:
        raise ValueError("max_write_length must be > 0")

    episodes: list[dict] = []
    for target_index in _select_supervised_assistant_indices(messages, assistant_loss_mode):
        prefix_messages = [dict(message) for message in messages[:target_index]]
        system_prefix, non_system_prefix = _split_system_prefix(prefix_messages)
        if episode_recent_messages == 0:
            visible_non_system = []
            write_non_system = non_system_prefix
        else:
            visible_non_system = non_system_prefix[-episode_recent_messages:]
            write_non_system = non_system_prefix[:-episode_recent_messages]

        write_messages = system_prefix + write_non_system if write_non_system else []
        write_input_ids: list[int] = []
        write_message_ids: list[int] = []
        write_sentence_ids: list[int] = []
        if write_messages:
            full_write_input_ids, write_message_ids, write_sentence_ids = _tokenize_chat_messages_with_write_span_ids(
                tokenizer,
                write_messages,
                include_sentence_ids=include_sentence_ids,
            )
            write_input_ids, write_message_ids = _truncate_sft_sequence(
                full_write_input_ids,
                write_message_ids,
                max_write_length,
            )
            _, write_sentence_ids = _truncate_sft_sequence(
                full_write_input_ids,
                write_sentence_ids,
                max_write_length,
            )

        read_messages = system_prefix + visible_non_system + [dict(messages[target_index])]
        read_features = tokenize_messages_for_sft(
            tokenizer,
            read_messages,
            max_length,
            assistant_loss_mode="final_assistant_only",
        )
        # Keep the immediate query turn visible during state-only dropout so memory focuses on
        # far-history recall instead of reconstructing the local prompt from state.
        state_only_visible_non_system = non_system_prefix[-1:] if non_system_prefix else []
        state_only_write_non_system = non_system_prefix[:-1] if non_system_prefix else []
        state_only_write_messages = (
            system_prefix + state_only_write_non_system if state_only_write_non_system else []
        )
        state_only_write_input_ids: list[int] = []
        state_only_write_message_ids: list[int] = []
        state_only_write_sentence_ids: list[int] = []
        if state_only_write_messages:
            (
                full_state_only_write_input_ids,
                state_only_write_message_ids,
                state_only_write_sentence_ids,
            ) = _tokenize_chat_messages_with_write_span_ids(
                tokenizer,
                state_only_write_messages,
                include_sentence_ids=include_sentence_ids,
            )
            state_only_write_input_ids, state_only_write_message_ids = _truncate_sft_sequence(
                full_state_only_write_input_ids,
                state_only_write_message_ids,
                max_write_length,
            )
            _, state_only_write_sentence_ids = _truncate_sft_sequence(
                full_state_only_write_input_ids,
                state_only_write_sentence_ids,
                max_write_length,
            )
        state_only_read_messages = (
            system_prefix + state_only_visible_non_system + [dict(messages[target_index])]
        )
        state_only_read_features = tokenize_messages_for_sft(
            tokenizer,
            state_only_read_messages,
            max_length,
            assistant_loss_mode="final_assistant_only",
        )

        episodes.append(
            {
                "write_input_ids": write_input_ids,
                "write_attention_mask": [1] * len(write_input_ids),
                "write_message_ids": write_message_ids,
                "write_sentence_ids": write_sentence_ids,
                "input_ids": read_features["input_ids"],
                "attention_mask": read_features["attention_mask"],
                "labels": read_features["labels"],
                "state_only_write_input_ids": state_only_write_input_ids,
                "state_only_write_attention_mask": [1] * len(state_only_write_input_ids),
                "state_only_write_message_ids": state_only_write_message_ids,
                "state_only_write_sentence_ids": state_only_write_sentence_ids,
                "state_only_input_ids": state_only_read_features["input_ids"],
                "state_only_attention_mask": state_only_read_features["attention_mask"],
                "state_only_labels": state_only_read_features["labels"],
                "episode_target_message_index": target_index,
                "write_message_count": len(write_messages),
                "visible_message_count": len(read_messages) - 1,
            }
        )
    return episodes


def tokenize_example(
    tokenizer,
    example: dict,
    max_length: int,
    *,
    assistant_loss_mode: str,
    training_mode: str,
    episode_recent_messages: int,
    max_write_length: int,
) -> dict:
    normalized = normalize_example(example)
    if training_mode == "episode":
        raise ValueError("tokenize_example does not support episode mode; use tokenize_examples_batch")
    if training_mode != "dialogue":
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    return tokenize_messages_for_sft(
        tokenizer,
        normalized.messages,
        max_length,
        assistant_loss_mode=assistant_loss_mode,
    )


def add_length_column(example: dict) -> dict[str, int]:
    total_length = len(example["input_ids"])
    if "write_input_ids" in example:
        total_length += len(example["write_input_ids"])
    return {"length": total_length}


def tokenize_examples_batch(
    tokenizer,
    batch: dict[str, list],
    max_length: int,
    *,
    assistant_loss_mode: str,
    training_mode: str,
    episode_recent_messages: int,
    max_write_length: int,
    include_sentence_ids: bool,
) -> dict[str, list]:
    tokenized: dict[str, list] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    if training_mode == "episode":
        tokenized["write_input_ids"] = []
        tokenized["write_attention_mask"] = []
        tokenized["write_message_ids"] = []
        tokenized["write_sentence_ids"] = []
        tokenized["state_only_write_input_ids"] = []
        tokenized["state_only_write_attention_mask"] = []
        tokenized["state_only_write_message_ids"] = []
        tokenized["state_only_write_sentence_ids"] = []
        tokenized["state_only_input_ids"] = []
        tokenized["state_only_attention_mask"] = []
        tokenized["state_only_labels"] = []
        tokenized["episode_target_message_index"] = []
        tokenized["write_message_count"] = []
        tokenized["visible_message_count"] = []

    batch_size = len(next(iter(batch.values())))
    for row_index in range(batch_size):
        example = {key: value[row_index] for key, value in batch.items()}
        normalized = normalize_example(example)
        if training_mode == "dialogue":
            features = tokenize_messages_for_sft(
                tokenizer,
                normalized.messages,
                max_length,
                assistant_loss_mode=assistant_loss_mode,
            )
            for key, value in features.items():
                tokenized[key].append(value)
            continue
        if training_mode != "episode":
            raise ValueError(f"Unsupported training_mode: {training_mode}")
        for episode in build_episode_training_examples(
            tokenizer,
            normalized.messages,
            max_length,
            assistant_loss_mode=assistant_loss_mode,
            episode_recent_messages=episode_recent_messages,
            max_write_length=max_write_length,
            include_sentence_ids=include_sentence_ids,
        ):
            for key, value in episode.items():
                tokenized[key].append(value)
    return tokenized


def _tokenized_dataset_cache_key(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
) -> str:
    code_hash = hashlib.sha256()
    for fn in (
        normalize_example,
        tokenize_messages_for_sft,
        _sentence_ids_for_message_delta,
        _tokenize_chat_messages_with_write_span_ids,
        build_episode_training_examples,
        tokenize_examples_batch,
        add_length_column,
        split_text_into_sentence_token_chunks,
    ):
        code_hash.update(inspect.getsource(fn).encode("utf-8"))
    include_sentence_ids = args.memory_write_granularity == "sentence_mean"
    payload = {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "train_file": None if args.train_file is None else str(args.train_file.resolve()),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", ""),
        "tokenizer_class": tokenizer.__class__.__name__,
        "max_length": args.max_length,
        "training_mode": args.training_mode,
        "assistant_loss_mode": args.assistant_loss_mode,
        "episode_recent_messages": args.episode_recent_messages,
        "max_write_length": args.max_write_length,
        "memory_write_granularity": args.memory_write_granularity,
        "include_sentence_ids": include_sentence_ids,
        "group_by_length": args.group_by_length,
        "code_hash": code_hash.hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _build_tokenized_dataset(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
) -> Dataset:
    if args.training_mode == "dialogue":
        tokenized = dataset.map(
            lambda example: tokenize_example(
                tokenizer,
                example,
                args.max_length,
                assistant_loss_mode=args.assistant_loss_mode,
                training_mode=args.training_mode,
                episode_recent_messages=args.episode_recent_messages,
                max_write_length=args.max_write_length,
            ),
            remove_columns=dataset.column_names,
            num_proc=None if args.dataset_num_proc <= 1 else args.dataset_num_proc,
        )
    elif args.training_mode == "episode":
        tokenized = dataset.map(
            lambda batch: tokenize_examples_batch(
                tokenizer,
                batch,
                args.max_length,
                assistant_loss_mode=args.assistant_loss_mode,
                training_mode=args.training_mode,
                episode_recent_messages=args.episode_recent_messages,
                max_write_length=args.max_write_length,
                include_sentence_ids=args.memory_write_granularity == "sentence_mean",
            ),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=None if args.dataset_num_proc <= 1 else args.dataset_num_proc,
        )
    else:
        raise ValueError(f"Unsupported training_mode: {args.training_mode}")
    if args.group_by_length:
        tokenized = tokenized.map(
            add_length_column,
            num_proc=None if args.dataset_num_proc <= 1 else args.dataset_num_proc,
        )
    return tokenized


def prepare_tokenized_dataset(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[Dataset, bool, Path | None]:
    if not args.tokenized_cache:
        return _build_tokenized_dataset(args, dataset, tokenizer), False, None
    if args.tokenized_dataset_root is None:
        raise ValueError("--tokenized-dataset-root is required when --tokenized-cache is enabled")

    args.tokenized_dataset_root.mkdir(parents=True, exist_ok=True)
    cache_key = _tokenized_dataset_cache_key(args, dataset, tokenizer)
    cache_dir = args.tokenized_dataset_root / cache_key
    ready_marker = cache_dir / "_READY"
    lock_dir = args.tokenized_dataset_root / f".{cache_key}.lock"
    is_builder = (not distributed) or local_rank in (-1, 0)

    if ready_marker.exists():
        return load_from_disk(str(cache_dir)), True, cache_dir

    if is_builder:
        while True:
            try:
                lock_dir.mkdir()
                break
            except FileExistsError:
                if ready_marker.exists():
                    return load_from_disk(str(cache_dir)), True, cache_dir
                time.sleep(2)

        try:
            if cache_dir.exists() and not ready_marker.exists():
                shutil.rmtree(cache_dir)
            temp_dir = args.tokenized_dataset_root / f".{cache_key}.tmp-{os.getpid()}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            tokenized = _build_tokenized_dataset(args, dataset, tokenizer)
            tokenized.save_to_disk(str(temp_dir))
            temp_dir.rename(cache_dir)
            ready_marker.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "created_at": time.time(),
                        "training_mode": args.training_mode,
                        "group_by_length": args.group_by_length,
                        "assistant_loss_mode": args.assistant_loss_mode,
                        "episode_recent_messages": args.episode_recent_messages,
                        "max_write_length": args.max_write_length,
                        "memory_write_granularity": args.memory_write_granularity,
                        "include_sentence_ids": args.memory_write_granularity == "sentence_mean",
                        "max_length": args.max_length,
                    },
                    indent=2,
                )
            )
            return tokenized, False, cache_dir
        finally:
            if lock_dir.exists():
                lock_dir.rmdir()

    waited = 0
    while not ready_marker.exists():
        time.sleep(2)
        waited += 2
        if waited > 7200:
            raise TimeoutError(f"Timed out waiting for tokenized dataset cache at {cache_dir}")
    return load_from_disk(str(cache_dir)), True, cache_dir


def detect_training_mode(tokenized: Dataset) -> str:
    if "write_input_ids" in tokenized.column_names:
        return "episode"
    return "dialogue"


def load_or_prepare_tokenized_dataset(
    args: argparse.Namespace,
    tokenizer,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[Dataset, dict[str, object]]:
    if args.tokenized_dataset_dir is not None:
        tokenized = load_from_disk(str(args.tokenized_dataset_dir))
        return tokenized, {
            "tokenized_cache_hit": True,
            "tokenized_cache_dir": str(args.tokenized_dataset_dir),
            "tokenized_dataset_source": "load_from_disk",
            "train_samples": len(tokenized),
            "training_mode": detect_training_mode(tokenized),
        }

    dataset = load_examples(args)
    tokenized, tokenized_cache_hit, tokenized_cache_dir = prepare_tokenized_dataset(
        args,
        dataset,
        tokenizer,
        distributed=distributed,
        local_rank=local_rank,
    )
    return tokenized, {
        "tokenized_cache_hit": tokenized_cache_hit,
        "tokenized_cache_dir": None if tokenized_cache_dir is None else str(tokenized_cache_dir),
        "tokenized_dataset_source": "prepared_cache" if args.tokenized_cache else "direct_map",
        "train_samples": len(tokenized),
        "training_mode": detect_training_mode(tokenized),
    }


class DialogueCausalLMCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class EpisodeCausalLMCollator(DialogueCausalLMCollator):
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        pad_token_id = self.tokenizer.pad_token_id

        def _pad_sequences(values: list[list[int]], pad_value: int) -> torch.Tensor | None:
            max_len = max(len(value) for value in values)
            if max_len == 0:
                return None
            padded = [value + [pad_value] * (max_len - len(value)) for value in values]
            return torch.tensor(padded, dtype=torch.long)

        write_lengths = [len(feature["write_input_ids"]) for feature in features]
        read_lengths = [len(feature["input_ids"]) for feature in features]
        write_input_ids = _pad_sequences(
            [feature["write_input_ids"] for feature in features],
            pad_token_id,
        )
        write_attention_mask = _pad_sequences(
            [feature["write_attention_mask"] for feature in features],
            0,
        )
        write_message_ids = _pad_sequences(
            [feature["write_message_ids"] for feature in features],
            -1,
        )
        write_sentence_ids = _pad_sequences(
            [feature["write_sentence_ids"] for feature in features],
            -1,
        )
        if (
            write_input_ids is not None
            and write_attention_mask is not None
            and write_message_ids is not None
            and write_sentence_ids is not None
        ):
            batch["write_input_ids"] = write_input_ids
            batch["write_attention_mask"] = write_attention_mask
            batch["write_message_ids"] = write_message_ids
            batch["write_sentence_ids"] = write_sentence_ids
        batch["write_lengths"] = torch.tensor(write_lengths, dtype=torch.long)
        batch["read_lengths"] = torch.tensor(read_lengths, dtype=torch.long)

        state_only_write_input_ids = _pad_sequences(
            [feature["state_only_write_input_ids"] for feature in features],
            pad_token_id,
        )
        state_only_write_attention_mask = _pad_sequences(
            [feature["state_only_write_attention_mask"] for feature in features],
            0,
        )
        state_only_write_message_ids = _pad_sequences(
            [feature["state_only_write_message_ids"] for feature in features],
            -1,
        )
        state_only_write_sentence_ids = _pad_sequences(
            [feature["state_only_write_sentence_ids"] for feature in features],
            -1,
        )
        state_only_input_ids = _pad_sequences(
            [feature["state_only_input_ids"] for feature in features],
            pad_token_id,
        )
        state_only_attention_mask = _pad_sequences(
            [feature["state_only_attention_mask"] for feature in features],
            0,
        )
        state_only_labels = _pad_sequences(
            [feature["state_only_labels"] for feature in features],
            -100,
        )
        if (
            state_only_write_input_ids is not None
            and state_only_write_attention_mask is not None
            and state_only_write_message_ids is not None
            and state_only_write_sentence_ids is not None
            and state_only_input_ids is not None
            and state_only_attention_mask is not None
            and state_only_labels is not None
        ):
            batch["state_only_write_input_ids"] = state_only_write_input_ids
            batch["state_only_write_attention_mask"] = state_only_write_attention_mask
            batch["state_only_write_message_ids"] = state_only_write_message_ids
            batch["state_only_write_sentence_ids"] = state_only_write_sentence_ids
            batch["state_only_input_ids"] = state_only_input_ids
            batch["state_only_attention_mask"] = state_only_attention_mask
            batch["state_only_labels"] = state_only_labels

        max_full_len = max(
            len(feature["write_input_ids"]) + len(feature["input_ids"]) for feature in features
        )
        full_input_ids = []
        full_attention_mask = []
        full_labels = []
        for feature in features:
            combined_input_ids = feature["write_input_ids"] + feature["input_ids"]
            combined_attention_mask = (
                feature["write_attention_mask"] + feature["attention_mask"]
            )
            combined_labels = ([-100] * len(feature["write_input_ids"])) + feature["labels"]
            pad_len = max_full_len - len(combined_input_ids)
            full_input_ids.append(combined_input_ids + [pad_token_id] * pad_len)
            full_attention_mask.append(combined_attention_mask + [0] * pad_len)
            full_labels.append(combined_labels + [-100] * pad_len)
        batch["full_input_ids"] = torch.tensor(full_input_ids, dtype=torch.long)
        batch["full_attention_mask"] = torch.tensor(full_attention_mask, dtype=torch.long)
        batch["full_labels"] = torch.tensor(full_labels, dtype=torch.long)
        return batch


def build_data_collator(training_mode: str, tokenizer):
    if training_mode == "episode":
        return EpisodeCausalLMCollator(tokenizer)
    if training_mode == "dialogue":
        return DialogueCausalLMCollator(tokenizer)
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def main() -> None:
    args = parse_args()
    if args.gradient_checkpointing:
        raise ValueError(
            "Gradient checkpointing is currently incompatible with Delta-Mem's stateful token updates. "
            "Disable --gradient-checkpointing before training."
        )
    dtype = get_dtype(args.dtype)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized, tokenized_meta = load_or_prepare_tokenized_dataset(
        args,
        tokenizer,
        distributed=distributed,
        local_rank=local_rank,
    )
    effective_training_mode = str(tokenized_meta["training_mode"])
    effective_group_by_length = args.group_by_length and effective_training_mode != "episode"
    if args.group_by_length and not effective_group_by_length and local_rank in (-1, 0):
        print(
            "Disabling group_by_length for episode training because startup sorting is prohibitively slow at episode scale."
        )

    suppress_non_actionable_accelerate_warnings()
    resolved_attn_implementation = resolve_attn_implementation(
        args.model_path,
        args.attn_implementation,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=resolved_attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    delta_config = HFDeltaMemConfig(
        rank=args.rank,
        alpha=args.alpha,
        num_state_heads=args.num_state_heads,
        num_memory_partitions=args.num_memory_partitions,
        num_global_memory_partitions=args.num_global_memory_partitions,
        memory_partition_routing=args.memory_partition_routing,
        memory_partition_basis=args.memory_partition_basis,
        tie_memory_partition_read_write=args.tie_memory_partition_read_write,
        memory_partition_read_mode=args.memory_partition_read_mode,
        memory_partition_sigmoid_gate_bias_init=args.memory_partition_sigmoid_gate_bias_init,
        slot_read_top_k=args.slot_read_top_k,
        global_memory_mode=args.global_memory_mode,
        global_memory_read_top_k=args.global_memory_read_top_k,
        global_memory_merge_mode=args.global_memory_merge_mode,
        global_memory_gate_bias_init=args.global_memory_gate_bias_init,
        global_memory_read_logit_bias=args.global_memory_read_logit_bias,
        beta_bias_init=args.beta_bias_init,
        couple_lambda=args.couple_lambda,
        state_update_mode=normalize_state_update_mode(args.state_update_mode),
        rankwise_gates=args.rankwise_gates,
        output_init=args.output_init,
        base_slice_ref_width=args.base_slice_ref_width,
        delta_heads=parse_delta_heads(args.delta_heads),
        trainable_delta_scale=args.trainable_delta_scale,
        delta_scale_init=args.delta_scale_init,
        delta_scale_max=args.delta_scale_max,
        delta_scale_granularity=args.delta_scale_granularity,
        delta_scale_parameterization=args.delta_scale_parameterization,
        delta_o_rmsnorm=args.delta_o_rmsnorm,
        delta_o_rmsnorm_eps=args.delta_o_rmsnorm_eps,
        online_gain=args.online_gain,
        target_layers=parse_layer_indices(args.target_layers),
        memory_readout_mode=normalize_memory_readout_mode(args.memory_readout_mode),
        memory_write_source=args.memory_write_source,
        memory_write_granularity=args.memory_write_granularity,
        memory_write_proposals_per_message=args.memory_write_proposals_per_message,
    )
    replaced = attach_delta_mem(model, delta_config)
    trainable_names = freeze_non_delta_mem_params(model)

    warmup_steps = compute_warmup_steps(
        train_samples=len(tokenized),
        per_device_train_batch_size=args.per_device_train_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
    )
    training_args_kwargs = dict(
        output_dir=str(args.output_dir / "trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.data_seed,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        length_column_name="length",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=args.torch_compile,
        tf32=args.tf32,
        deepspeed=None if args.deepspeed_config is None else str(args.deepspeed_config),
        ddp_find_unused_parameters=False,
        ddp_backend=args.ddp_backend if distributed else None,
        local_rank=local_rank if distributed else -1,
        bf16=args.bf16 or args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        report_to=["wandb"] if args.wandb else ["none"],
        run_name=(args.wandb_run_name or args.wandb_project) if args.wandb else None,
        remove_unused_columns=False,
    )
    if "group_by_length" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["group_by_length"] = effective_group_by_length
    training_args = TrainingArguments(**training_args_kwargs)

    if args.wandb:
        if args.wandb_dir is None:
            raise ValueError("--wandb-dir is required when --wandb is enabled")
        args.wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_DIR"] = str(args.wandb_dir)
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = args.wandb_tags
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
    trainer = DeltaMemTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=build_data_collator(effective_training_mode, tokenizer),
        delta_config=delta_config,
        write_sparsity_weight=args.write_sparsity_weight,
        write_sparsity_target=args.write_sparsity_target,
        memory_loss_mode="context_dropout_ce",
        memory_contrast_weight=args.memory_contrast_weight,
        memory_kl_weight=args.memory_kl_weight,
        memory_margin=args.memory_margin,
        memory_causal_weight=args.memory_causal_weight,
        memory_anchor_weight=args.memory_anchor_weight,
        memory_anchor_margin=args.memory_anchor_margin,
        memory_full_ce_weight=args.memory_full_ce_weight,
        memory_full_ce_max_length=args.memory_full_ce_max_length,
        memory_recover_weight=args.memory_recover_weight,
        memory_need_floor=args.memory_need_floor,
        memory_probe_weight=args.memory_probe_weight,
        memory_probe_alpha=args.memory_probe_alpha,
        memory_probe_margin=args.memory_probe_margin,
        memory_partition_alignment_weight=args.memory_partition_alignment_weight,
        memory_partition_entropy_weight=args.memory_partition_entropy_weight,
        memory_partition_balance_weight=args.memory_partition_balance_weight,
        memory_dropout_no_memory_prob=args.memory_dropout_no_memory_prob,
        memory_dropout_state_only_prob=args.memory_dropout_state_only_prob,
        context_ablation_mode=args.context_ablation_mode,
        context_ablation_no_state_prob=args.context_ablation_no_state_prob,
        context_ablation_state_only_prob=args.context_ablation_state_only_prob,
    )
    trainer.log_delta_debug_stats = args.log_delta_debug_stats
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    base_model = trainer.accelerator.unwrap_model(trainer.model)
    if trainer.is_world_process_zero():
        save_delta_mem_adapter(base_model, args.output_dir, delta_config)
        summary = {
            "output_dir": str(args.output_dir),
            "num_replaced_modules": len(replaced),
            "num_trainable_tensors": len(trainable_names),
            "first_replaced_modules": replaced[:8],
            "first_trainable_tensors": trainable_names[:8],
            "train_samples": len(tokenized),
            "training_mode": effective_training_mode,
            "assistant_loss_mode": args.assistant_loss_mode,
            "episode_recent_messages": args.episode_recent_messages,
            "max_write_length": args.max_write_length,
            "memory_loss_mode": "context_dropout_ce",
            "memory_write_source": args.memory_write_source,
            "memory_write_granularity": args.memory_write_granularity,
            "delta_o_rmsnorm": args.delta_o_rmsnorm,
            "delta_o_rmsnorm_eps": args.delta_o_rmsnorm_eps,
            "target_layers": args.target_layers,
            "memory_contrast_weight": args.memory_contrast_weight,
            "memory_kl_weight": args.memory_kl_weight,
            "memory_margin": args.memory_margin,
            "memory_causal_weight": args.memory_causal_weight,
            "memory_anchor_weight": args.memory_anchor_weight,
            "memory_anchor_margin": args.memory_anchor_margin,
            "memory_recover_weight": args.memory_recover_weight,
            "memory_need_floor": args.memory_need_floor,
            "memory_dropout_no_memory_prob": args.memory_dropout_no_memory_prob,
            "memory_dropout_state_only_prob": args.memory_dropout_state_only_prob,
            "context_ablation_mode": args.context_ablation_mode,
            "context_ablation_no_state_prob": args.context_ablation_no_state_prob,
            "context_ablation_state_only_prob": args.context_ablation_state_only_prob,
            "memory_full_ce_weight": args.memory_full_ce_weight,
            "memory_full_ce_max_length": args.memory_full_ce_max_length,
            "output_init": args.output_init,
            "base_slice_ref_width": args.base_slice_ref_width,
            "memory_readout_mode": args.memory_readout_mode,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "optim": args.optim,
            "requested_attn_implementation": args.attn_implementation,
            "attn_implementation": resolved_attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "torch_compile": args.torch_compile,
            "tf32": args.tf32,
            "group_by_length": effective_group_by_length,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataset_num_proc": args.dataset_num_proc,
            "hf_cache_dir": str(args.hf_cache_dir),
            "tokenized_dataset_dir": None if args.tokenized_dataset_dir is None else str(args.tokenized_dataset_dir),
            "tokenized_dataset_root": str(args.tokenized_dataset_root),
            "tokenized_cache": args.tokenized_cache,
            "tokenized_cache_hit": tokenized_meta["tokenized_cache_hit"],
            "tokenized_cache_dir": tokenized_meta["tokenized_cache_dir"],
            "tokenized_dataset_source": tokenized_meta["tokenized_dataset_source"],
            "ddp_backend": args.ddp_backend if distributed else None,
            "local_rank": local_rank if distributed else -1,
            "world_size": world_size,
            "deepspeed_config": None if args.deepspeed_config is None else str(args.deepspeed_config),
            "write_sparsity_weight": args.write_sparsity_weight,
            "write_sparsity_target": args.write_sparsity_target,
            "wandb_project": args.wandb_project if args.wandb else None,
            "wandb_entity": args.wandb_entity if args.wandb else None,
            "wandb_run_name": args.wandb_run_name if args.wandb else None,
            "wandb_group": args.wandb_group if args.wandb else None,
            "wandb_tags": args.wandb_tags if args.wandb else None,
            "wandb_mode": args.wandb_mode if args.wandb else None,
            "wandb_dir": str(args.wandb_dir) if args.wandb else None,
            "gate_stats": collect_delta_mem_gate_stats(base_model),
            "config": asdict(delta_config),
        }
        (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


def _destroy_process_group_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        _destroy_process_group_if_needed()
