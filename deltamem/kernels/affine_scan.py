from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised through the fallback path.
    triton = None
    tl = None


@dataclass(frozen=True)
class TritonScanSupport:
    supported: bool
    reason: str = ""


def _next_power_of_2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def triton_scan_support(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
) -> TritonScanSupport:
    if triton is None or tl is None:
        return TritonScanSupport(False, "triton is unavailable")
    if not state.is_cuda:
        return TritonScanSupport(False, "state tensor is not on CUDA")
    if any(tensor.device != state.device for tensor in (q, k, v, keep, erase, write)):
        return TritonScanSupport(False, "all tensors must be on the same CUDA device")
    if state.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return TritonScanSupport(False, f"unsupported dtype {state.dtype}")
    if (
        state.ndim != 3
        or q.ndim != 3
        or k.ndim != 3
        or v.ndim != 3
        or keep.ndim != 3
        or erase.ndim != 3
        or write.ndim != 3
    ):
        return TritonScanSupport(False, "expected 3D tensors")
    batch_size, rank, rank2 = state.shape
    if rank != rank2:
        return TritonScanSupport(False, "state must be square in the last two dims")
    if q.size(0) != batch_size or q.size(2) != rank:
        return TritonScanSupport(False, "q shape does not match state rank")
    if k.shape != q.shape or v.shape != q.shape or keep.shape != q.shape or erase.shape != q.shape or write.shape != q.shape:
        return TritonScanSupport(False, "q/k/v/keep/erase/write must share shape [batch, seq, rank]")
    return TritonScanSupport(True)


if triton is not None and tl is not None:

    @triton.jit
    def _affine_scan_forward_kernel(
    state0_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    keep_ptr,
    erase_ptr,
    write_ptr,
    mask_ptr,
    state_hist_ptr,
    reads_ptr,
    state_out_ptr,
    batch_size,
    seq_len,
    rank,
    state_b_stride,
    state_row_stride,
    state_col_stride,
    q_b_stride,
    q_t_stride,
    q_r_stride,
    scalar_b_stride,
    scalar_t_stride,
    scalar_r_stride,
    mask_b_stride,
    mask_t_stride,
    hist_b_stride,
    hist_t_stride,
    hist_row_stride,
    hist_col_stride,
    reads_b_stride,
    reads_t_stride,
    reads_r_stride,
    HAS_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx = pid // rank
        row_idx = pid % rank
        if batch_idx >= batch_size:
            return

        cols = tl.arange(0, BLOCK_R)
        col_mask = cols < rank

        state_offsets = (
            batch_idx * state_b_stride
            + row_idx * state_row_stride
            + cols * state_col_stride
        )
        state_vec = tl.load(state0_ptr + state_offsets, mask=col_mask, other=0.0).to(tl.float32)

        t = 0
        while t < seq_len:
            hist_offsets = (
                batch_idx * hist_b_stride
                + t * hist_t_stride
                + row_idx * hist_row_stride
                + cols * hist_col_stride
            )
            tl.store(state_hist_ptr + hist_offsets, state_vec, mask=col_mask)

            valid = tl.full((), 1, tl.int32)
            if HAS_MASK:
                valid = tl.load(mask_ptr + batch_idx * mask_b_stride + t * mask_t_stride)
            valid_f = valid.to(tl.float32)

            q_offsets = batch_idx * q_b_stride + t * q_t_stride + cols * q_r_stride
            q_vec = tl.load(q_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
            read = tl.sum(state_vec * q_vec, axis=0) * valid_f
            tl.store(
                reads_ptr + batch_idx * reads_b_stride + t * reads_t_stride + row_idx * reads_r_stride,
                read,
            )

            k_vec = tl.load(k_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
            scalar_offset = batch_idx * scalar_b_stride + t * scalar_t_stride + row_idx * scalar_r_stride
            v_scalar = tl.load(v_ptr + scalar_offset).to(tl.float32)
            keep_scalar = tl.load(keep_ptr + scalar_offset).to(tl.float32)
            erase_scalar = tl.load(erase_ptr + scalar_offset).to(tl.float32)
            write_scalar = tl.load(write_ptr + scalar_offset).to(tl.float32)
            dot = tl.sum(state_vec * k_vec, axis=0)
            updated = (
                keep_scalar * state_vec
                - erase_scalar * dot * k_vec
                + write_scalar * v_scalar * k_vec
            )
            state_vec = tl.where(valid != 0, updated, state_vec)
            t += 1

        tl.store(state_out_ptr + state_offsets, state_vec, mask=col_mask)


    @triton.jit
    def _affine_scan_backward_kernel(
    state_hist_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    keep_ptr,
    erase_ptr,
    write_ptr,
    mask_ptr,
    grad_reads_ptr,
    grad_state_out_ptr,
    grad_state0_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    grad_keep_ptr,
    grad_erase_ptr,
    grad_write_ptr,
    batch_size,
    seq_len,
    rank,
    hist_b_stride,
    hist_t_stride,
    hist_row_stride,
    hist_col_stride,
    q_b_stride,
    q_t_stride,
    q_r_stride,
    scalar_b_stride,
    scalar_t_stride,
    scalar_r_stride,
    mask_b_stride,
    mask_t_stride,
    grad_reads_b_stride,
    grad_reads_t_stride,
    grad_reads_r_stride,
    grad_state_b_stride,
    grad_state_row_stride,
    grad_state_col_stride,
    grad_q_b_stride,
    grad_q_t_stride,
    grad_q_r_stride,
    HAS_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx = pid // rank
        row_idx = pid % rank
        if batch_idx >= batch_size:
            return

        cols = tl.arange(0, BLOCK_R)
        col_mask = cols < rank

        grad_state_offsets = (
            batch_idx * grad_state_b_stride
            + row_idx * grad_state_row_stride
            + cols * grad_state_col_stride
        )
        grad_state = tl.load(grad_state_out_ptr + grad_state_offsets, mask=col_mask, other=0.0).to(tl.float32)

        t = seq_len - 1
        while t >= 0:
            valid = tl.full((), 1, tl.int32)
            if HAS_MASK:
                valid = tl.load(mask_ptr + batch_idx * mask_b_stride + t * mask_t_stride)

            if valid != 0:
                hist_offsets = (
                    batch_idx * hist_b_stride
                    + t * hist_t_stride
                    + row_idx * hist_row_stride
                    + cols * hist_col_stride
                )
                state_prev = tl.load(state_hist_ptr + hist_offsets, mask=col_mask, other=0.0).to(tl.float32)

                q_offsets = batch_idx * q_b_stride + t * q_t_stride + cols * q_r_stride
                q_vec = tl.load(q_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
                k_vec = tl.load(k_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)

                scalar_offset = batch_idx * scalar_b_stride + t * scalar_t_stride + row_idx * scalar_r_stride
                v_scalar = tl.load(v_ptr + scalar_offset).to(tl.float32)
                keep_scalar = tl.load(keep_ptr + scalar_offset).to(tl.float32)
                erase_scalar = tl.load(erase_ptr + scalar_offset).to(tl.float32)
                write_scalar = tl.load(write_ptr + scalar_offset).to(tl.float32)
                grad_read = tl.load(
                    grad_reads_ptr
                    + batch_idx * grad_reads_b_stride
                    + t * grad_reads_t_stride
                    + row_idx * grad_reads_r_stride
                ).to(tl.float32)

                tl.atomic_add(
                    grad_q_ptr + batch_idx * grad_q_b_stride + t * grad_q_t_stride + cols * grad_q_r_stride,
                    grad_read * state_prev,
                    mask=col_mask,
                )

                grad_state_dot_k = tl.sum(grad_state * k_vec, axis=0)
                state_prev_dot_k = tl.sum(state_prev * k_vec, axis=0)

                tl.atomic_add(
                    grad_k_ptr + batch_idx * grad_q_b_stride + t * grad_q_t_stride + cols * grad_q_r_stride,
                    (write_scalar * v_scalar - erase_scalar * state_prev_dot_k) * grad_state
                    - erase_scalar * grad_state_dot_k * state_prev,
                    mask=col_mask,
                )
                tl.store(grad_v_ptr + scalar_offset, write_scalar * grad_state_dot_k)
                tl.store(grad_keep_ptr + scalar_offset, tl.sum(grad_state * state_prev, axis=0))
                tl.store(grad_erase_ptr + scalar_offset, -state_prev_dot_k * grad_state_dot_k)
                tl.store(grad_write_ptr + scalar_offset, v_scalar * grad_state_dot_k)

                grad_state = (
                    grad_read * q_vec
                    + keep_scalar * grad_state
                    - erase_scalar * grad_state_dot_k * k_vec
                )

            t -= 1

        tl.store(grad_state0_ptr + grad_state_offsets, grad_state, mask=col_mask)


else:

    _affine_scan_forward_kernel = None
    _affine_scan_backward_kernel = None


class _TritonAffineScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state0: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state0 = state0.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        keep = keep.contiguous()
        erase = erase.contiguous()
        write = write.contiguous()
        mask_tensor = None if token_mask is None else token_mask.to(device=state0.device, dtype=torch.int32).contiguous()

        batch_size, rank, _ = state0.shape
        seq_len = q.size(1)
        reads = torch.empty_like(v)
        state_out = torch.empty_like(state0)
        state_hist = torch.empty((batch_size, seq_len, rank, rank), device=state0.device, dtype=state0.dtype)

        block_r = _next_power_of_2(rank)
        grid = (batch_size * rank,)
        _affine_scan_forward_kernel[grid](
            state0,
            q,
            k,
            v,
            keep,
            erase,
            write,
            mask_tensor if mask_tensor is not None else state0,
            state_hist,
            reads,
            state_out,
            batch_size,
            seq_len,
            rank,
            state0.stride(0),
            state0.stride(1),
            state0.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            0 if mask_tensor is None else mask_tensor.stride(0),
            0 if mask_tensor is None else mask_tensor.stride(1),
            state_hist.stride(0),
            state_hist.stride(1),
            state_hist.stride(2),
            state_hist.stride(3),
            reads.stride(0),
            reads.stride(1),
            reads.stride(2),
            HAS_MASK=mask_tensor is not None,
            BLOCK_R=block_r,
        )
        ctx.has_token_mask = mask_tensor is not None
        ctx.input_dtypes = (
            state0.dtype,
            q.dtype,
            k.dtype,
            v.dtype,
            keep.dtype,
            erase.dtype,
            write.dtype,
        )
        ctx.save_for_backward(
            state_hist,
            q,
            k,
            v,
            keep,
            erase,
            write,
            torch.empty(0, device=state0.device, dtype=torch.int32) if mask_tensor is None else mask_tensor,
        )
        return state_out, reads

    @staticmethod
    def backward(ctx, grad_state_out: torch.Tensor | None, grad_reads: torch.Tensor | None):
        state_hist, q, k, v, keep, erase, write, mask_tensor = ctx.saved_tensors
        state_dtype, q_dtype, k_dtype, v_dtype, keep_dtype, erase_dtype, write_dtype = ctx.input_dtypes
        if grad_reads is None:
            grad_reads = torch.zeros_like(v)
        else:
            grad_reads = grad_reads.contiguous()
        if grad_state_out is None:
            grad_state_out = torch.zeros(
                state_hist.size(0),
                state_hist.size(2),
                state_hist.size(3),
                device=state_hist.device,
                dtype=state_hist.dtype,
            )
        else:
            grad_state_out = grad_state_out.contiguous()

        batch_size, seq_len, rank, _ = state_hist.shape
        grad_state0 = torch.empty_like(grad_state_out)
        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        grad_v = torch.zeros_like(v, dtype=torch.float32)
        grad_keep = torch.zeros_like(keep, dtype=torch.float32)
        grad_erase = torch.zeros_like(erase, dtype=torch.float32)
        grad_write = torch.zeros_like(write, dtype=torch.float32)

        block_r = _next_power_of_2(rank)
        grid = (batch_size * rank,)
        _affine_scan_backward_kernel[grid](
            state_hist,
            q,
            k,
            v,
            keep,
            erase,
            write,
            mask_tensor if ctx.has_token_mask else q,
            grad_reads,
            grad_state_out,
            grad_state0,
            grad_q,
            grad_k,
            grad_v,
            grad_keep,
            grad_erase,
            grad_write,
            batch_size,
            seq_len,
            rank,
            state_hist.stride(0),
            state_hist.stride(1),
            state_hist.stride(2),
            state_hist.stride(3),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            0 if not ctx.has_token_mask else mask_tensor.stride(0),
            0 if not ctx.has_token_mask else mask_tensor.stride(1),
            grad_reads.stride(0),
            grad_reads.stride(1),
            grad_reads.stride(2),
            grad_state0.stride(0),
            grad_state0.stride(1),
            grad_state0.stride(2),
            grad_q.stride(0),
            grad_q.stride(1),
            grad_q.stride(2),
            HAS_MASK=ctx.has_token_mask,
            BLOCK_R=block_r,
        )

        return (
            grad_state0.to(dtype=state_dtype),
            grad_q.to(dtype=q_dtype),
            grad_k.to(dtype=k_dtype),
            grad_v.to(dtype=v_dtype),
            grad_keep.to(dtype=keep_dtype),
            grad_erase.to(dtype=erase_dtype),
            grad_write.to(dtype=write_dtype),
            None,
        )


def triton_affine_scan(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = triton_scan_support(state, q, k, v, keep, erase, write)
    if not support.supported:
        raise RuntimeError(f"Triton affine scan is unavailable: {support.reason}")
    return _TritonAffineScanFn.apply(state, q, k, v, keep, erase, write, token_mask)
