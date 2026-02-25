"""Register natten-mps operations as torch.library custom ops.

This enables torch.compile to trace through neighborhood attention calls
by providing:
  1. Custom op definitions with proper schemas
  2. Fake (meta) tensor implementations for shape inference
  3. Autograd formulas for backward pass

The registered ops are the split QK/AV primitives (na{1,2,3}d_qk and
na{1,2,3}d_av) plus the fused forward (na{1,2,3}d).  The higher-level
functional API (GQA expansion, FMHA fast path, additional_kv merging)
is pure Python that torch.compile can trace natively.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from natten_mps._core import ops
from natten_mps._core import pure as _pure

NATTEN_MPS_LIB = torch.library.Library("natten_mps", "DEF")

# ---------------------------------------------------------------------------
# Helper: output spatial size after stride
# ---------------------------------------------------------------------------


def _strided_size(spatial: int, stride: int) -> int:
    return (spatial - 1) // stride + 1


# ---------------------------------------------------------------------------
# 1D QK
# ---------------------------------------------------------------------------

NATTEN_MPS_LIB.define(
    "na1d_qk(Tensor query, Tensor key, int[] kernel_size, int[] dilation, "
    "int[] stride, bool[] is_causal, float? scale) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na1d_qk", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na1d_qk", "CPU")
def _na1d_qk_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    ks = tuple(kernel_size)
    dil = tuple(dilation)
    st = tuple(stride)
    causal = tuple(is_causal)
    return ops.na1d_qk_forward(query, key, ks, dil, st, causal, scale)


@torch.library.impl(NATTEN_MPS_LIB, "na1d_qk", "Meta")
def _na1d_qk_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    B, L, H, D = query.shape
    L_out = _strided_size(L, stride[0])
    return query.new_empty(B, L_out, H, kernel_size[0])


# ---------------------------------------------------------------------------
# 1D AV
# ---------------------------------------------------------------------------

NATTEN_MPS_LIB.define(
    "na1d_av(Tensor attn, Tensor value, int[] kernel_size, int[] dilation, "
    "int[] stride, bool[] is_causal) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na1d_av", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na1d_av", "CPU")
def _na1d_av_impl(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
) -> torch.Tensor:
    return ops.na1d_av_forward(
        attn, value, tuple(kernel_size), tuple(dilation), tuple(stride), tuple(is_causal)
    )


@torch.library.impl(NATTEN_MPS_LIB, "na1d_av", "Meta")
def _na1d_av_meta(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
) -> torch.Tensor:
    B, L_out, H, _K = attn.shape
    D = value.shape[-1]
    return value.new_empty(B, L_out, H, D)


# ---------------------------------------------------------------------------
# 2D QK
# ---------------------------------------------------------------------------

NATTEN_MPS_LIB.define(
    "na2d_qk(Tensor query, Tensor key, int[] kernel_size, int[] dilation, "
    "int[] stride, bool[] is_causal, float? scale) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na2d_qk", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na2d_qk", "CPU")
def _na2d_qk_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    return ops.na2d_qk_forward(
        query, key, tuple(kernel_size), tuple(dilation), tuple(stride), tuple(is_causal), scale
    )


@torch.library.impl(NATTEN_MPS_LIB, "na2d_qk", "Meta")
def _na2d_qk_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    B, Hsp, Wsp, H, D = query.shape
    H_out = _strided_size(Hsp, stride[0])
    W_out = _strided_size(Wsp, stride[1])
    kernel_area = kernel_size[0] * kernel_size[1]
    return query.new_empty(B, H_out, W_out, H, kernel_area)


# ---------------------------------------------------------------------------
# 2D AV
# ---------------------------------------------------------------------------

NATTEN_MPS_LIB.define(
    "na2d_av(Tensor attn, Tensor value, int[] kernel_size, int[] dilation, "
    "int[] stride, bool[] is_causal) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na2d_av", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na2d_av", "CPU")
def _na2d_av_impl(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
) -> torch.Tensor:
    return ops.na2d_av_forward(
        attn, value, tuple(kernel_size), tuple(dilation), tuple(stride), tuple(is_causal)
    )


@torch.library.impl(NATTEN_MPS_LIB, "na2d_av", "Meta")
def _na2d_av_meta(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
) -> torch.Tensor:
    B, H_out, W_out, H, _K = attn.shape
    D = value.shape[-1]
    return value.new_empty(B, H_out, W_out, H, D)


# ---------------------------------------------------------------------------
# 3D QK
# ---------------------------------------------------------------------------

NATTEN_MPS_LIB.define(
    "na3d_qk(Tensor query, Tensor key, int[] kernel_size, int[] dilation, "
    "int[] stride, bool[] is_causal, float? scale) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na3d_qk", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na3d_qk", "CPU")
def _na3d_qk_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    return ops.na3d_qk_forward(
        query, key, tuple(kernel_size), tuple(dilation), tuple(stride), tuple(is_causal), scale
    )


@torch.library.impl(NATTEN_MPS_LIB, "na3d_qk", "Meta")
def _na3d_qk_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    B, D1, D2, D3, H, D = query.shape
    D1_out = _strided_size(D1, stride[0])
    D2_out = _strided_size(D2, stride[1])
    D3_out = _strided_size(D3, stride[2])
    kernel_vol = kernel_size[0] * kernel_size[1] * kernel_size[2]
    return query.new_empty(B, D1_out, D2_out, D3_out, H, kernel_vol)


# ---------------------------------------------------------------------------
# 3D AV
# ---------------------------------------------------------------------------

NATTEN_MPS_LIB.define(
    "na3d_av(Tensor attn, Tensor value, int[] kernel_size, int[] dilation, "
    "int[] stride, bool[] is_causal) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na3d_av", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na3d_av", "CPU")
def _na3d_av_impl(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
) -> torch.Tensor:
    return ops.na3d_av_forward(
        attn, value, tuple(kernel_size), tuple(dilation), tuple(stride), tuple(is_causal)
    )


@torch.library.impl(NATTEN_MPS_LIB, "na3d_av", "Meta")
def _na3d_av_meta(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
    is_causal: Sequence[bool],
) -> torch.Tensor:
    B, D1_out, D2_out, D3_out, H, _K = attn.shape
    D = value.shape[-1]
    return value.new_empty(B, D1_out, D2_out, D3_out, H, D)


# ---------------------------------------------------------------------------
# Fused forward ops (na{1,2,3}d)
# ---------------------------------------------------------------------------

NATTEN_MPS_LIB.define(
    "na1d_fwd(Tensor query, Tensor key, Tensor value, int[] kernel_size, "
    "int[] stride, int[] dilation, bool[] is_causal, float? scale) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na1d_fwd", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na1d_fwd", "CPU")
def _na1d_fwd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    return ops.na1d_forward(
        query, key, value,
        tuple(kernel_size), tuple(stride), tuple(dilation), tuple(is_causal),
        float(query.shape[-1] ** -0.5) if scale is None else scale,
    )


@torch.library.impl(NATTEN_MPS_LIB, "na1d_fwd", "Meta")
def _na1d_fwd_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    B, L, H, D = query.shape
    L_out = _strided_size(L, stride[0])
    return query.new_empty(B, L_out, H, D)


NATTEN_MPS_LIB.define(
    "na2d_fwd(Tensor query, Tensor key, Tensor value, int[] kernel_size, "
    "int[] stride, int[] dilation, bool[] is_causal, float? scale) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na2d_fwd", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na2d_fwd", "CPU")
def _na2d_fwd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    return ops.na2d_forward(
        query, key, value,
        tuple(kernel_size), tuple(stride), tuple(dilation), tuple(is_causal),
        float(query.shape[-1] ** -0.5) if scale is None else scale,
    )


@torch.library.impl(NATTEN_MPS_LIB, "na2d_fwd", "Meta")
def _na2d_fwd_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    B, Hsp, Wsp, H, D = query.shape
    H_out = _strided_size(Hsp, stride[0])
    W_out = _strided_size(Wsp, stride[1])
    return query.new_empty(B, H_out, W_out, H, D)


NATTEN_MPS_LIB.define(
    "na3d_fwd(Tensor query, Tensor key, Tensor value, int[] kernel_size, "
    "int[] stride, int[] dilation, bool[] is_causal, float? scale) -> Tensor"
)


@torch.library.impl(NATTEN_MPS_LIB, "na3d_fwd", "MPS")
@torch.library.impl(NATTEN_MPS_LIB, "na3d_fwd", "CPU")
def _na3d_fwd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    return ops.na3d_forward(
        query, key, value,
        tuple(kernel_size), tuple(stride), tuple(dilation), tuple(is_causal),
        float(query.shape[-1] ** -0.5) if scale is None else scale,
    )


@torch.library.impl(NATTEN_MPS_LIB, "na3d_fwd", "Meta")
def _na3d_fwd_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: Optional[float],
) -> torch.Tensor:
    B, D1, D2, D3, H, D = query.shape
    D1_out = _strided_size(D1, stride[0])
    D2_out = _strided_size(D2, stride[1])
    D3_out = _strided_size(D3, stride[2])
    return query.new_empty(B, D1_out, D2_out, D3_out, H, D)


# ---------------------------------------------------------------------------
# Autograd setup_context + backward for fused ops
# ---------------------------------------------------------------------------


def _setup_fused_ctx(ctx, inputs, output):
    """Save tensors and params for fused backward."""
    q, k, v, kernel_size, stride, dilation, is_causal, scale = inputs
    ctx.kernel_size = list(kernel_size)
    ctx.stride = list(stride)
    ctx.dilation = list(dilation)
    ctx.is_causal = list(is_causal)
    ctx.scale = scale
    ctx.save_for_backward(q, k, v)


def _fused_backward(ctx, grad_output, rank: int):
    """Generic fused backward for any rank."""
    q, k, v = ctx.saved_tensors
    ks = tuple(ctx.kernel_size)
    dil = tuple(ctx.dilation)
    st = tuple(ctx.stride)
    causal = tuple(ctx.is_causal)
    scale = ctx.scale
    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)

    qk_fwd = getattr(ops, f"na{rank}d_qk_forward")
    av_bwd = getattr(ops, f"na{rank}d_av_backward")
    qk_bwd = getattr(ops, f"na{rank}d_qk_backward")

    logits = qk_fwd(q, k, ks, dil, st, causal)
    attn_weights = torch.softmax(logits * scale_value, dim=-1)

    av_result = av_bwd(grad_output, attn_weights, v, ks, dil, st, causal)
    if av_result is not None:
        d_attn, d_v = av_result
        d_logits = attn_weights * (d_attn - (attn_weights * d_attn).sum(-1, keepdim=True))
        d_logits = d_logits * scale_value
        qk_result = qk_bwd(d_logits, q, k, ks, dil, st, causal)
        if qk_result is not None:
            return qk_result[0], qk_result[1], d_v, None, None, None, None, None

    # Fallback to pure backend
    pure_fwd = getattr(_pure, f"na{rank}d_forward")
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = pure_fwd(q_, k_, v_, ks, st, dil, causal, scale)
        grads = torch.autograd.grad(out, (q_, k_, v_), grad_output, allow_unused=True)
    return grads[0], grads[1], grads[2], None, None, None, None, None


def _register_fused_autograd(op_name: str, rank: int):
    """Register autograd for a fused forward op."""
    def backward(ctx, grad_output):
        return _fused_backward(ctx, grad_output, rank)

    torch.library.register_autograd(
        f"natten_mps::{op_name}", backward, setup_context=_setup_fused_ctx
    )


_register_fused_autograd("na1d_fwd", 1)
_register_fused_autograd("na2d_fwd", 2)
_register_fused_autograd("na3d_fwd", 3)


# ---------------------------------------------------------------------------
# Autograd for split QK ops
# ---------------------------------------------------------------------------


def _setup_qk_ctx(ctx, inputs, output):
    q, k, kernel_size, dilation, stride, is_causal, scale = inputs
    ctx.kernel_size = list(kernel_size)
    ctx.dilation = list(dilation)
    ctx.stride = list(stride)
    ctx.is_causal = list(is_causal)
    ctx.scale = scale
    ctx.save_for_backward(q, k)


def _qk_backward(ctx, grad_output, rank: int):
    q, k = ctx.saved_tensors
    ks = tuple(ctx.kernel_size)
    dil = tuple(ctx.dilation)
    st = tuple(ctx.stride)
    causal = tuple(ctx.is_causal)

    qk_bwd = getattr(ops, f"na{rank}d_qk_backward")
    result = qk_bwd(grad_output, q, k, ks, dil, st, causal)
    if result is not None:
        return result[0], result[1], None, None, None, None, None

    pure_qk_fwd = getattr(_pure, f"na{rank}d_qk_forward")
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        out = pure_qk_fwd(q_, k_, ks, dil, st, causal, ctx.scale)
        grads = torch.autograd.grad(out, (q_, k_), grad_output, allow_unused=True)
    return grads[0], grads[1], None, None, None, None, None


def _register_qk_autograd(op_name: str, rank: int):
    def backward(ctx, grad_output):
        return _qk_backward(ctx, grad_output, rank)

    torch.library.register_autograd(
        f"natten_mps::{op_name}", backward, setup_context=_setup_qk_ctx
    )


_register_qk_autograd("na1d_qk", 1)
_register_qk_autograd("na2d_qk", 2)
_register_qk_autograd("na3d_qk", 3)


# ---------------------------------------------------------------------------
# Autograd for split AV ops
# ---------------------------------------------------------------------------


def _setup_av_ctx(ctx, inputs, output):
    attn, v, kernel_size, dilation, stride, is_causal = inputs
    ctx.kernel_size = list(kernel_size)
    ctx.dilation = list(dilation)
    ctx.stride = list(stride)
    ctx.is_causal = list(is_causal)
    ctx.save_for_backward(attn, v)


def _av_backward(ctx, grad_output, rank: int):
    attn, v = ctx.saved_tensors
    ks = tuple(ctx.kernel_size)
    dil = tuple(ctx.dilation)
    st = tuple(ctx.stride)
    causal = tuple(ctx.is_causal)

    av_bwd = getattr(ops, f"na{rank}d_av_backward")
    result = av_bwd(grad_output, attn, v, ks, dil, st, causal)
    if result is not None:
        return result[0], result[1], None, None, None, None

    pure_av_fwd = getattr(_pure, f"na{rank}d_av_forward")
    with torch.enable_grad():
        attn_ = attn.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = pure_av_fwd(attn_, v_, ks, dil, st, causal)
        grads = torch.autograd.grad(out, (attn_, v_), grad_output, allow_unused=True)
    return grads[0], grads[1], None, None, None, None


def _register_av_autograd(op_name: str, rank: int):
    def backward(ctx, grad_output):
        return _av_backward(ctx, grad_output, rank)

    torch.library.register_autograd(
        f"natten_mps::{op_name}", backward, setup_context=_setup_av_ctx
    )


_register_av_autograd("na1d_av", 1)
_register_av_autograd("na2d_av", 2)
_register_av_autograd("na3d_av", 3)
