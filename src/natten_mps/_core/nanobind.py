"""
Tier 2: Nanobind Metal-extension backend for natten-mps.

Attempts to import the compiled _nanobind_ext module. If available,
delegates 1D QK/AV kernel dispatch to precompiled Metal shaders loaded via
the Metal API. Other ops raise NotImplementedError.

The nanobind backend dispatches directly to precompiled .metallib shaders,
bypassing torch.mps.compile_shader() runtime compilation. This eliminates
shader compilation overhead at the cost of requiring a C++ build step.

Architecture note: Since nanobind lacks native torch::Tensor type casters,
the C++ extension accepts raw MTLBuffer pointers (via tensor.data_ptr())
and shape parameters. This Python module handles tensor creation, layout
conversion, and synchronization.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

# The nanobind extension is loaded if present, but only reports as "available"
# when explicitly requested via NATTEN_BACKEND=nanobind. Auto-detection is
# disabled because the synchronous dispatch (separate command queue +
# waitUntilCompleted) is slower than compile_shader and can conflict with
# PyTorch's async MPS pipeline.
_EXT_LOADED = False
_dispatch_1d = None

try:
    from natten_mps._core._nanobind_ext import (
        is_available as _ext_available,
        dispatch_1d as _dispatch_1d,
    )
    _EXT_LOADED = _ext_available()
except ImportError:
    pass

# Only auto-select nanobind if explicitly requested
_AVAILABLE = _EXT_LOADED and os.environ.get("NATTEN_BACKEND") == "nanobind"


def is_available():
    return _AVAILABLE


def ext_loaded():
    """Check if the nanobind extension .so is loaded (for benchmarking)."""
    return _EXT_LOADED


def _not_implemented() -> NotImplementedError:
    return NotImplementedError(
        "Nanobind backend is not yet available for this op. "
        "Use set_backend('pure') or install natten-mps with nanobind extension support: "
        "pip install -e '.[nanobind]'"
    )


# ---------------------------------------------------------------------------
# Layout helpers (spatial-first ↔ heads-first)
# ---------------------------------------------------------------------------


def _to_heads_first_1d(t: torch.Tensor) -> torch.Tensor:
    """[B, L, H, D] → [B, H, L, D]"""
    return t.permute(0, 2, 1, 3).contiguous()


def _from_heads_first_1d(t: torch.Tensor) -> torch.Tensor:
    """[B, H, L, D] → [B, L, H, D]"""
    return t.permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# Low-level dispatch helper
# ---------------------------------------------------------------------------


def _dispatch_kernel_1d(
    kernel_name: str,
    buf0: torch.Tensor,
    buf1: torch.Tensor,
    out: torch.Tensor,
    batch_size: int,
    heads: int,
    length: int,
    dim: int,
    kernel_size: int,
    dilation: int,
) -> None:
    """Dispatch a 1D Metal kernel via the nanobind extension.

    Synchronizes MPS before dispatch to ensure input data is ready,
    then runs the kernel synchronously on a separate command buffer.
    """
    # Ensure any pending MPS operations on inputs are complete
    torch.mps.synchronize()

    _dispatch_1d(
        kernel_name,
        buf0.data_ptr(),
        buf1.data_ptr(),
        out.data_ptr(),
        batch_size, heads, length, dim,
        kernel_size, dilation,
    )


# ---------------------------------------------------------------------------
# 1D QK forward — uses precompiled Metal shader
# ---------------------------------------------------------------------------


def na1d_qk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    stride: Tuple[int] = (1,),
    is_causal: Tuple[bool] = (False,),
    scale: Optional[float] = None,
) -> torch.Tensor:
    """1D QK forward via precompiled Metal shaders.

    Inputs in spatial-first layout [B, L, H, D].
    Returns attention weights [B, L, H, K] (spatial-first).
    """
    if not _EXT_LOADED:
        raise _not_implemented()

    if any(is_causal):
        raise NotImplementedError("Nanobind backend does not support causal masking")
    if stride[0] != 1:
        raise NotImplementedError("Nanobind backend does not support strided output")

    B, L, H, D = q.shape
    K = kernel_size[0]
    dil = dilation[0]

    q_hf = _to_heads_first_1d(q)
    k_hf = _to_heads_first_1d(k)

    # Allocate output in heads-first layout [B, H, L, K]
    attn_hf = torch.empty(B, H, L, K, dtype=q.dtype, device=q.device)

    _dispatch_kernel_1d(
        "natten1d_qk_forward",
        q_hf, k_hf, attn_hf,
        B, H, L, D, K, dil,
    )

    # Convert back to spatial-first [B, L, H, K]
    result = attn_hf.permute(0, 2, 1, 3).contiguous()
    if scale is not None:
        result = result * scale
    return result


# ---------------------------------------------------------------------------
# 1D AV forward — uses precompiled Metal shader
# ---------------------------------------------------------------------------


def na1d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    stride: Tuple[int] = (1,),
    is_causal: Tuple[bool] = (False,),
) -> torch.Tensor:
    """1D AV forward via precompiled Metal shaders.

    attn: [B, L, H, K] (spatial-first), v: [B, L, H, D] (spatial-first).
    Returns: [B, L, H, D] (spatial-first).
    """
    if not _EXT_LOADED:
        raise _not_implemented()

    if any(is_causal):
        raise NotImplementedError("Nanobind backend does not support causal masking")
    if stride[0] != 1:
        raise NotImplementedError("Nanobind backend does not support strided output")

    B, L, H, D = v.shape
    K = kernel_size[0]
    dil = dilation[0]

    # Convert to heads-first [B, H, L, K/D]
    attn_hf = _to_heads_first_1d(attn)
    v_hf = _to_heads_first_1d(v)

    # Allocate output in heads-first layout [B, H, L, D]
    out_hf = torch.empty(B, H, L, D, dtype=v.dtype, device=v.device)

    _dispatch_kernel_1d(
        "natten1d_av_forward",
        attn_hf, v_hf, out_hf,
        B, H, L, D, K, dil,
    )

    return _from_heads_first_1d(out_hf)


# ---------------------------------------------------------------------------
# Fused forward — compose QK + softmax + AV
# ---------------------------------------------------------------------------


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    """Fused 1D NA forward via nanobind (QK → softmax → AV)."""
    if not _EXT_LOADED:
        raise _not_implemented()

    import math
    D = q.shape[-1]
    _scale = scale if scale is not None else 1.0 / math.sqrt(D)

    attn = na1d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, _scale)
    attn = torch.softmax(attn, dim=-1)
    return na1d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)


# ---------------------------------------------------------------------------
# 2D/3D stubs — not implemented
# ---------------------------------------------------------------------------


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    raise _not_implemented()


def na2d_qk_forward(q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False), scale=None):
    raise _not_implemented()


def na2d_av_forward(attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    raise _not_implemented()


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    raise _not_implemented()


def na3d_qk_forward(q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False), scale=None):
    raise _not_implemented()


def na3d_av_forward(attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    raise _not_implemented()


# ---------------------------------------------------------------------------
# Backward stubs — return None (use re-differentiation)
# ---------------------------------------------------------------------------


def na1d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return None


def na1d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return None


def na2d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return None


def na2d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return None


def na3d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return None


def na3d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return None
