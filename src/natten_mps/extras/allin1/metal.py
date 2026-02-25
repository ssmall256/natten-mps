"""Metal dispatch for extras/allin1 QK+RPB and AV kernels.

Uses torch.mps.compile_shader() to compile and cache the shader library,
then dispatches the appropriate kernel by name. Falls back to None when
Metal is unavailable or kernel_size is unsupported.
"""

from __future__ import annotations

from typing import Optional

import torch

_LIB = None
_SUPPORTED_K = {3, 5, 7}


def _get_library():
    global _LIB
    if _LIB is None:
        from natten_mps.extras.allin1._metal_shaders import ALLIN1_METAL_SOURCE
        _LIB = torch.mps.compile_shader(ALLIN1_METAL_SOURCE)
    return _LIB


def _is_available(tensor: torch.Tensor, kernel_size: int) -> bool:
    """Check if Metal dispatch is possible."""
    return (
        tensor.is_mps
        and tensor.dtype == torch.float32
        and kernel_size in _SUPPORTED_K
        and torch.backends.mps.is_available()
    )


# ---------------------------------------------------------------------------
# 1D dispatch
# ---------------------------------------------------------------------------


def metal_1d_qkrpb(
    query_hf: torch.Tensor,
    key_hf: torch.Tensor,
    rpb: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[torch.Tensor]:
    """1D QK+RPB via Metal. Returns None if not available.

    Inputs in heads-first layout [B, H, L, D].
    RPB: [H, 2*K-1].
    Returns: [B, H, L, K].
    """
    if not _is_available(query_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, L, D = query_hf.shape
    K = kernel_size

    q = query_hf.contiguous()
    k = key_hf.contiguous()
    rpb_c = rpb.contiguous()
    out = torch.empty(B, H, L, K, dtype=q.dtype, device=q.device)

    kernel = getattr(lib, f"natten1d_qkrpb_k{K}")
    kernel(q, k, rpb_c, out, B, H, L, D, dilation, threads=(L, 1, B * H))
    return out


def metal_1d_av(
    attn_hf: torch.Tensor,
    value_hf: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[torch.Tensor]:
    """1D AV via Metal. Returns None if not available.

    attn: heads-first [B, H, L, K].
    value: heads-first [B, H, L, D].
    Returns: [B, H, L, D].
    """
    if not _is_available(attn_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, L, K = attn_hf.shape
    D = value_hf.shape[3]

    attn = attn_hf.contiguous()
    v = value_hf.contiguous()
    out = torch.empty(B, H, L, D, dtype=v.dtype, device=v.device)

    kernel = getattr(lib, f"natten1d_av_k{K}")
    kernel(attn, v, out, B, H, L, D, dilation, threads=(L, 1, B * H))
    return out


# ---------------------------------------------------------------------------
# 2D dispatch
# ---------------------------------------------------------------------------


def metal_2d_qkrpb(
    query_hf: torch.Tensor,
    key_hf: torch.Tensor,
    rpb: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[torch.Tensor]:
    """2D QK+RPB via Metal. Returns None if not available.

    Inputs in heads-first layout [B, H, Ht, W, D].
    RPB: [H, 2*K-1, 2*K-1].
    Returns: [B, H, Ht, W, K*K].
    """
    if not _is_available(query_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, Ht, W, D = query_hf.shape
    K = kernel_size

    q = query_hf.contiguous()
    k = key_hf.contiguous()
    rpb_c = rpb.contiguous()
    out = torch.empty(B, H, Ht, W, K * K, dtype=q.dtype, device=q.device)

    kernel = getattr(lib, f"natten2d_qkrpb_k{K}")
    kernel(q, k, rpb_c, out, B, H, Ht, W, D, dilation, threads=(W, Ht, B * H))
    return out


def metal_2d_av(
    attn_hf: torch.Tensor,
    value_hf: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[torch.Tensor]:
    """2D AV via Metal. Returns None if not available.

    attn: heads-first [B, H, Ht, W, K*K].
    value: heads-first [B, H, Ht, W, D].
    Returns: [B, H, Ht, W, D].
    """
    if not _is_available(attn_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, Ht, W, _ = attn_hf.shape
    D = value_hf.shape[4]
    K = kernel_size

    attn = attn_hf.contiguous()
    v = value_hf.contiguous()
    out = torch.empty(B, H, Ht, W, D, dtype=v.dtype, device=v.device)

    kernel = getattr(lib, f"natten2d_av_k{K}")
    kernel(attn, v, out, B, H, Ht, W, D, dilation, threads=(W, Ht, B * H))
    return out
