"""Metal dispatch for extras/allin1 QK+RPB and AV kernels (forward + backward).

Uses torch.mps.compile_shader() to compile and cache the shader library,
then dispatches the appropriate kernel by name. Falls back to None when
Metal is unavailable or kernel_size is unsupported.
"""

from __future__ import annotations

from typing import Optional, Tuple

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


# ---------------------------------------------------------------------------
# 1D backward dispatch
# ---------------------------------------------------------------------------


def metal_1d_qkrpb_backward(
    query_hf: torch.Tensor,
    key_hf: torch.Tensor,
    rpb: Optional[torch.Tensor],
    d_attn: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """1D QK+RPB backward via Metal. Returns (dQ, dK, dRPB) or None.

    All inputs in heads-first layout [B, H, L, D/K].
    RPB: [H, 2*K-1] or None.
    """
    if not _is_available(query_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, L, D = query_hf.shape
    K = kernel_size

    q = query_hf.contiguous()
    k = key_hf.contiguous()
    da = d_attn.contiguous()

    # dQuery — per-element, no reduction
    dq = torch.empty(B, H, L, D, dtype=q.dtype, device=q.device)
    dq_kernel = getattr(lib, f"natten1d_dq_k{K}")
    dq_kernel(q, k, da, dq, B, H, L, D, dilation, threads=(L, 1, B * H))

    # dKey — reduction over query positions
    dk = torch.zeros(B, H, L, D, dtype=q.dtype, device=q.device)
    dk_kernel = getattr(lib, f"natten1d_dk_k{K}")
    dk_kernel(q, k, da, dk, B, H, L, D, dilation, threads=(L, 1, B * H))

    # dRPB — reduction over batch and spatial
    d_rpb: Optional[torch.Tensor] = None
    if rpb is not None:
        rpb_len = 2 * K - 1
        d_rpb = torch.zeros(H, rpb_len, dtype=q.dtype, device=q.device)
        drpb_kernel = getattr(lib, f"natten1d_drpb_k{K}")
        drpb_kernel(da, d_rpb, B, H, L, dilation, threads=(rpb_len, 1, H))

    return dq, dk, d_rpb


def metal_1d_av_backward(
    attn_hf: torch.Tensor,
    value_hf: torch.Tensor,
    d_out: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """1D AV backward via Metal. Returns (dAttn, dVal) or None.

    attn: [B, H, L, K], value: [B, H, L, D], d_out: [B, H, L, D].
    """
    if not _is_available(attn_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, L, K = attn_hf.shape
    D = value_hf.shape[3]

    a = attn_hf.contiguous()
    v = value_hf.contiguous()
    do = d_out.contiguous()

    # dAttn — per-element dot product
    d_attn = torch.empty(B, H, L, K, dtype=v.dtype, device=v.device)
    dattn_kernel = getattr(lib, f"natten1d_dattn_k{K}")
    dattn_kernel(do, v, d_attn, B, H, L, D, dilation, threads=(L, 1, B * H))

    # dValue — reduction over query positions
    d_val = torch.zeros(B, H, L, D, dtype=v.dtype, device=v.device)
    dval_kernel = getattr(lib, f"natten1d_dval_k{K}")
    dval_kernel(do, a, d_val, B, H, L, D, dilation, threads=(L, 1, B * H))

    return d_attn, d_val


# ---------------------------------------------------------------------------
# 2D backward dispatch
# ---------------------------------------------------------------------------


def metal_2d_qkrpb_backward(
    query_hf: torch.Tensor,
    key_hf: torch.Tensor,
    rpb: Optional[torch.Tensor],
    d_attn: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """2D QK+RPB backward via Metal. Returns (dQ, dK, dRPB) or None.

    All inputs in heads-first layout [B, H, Ht, W, D/K*K].
    RPB: [H, 2*K-1, 2*K-1] or None.
    """
    if not _is_available(query_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, Ht, W, D = query_hf.shape
    K = kernel_size

    q = query_hf.contiguous()
    k = key_hf.contiguous()
    da = d_attn.contiguous()

    # dQuery
    dq = torch.empty(B, H, Ht, W, D, dtype=q.dtype, device=q.device)
    dq_kernel = getattr(lib, f"natten2d_dq_k{K}")
    dq_kernel(q, k, da, dq, B, H, Ht, W, D, dilation, threads=(W, Ht, B * H))

    # dKey — reduction
    dk = torch.zeros(B, H, Ht, W, D, dtype=q.dtype, device=q.device)
    dk_kernel = getattr(lib, f"natten2d_dk_k{K}")
    dk_kernel(q, k, da, dk, B, H, Ht, W, D, dilation, threads=(W, Ht, B * H))

    # dRPB — reduction over batch and spatial
    d_rpb: Optional[torch.Tensor] = None
    if rpb is not None:
        rpb_size = 2 * K - 1
        d_rpb = torch.zeros(H, rpb_size, rpb_size, dtype=q.dtype, device=q.device)
        drpb_kernel = getattr(lib, f"natten2d_drpb_k{K}")
        drpb_kernel(da, d_rpb, B, H, Ht, W, dilation, threads=(rpb_size, rpb_size, H))

    return dq, dk, d_rpb


def metal_2d_av_backward(
    attn_hf: torch.Tensor,
    value_hf: torch.Tensor,
    d_out: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """2D AV backward via Metal. Returns (dAttn, dVal) or None.

    attn: [B, H, Ht, W, K*K], value: [B, H, Ht, W, D], d_out: [B, H, Ht, W, D].
    """
    if not _is_available(attn_hf, kernel_size):
        return None

    lib = _get_library()
    B, H, Ht, W, _ = attn_hf.shape
    K = kernel_size
    D = value_hf.shape[4]

    a = attn_hf.contiguous()
    v = value_hf.contiguous()
    do = d_out.contiguous()

    # dAttn — per-element
    d_attn = torch.empty(B, H, Ht, W, K * K, dtype=v.dtype, device=v.device)
    dattn_kernel = getattr(lib, f"natten2d_dattn_k{K}")
    dattn_kernel(do, v, d_attn, B, H, Ht, W, D, dilation, threads=(W, Ht, B * H))

    # dValue — reduction
    d_val = torch.zeros(B, H, Ht, W, D, dtype=v.dtype, device=v.device)
    dval_kernel = getattr(lib, f"natten2d_dval_k{K}")
    dval_kernel(do, a, d_val, B, H, Ht, W, D, dilation, threads=(W, Ht, B * H))

    return d_attn, d_val
