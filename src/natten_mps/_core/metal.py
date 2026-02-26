"""
Tier 1: MPS Metal kernel backend for natten-mps.

Dispatches custom Metal compute shaders via ``torch.mps.compile_shader``.
Kernels operate on heads-first layout [B, H, ..., D]; this module handles
permutation from the spatial-first layout used by the rest of natten-mps.

1D, 2D, and 3D operations are GPU-accelerated when inputs are on MPS and
float32/float16, including causal masking, strided output, and non-uniform
kernel/dilation configurations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from natten_mps._core import inverse_maps as _inv

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

_AVAILABLE: Optional[bool] = None  # lazy probe


def is_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = _probe_metal()
    return _AVAILABLE


def _probe_metal() -> bool:
    """Return True if MPS is available and compile_shader works."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    if not hasattr(torch.mps, "compile_shader"):
        return False
    try:
        _get_library()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lazy shader compilation (singleton)
# ---------------------------------------------------------------------------

_LIB = None
_FUSED_LIB = None
_VARLEN_LIB = None
_VARLEN_2D_LIB = None
_VARLEN_3D_LIB = None


def _get_library():
    """Compile and cache the Metal shader library."""
    global _LIB
    if _LIB is None:
        from natten_mps._core._metal_shaders import NATTEN_METAL_SOURCE

        _LIB = torch.mps.compile_shader(NATTEN_METAL_SOURCE)
    return _LIB


def _get_fused_library():
    """Compile and cache the fused SIMD Metal shader library."""
    global _FUSED_LIB
    if _FUSED_LIB is None:
        from natten_mps._core._metal_shaders import NATTEN_FUSED_SIMD_SOURCE

        _FUSED_LIB = torch.mps.compile_shader(NATTEN_FUSED_SIMD_SOURCE)
    return _FUSED_LIB


def _get_varlen_library():
    """Compile and cache the varlen 1D Metal shader library."""
    global _VARLEN_LIB
    if _VARLEN_LIB is None:
        from natten_mps._core._metal_shaders import NATTEN_VARLEN_1D_SOURCE

        _VARLEN_LIB = torch.mps.compile_shader(NATTEN_VARLEN_1D_SOURCE)
    return _VARLEN_LIB


def _get_varlen_2d_library():
    """Compile and cache the varlen 2D Metal shader library."""
    global _VARLEN_2D_LIB
    if _VARLEN_2D_LIB is None:
        from natten_mps._core._metal_shaders import NATTEN_VARLEN_2D_SOURCE

        _VARLEN_2D_LIB = torch.mps.compile_shader(NATTEN_VARLEN_2D_SOURCE)
    return _VARLEN_2D_LIB


def _get_varlen_3d_library():
    """Compile and cache the varlen 3D Metal shader library."""
    global _VARLEN_3D_LIB
    if _VARLEN_3D_LIB is None:
        from natten_mps._core._metal_shaders import NATTEN_VARLEN_3D_SOURCE

        _VARLEN_3D_LIB = torch.mps.compile_shader(NATTEN_VARLEN_3D_SOURCE)
    return _VARLEN_3D_LIB


# ---------------------------------------------------------------------------
# Device / dtype eligibility
# ---------------------------------------------------------------------------


def _on_mps(*tensors: torch.Tensor) -> bool:
    return all(t.device.type == "mps" for t in tensors)


def _supported_dtype(*tensors: torch.Tensor) -> bool:
    return all(t.dtype in (torch.float32, torch.float16, torch.bfloat16) for t in tensors)


def _upcast_bf16(*tensors: torch.Tensor) -> tuple[list[torch.Tensor], torch.dtype]:
    """If any tensor is bf16, upcast all to fp32. Returns (tensors, original_dtype)."""
    orig_dtype = tensors[0].dtype
    if orig_dtype == torch.bfloat16:
        return [t.float() for t in tensors], orig_dtype
    return list(tensors), orig_dtype


def _kernel_suffix(dtype: torch.dtype) -> str:
    """Return kernel name suffix for the given dtype."""
    if dtype == torch.float16:
        return "_f16"
    return ""


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Layout helpers  (spatial-first ↔ heads-first)
# ---------------------------------------------------------------------------


def _to_heads_first_1d(t: torch.Tensor) -> torch.Tensor:
    """[B, L, H, D] → [B, H, L, D]"""
    return t.permute(0, 2, 1, 3).contiguous()


def _from_heads_first_1d(t: torch.Tensor) -> torch.Tensor:
    """[B, H, L, D] → [B, L, H, D]"""
    return t.permute(0, 2, 1, 3).contiguous()


def _to_heads_first_2d(t: torch.Tensor) -> torch.Tensor:
    """[B, Hi, Wi, H, D] → [B, H, Hi, Wi, D]"""
    return t.permute(0, 3, 1, 2, 4).contiguous()


def _from_heads_first_2d(t: torch.Tensor) -> torch.Tensor:
    """[B, H, Hi, Wi, D] → [B, Hi, Wi, H, D]"""
    return t.permute(0, 2, 3, 1, 4).contiguous()


def _to_heads_first_3d(t: torch.Tensor) -> torch.Tensor:
    """[B, Dp, Hi, Wi, H, D] → [B, H, Dp, Hi, Wi, D]"""
    return t.permute(0, 4, 1, 2, 3, 5).contiguous()


def _from_heads_first_3d(t: torch.Tensor) -> torch.Tensor:
    """[B, H, Dp, Hi, Wi, D] → [B, Dp, Hi, Wi, H, D]"""
    return t.permute(0, 2, 3, 4, 1, 5).contiguous()


# ---------------------------------------------------------------------------
# 1D split ops
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
    if not (_on_mps(q, k) and _supported_dtype(q, k)):
        from natten_mps._core import pure

        return pure.na1d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale)

    [q, k], orig_dtype = _upcast_bf16(q, k)

    lib = _get_library()
    B, L, H, D = q.shape
    K = kernel_size[0]
    dil = dilation[0]
    s = stride[0]
    L_out = _ceil_div(L, s)

    q_hf = _to_heads_first_1d(q)
    k_hf = _to_heads_first_1d(k)
    attn_hf = torch.zeros(B, H, L_out, K, device=q.device, dtype=q.dtype)

    if any(is_causal) and s != 1:
        kernel = getattr(lib, "natten1d_qk_causal_strided_forward" + _kernel_suffix(q.dtype))
        kernel(q_hf, k_hf, attn_hf, B, H, L, D, K, dil, s, L_out, threads=(L_out, H, B))
    elif any(is_causal):
        kernel = getattr(lib, "natten1d_qk_causal_forward" + _kernel_suffix(q.dtype))
        kernel(q_hf, k_hf, attn_hf, B, H, L, D, K, dil, threads=(L_out, H, B))
    elif s != 1:
        kernel = getattr(lib, "natten1d_qk_strided_forward" + _kernel_suffix(q.dtype))
        kernel(q_hf, k_hf, attn_hf, B, H, L, D, K, dil, s, L_out, threads=(L_out, H, B))
    else:
        kernel = getattr(lib, "natten1d_qk_forward" + _kernel_suffix(q.dtype))
        kernel(q_hf, k_hf, attn_hf, B, H, L, D, K, dil, threads=(L, H, B))

    result = attn_hf.permute(0, 2, 1, 3).contiguous()
    if scale is not None:
        result = result * scale
    return result.to(orig_dtype)


def na1d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    stride: Tuple[int] = (1,),
    is_causal: Tuple[bool] = (False,),
) -> torch.Tensor:
    if not (_on_mps(attn, v) and _supported_dtype(attn, v)):
        from natten_mps._core import pure

        return pure.na1d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)

    [attn, v], orig_dtype = _upcast_bf16(attn, v)

    lib = _get_library()
    B, L_out, H, K = attn.shape
    _, L, _, D = v.shape
    dil = dilation[0]
    s = stride[0]

    attn_hf = _to_heads_first_1d(attn)
    v_hf = _to_heads_first_1d(v)
    out_hf = torch.zeros(B, H, L_out, D, device=v.device, dtype=v.dtype)

    if any(is_causal) and s != 1:
        kernel = getattr(lib, "natten1d_av_causal_strided_forward" + _kernel_suffix(v.dtype))
        kernel(attn_hf, v_hf, out_hf, B, H, L, D, K, dil, s, L_out, threads=(L_out, H, B))
    elif any(is_causal):
        kernel = getattr(lib, "natten1d_av_causal_forward" + _kernel_suffix(v.dtype))
        kernel(attn_hf, v_hf, out_hf, B, H, L, D, K, dil, threads=(L_out, H, B))
    elif s != 1:
        kernel = getattr(lib, "natten1d_av_strided_forward" + _kernel_suffix(v.dtype))
        kernel(attn_hf, v_hf, out_hf, B, H, L, D, K, dil, s, L_out, threads=(L_out, H, B))
    else:
        kernel = getattr(lib, "natten1d_av_forward" + _kernel_suffix(v.dtype))
        kernel(attn_hf, v_hf, out_hf, B, H, L, D, K, dil, threads=(L_out, H, B))

    return _from_heads_first_1d(out_hf).to(orig_dtype)


# ---------------------------------------------------------------------------
# 2D split ops
# ---------------------------------------------------------------------------


def na2d_qk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    is_causal: Tuple[bool, bool] = (False, False),
    scale: Optional[float] = None,
) -> torch.Tensor:
    if not (_on_mps(q, k) and _supported_dtype(q, k)):
        from natten_mps._core import pure

        return pure.na2d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale)

    [q, k], orig_dtype = _upcast_bf16(q, k)

    lib = _get_library()
    B, Hi, Wi, H, D = q.shape
    Kh, Kw = kernel_size
    dil_h, dil_w = dilation
    sh, sw = stride
    strided = sh != 1 or sw != 1
    Hi_out = _ceil_div(Hi, sh)
    Wi_out = _ceil_div(Wi, sw)

    q_hf = _to_heads_first_2d(q)
    k_hf = _to_heads_first_2d(k)
    attn_hf = torch.zeros(B, H, Hi_out, Wi_out, Kh * Kw, device=q.device, dtype=q.dtype)

    if any(is_causal) and strided:
        kernel = getattr(lib, "natten2d_qk_causal_strided_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]),
            sh, sw, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H),
        )
    elif any(is_causal):
        kernel = getattr(lib, "natten2d_qk_causal_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]),
            threads=(Wi_out, Hi_out, B * H),
        )
    elif strided:
        kernel = getattr(lib, "natten2d_qk_strided_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            sh, sw, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H),
        )
    else:
        kernel = getattr(lib, "natten2d_qk_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            threads=(Wi, Hi, B * H),
        )

    result = attn_hf.permute(0, 2, 3, 1, 4).contiguous()
    if scale is not None:
        result = result * scale
    return result.to(orig_dtype)


def na2d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    is_causal: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    if not (_on_mps(attn, v) and _supported_dtype(attn, v)):
        from natten_mps._core import pure

        return pure.na2d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)

    [attn, v], orig_dtype = _upcast_bf16(attn, v)

    lib = _get_library()
    B, Hi_out, Wi_out, H, KK = attn.shape
    _, Hi, Wi, _, D = v.shape
    Kh, Kw = kernel_size
    dil_h, dil_w = dilation
    sh, sw = stride
    strided = sh != 1 or sw != 1

    attn_hf = _to_heads_first_2d(attn)
    v_hf = _to_heads_first_2d(v)
    out_hf = torch.zeros(B, H, Hi_out, Wi_out, D, device=v.device, dtype=v.dtype)

    if any(is_causal) and strided:
        kernel = getattr(lib, "natten2d_av_causal_strided_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]),
            sh, sw, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H),
        )
    elif any(is_causal):
        kernel = getattr(lib, "natten2d_av_causal_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]),
            threads=(Wi_out, Hi_out, B * H),
        )
    elif strided:
        kernel = getattr(lib, "natten2d_av_strided_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            sh, sw, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H),
        )
    else:
        kernel = getattr(lib, "natten2d_av_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
            threads=(Wi_out, Hi_out, B * H),
        )

    return _from_heads_first_2d(out_hf).to(orig_dtype)


# ---------------------------------------------------------------------------
# Fused forward (QK → softmax → AV)
# ---------------------------------------------------------------------------


def _can_use_metal_1d(
    q: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    is_causal: Tuple[bool],
    stride: Tuple[int],
) -> bool:
    """Check whether the 1D Metal path supports this configuration."""
    return _on_mps(q) and _supported_dtype(q)


def _can_use_metal_2d(
    q: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    is_causal: Tuple[bool, bool],
    stride: Tuple[int, int],
) -> bool:
    """Check whether the 2D Metal path supports this configuration."""
    return _on_mps(q) and _supported_dtype(q)


def _can_use_metal_3d(
    q: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    stride: Tuple[int, int, int],
) -> bool:
    """Check whether the 3D Metal path supports this configuration."""
    return _on_mps(q) and _supported_dtype(q)


def _can_use_fused(D: int) -> bool:
    """Check if the fused SIMD kernel path can be used."""
    return D >= 32 and D % 32 == 0 and D <= 128


# ---------------------------------------------------------------------------
# Fused SIMD forward dispatch (QK→softmax→AV in one pass)
# ---------------------------------------------------------------------------


def na1d_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    stride: Tuple[int],
    is_causal: Tuple[bool],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused QK→softmax→AV forward using SIMD cooperative kernel.
    Returns (output, lse).
    """
    lib = _get_fused_library()
    B, L, H, D = q.shape
    K = kernel_size[0]
    dil = dilation[0]
    s = stride[0]
    L_out = _ceil_div(L, s)

    q_hf = _to_heads_first_1d(q)
    k_hf = _to_heads_first_1d(k)
    v_hf = _to_heads_first_1d(v)
    out_hf = torch.zeros(B, H, L_out, D, device=q.device, dtype=q.dtype)
    lse = torch.zeros(B, H, L_out, device=q.device, dtype=torch.float32)

    kernel = getattr(lib, "natten1d_fused_forward" + _kernel_suffix(q.dtype))
    kernel(
        q_hf, k_hf, v_hf, out_hf, lse,
        B, H, L, D, K, dil, s, L_out,
        scale, int(is_causal[0]),
        threads=(L_out * 32, H, B),
    )
    return _from_heads_first_1d(out_hf), lse


def na2d_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    stride: Tuple[int, int],
    is_causal: Tuple[bool, bool],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused QK→softmax→AV forward for 2D. Returns (output, lse)."""
    lib = _get_fused_library()
    B, Hi, Wi, H, D = q.shape
    Kh, Kw = kernel_size
    dil_h, dil_w = dilation
    sh, sw = stride
    Hi_out = _ceil_div(Hi, sh)
    Wi_out = _ceil_div(Wi, sw)

    q_hf = _to_heads_first_2d(q)
    k_hf = _to_heads_first_2d(k)
    v_hf = _to_heads_first_2d(v)
    out_hf = torch.zeros(B, H, Hi_out, Wi_out, D, device=q.device, dtype=q.dtype)
    lse = torch.zeros(B, H, Hi_out, Wi_out, device=q.device, dtype=torch.float32)

    kernel = getattr(lib, "natten2d_fused_forward" + _kernel_suffix(q.dtype))
    kernel(
        q_hf, k_hf, v_hf, out_hf, lse,
        B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
        sh, sw, Hi_out, Wi_out,
        scale, int(is_causal[0]), int(is_causal[1]),
        threads=(Wi_out * 32, Hi_out, B * H),
    )
    return _from_heads_first_2d(out_hf), lse


def na3d_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused QK→softmax→AV forward for 3D. Returns (output, lse)."""
    lib = _get_fused_library()
    B, Dp, Hi, Wi, H, D = q.shape
    Kd, Kh, Kw = kernel_size
    dil_d, dil_h, dil_w = dilation
    sd, sh, sw = stride
    Dp_out = _ceil_div(Dp, sd)
    Hi_out = _ceil_div(Hi, sh)
    Wi_out = _ceil_div(Wi, sw)

    q_hf = _to_heads_first_3d(q)
    k_hf = _to_heads_first_3d(k)
    v_hf = _to_heads_first_3d(v)
    out_hf = torch.zeros(B, H, Dp_out, Hi_out, Wi_out, D, device=q.device, dtype=q.dtype)
    lse = torch.zeros(B, H, Dp_out, Hi_out, Wi_out, device=q.device, dtype=torch.float32)

    kernel = getattr(lib, "natten3d_fused_forward" + _kernel_suffix(q.dtype))
    kernel(
        q_hf, k_hf, v_hf, out_hf, lse,
        B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
        sd, sh, sw, Dp_out, Hi_out, Wi_out,
        scale, int(is_causal[0]), int(is_causal[1]), int(is_causal[2]),
        threads=(Wi_out * 32, Hi_out, B * H * Dp_out),
    )
    return _from_heads_first_3d(out_hf), lse


def na1d_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    stride: Tuple[int],
    dilation: Tuple[int],
    is_causal: Tuple[bool],
    scale: Optional[float],
) -> torch.Tensor:
    if not _can_use_metal_1d(q, kernel_size, dilation, is_causal, stride):
        from natten_mps._core import pure

        return pure.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)

    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)
    D = q.shape[-1]

    if _can_use_fused(D):
        out, _lse = na1d_fused_forward(q, k, v, kernel_size, dilation, stride, is_causal, scale_value)
        return out.to(orig_dtype)

    logits = na1d_qk_forward(q, k, kernel_size, dilation, stride=stride, is_causal=is_causal)
    logits = logits * scale_value
    attn_weights = torch.softmax(logits, dim=-1)
    return na1d_av_forward(attn_weights, v, kernel_size, dilation, stride=stride, is_causal=is_causal).to(orig_dtype)


def na1d_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    stride: Tuple[int],
    dilation: Tuple[int],
    is_causal: Tuple[bool],
    scale: Optional[float],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Same as na1d_forward but returns (output, lse_or_None)."""
    if not _can_use_metal_1d(q, kernel_size, dilation, is_causal, stride):
        from natten_mps._core import pure

        return pure.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale), None

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)
    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)
    D = q.shape[-1]

    if _can_use_fused(D):
        out, lse = na1d_fused_forward(q, k, v, kernel_size, dilation, stride, is_causal, scale_value)
        return out.to(orig_dtype), lse

    logits = na1d_qk_forward(q, k, kernel_size, dilation, stride=stride, is_causal=is_causal)
    logits = logits * scale_value
    attn_weights = torch.softmax(logits, dim=-1)
    out = na1d_av_forward(attn_weights, v, kernel_size, dilation, stride=stride, is_causal=is_causal)
    return out.to(orig_dtype), None


def na1d_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    scale: Optional[float],
) -> torch.Tensor:
    """Variable-length 1D NA via Metal varlen kernels."""
    if not (_on_mps(q, k, v) and _supported_dtype(q, k, v)):
        from natten_mps._core import pure

        return pure.na1d_varlen_forward(q, k, v, seq_lens, kernel_size, dilation, scale)

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)

    lib = _get_varlen_library()
    B, L_max, H, D = q.shape
    K = kernel_size[0]
    dil = dilation[0]
    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)

    # Ensure seq_lens is int32 on MPS
    sl = seq_lens.to(dtype=torch.int32, device=q.device)

    q_hf = _to_heads_first_1d(q)
    k_hf = _to_heads_first_1d(k)
    v_hf = _to_heads_first_1d(v)

    # QK varlen
    attn_hf = torch.zeros(B, H, L_max, K, device=q.device, dtype=q.dtype)
    qk_kernel = getattr(lib, "natten1d_qk_varlen_forward" + _kernel_suffix(q.dtype))
    qk_kernel(q_hf, k_hf, attn_hf, sl, B, H, L_max, D, K, dil, threads=(L_max, H, B))

    # Scale + softmax (mask padding rows to -inf so softmax gives 0)
    attn_out = attn_hf.permute(0, 2, 1, 3).contiguous()  # [B, L_max, H, K]
    attn_out = attn_out * scale_value

    # Build padding mask: positions >= seq_lens[b] should be -inf
    pos = torch.arange(L_max, device=q.device).unsqueeze(0)  # [1, L_max]
    pad_mask = pos >= sl.unsqueeze(1)  # [B, L_max]
    attn_out[pad_mask] = float("-inf")

    attn_weights = torch.softmax(attn_out, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # AV varlen
    attn_hf2 = attn_weights.permute(0, 2, 1, 3).contiguous()  # [B, H, L_max, K]
    out_hf = torch.zeros(B, H, L_max, D, device=v.device, dtype=v.dtype)
    av_kernel = getattr(lib, "natten1d_av_varlen_forward" + _kernel_suffix(v.dtype))
    av_kernel(attn_hf2, v_hf, out_hf, sl, B, H, L_max, D, K, dil, threads=(L_max, H, B))

    return _from_heads_first_1d(out_hf).to(orig_dtype)


def na2d_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spatial_sizes: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    scale: Optional[float],
) -> torch.Tensor:
    """Variable-length 2D NA via Metal varlen kernels."""
    if not (_on_mps(q, k, v) and _supported_dtype(q, k, v)):
        from natten_mps._core import pure

        return pure.na2d_varlen_forward(q, k, v, spatial_sizes, kernel_size, dilation, scale)

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)

    lib = _get_varlen_2d_library()
    B, H_max, W_max, H, D = q.shape
    Kh, Kw = kernel_size
    K_area = Kh * Kw
    dil_h, dil_w = dilation
    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)

    ss = spatial_sizes.to(dtype=torch.int32, device=q.device).reshape(-1)

    q_hf = _to_heads_first_2d(q)
    k_hf = _to_heads_first_2d(k)
    v_hf = _to_heads_first_2d(v)

    # QK varlen
    attn_hf = torch.zeros(B, H, H_max, W_max, K_area, device=q.device, dtype=q.dtype)
    qk_kernel = getattr(lib, "natten2d_qk_varlen_forward" + _kernel_suffix(q.dtype))
    qk_kernel(
        q_hf, k_hf, attn_hf, ss,
        B, H, H_max, W_max, D, Kh, Kw, dil_h, dil_w,
        threads=(W_max, H_max, B * H),
    )

    # Scale + softmax (mask padding positions to -inf)
    attn_out = _from_heads_first_2d(attn_hf)  # [B, H_max, W_max, H, K_area]
    attn_out = attn_out * scale_value

    h_pos = torch.arange(H_max, device=q.device).view(1, H_max, 1)
    w_pos = torch.arange(W_max, device=q.device).view(1, 1, W_max)
    H_b = spatial_sizes[:, 0:1].to(device=q.device).unsqueeze(-1)  # [B, 1, 1]
    W_b = spatial_sizes[:, 1:2].to(device=q.device).unsqueeze(-1)  # [B, 1, 1]
    pad_mask = (h_pos >= H_b) | (w_pos >= W_b)  # [B, H_max, W_max]
    attn_out[pad_mask] = float("-inf")

    attn_weights = torch.softmax(attn_out, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # AV varlen
    attn_hf2 = _to_heads_first_2d(attn_weights)  # [B, H, H_max, W_max, K_area]
    out_hf = torch.zeros(B, H, H_max, W_max, D, device=v.device, dtype=v.dtype)
    av_kernel = getattr(lib, "natten2d_av_varlen_forward" + _kernel_suffix(v.dtype))
    av_kernel(
        attn_hf2, v_hf, out_hf, ss,
        B, H, H_max, W_max, D, Kh, Kw, dil_h, dil_w,
        threads=(W_max, H_max, B * H),
    )

    return _from_heads_first_2d(out_hf).to(orig_dtype)


def na3d_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spatial_sizes: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    scale: Optional[float],
) -> torch.Tensor:
    """Variable-length 3D NA via Metal varlen kernels."""
    if not (_on_mps(q, k, v) and _supported_dtype(q, k, v)):
        from natten_mps._core import pure

        return pure.na3d_varlen_forward(q, k, v, spatial_sizes, kernel_size, dilation, scale)

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)

    lib = _get_varlen_3d_library()
    B, D_max, H_max, W_max, heads, D = q.shape
    Kd, Kh, Kw = kernel_size
    K_vol = Kd * Kh * Kw
    dil_d, dil_h, dil_w = dilation
    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)

    ss = spatial_sizes.to(dtype=torch.int32, device=q.device).reshape(-1)

    q_hf = _to_heads_first_3d(q)
    k_hf = _to_heads_first_3d(k)
    v_hf = _to_heads_first_3d(v)

    # QK varlen
    attn_hf = torch.zeros(B, heads, D_max, H_max, W_max, K_vol, device=q.device, dtype=q.dtype)
    qk_kernel = getattr(lib, "natten3d_qk_varlen_forward" + _kernel_suffix(q.dtype))
    qk_kernel(
        q_hf, k_hf, attn_hf, ss,
        B, heads, D_max, H_max, W_max, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
        threads=(W_max, H_max, B * heads * D_max),
    )

    # Scale + softmax (mask padding positions to -inf)
    attn_out = _from_heads_first_3d(attn_hf)  # [B, D_max, H_max, W_max, heads, K_vol]
    attn_out = attn_out * scale_value

    d_pos = torch.arange(D_max, device=q.device).view(1, D_max, 1, 1)
    h_pos = torch.arange(H_max, device=q.device).view(1, 1, H_max, 1)
    w_pos = torch.arange(W_max, device=q.device).view(1, 1, 1, W_max)
    D_b = spatial_sizes[:, 0:1].to(device=q.device).view(B, 1, 1, 1)
    H_b = spatial_sizes[:, 1:2].to(device=q.device).view(B, 1, 1, 1)
    W_b = spatial_sizes[:, 2:3].to(device=q.device).view(B, 1, 1, 1)
    pad_mask = (d_pos >= D_b) | (h_pos >= H_b) | (w_pos >= W_b)  # [B, D_max, H_max, W_max]
    attn_out[pad_mask] = float("-inf")

    attn_weights = torch.softmax(attn_out, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # AV varlen
    attn_hf2 = _to_heads_first_3d(attn_weights)
    out_hf = torch.zeros(B, heads, D_max, H_max, W_max, D, device=v.device, dtype=v.dtype)
    av_kernel = getattr(lib, "natten3d_av_varlen_forward" + _kernel_suffix(v.dtype))
    av_kernel(
        attn_hf2, v_hf, out_hf, ss,
        B, heads, D_max, H_max, W_max, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
        threads=(W_max, H_max, B * heads * D_max),
    )

    return _from_heads_first_3d(out_hf).to(orig_dtype)


def na2d_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    is_causal: Tuple[bool, bool],
    scale: Optional[float],
) -> torch.Tensor:
    if not _can_use_metal_2d(q, kernel_size, dilation, is_causal, stride):
        from natten_mps._core import pure

        return pure.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)

    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)
    D = q.shape[-1]

    if _can_use_fused(D):
        out, _lse = na2d_fused_forward(q, k, v, kernel_size, dilation, stride, is_causal, scale_value)
        return out.to(orig_dtype)

    logits = na2d_qk_forward(q, k, kernel_size, dilation, stride=stride, is_causal=is_causal)
    logits = logits * scale_value
    attn_weights = torch.softmax(logits, dim=-1)
    return na2d_av_forward(attn_weights, v, kernel_size, dilation, stride=stride, is_causal=is_causal).to(orig_dtype)


def na2d_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    is_causal: Tuple[bool, bool],
    scale: Optional[float],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Same as na2d_forward but returns (output, lse_or_None)."""
    if not _can_use_metal_2d(q, kernel_size, dilation, is_causal, stride):
        from natten_mps._core import pure

        return pure.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale), None

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)
    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)
    D = q.shape[-1]

    if _can_use_fused(D):
        out, lse = na2d_fused_forward(q, k, v, kernel_size, dilation, stride, is_causal, scale_value)
        return out.to(orig_dtype), lse

    logits = na2d_qk_forward(q, k, kernel_size, dilation, stride=stride, is_causal=is_causal)
    logits = logits * scale_value
    attn_weights = torch.softmax(logits, dim=-1)
    out = na2d_av_forward(attn_weights, v, kernel_size, dilation, stride=stride, is_causal=is_causal)
    return out.to(orig_dtype), None


# ---------------------------------------------------------------------------
# 3D split ops
# ---------------------------------------------------------------------------


def na3d_qk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    stride: Tuple[int, int, int] = (1, 1, 1),
    is_causal: Tuple[bool, bool, bool] = (False, False, False),
    scale: Optional[float] = None,
) -> torch.Tensor:
    if not (_on_mps(q, k) and _supported_dtype(q, k)):
        from natten_mps._core import pure

        return pure.na3d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale)

    [q, k], orig_dtype = _upcast_bf16(q, k)

    lib = _get_library()
    B, Dp, Hi, Wi, H, D = q.shape
    Kd, Kh, Kw = kernel_size
    dil_d, dil_h, dil_w = dilation
    K3 = Kd * Kh * Kw
    sd, sh, sw = stride
    strided = sd != 1 or sh != 1 or sw != 1
    Dp_out = _ceil_div(Dp, sd)
    Hi_out = _ceil_div(Hi, sh)
    Wi_out = _ceil_div(Wi, sw)

    q_hf = _to_heads_first_3d(q)
    k_hf = _to_heads_first_3d(k)
    attn_hf = torch.zeros(B, H, Dp_out, Hi_out, Wi_out, K3, device=q.device, dtype=q.dtype)

    if any(is_causal) and strided:
        kernel = getattr(lib, "natten3d_qk_causal_strided_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]), int(is_causal[2]),
            sd, sh, sw, Dp_out, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H * Dp_out),
        )
    elif any(is_causal):
        kernel = getattr(lib, "natten3d_qk_causal_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]), int(is_causal[2]),
            threads=(Wi_out, Hi_out, B * H * Dp_out),
        )
    elif strided:
        kernel = getattr(lib, "natten3d_qk_strided_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            sd, sh, sw, Dp_out, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H * Dp_out),
        )
    else:
        kernel = getattr(lib, "natten3d_qk_forward" + _kernel_suffix(q.dtype))
        kernel(
            q_hf, k_hf, attn_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            threads=(Wi, Hi, B * H * Dp),
        )

    result = attn_hf.permute(0, 2, 3, 4, 1, 5).contiguous()
    if scale is not None:
        result = result * scale
    return result.to(orig_dtype)


def na3d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    stride: Tuple[int, int, int] = (1, 1, 1),
    is_causal: Tuple[bool, bool, bool] = (False, False, False),
) -> torch.Tensor:
    if not (_on_mps(attn, v) and _supported_dtype(attn, v)):
        from natten_mps._core import pure

        return pure.na3d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)

    [attn, v], orig_dtype = _upcast_bf16(attn, v)

    lib = _get_library()
    B, Dp_out, Hi_out, Wi_out, H, K3 = attn.shape
    _, Dp, Hi, Wi, _, D = v.shape
    Kd, Kh, Kw = kernel_size
    dil_d, dil_h, dil_w = dilation
    sd, sh, sw = stride
    strided = sd != 1 or sh != 1 or sw != 1

    attn_hf = _to_heads_first_3d(attn)
    v_hf = _to_heads_first_3d(v)
    out_hf = torch.zeros(B, H, Dp_out, Hi_out, Wi_out, D, device=v.device, dtype=v.dtype)

    if any(is_causal) and strided:
        kernel = getattr(lib, "natten3d_av_causal_strided_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]), int(is_causal[2]),
            sd, sh, sw, Dp_out, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H * Dp_out),
        )
    elif any(is_causal):
        kernel = getattr(lib, "natten3d_av_causal_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            int(is_causal[0]), int(is_causal[1]), int(is_causal[2]),
            threads=(Wi_out, Hi_out, B * H * Dp_out),
        )
    elif strided:
        kernel = getattr(lib, "natten3d_av_strided_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            sd, sh, sw, Dp_out, Hi_out, Wi_out,
            threads=(Wi_out, Hi_out, B * H * Dp_out),
        )
    else:
        kernel = getattr(lib, "natten3d_av_forward" + _kernel_suffix(v.dtype))
        kernel(
            attn_hf, v_hf, out_hf,
            B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
            threads=(Wi_out, Hi_out, B * H * Dp_out),
        )

    return _from_heads_first_3d(out_hf).to(orig_dtype)


# ---------------------------------------------------------------------------
# 3D fused forward
# ---------------------------------------------------------------------------


def na3d_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    scale: Optional[float],
) -> torch.Tensor:
    if not _can_use_metal_3d(q, kernel_size, dilation, is_causal, stride):
        from natten_mps._core import pure

        return pure.na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)

    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)
    D = q.shape[-1]

    if _can_use_fused(D):
        out, _lse = na3d_fused_forward(q, k, v, kernel_size, dilation, stride, is_causal, scale_value)
        return out.to(orig_dtype)

    logits = na3d_qk_forward(q, k, kernel_size, dilation, stride=stride, is_causal=is_causal)
    logits = logits * scale_value
    attn_weights = torch.softmax(logits, dim=-1)
    return na3d_av_forward(attn_weights, v, kernel_size, dilation, stride=stride, is_causal=is_causal).to(orig_dtype)


def na3d_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    scale: Optional[float],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Same as na3d_forward but returns (output, lse_or_None)."""
    if not _can_use_metal_3d(q, kernel_size, dilation, is_causal, stride):
        from natten_mps._core import pure

        return pure.na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale), None

    [q, k, v], orig_dtype = _upcast_bf16(q, k, v)
    scale_value = float(q.shape[-1] ** -0.5 if scale is None else scale)
    D = q.shape[-1]

    if _can_use_fused(D):
        out, lse = na3d_fused_forward(q, k, v, kernel_size, dilation, stride, is_causal, scale_value)
        return out.to(orig_dtype), lse

    logits = na3d_qk_forward(q, k, kernel_size, dilation, stride=stride, is_causal=is_causal)
    logits = logits * scale_value
    attn_weights = torch.softmax(logits, dim=-1)
    out = na3d_av_forward(attn_weights, v, kernel_size, dilation, stride=stride, is_causal=is_causal)
    return out.to(orig_dtype), None


# ---------------------------------------------------------------------------
# Fused SIMD backward dispatch
# ---------------------------------------------------------------------------


def na1d_fused_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fwd_out: torch.Tensor,
    lse: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    stride: Tuple[int],
    is_causal: Tuple[bool],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SIMD backward: returns (d_q, d_k, d_v) in spatial-first layout."""
    [grad_out, q, k, v, fwd_out], orig_dtype = _upcast_bf16(grad_out, q, k, v, fwd_out)

    lib = _get_fused_library()
    B, L, H, D = q.shape
    K = kernel_size[0]
    dil = dilation[0]
    s = stride[0]
    L_out = _ceil_div(L, s)

    go_hf = _to_heads_first_1d(grad_out)
    q_hf = _to_heads_first_1d(q)
    k_hf = _to_heads_first_1d(k)
    v_hf = _to_heads_first_1d(v)
    out_hf = _to_heads_first_1d(fwd_out)

    dq_hf = torch.zeros_like(q_hf)
    attn_hf = torch.zeros(B, H, L_out, K, device=q.device, dtype=torch.float32)
    dlogits_hf = torch.zeros(B, H, L_out, K, device=q.device, dtype=torch.float32)

    kernel = getattr(lib, "natten1d_fused_backward" + _kernel_suffix(q.dtype))
    kernel(
        q_hf, k_hf, v_hf, go_hf, out_hf, lse,
        dq_hf, attn_hf, dlogits_hf,
        B, H, L, D, K, dil, s, L_out,
        scale, int(is_causal[0]),
        threads=(L_out * 32, H, B),
    )

    # Use existing inverse-map kernels for d_k and d_v
    dk_hf = torch.zeros_like(k_hf)
    dv_hf = torch.zeros_like(v_hf)

    main_lib = _get_library()
    sfx = _kernel_suffix(q.dtype)

    # d_k via inverse-map kernel
    inv_off, inv_attn, inv_qbase = _inv.inverse_map_1d_qk(
        L, L_out, K, s, dil, bool(is_causal[0]), D)
    # dlogits is in float32; for f16 kernels, attn and dlogits need matching dtype
    if q.dtype == torch.float16:
        dlogits_hf_cast = dlogits_hf.half()
        attn_hf_cast = attn_hf.half()
    else:
        dlogits_hf_cast = dlogits_hf
        attn_hf_cast = attn_hf
    getattr(main_lib, "natten1d_k_backward_inv" + sfx)(
        dlogits_hf_cast, q_hf, dk_hf, inv_off, inv_attn, inv_qbase,
        B, H, L, D, L_out, K,
        threads=(D, L, B * H))

    # d_v via inverse-map kernel
    inv_off_v, inv_attn_v, inv_gbase = _inv.inverse_map_1d(
        L, L_out, K, s, dil, bool(is_causal[0]), D)
    getattr(main_lib, "natten1d_v_backward_inv" + sfx)(
        go_hf, attn_hf_cast, dv_hf, inv_off_v, inv_attn_v, inv_gbase,
        B, H, L, D, L_out, K,
        threads=(D, L, B * H))

    d_q = _from_heads_first_1d(dq_hf).to(orig_dtype)
    d_k = _from_heads_first_1d(dk_hf).to(orig_dtype)
    d_v = _from_heads_first_1d(dv_hf).to(orig_dtype)
    return d_q, d_k, d_v


def na2d_fused_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fwd_out: torch.Tensor,
    lse: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    stride: Tuple[int, int],
    is_causal: Tuple[bool, bool],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SIMD backward for 2D: returns (d_q, d_k, d_v) in spatial-first layout."""
    [grad_out, q, k, v, fwd_out], orig_dtype = _upcast_bf16(grad_out, q, k, v, fwd_out)

    lib = _get_fused_library()
    B, Hi, Wi, H, D = q.shape
    Kh, Kw = kernel_size
    KK = Kh * Kw
    dil_h, dil_w = dilation
    sh, sw = stride
    Hi_out = _ceil_div(Hi, sh)
    Wi_out = _ceil_div(Wi, sw)

    go_hf = _to_heads_first_2d(grad_out)
    q_hf = _to_heads_first_2d(q)
    k_hf = _to_heads_first_2d(k)
    v_hf = _to_heads_first_2d(v)
    out_hf = _to_heads_first_2d(fwd_out)

    dq_hf = torch.zeros_like(q_hf)
    attn_hf = torch.zeros(B, H, Hi_out, Wi_out, KK, device=q.device, dtype=torch.float32)
    dlogits_hf = torch.zeros(B, H, Hi_out, Wi_out, KK, device=q.device, dtype=torch.float32)

    kernel = getattr(lib, "natten2d_fused_backward" + _kernel_suffix(q.dtype))
    kernel(
        q_hf, k_hf, v_hf, go_hf, out_hf, lse,
        dq_hf, attn_hf, dlogits_hf,
        B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
        sh, sw, Hi_out, Wi_out,
        scale, int(is_causal[0]), int(is_causal[1]),
        threads=(Wi_out * 32, Hi_out, B * H),
    )

    dk_hf = torch.zeros_like(k_hf)
    dv_hf = torch.zeros_like(v_hf)

    main_lib = _get_library()
    sfx = _kernel_suffix(q.dtype)
    HW = Hi * Wi
    out_count = Hi_out * Wi_out

    if q.dtype == torch.float16:
        dlogits_hf_cast = dlogits_hf.half()
        attn_hf_cast = attn_hf.half()
    else:
        dlogits_hf_cast = dlogits_hf
        attn_hf_cast = attn_hf

    inv_off, inv_attn, inv_qbase = _inv.inverse_map_2d_qk(
        Hi, Wi, Hi_out, Wi_out, Kh, Kw, sh, sw, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), D)
    getattr(main_lib, "natten2d_k_backward_inv" + sfx)(
        dlogits_hf_cast, q_hf, dk_hf, inv_off, inv_attn, inv_qbase,
        B, H, Hi, Wi, D, out_count, KK,
        threads=(D, HW, B * H))

    inv_off_v, inv_attn_v, inv_gbase = _inv.inverse_map_2d(
        Hi, Wi, Hi_out, Wi_out, Kh, Kw, sh, sw, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), D)
    getattr(main_lib, "natten2d_v_backward_inv" + sfx)(
        go_hf, attn_hf_cast, dv_hf, inv_off_v, inv_attn_v, inv_gbase,
        B, H, Hi, Wi, D, out_count, KK,
        threads=(D, HW, B * H))

    d_q = _from_heads_first_2d(dq_hf).to(orig_dtype)
    d_k = _from_heads_first_2d(dk_hf).to(orig_dtype)
    d_v = _from_heads_first_2d(dv_hf).to(orig_dtype)
    return d_q, d_k, d_v


def na3d_fused_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fwd_out: torch.Tensor,
    lse: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SIMD backward for 3D: returns (d_q, d_k, d_v) in spatial-first layout."""
    [grad_out, q, k, v, fwd_out], orig_dtype = _upcast_bf16(grad_out, q, k, v, fwd_out)

    lib = _get_fused_library()
    B, Dp, Hi, Wi, H, D = q.shape
    Kd, Kh, Kw = kernel_size
    KKK = Kd * Kh * Kw
    dil_d, dil_h, dil_w = dilation
    sd, sh, sw = stride
    Dp_out = _ceil_div(Dp, sd)
    Hi_out = _ceil_div(Hi, sh)
    Wi_out = _ceil_div(Wi, sw)

    go_hf = _to_heads_first_3d(grad_out)
    q_hf = _to_heads_first_3d(q)
    k_hf = _to_heads_first_3d(k)
    v_hf = _to_heads_first_3d(v)
    out_hf = _to_heads_first_3d(fwd_out)

    dq_hf = torch.zeros_like(q_hf)
    attn_hf = torch.zeros(B, H, Dp_out, Hi_out, Wi_out, KKK, device=q.device, dtype=torch.float32)
    dlogits_hf = torch.zeros(B, H, Dp_out, Hi_out, Wi_out, KKK, device=q.device, dtype=torch.float32)

    kernel = getattr(lib, "natten3d_fused_backward" + _kernel_suffix(q.dtype))
    kernel(
        q_hf, k_hf, v_hf, go_hf, out_hf, lse,
        dq_hf, attn_hf, dlogits_hf,
        B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
        sd, sh, sw, Dp_out, Hi_out, Wi_out,
        scale, int(is_causal[0]), int(is_causal[1]), int(is_causal[2]),
        threads=(Wi_out * 32, Hi_out, B * H * Dp_out),
    )

    dk_hf = torch.zeros_like(k_hf)
    dv_hf = torch.zeros_like(v_hf)

    main_lib = _get_library()
    sfx = _kernel_suffix(q.dtype)
    vol = Dp * Hi * Wi
    out_count = Dp_out * Hi_out * Wi_out

    if q.dtype == torch.float16:
        dlogits_hf_cast = dlogits_hf.half()
        attn_hf_cast = attn_hf.half()
    else:
        dlogits_hf_cast = dlogits_hf
        attn_hf_cast = attn_hf

    inv_off, inv_attn, inv_qbase = _inv.inverse_map_3d_qk(
        Dp, Hi, Wi, Dp_out, Hi_out, Wi_out, Kd, Kh, Kw,
        sd, sh, sw, dil_d, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2]), D)
    getattr(main_lib, "natten3d_k_backward_inv" + sfx)(
        dlogits_hf_cast, q_hf, dk_hf, inv_off, inv_attn, inv_qbase,
        B, H, vol, D, out_count, KKK,
        threads=(D, vol, B * H))

    inv_off_v, inv_attn_v, inv_gbase = _inv.inverse_map_3d(
        Dp, Hi, Wi, Dp_out, Hi_out, Wi_out, Kd, Kh, Kw,
        sd, sh, sw, dil_d, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2]), D)
    getattr(main_lib, "natten3d_v_backward_inv" + sfx)(
        go_hf, attn_hf_cast, dv_hf, inv_off_v, inv_attn_v, inv_gbase,
        B, H, vol, D, out_count, KKK,
        threads=(D, vol, B * H))

    d_q = _from_heads_first_3d(dq_hf).to(orig_dtype)
    d_k = _from_heads_first_3d(dk_hf).to(orig_dtype)
    d_v = _from_heads_first_3d(dv_hf).to(orig_dtype)
    return d_q, d_k, d_v


# ---------------------------------------------------------------------------
# Backward dispatch — return None if Metal backward not available
# ---------------------------------------------------------------------------


def _can_use_metal_backward_1d(t, kernel_size, dilation, stride, is_causal):
    return _on_mps(t) and _supported_dtype(t)


def _can_use_metal_backward_2d(t, kernel_size, dilation, stride, is_causal):
    return _on_mps(t) and _supported_dtype(t)


def _can_use_metal_backward_3d(t, kernel_size, dilation, stride, is_causal):
    return _on_mps(t) and _supported_dtype(t)


# -- 1D backward --

def na1d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    if not _can_use_metal_backward_1d(d_attn, kernel_size, dilation, stride, is_causal):
        return None

    [d_attn, q, k], orig_dtype = _upcast_bf16(d_attn, q, k)

    lib = _get_library()
    B, L_out, H, K = d_attn.shape
    L = q.shape[1]
    D = q.shape[-1]
    dil = dilation[0]
    s = stride[0]
    causal_flag = 1 if is_causal[0] else 0

    da_hf = _to_heads_first_1d(d_attn)
    q_hf = _to_heads_first_1d(q)
    k_hf = _to_heads_first_1d(k)

    dq_hf = torch.zeros_like(q_hf)
    dk_hf = torch.zeros_like(k_hf)

    sfx = _kernel_suffix(q.dtype)
    # q_backward: unified kernel with stride/causal params
    getattr(lib, "natten1d_q_backward" + sfx)(
        da_hf, k_hf, dq_hf, B, H, L, D, K, dil, s, L_out, causal_flag,
        threads=(L, H, B))
    # k_backward: inverse-map kernel (CSR lookup, no brute-force iteration)
    inv_off, inv_attn, inv_qbase = _inv.inverse_map_1d_qk(
        L, L_out, K, s, dil, bool(is_causal[0]), D)
    getattr(lib, "natten1d_k_backward_inv" + sfx)(
        da_hf, q_hf, dk_hf, inv_off, inv_attn, inv_qbase,
        B, H, L, D, L_out, K,
        threads=(D, L, B * H))

    return _from_heads_first_1d(dq_hf).to(orig_dtype), _from_heads_first_1d(dk_hf).to(orig_dtype)


def na1d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    if not _can_use_metal_backward_1d(d_out, kernel_size, dilation, stride, is_causal):
        return None

    [d_out, attn, v], orig_dtype = _upcast_bf16(d_out, attn, v)

    lib = _get_library()
    B, L_out, H, D = d_out.shape
    L = v.shape[1]
    K = kernel_size[0]
    dil = dilation[0]
    s = stride[0]
    causal_flag = 1 if is_causal[0] else 0

    do_hf = _to_heads_first_1d(d_out)
    a_hf = _to_heads_first_1d(attn)
    v_hf = _to_heads_first_1d(v)

    da_hf = torch.zeros(B, H, L_out, K, device=d_out.device, dtype=d_out.dtype)
    dv_hf = torch.zeros_like(v_hf)

    sfx = _kernel_suffix(d_out.dtype)
    # a_backward: unified kernel with stride/causal params
    getattr(lib, "natten1d_a_backward" + sfx)(
        do_hf, v_hf, da_hf, B, H, L, D, K, dil, s, L_out, causal_flag,
        threads=(L_out, H, B))
    # v_backward: inverse-map kernel
    inv_off, inv_attn, inv_gbase = _inv.inverse_map_1d(
        L, L_out, K, s, dil, bool(is_causal[0]), D)
    getattr(lib, "natten1d_v_backward_inv" + sfx)(
        do_hf, a_hf, dv_hf, inv_off, inv_attn, inv_gbase,
        B, H, L, D, L_out, K,
        threads=(D, L, B * H))

    return da_hf.permute(0, 2, 1, 3).contiguous().to(orig_dtype), _from_heads_first_1d(dv_hf).to(orig_dtype)


# -- 2D backward --

def na2d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    if not _can_use_metal_backward_2d(d_attn, kernel_size, dilation, stride, is_causal):
        return None

    [d_attn, q, k], orig_dtype = _upcast_bf16(d_attn, q, k)

    lib = _get_library()
    # d_attn shape: [B, Hi_out, Wi_out, H, KK]
    B, Hi_out, Wi_out, H, KK = d_attn.shape
    # q/k shape: [B, Hi, Wi, H, D]
    Hi, Wi = q.shape[1], q.shape[2]
    D = q.shape[-1]
    Kh, Kw = kernel_size
    dil_h, dil_w = dilation
    sh, sw = stride
    causal_h = 1 if is_causal[0] else 0
    causal_w = 1 if is_causal[1] else 0

    da_hf = _to_heads_first_2d(d_attn)
    q_hf = _to_heads_first_2d(q)
    k_hf = _to_heads_first_2d(k)

    dq_hf = torch.zeros_like(q_hf)
    dk_hf = torch.zeros_like(k_hf)

    sfx = _kernel_suffix(q.dtype)
    # q_backward: unified kernel with stride/causal params
    getattr(lib, "natten2d_q_backward" + sfx)(
        da_hf, k_hf, dq_hf, B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
        sh, sw, Hi_out, Wi_out, causal_h, causal_w,
        threads=(Wi, Hi, B * H))
    # k_backward: inverse-map kernel
    HW = Hi * Wi
    out_count = Hi_out * Wi_out
    inv_off, inv_attn, inv_qbase = _inv.inverse_map_2d_qk(
        Hi, Wi, Hi_out, Wi_out, Kh, Kw, sh, sw, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), D)
    getattr(lib, "natten2d_k_backward_inv" + sfx)(
        da_hf, q_hf, dk_hf, inv_off, inv_attn, inv_qbase,
        B, H, Hi, Wi, D, out_count, KK,
        threads=(D, HW, B * H))

    return _from_heads_first_2d(dq_hf).to(orig_dtype), _from_heads_first_2d(dk_hf).to(orig_dtype)


def na2d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    if not _can_use_metal_backward_2d(d_out, kernel_size, dilation, stride, is_causal):
        return None

    [d_out, attn, v], orig_dtype = _upcast_bf16(d_out, attn, v)

    lib = _get_library()
    # d_out shape: [B, Hi_out, Wi_out, H, D]
    B, Hi_out, Wi_out, H, D = d_out.shape
    # v shape: [B, Hi, Wi, H, D]
    Hi, Wi = v.shape[1], v.shape[2]
    Kh, Kw = kernel_size
    KK = Kh * Kw
    dil_h, dil_w = dilation
    sh, sw = stride
    causal_h = 1 if is_causal[0] else 0
    causal_w = 1 if is_causal[1] else 0

    do_hf = _to_heads_first_2d(d_out)
    a_hf = _to_heads_first_2d(attn)
    v_hf = _to_heads_first_2d(v)

    da_hf = torch.zeros(B, H, Hi_out, Wi_out, KK, device=d_out.device, dtype=d_out.dtype)
    dv_hf = torch.zeros_like(v_hf)

    sfx = _kernel_suffix(d_out.dtype)
    # a_backward: unified kernel with stride/causal params
    getattr(lib, "natten2d_a_backward" + sfx)(
        do_hf, v_hf, da_hf, B, H, Hi, Wi, D, Kh, Kw, dil_h, dil_w,
        sh, sw, Hi_out, Wi_out, causal_h, causal_w,
        threads=(Wi_out, Hi_out, B * H))
    # v_backward: inverse-map kernel
    HW = Hi * Wi
    out_count = Hi_out * Wi_out
    inv_off, inv_attn, inv_gbase = _inv.inverse_map_2d(
        Hi, Wi, Hi_out, Wi_out, Kh, Kw, sh, sw, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), D)
    getattr(lib, "natten2d_v_backward_inv" + sfx)(
        do_hf, a_hf, dv_hf, inv_off, inv_attn, inv_gbase,
        B, H, Hi, Wi, D, out_count, KK,
        threads=(D, HW, B * H))

    return da_hf.permute(0, 2, 3, 1, 4).contiguous().to(orig_dtype), _from_heads_first_2d(dv_hf).to(orig_dtype)


# -- 3D backward --

def na3d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    if not _can_use_metal_backward_3d(d_attn, kernel_size, dilation, stride, is_causal):
        return None

    [d_attn, q, k], orig_dtype = _upcast_bf16(d_attn, q, k)

    lib = _get_library()
    # d_attn shape: [B, Dp_out, Hi_out, Wi_out, H, KKK]
    B, Dp_out, Hi_out, Wi_out, H, KKK = d_attn.shape
    # q/k shape: [B, Dp, Hi, Wi, H, D]
    Dp, Hi, Wi = q.shape[1], q.shape[2], q.shape[3]
    D = q.shape[-1]
    Kd, Kh, Kw = kernel_size
    dil_d, dil_h, dil_w = dilation
    sd, sh, sw = stride
    causal_d = 1 if is_causal[0] else 0
    causal_h = 1 if is_causal[1] else 0
    causal_w = 1 if is_causal[2] else 0

    da_hf = _to_heads_first_3d(d_attn)
    q_hf = _to_heads_first_3d(q)
    k_hf = _to_heads_first_3d(k)

    dq_hf = torch.zeros_like(q_hf)
    dk_hf = torch.zeros_like(k_hf)

    sfx = _kernel_suffix(q.dtype)
    # q_backward: unified kernel with stride/causal params
    getattr(lib, "natten3d_q_backward" + sfx)(
        da_hf, k_hf, dq_hf, B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
        sd, sh, sw, Dp_out, Hi_out, Wi_out, causal_d, causal_h, causal_w,
        threads=(Wi, Hi, B * H * Dp))
    # k_backward: inverse-map kernel
    vol = Dp * Hi * Wi
    out_count = Dp_out * Hi_out * Wi_out
    inv_off, inv_attn, inv_qbase = _inv.inverse_map_3d_qk(
        Dp, Hi, Wi, Dp_out, Hi_out, Wi_out, Kd, Kh, Kw,
        sd, sh, sw, dil_d, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2]), D)
    getattr(lib, "natten3d_k_backward_inv" + sfx)(
        da_hf, q_hf, dk_hf, inv_off, inv_attn, inv_qbase,
        B, H, vol, D, out_count, KKK,
        threads=(D, vol, B * H))

    return _from_heads_first_3d(dq_hf).to(orig_dtype), _from_heads_first_3d(dk_hf).to(orig_dtype)


def na3d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    if not _can_use_metal_backward_3d(d_out, kernel_size, dilation, stride, is_causal):
        return None

    [d_out, attn, v], orig_dtype = _upcast_bf16(d_out, attn, v)

    lib = _get_library()
    # d_out shape: [B, Dp_out, Hi_out, Wi_out, H, D]
    B, Dp_out, Hi_out, Wi_out, H, D = d_out.shape
    # v shape: [B, Dp, Hi, Wi, H, D]
    Dp, Hi, Wi = v.shape[1], v.shape[2], v.shape[3]
    Kd, Kh, Kw = kernel_size
    KKK = Kd * Kh * Kw
    dil_d, dil_h, dil_w = dilation
    sd, sh, sw = stride
    causal_d = 1 if is_causal[0] else 0
    causal_h = 1 if is_causal[1] else 0
    causal_w = 1 if is_causal[2] else 0

    do_hf = _to_heads_first_3d(d_out)
    a_hf = _to_heads_first_3d(attn)
    v_hf = _to_heads_first_3d(v)

    da_hf = torch.zeros(B, H, Dp_out, Hi_out, Wi_out, KKK, device=d_out.device, dtype=d_out.dtype)
    dv_hf = torch.zeros_like(v_hf)

    sfx = _kernel_suffix(d_out.dtype)
    # a_backward: unified kernel with stride/causal params
    getattr(lib, "natten3d_a_backward" + sfx)(
        do_hf, v_hf, da_hf, B, H, Dp, Hi, Wi, D, Kd, Kh, Kw, dil_d, dil_h, dil_w,
        sd, sh, sw, Dp_out, Hi_out, Wi_out, causal_d, causal_h, causal_w,
        threads=(Wi_out, Hi_out, B * H * Dp_out))
    # v_backward: inverse-map kernel
    vol = Dp * Hi * Wi
    out_count = Dp_out * Hi_out * Wi_out
    inv_off, inv_attn, inv_gbase = _inv.inverse_map_3d(
        Dp, Hi, Wi, Dp_out, Hi_out, Wi_out, Kd, Kh, Kw,
        sd, sh, sw, dil_d, dil_h, dil_w,
        bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2]), D)
    getattr(lib, "natten3d_v_backward_inv" + sfx)(
        do_hf, a_hf, dv_hf, inv_off, inv_attn, inv_gbase,
        B, H, vol, D, out_count, KKK,
        threads=(D, vol, B * H))

    return da_hf.permute(0, 2, 3, 4, 1, 5).contiguous().to(orig_dtype), _from_heads_first_3d(dv_hf).to(orig_dtype)
