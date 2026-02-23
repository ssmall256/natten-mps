"""Fused QK+RPB and AV operations for DiNAT-style models.

These provide fused neighborhood attention ops with relative position bias,
using pure PyTorch implementations.

Layout: spatial-first [B, ..., H, D] — transposition to heads-first
is handled internally.

Example usage::

    from natten_mps.extras.allin1 import na1d_qk_rpb, na1d_av_fused

    logits = na1d_qk_rpb(q, k, rpb, kernel_size=5, dilation=2, scale=0.288)
    attn = torch.softmax(logits, dim=-1)
    out = na1d_av_fused(attn, v, kernel_size=5, dilation=2)
"""

from __future__ import annotations

from typing import Optional

import torch

from natten_mps.extras.allin1.functional import (
    natten1dav,
    natten1dqkrpb,
    natten2dav,
    natten2dqkrpb,
)


def na1d_qk_rpb(
    query: torch.Tensor,
    key: torch.Tensor,
    rpb: Optional[torch.Tensor],
    kernel_size: int,
    dilation: int = 1,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused 1D QK + RPB.

    Layout: spatial-first [B, L, H, D].
    RPB: [H, 2*kernel_size - 1] or None.
    Returns: [B, L, H, K] (attention logits before softmax).
    """
    if query.ndim != 4:
        raise ValueError(f"query must be 4D [B, L, H, D], got shape {query.shape}")

    if scale is not None:
        query = query * scale

    H = query.shape[2]
    if rpb is None:
        rpb = torch.zeros(H, 2 * kernel_size - 1, device=query.device, dtype=query.dtype)

    return natten1dqkrpb(query, key, rpb, kernel_size, dilation)


def na1d_av_fused(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
) -> torch.Tensor:
    """Fused 1D AV.

    Layout: spatial-first [B, L, H, K] for attn, [B, L, H, D] for value.
    Returns: [B, L, H, D].
    """
    if attn.ndim != 4 or value.ndim != 4:
        raise ValueError("attn and value must be 4D for na1d_av_fused")

    return natten1dav(attn, value, kernel_size, dilation)


def na2d_qk_rpb(
    query: torch.Tensor,
    key: torch.Tensor,
    rpb: Optional[torch.Tensor],
    kernel_size: int,
    dilation: int = 1,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused 2D QK + RPB.

    Layout: spatial-first [B, Hh, Hw, H, D].
    RPB: [H, 2*K-1, 2*K-1] or None.
    Returns: [B, Hh, Hw, H, K*K].
    """
    if query.ndim != 5:
        raise ValueError(f"query must be 5D [B, Hh, Hw, H, D], got shape {query.shape}")

    if scale is not None:
        query = query * scale

    H = query.shape[3]
    if rpb is None:
        rpb = torch.zeros(H, 2 * kernel_size - 1, 2 * kernel_size - 1, device=query.device, dtype=query.dtype)

    return natten2dqkrpb(query, key, rpb, kernel_size, dilation)


def na2d_av_fused(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
) -> torch.Tensor:
    """Fused 2D AV.

    Layout: spatial-first [B, Hh, Hw, H, K*K] for attn, [B, Hh, Hw, H, D] for value.
    Returns: [B, Hh, Hw, H, D].
    """
    if attn.ndim != 5 or value.ndim != 5:
        raise ValueError("attn and value must be 5D for na2d_av_fused")

    return natten2dav(attn, value, kernel_size, dilation)


__all__ = [
    "na1d_qk_rpb",
    "na1d_av_fused",
    "na2d_qk_rpb",
    "na2d_av_fused",
]
