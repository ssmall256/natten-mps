from __future__ import annotations

import torch

from natten_mps.functional import na1d_av as _na1d_av
from natten_mps.functional import na1d_qk as _na1d_qk
from natten_mps.functional import na2d_av as _na2d_av
from natten_mps.functional import na2d_qk as _na2d_qk
from natten_mps.nn.na1d import NeighborhoodAttention1D as _ModernNA1D
from natten_mps.nn.na2d import NeighborhoodAttention2D as _ModernNA2D
from natten_mps.utils.window import get_pb_start_vectorized


class NeighborhoodAttention1D(_ModernNA1D):
    def __init__(
        self,
        dim,
        kernel_size,
        dilation=1,
        num_heads=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            embed_dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            is_causal=False,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )


class NeighborhoodAttention2D(_ModernNA2D):
    def __init__(
        self,
        dim,
        kernel_size,
        dilation=1,
        num_heads=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            embed_dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            is_causal=False,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """
    query: [B, H, L, D]
    key:   [B, H, L, D]
    rpb:   [H, 2*K-1]
    return: [B, H, L, K]
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("natten1dqkrpb expects query/key shape [B, H, L, D].")
    if query.shape != key.shape:
        raise ValueError("query and key must have identical shapes.")

    bsz, heads, length, _ = query.shape
    q_hl = query.permute(0, 2, 1, 3)
    k_hl = key.permute(0, 2, 1, 3)

    logits = _na1d_qk(q_hl, k_hl, kernel_size=kernel_size, dilation=dilation).permute(0, 2, 1, 3)

    ksize = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    dil = dilation if isinstance(dilation, int) else dilation[0]

    if rpb.shape != (heads, 2 * ksize - 1):
        raise ValueError(f"rpb must have shape [{heads}, {2 * ksize - 1}] for 1D.")

    seq_indices = torch.arange(length, device=query.device, dtype=torch.long)
    pb_start = get_pb_start_vectorized(
        seq_indices,
        length=length,
        kernel_size=ksize,
        dilation=dil,
    )
    pb_indices = pb_start.unsqueeze(1) + torch.arange(ksize, device=query.device, dtype=torch.long).unsqueeze(0)
    bias = rpb[:, pb_indices]  # [H, L, K]
    return logits + bias.unsqueeze(0)


def natten1dav(attn, value, kernel_size, dilation):
    """
    attn:  [B, H, L, K]
    value: [B, H, L, D]
    return: [B, H, L, D]
    """
    if attn.ndim != 4 or value.ndim != 4:
        raise ValueError("natten1dav expects attn [B,H,L,K] and value [B,H,L,D].")

    attn_hl = attn.permute(0, 2, 1, 3)
    value_hl = value.permute(0, 2, 1, 3)
    out = _na1d_av(attn_hl, value_hl, kernel_size=kernel_size, dilation=dilation)
    return out.permute(0, 2, 1, 3)


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """
    query: [B, H, Hh, Hw, D]
    key:   [B, H, Hh, Hw, D]
    rpb:   [H, 2*Kh-1, 2*Kw-1]
    return: [B, H, Hh, Hw, Kh*Kw]
    """
    if query.ndim != 5 or key.ndim != 5:
        raise ValueError("natten2dqkrpb expects query/key shape [B, H, Hh, Hw, D].")
    if query.shape != key.shape:
        raise ValueError("query and key must have identical shapes.")

    bsz, heads, height, width, _ = query.shape
    q_hl = query.permute(0, 2, 3, 1, 4)
    k_hl = key.permute(0, 2, 3, 1, 4)

    logits = _na2d_qk(q_hl, k_hl, kernel_size=kernel_size, dilation=dilation).permute(0, 3, 1, 2, 4)

    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if isinstance(dilation, int):
        dil = (dilation, dilation)
    else:
        dil = tuple(dilation)

    if rpb.shape != (heads, 2 * kh - 1, 2 * kw - 1):
        raise ValueError(f"rpb must have shape [{heads}, {2 * kh - 1}, {2 * kw - 1}] for 2D.")

    h_indices = torch.arange(height, device=query.device, dtype=torch.long)
    w_indices = torch.arange(width, device=query.device, dtype=torch.long)
    pb_start_h = get_pb_start_vectorized(
        h_indices,
        length=height,
        kernel_size=kh,
        dilation=dil[0],
    )
    pb_start_w = get_pb_start_vectorized(
        w_indices,
        length=width,
        kernel_size=kw,
        dilation=dil[1],
    )
    pb_h = pb_start_h.unsqueeze(1) + torch.arange(kh, device=query.device, dtype=torch.long).unsqueeze(0)
    pb_w = pb_start_w.unsqueeze(1) + torch.arange(kw, device=query.device, dtype=torch.long).unsqueeze(0)
    pb_h_grid = pb_h[:, None, :, None].expand(height, width, kh, kw)
    pb_w_grid = pb_w[None, :, None, :].expand(height, width, kh, kw)

    bias = rpb[:, pb_h_grid, pb_w_grid].reshape(heads, height, width, kh * kw)
    return logits + bias.unsqueeze(0)


def natten2dav(attn, value, kernel_size, dilation):
    """
    attn:  [B, H, Hh, Hw, Kh*Kw]
    value: [B, H, Hh, Hw, D]
    return: [B, H, Hh, Hw, D]
    """
    if attn.ndim != 5 or value.ndim != 5:
        raise ValueError("natten2dav expects attn [B,H,Hh,Hw,K] and value [B,H,Hh,Hw,D].")

    attn_hl = attn.permute(0, 2, 3, 1, 4)
    value_hl = value.permute(0, 2, 3, 1, 4)
    out = _na2d_av(attn_hl, value_hl, kernel_size=kernel_size, dilation=dilation)
    return out.permute(0, 3, 1, 2, 4)


class _FlopHandler:
    def __call__(self, *args, **kwargs):
        return 0


def add_natten_handle(flop_counter):
    """Compatibility stub for fvcore flop counting."""
    return flop_counter


__all__ = [
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "natten1dqkrpb",
    "natten1dav",
    "natten2dqkrpb",
    "natten2dav",
    "_FlopHandler",
    "add_natten_handle",
]
