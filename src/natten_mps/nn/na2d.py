from __future__ import annotations

import torch

from natten_mps import functional as F
from natten_mps.utils.params import normalize_tuple_param


class NeighborhoodAttention2D(torch.nn.Module):
    """2-D Neighborhood Attention module."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        kernel_size,
        stride=1,
        dilation=1,
        is_causal=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        self.kernel_size = normalize_tuple_param(kernel_size, 2, "kernel_size")
        self.stride = normalize_tuple_param(stride, 2, "stride")
        self.dilation = normalize_tuple_param(dilation, 2, "dilation")
        self.is_causal = tuple(bool(v) for v in normalize_tuple_param(is_causal, 2, "is_causal"))

        self.scale = float(qk_scale) if qk_scale is not None else self.head_dim ** -0.5

        self.qkv = torch.nn.Linear(self.embed_dim, self.embed_dim * 3, bias=qkv_bias)
        self.attn_drop_p = float(attn_drop)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("NeighborhoodAttention2D expects input shape [B, H, W, C].")

        bsz, height, width, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Input channel dim ({channels}) must equal embed_dim ({self.embed_dim})."
            )

        qkv = self.qkv(x).reshape(bsz, height, width, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 2, 4, 5)
        q, k, v = qkv.unbind(0)

        if self.attn_drop_p > 0.0:
            logits = F.na2d_qk(q, k, kernel_size=self.kernel_size, dilation=self.dilation)
            default_scale = self.head_dim ** -0.5
            if self.scale != default_scale:
                logits = logits * (self.scale / default_scale)
            attn = self.attn_drop(torch.softmax(logits, dim=-1))
            out = F.na2d_av(attn, v, kernel_size=self.kernel_size, dilation=self.dilation)
        else:
            out = F.na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                is_causal=self.is_causal,
                scale=self.scale,
            )

        out = out.reshape(bsz, out.shape[1], out.shape[2], channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
