from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from natten_mps import functional as F
from natten_mps.utils.params import normalize_tuple_param


class NeighborhoodAttention3D(torch.nn.Module):
    """3-D Neighborhood Attention module.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of query attention heads.
        kernel_size: Neighborhood window size (scalar or 3-tuple).
        num_kv_heads: Number of key/value heads for GQA/MQA.  When ``None``
            (default) it equals ``num_heads`` (standard MHA).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        is_causal: Union[bool, Tuple[bool, ...]] = False,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else self.num_heads

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        self.kernel_size = normalize_tuple_param(kernel_size, 3, "kernel_size")
        self.stride = normalize_tuple_param(stride, 3, "stride")
        self.dilation = normalize_tuple_param(dilation, 3, "dilation")
        self.is_causal = tuple(bool(v) for v in normalize_tuple_param(is_causal, 3, "is_causal"))

        self.scale = float(qk_scale) if qk_scale is not None else self.head_dim ** -0.5

        self._use_gqa = self.num_kv_heads != self.num_heads
        if self._use_gqa:
            self.q_proj = torch.nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=qkv_bias)
            self.kv_proj = torch.nn.Linear(self.embed_dim, 2 * self.num_kv_heads * self.head_dim, bias=qkv_bias)
        else:
            self.qkv = torch.nn.Linear(self.embed_dim, self.embed_dim * 3, bias=qkv_bias)

        self.attn_drop_p = float(attn_drop)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x, spatial_sizes=None):
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, D, H, W, C]``.
            spatial_sizes: Optional ``[B, 3]`` int tensor of actual (D, H, W)
                per batch element for variable-length attention.  Positions
                beyond the per-sample sizes produce zero output.
        """
        if x.ndim != 5:
            raise ValueError("NeighborhoodAttention3D expects input shape [B, D, H, W, C].")

        bsz, depth, height, width, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Input channel dim ({channels}) must equal embed_dim ({self.embed_dim})."
            )

        if self._use_gqa:
            q = self.q_proj(x).reshape(bsz, depth, height, width, self.num_heads, self.head_dim)
            kv = self.kv_proj(x).reshape(bsz, depth, height, width, 2, self.num_kv_heads, self.head_dim)
            kv = kv.permute(4, 0, 1, 2, 3, 5, 6)
            k, v = kv.unbind(0)
        else:
            qkv = self.qkv(x).reshape(bsz, depth, height, width, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(4, 0, 1, 2, 3, 5, 6)
            q, k, v = qkv.unbind(0)

        if spatial_sizes is not None:
            out = F.na3d_varlen(
                q, k, v, spatial_sizes,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                scale=self.scale,
            )
        elif self.attn_drop_p > 0.0:
            logits = F.na3d_qk(q, k, kernel_size=self.kernel_size, dilation=self.dilation)
            default_scale = self.head_dim ** -0.5
            if self.scale != default_scale:
                logits = logits * (self.scale / default_scale)
            attn = self.attn_drop(torch.softmax(logits, dim=-1))
            out = F.na3d_av(attn, v, kernel_size=self.kernel_size, dilation=self.dilation)
        else:
            out = F.na3d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                is_causal=self.is_causal,
                scale=self.scale,
            )

        out = out.reshape(bsz, out.shape[1], out.shape[2], out.shape[3], self.embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
