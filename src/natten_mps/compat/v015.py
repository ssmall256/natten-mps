from __future__ import annotations

import torch

from natten_mps.functional import na1d as _na1d
from natten_mps.functional import na1d_av, na1d_qk, na2d as _na2d
from natten_mps.functional import na2d_av, na2d_qk

from .v014 import (
    NeighborhoodAttention1D,
    NeighborhoodAttention2D,
    _FlopHandler,
    add_natten_handle,
    natten1dav,
    natten1dqkrpb,
    natten2dav,
    natten2dqkrpb,
)


class NeighborhoodAttention3D(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("3D neighborhood attention is not supported in natten-mps")



def na1d(query, key, value, kernel_size, dilation=1, is_causal=False):
    return _na1d(
        query,
        key,
        value,
        kernel_size=kernel_size,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
    )


def na2d(query, key, value, kernel_size, dilation=1, is_causal=False):
    return _na2d(
        query,
        key,
        value,
        kernel_size=kernel_size,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
    )


def has_cuda():
    return False


def has_mps():
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def has_gemm():
    return False


def has_fna():
    return False


__all__ = [
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "natten1dqkrpb",
    "natten1dav",
    "natten2dqkrpb",
    "natten2dav",
    "na1d",
    "na2d",
    "na1d_qk",
    "na1d_av",
    "na2d_qk",
    "na2d_av",
    "has_cuda",
    "has_mps",
    "has_gemm",
    "has_fna",
    "_FlopHandler",
    "add_natten_handle",
]
