from __future__ import annotations

import torch

from natten_mps.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk
from natten_mps.nn import NeighborhoodAttention1D, NeighborhoodAttention2D


def has_cuda():
    return False


def has_mps():
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def has_fna():
    return False


__all__ = [
    "na1d",
    "na2d",
    "na1d_qk",
    "na1d_av",
    "na2d_qk",
    "na2d_av",
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "has_cuda",
    "has_mps",
    "has_fna",
]
