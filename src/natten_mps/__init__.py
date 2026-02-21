from __future__ import annotations

import torch

from natten_mps._core import ops
from natten_mps._core import metal as _metal
from natten_mps._core import nanobind as _nanobind
from natten_mps.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk
from natten_mps.nn import NeighborhoodAttention1D, NeighborhoodAttention2D
from natten_mps.version import __version__


def has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def has_metal() -> bool:
    return bool(_metal.is_available())


def has_nanobind() -> bool:
    return bool(_nanobind.is_available())


def get_backend() -> str:
    return ops.get_backend()


def set_backend(name: str) -> None:
    ops.set_backend(name)


set_backend("auto")


__all__ = [
    "na1d",
    "na2d",
    "na1d_qk",
    "na1d_av",
    "na2d_qk",
    "na2d_av",
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "has_mps",
    "has_metal",
    "has_nanobind",
    "get_backend",
    "set_backend",
    "__version__",
]
