"""Neighborhood Attention for Apple Silicon — PyTorch MPS backend.

torch is imported lazily ("as needed"): ``import natten_mps`` (and reading
``natten_mps.__version__``) does NOT import torch. The torch-dependent machinery
is loaded on first access to a public attribute via PEP 562 ``__getattr__``, or
when one of the helper functions below is called. torch remains a hard runtime
dependency (declared in pyproject); it is simply not required merely to import
or introspect the package — which keeps build-time metadata resolution and
torch-free environments working.
"""
from __future__ import annotations

from natten_mps.version import __version__

__all__ = [
    "na1d",
    "na1d_varlen",
    "na2d",
    "na2d_varlen",
    "na3d",
    "na3d_varlen",
    "na1d_qk",
    "na1d_av",
    "na2d_qk",
    "na2d_av",
    "na3d_qk",
    "na3d_av",
    "merge_attentions",
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "has_mps",
    "has_metal",
    "get_backend",
    "set_backend",
    "get_support_matrix",
    "__version__",
]

# Public attribute -> submodule that defines it (imported lazily on first access).
_LAZY_ATTRS = {
    "na1d": "natten_mps.functional",
    "na1d_varlen": "natten_mps.functional",
    "na1d_qk": "natten_mps.functional",
    "na1d_av": "natten_mps.functional",
    "na2d": "natten_mps.functional",
    "na2d_varlen": "natten_mps.functional",
    "na2d_qk": "natten_mps.functional",
    "na2d_av": "natten_mps.functional",
    "na3d": "natten_mps.functional",
    "na3d_varlen": "natten_mps.functional",
    "na3d_qk": "natten_mps.functional",
    "na3d_av": "natten_mps.functional",
    "merge_attentions": "natten_mps.merge",
    "NeighborhoodAttention1D": "natten_mps.nn",
    "NeighborhoodAttention2D": "natten_mps.nn",
    "NeighborhoodAttention3D": "natten_mps.nn",
    "get_support_matrix": "natten_mps.support_matrix",
}


def __getattr__(name: str):  # PEP 562 — lazy attribute resolution
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    # Register the torch.library custom ops (for torch.compile) the first time
    # any public op/class is touched — preserves the old import-time behaviour,
    # just deferred until torch is actually needed.
    importlib.import_module("natten_mps._torch_ops")
    module = importlib.import_module(target)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)


def has_mps() -> bool:
    import torch

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def has_metal() -> bool:
    from natten_mps._core import metal as _metal

    return bool(_metal.is_available())


def get_backend() -> str:
    from natten_mps._core import ops

    return ops.get_backend()


def set_backend(name: str) -> None:
    from natten_mps._core import ops

    ops.set_backend(name)
