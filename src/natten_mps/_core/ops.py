"""
Backend dispatch for natten-mps.

Each backend module must provide these functions:
  - na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale) -> output
  - na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale) -> output
  - na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale) -> output
  - na1d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale) -> attn_weights
  - na1d_av_forward(attn, v, kernel_size, dilation, stride, is_causal) -> output
  - na2d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale) -> attn_weights
  - na2d_av_forward(attn, v, kernel_size, dilation, stride, is_causal) -> output
  - na3d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale) -> attn_weights
  - na3d_av_forward(attn, v, kernel_size, dilation, stride, is_causal) -> output
"""

from __future__ import annotations

import os

from . import metal, nanobind, pure

_REQUIRED_BACKEND_FUNCTIONS = (
    "na1d_forward",
    "na2d_forward",
    "na3d_forward",
    "na1d_qk_forward",
    "na1d_av_forward",
    "na2d_qk_forward",
    "na2d_av_forward",
    "na3d_qk_forward",
    "na3d_av_forward",
)

_ACTIVE_BACKEND = os.environ.get("NATTEN_BACKEND", "auto")
_BACKEND_REGISTRY = {}


def register_backend(name, module):
    for fn_name in _REQUIRED_BACKEND_FUNCTIONS:
        if not hasattr(module, fn_name):
            raise ValueError(f"Backend '{name}' does not define required function '{fn_name}'.")
    _BACKEND_REGISTRY[name] = module


def _backend_available(name: str) -> bool:
    module = _BACKEND_REGISTRY[name]
    is_available_fn = getattr(module, "is_available", None)
    if callable(is_available_fn):
        return bool(is_available_fn())
    return True


def set_backend(name):
    global _ACTIVE_BACKEND

    if name == "auto":
        _ACTIVE_BACKEND = "auto"
        return

    if name not in _BACKEND_REGISTRY:
        valid = ["auto", *_BACKEND_REGISTRY.keys()]
        raise ValueError(f"Unknown backend '{name}'. Expected one of: {valid}.")

    if not _backend_available(name):
        raise NotImplementedError(
            f"Backend '{name}' is not available in this build. "
            "Use set_backend('pure') or set_backend('auto')."
        )

    _ACTIVE_BACKEND = name


def get_backend():
    if _ACTIVE_BACKEND == "auto":
        return _resolve_backend()
    return _ACTIVE_BACKEND


def _resolve_backend():
    for name in ("nanobind", "metal", "pure"):
        if name in _BACKEND_REGISTRY and _backend_available(name):
            return name
    raise RuntimeError("No valid backend is registered for natten-mps.")


def _active_module():
    return _BACKEND_REGISTRY[get_backend()]


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _active_module().na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _active_module().na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na1d_qk_forward(q, k, kernel_size, dilation, stride=(1,), is_causal=(False,), scale=None):
    return _active_module().na1d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale)


def na1d_av_forward(attn, v, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return _active_module().na1d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)


def na2d_qk_forward(q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False), scale=None):
    return _active_module().na2d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale)


def na2d_av_forward(attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return _active_module().na2d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _active_module().na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na3d_qk_forward(q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False), scale=None):
    return _active_module().na3d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale)


def na3d_av_forward(attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return _active_module().na3d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)


# ---------------------------------------------------------------------------
# Backward dispatch — returns None if Metal backward unavailable
# ---------------------------------------------------------------------------


def na1d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return _active_module().na1d_qk_backward(d_attn, q, k, kernel_size, dilation, stride, is_causal)


def na1d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return _active_module().na1d_av_backward(d_out, attn, v, kernel_size, dilation, stride, is_causal)


def na2d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return _active_module().na2d_qk_backward(d_attn, q, k, kernel_size, dilation, stride, is_causal)


def na2d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return _active_module().na2d_av_backward(d_out, attn, v, kernel_size, dilation, stride, is_causal)


def na3d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return _active_module().na3d_qk_backward(d_attn, q, k, kernel_size, dilation, stride, is_causal)


def na3d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return _active_module().na3d_av_backward(d_out, attn, v, kernel_size, dilation, stride, is_causal)


register_backend("pure", pure)
register_backend("metal", metal)
register_backend("nanobind", nanobind)

if _ACTIVE_BACKEND not in {"auto", *list(_BACKEND_REGISTRY.keys())}:
    _ACTIVE_BACKEND = "auto"
