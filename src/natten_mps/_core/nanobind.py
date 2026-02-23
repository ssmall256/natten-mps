"""
Tier 2: Nanobind Metal-extension backend for natten-mps.

Attempts to import the compiled _nanobind_ext module. If available,
delegates kernel dispatch to precompiled Metal shaders loaded via
the Metal API. Otherwise, all functions raise NotImplementedError.
"""

try:
    from natten_mps._core._nanobind_ext import is_available as _ext_available
    _AVAILABLE = _ext_available()
except ImportError:
    _AVAILABLE = False


def is_available():
    return _AVAILABLE


def _not_implemented() -> NotImplementedError:
    return NotImplementedError(
        "Nanobind backend is not yet available. "
        "Use set_backend('pure') or install natten-mps with nanobind extension support: "
        "pip install -e '.[nanobind]'"
    )


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    raise _not_implemented()


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    raise _not_implemented()


def na1d_qk_forward(q, k, kernel_size, dilation, stride=(1,), is_causal=(False,), scale=None):
    raise _not_implemented()


def na1d_av_forward(attn, v, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    raise _not_implemented()


def na2d_qk_forward(q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False), scale=None):
    raise _not_implemented()


def na2d_av_forward(attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    raise _not_implemented()


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    raise _not_implemented()


def na3d_qk_forward(q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False), scale=None):
    raise _not_implemented()


def na3d_av_forward(attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    raise _not_implemented()


# Backward stubs — return None (use re-differentiation)

def na1d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return None

def na1d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return None

def na2d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return None

def na2d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return None

def na3d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return None

def na3d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return None
