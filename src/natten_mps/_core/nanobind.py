"""
Tier 2: Nanobind Metal-extension backend for natten-mps.
"""

_AVAILABLE = False


def is_available():
    return _AVAILABLE


def _not_implemented() -> NotImplementedError:
    return NotImplementedError(
        "Nanobind backend is not yet available. "
        "Use set_backend('pure') or install natten-mps with nanobind extension support."
    )


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    raise _not_implemented()


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    raise _not_implemented()


def na1d_qk_forward(q, k, kernel_size, dilation):
    raise _not_implemented()


def na1d_av_forward(attn, v, kernel_size, dilation):
    raise _not_implemented()


def na2d_qk_forward(q, k, kernel_size, dilation):
    raise _not_implemented()


def na2d_av_forward(attn, v, kernel_size, dilation):
    raise _not_implemented()
