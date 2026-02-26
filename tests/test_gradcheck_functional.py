"""Numerical gradient checks (torch.autograd.gradcheck) on CPU with float64.

These complement the MPS-device backward tests by verifying analytical
gradients against finite-difference approximations at double precision.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from natten_mps.functional import na1d, na1d_av, na1d_qk, na2d, na3d


_DEVICE = "cpu"
_DTYPE = torch.float64
_EPS = 1e-6
_ATOL = 1e-4
_RTOL = 1e-3


def _randn(*shape):
    return torch.randn(*shape, device=_DEVICE, dtype=_DTYPE, requires_grad=True)


# -------------------------------------------------------------------
# Fused forward
# -------------------------------------------------------------------


class TestFusedGradcheck:

    def test_na1d_gradcheck(self):
        q, k, v = _randn(1, 8, 2, 4), _randn(1, 8, 2, 4), _randn(1, 8, 2, 4)

        def fn(q_in, k_in, v_in):
            return na1d(q_in, k_in, v_in, kernel_size=3)

        assert gradcheck(fn, (q, k, v), eps=_EPS, atol=_ATOL, rtol=_RTOL)

    def test_na2d_gradcheck(self):
        q = _randn(1, 4, 4, 2, 4)
        k = _randn(1, 4, 4, 2, 4)
        v = _randn(1, 4, 4, 2, 4)

        def fn(q_in, k_in, v_in):
            return na2d(q_in, k_in, v_in, kernel_size=3)

        assert gradcheck(fn, (q, k, v), eps=_EPS, atol=_ATOL, rtol=_RTOL)

    def test_na3d_gradcheck(self):
        q = _randn(1, 4, 4, 4, 2, 4)
        k = _randn(1, 4, 4, 4, 2, 4)
        v = _randn(1, 4, 4, 4, 2, 4)

        def fn(q_in, k_in, v_in):
            return na3d(q_in, k_in, v_in, kernel_size=3)

        assert gradcheck(fn, (q, k, v), eps=_EPS, atol=_ATOL, rtol=_RTOL)


# -------------------------------------------------------------------
# Split QK / AV
# -------------------------------------------------------------------


class TestSplitGradcheck:

    def test_na1d_qk_gradcheck(self):
        q, k = _randn(1, 8, 2, 4), _randn(1, 8, 2, 4)

        def fn(q_in, k_in):
            return na1d_qk(q_in, k_in, kernel_size=3)

        assert gradcheck(fn, (q, k), eps=_EPS, atol=_ATOL, rtol=_RTOL)

    def test_na1d_av_gradcheck(self):
        # AV takes attention weights [B, L, H, K] and values
        attn = torch.randn(1, 8, 2, 3, device=_DEVICE, dtype=_DTYPE,
                           requires_grad=True).softmax(dim=-1).detach().requires_grad_(True)
        v = _randn(1, 8, 2, 4)

        def fn(a_in, v_in):
            return na1d_av(a_in, v_in, kernel_size=3)

        assert gradcheck(fn, (attn, v), eps=_EPS, atol=_ATOL, rtol=_RTOL)


# -------------------------------------------------------------------
# Fused with features (causal, dilation)
# -------------------------------------------------------------------


class TestFeatureGradcheck:

    def test_na1d_causal_gradcheck(self):
        q, k, v = _randn(1, 8, 2, 4), _randn(1, 8, 2, 4), _randn(1, 8, 2, 4)

        def fn(q_in, k_in, v_in):
            return na1d(q_in, k_in, v_in, kernel_size=3, is_causal=True)

        assert gradcheck(fn, (q, k, v), eps=_EPS, atol=_ATOL, rtol=_RTOL)

    def test_na1d_dilation_gradcheck(self):
        q, k, v = _randn(1, 12, 2, 4), _randn(1, 12, 2, 4), _randn(1, 12, 2, 4)

        def fn(q_in, k_in, v_in):
            return na1d(q_in, k_in, v_in, kernel_size=3, dilation=2)

        assert gradcheck(fn, (q, k, v), eps=_EPS, atol=_ATOL, rtol=_RTOL)

    def test_na2d_causal_gradcheck(self):
        q = _randn(1, 4, 4, 2, 4)
        k = _randn(1, 4, 4, 2, 4)
        v = _randn(1, 4, 4, 2, 4)

        def fn(q_in, k_in, v_in):
            return na2d(q_in, k_in, v_in, kernel_size=3, is_causal=True)

        assert gradcheck(fn, (q, k, v), eps=_EPS, atol=_ATOL, rtol=_RTOL)
