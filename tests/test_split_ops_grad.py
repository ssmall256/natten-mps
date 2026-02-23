"""Gradient checks on individual split QK/AV ops."""

import pytest
import torch

from natten_mps.functional import (
    na1d_av,
    na1d_qk,
    na2d_av,
    na2d_qk,
    na3d_av,
    na3d_qk,
)


class TestNA1DQKGrad:
    def test_gradcheck(self):
        q = torch.randn(1, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda q, k: na1d_qk(q, k, 3), (q, k))

    def test_gradcheck_with_dilation(self):
        q = torch.randn(1, 8, 2, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 8, 2, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda q, k: na1d_qk(q, k, 3, dilation=2), (q, k))

    def test_gradcheck_with_scale(self):
        q = torch.randn(1, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda q, k: na1d_qk(q, k, 3, scale=0.5), (q, k))


class TestNA1DAVGrad:
    def test_gradcheck(self):
        attn = torch.randn(1, 6, 2, 3, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda a, v: na1d_av(a, v, 3), (attn, v))


class TestNA2DQKGrad:
    def test_gradcheck(self):
        q = torch.randn(1, 6, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 6, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda q, k: na2d_qk(q, k, 3), (q, k))

    def test_gradcheck_with_scale(self):
        q = torch.randn(1, 6, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 6, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda q, k: na2d_qk(q, k, 3, scale=0.25), (q, k))


class TestNA2DAVGrad:
    def test_gradcheck(self):
        attn = torch.randn(1, 6, 6, 2, 9, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 6, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda a, v: na2d_av(a, v, 3), (attn, v))


class TestNA3DQKGrad:
    def test_gradcheck(self):
        q = torch.randn(1, 4, 4, 4, 1, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 1, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda q, k: na3d_qk(q, k, 3), (q, k), atol=1e-4)


class TestNA3DAVGrad:
    def test_gradcheck(self):
        attn = torch.randn(1, 4, 4, 4, 1, 27, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 4, 4, 4, 1, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda a, v: na3d_av(a, v, 3), (attn, v), atol=1e-4)
