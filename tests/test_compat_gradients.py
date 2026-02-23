"""Gradient flow through compat layer modules."""

import torch

from natten_mps.compat import for_version


class TestV014Gradients:
    def test_na1d_gradient_flows(self):
        mod_v014 = for_version("0.14.0")
        na1d = mod_v014.NeighborhoodAttention1D(dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 8, 16, requires_grad=True)
        out = na1d(x)
        out.sum().backward()
        assert x.grad is not None

    def test_na2d_gradient_flows(self):
        mod_v014 = for_version("0.14.0")
        na2d = mod_v014.NeighborhoodAttention2D(dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 6, 6, 16, requires_grad=True)
        out = na2d(x)
        out.sum().backward()
        assert x.grad is not None


class TestV017Gradients:
    def test_na1d_gradient_flows(self):
        mod_v017 = for_version("0.17.0")
        na1d = mod_v017.NeighborhoodAttention1D(dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 8, 16, requires_grad=True)
        out = na1d(x)
        out.sum().backward()
        assert x.grad is not None

    def test_na2d_gradient_flows(self):
        mod_v017 = for_version("0.17.0")
        na2d = mod_v017.NeighborhoodAttention2D(dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 6, 6, 16, requires_grad=True)
        out = na2d(x)
        out.sum().backward()
        assert x.grad is not None


class TestV020Gradients:
    def test_na1d_functional_gradient(self):
        mod_v020 = for_version("0.20.0")
        q = torch.randn(1, 8, 2, 8, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        out = mod_v020.na1d(q, k, v, kernel_size=3)
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_na2d_functional_gradient(self):
        mod_v020 = for_version("0.20.0")
        q = torch.randn(1, 6, 6, 2, 8, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        out = mod_v020.na2d(q, k, v, kernel_size=3)
        out.sum().backward()
        assert q.grad is not None
