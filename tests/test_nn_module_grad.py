"""End-to-end gradient flow through nn.Module classes."""

import pytest
import torch

from natten_mps.nn import (
    NeighborhoodAttention1D,
    NeighborhoodAttention2D,
    NeighborhoodAttention3D,
)


class TestNA1DModuleGrad:
    def test_gradient_flows_to_input(self):
        mod = NeighborhoodAttention1D(embed_dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 8, 16, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flows_to_parameters(self):
        mod = NeighborhoodAttention1D(embed_dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 8, 16)
        out = mod(x)
        out.sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


class TestNA2DModuleGrad:
    def test_gradient_flows_to_input(self):
        mod = NeighborhoodAttention2D(embed_dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 6, 6, 16, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flows_to_parameters(self):
        mod = NeighborhoodAttention2D(embed_dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 6, 6, 16)
        out = mod(x)
        out.sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


class TestNA3DModuleGrad:
    def test_gradient_flows_to_input(self):
        mod = NeighborhoodAttention3D(embed_dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 4, 4, 4, 16, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flows_to_parameters(self):
        mod = NeighborhoodAttention3D(embed_dim=16, num_heads=2, kernel_size=3)
        x = torch.randn(1, 4, 4, 4, 16)
        out = mod(x)
        out.sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


# ===================================================================
# GQA / MQA nn module tests
# ===================================================================


class TestNA1DModuleGQA:
    def test_gqa_forward_shape(self):
        mod = NeighborhoodAttention1D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 8, 16)
        out = mod(x)
        assert out.shape == x.shape

    def test_mqa_forward_shape(self):
        mod = NeighborhoodAttention1D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=1
        )
        x = torch.randn(1, 8, 16)
        out = mod(x)
        assert out.shape == x.shape

    def test_gqa_gradient_flows_to_input(self):
        mod = NeighborhoodAttention1D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 8, 16, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape

    def test_gqa_gradient_flows_to_parameters(self):
        mod = NeighborhoodAttention1D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 8, 16)
        out = mod(x)
        out.sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gqa_has_separate_projections(self):
        mod = NeighborhoodAttention1D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        assert hasattr(mod, "q_proj")
        assert hasattr(mod, "kv_proj")
        assert not hasattr(mod, "qkv")

    def test_indivisible_kv_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            NeighborhoodAttention1D(
                embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=3
            )


class TestNA2DModuleGQA:
    def test_gqa_forward_shape(self):
        mod = NeighborhoodAttention2D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 6, 6, 16)
        out = mod(x)
        assert out.shape == x.shape

    def test_gqa_gradient_flows_to_input(self):
        mod = NeighborhoodAttention2D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 6, 6, 16, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape

    def test_gqa_gradient_flows_to_parameters(self):
        mod = NeighborhoodAttention2D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 6, 6, 16)
        out = mod(x)
        out.sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


class TestNA3DModuleGQA:
    def test_gqa_forward_shape(self):
        mod = NeighborhoodAttention3D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 4, 4, 4, 16)
        out = mod(x)
        assert out.shape == x.shape

    def test_gqa_gradient_flows_to_input(self):
        mod = NeighborhoodAttention3D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 4, 4, 4, 16, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape

    def test_gqa_gradient_flows_to_parameters(self):
        mod = NeighborhoodAttention3D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
        )
        x = torch.randn(1, 4, 4, 4, 16)
        out = mod(x)
        out.sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
