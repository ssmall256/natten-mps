"""Expanded 3D functional tests with various kernel/dilation/stride combos."""

import pytest
import torch

from natten_mps.functional import na3d, na3d_av, na3d_qk


class TestNA3DShapes:
    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_basic_shapes(self, kernel_size):
        q = torch.randn(1, 6, 6, 6, 2, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = na3d(q, k, v, kernel_size)
        assert out.shape == q.shape

    @pytest.mark.parametrize("dilation", [1, 2])
    def test_with_dilation(self, dilation):
        q = torch.randn(1, 8, 8, 8, 2, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = na3d(q, k, v, 3, dilation=dilation)
        assert out.shape == q.shape

    def test_stride_reduces_spatial(self):
        q = torch.randn(1, 8, 8, 8, 2, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = na3d(q, k, v, 3, stride=2)
        assert out.shape == (1, 4, 4, 4, 2, 8)

    def test_non_uniform_kernel(self):
        q = torch.randn(1, 6, 6, 6, 2, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = na3d(q, k, v, (3, 5, 3))
        assert out.shape == q.shape


class TestNA3DSplitOps:
    def test_qk_shape(self):
        q = torch.randn(1, 4, 4, 4, 2, 8)
        k = torch.randn_like(q)
        logits = na3d_qk(q, k, 3)
        assert logits.shape == (1, 4, 4, 4, 2, 27)

    def test_av_shape(self):
        attn = torch.softmax(torch.randn(1, 4, 4, 4, 2, 27), dim=-1)
        v = torch.randn(1, 4, 4, 4, 2, 8)
        out = na3d_av(attn, v, 3)
        assert out.shape == v.shape

    def test_split_matches_fused(self):
        q = torch.randn(1, 4, 4, 4, 2, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_fused = na3d(q, k, v, 3)
        scale = q.shape[-1] ** -0.5
        logits = na3d_qk(q, k, 3)
        attn = torch.softmax(logits * scale, dim=-1)
        out_split = na3d_av(attn, v, 3)
        torch.testing.assert_close(out_fused, out_split, atol=1e-6, rtol=1e-6)


class TestNA3DGlobalAttention:
    def test_full_kernel_equals_global(self):
        """When kernel covers full volume, NA3D should equal global attention."""
        B, D, H, W, heads, dim = 1, 4, 4, 4, 2, 8
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_na = na3d(q, k, v, kernel_size=(D, H, W))

        # Manual global attention
        scale = dim ** -0.5
        q_flat = q.reshape(B, D * H * W, heads, dim)
        k_flat = k.reshape(B, D * H * W, heads, dim)
        v_flat = v.reshape(B, D * H * W, heads, dim)
        logits = torch.einsum("bihd,bjhd->bhij", q_flat, k_flat) * scale
        attn = torch.softmax(logits, dim=-1)
        out_global = torch.einsum("bhij,bjhd->bihd", attn, v_flat).reshape(B, D, H, W, heads, dim)

        torch.testing.assert_close(out_na, out_global, atol=1e-5, rtol=1e-5)
