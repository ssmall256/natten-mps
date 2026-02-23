"""Compare against global attention reference when kernel covers full spatial extent."""

import pytest
import torch

from natten_mps.functional import na1d, na2d, na3d


class TestGlobalAttention1D:
    @pytest.mark.parametrize("length", [4, 8, 12])
    def test_full_kernel_matches_global(self, length):
        B, heads, dim = 1, 2, 8
        q = torch.randn(B, length, heads, dim)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_na = na1d(q, k, v, kernel_size=length)

        scale = dim ** -0.5
        logits = torch.einsum("bihd,bjhd->bhij", q, k) * scale
        attn = torch.softmax(logits, dim=-1)
        out_global = torch.einsum("bhij,bjhd->bihd", attn, v)

        torch.testing.assert_close(out_na, out_global, atol=1e-6, rtol=1e-6)


class TestGlobalAttention2D:
    @pytest.mark.parametrize("size", [4, 6])
    def test_full_kernel_matches_global(self, size):
        B, heads, dim = 1, 2, 8
        q = torch.randn(B, size, size, heads, dim)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_na = na2d(q, k, v, kernel_size=size)

        scale = dim ** -0.5
        q_flat = q.reshape(B, size * size, heads, dim)
        k_flat = k.reshape(B, size * size, heads, dim)
        v_flat = v.reshape(B, size * size, heads, dim)
        logits = torch.einsum("bihd,bjhd->bhij", q_flat, k_flat) * scale
        attn = torch.softmax(logits, dim=-1)
        out_global = torch.einsum("bhij,bjhd->bihd", attn, v_flat).reshape(B, size, size, heads, dim)

        torch.testing.assert_close(out_na, out_global, atol=1e-5, rtol=1e-5)


class TestGlobalAttention3D:
    def test_full_kernel_matches_global(self):
        B, D, H, W, heads, dim = 1, 3, 3, 3, 2, 8
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_na = na3d(q, k, v, kernel_size=3)

        scale = dim ** -0.5
        n = D * H * W
        q_flat = q.reshape(B, n, heads, dim)
        k_flat = k.reshape(B, n, heads, dim)
        v_flat = v.reshape(B, n, heads, dim)
        logits = torch.einsum("bihd,bjhd->bhij", q_flat, k_flat) * scale
        attn = torch.softmax(logits, dim=-1)
        out_global = torch.einsum("bhij,bjhd->bihd", attn, v_flat).reshape(B, D, H, W, heads, dim)

        torch.testing.assert_close(out_na, out_global, atol=1e-5, rtol=1e-5)
