"""Tests for variable-length 3D neighborhood attention."""

import pytest
import torch
import numpy as np

from natten_mps.functional import na3d, na3d_varlen


def _varlen_reference(q, k, v, spatial_sizes, kernel_size, dilation, scale):
    """Per-sample reference: slice each batch element and run na3d independently."""
    B = q.shape[0]
    out = torch.zeros_like(q)
    for b in range(B):
        D_b = int(spatial_sizes[b, 0].item())
        H_b = int(spatial_sizes[b, 1].item())
        W_b = int(spatial_sizes[b, 2].item())
        out_b = na3d(
            q[b:b+1, :D_b, :H_b, :W_b], k[b:b+1, :D_b, :H_b, :W_b], v[b:b+1, :D_b, :H_b, :W_b],
            kernel_size=kernel_size, dilation=dilation, scale=scale,
        )
        out[b, :D_b, :H_b, :W_b] = out_b[0]
    return out


class TestVarlen3DForward:
    """Forward-pass correctness tests."""

    def test_uniform_sizes(self):
        """All spatial_sizes == max must match na3d exactly."""
        B, D, H, W, heads, dim, K = 2, 4, 4, 4, 2, 8, 3
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn(B, D, H, W, heads, dim)
        v = torch.randn(B, D, H, W, heads, dim)
        spatial_sizes = torch.tensor([[D, H, W], [D, H, W]], dtype=torch.int32)

        out_varlen = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_na3d = na3d(q, k, v, kernel_size=K)
        torch.testing.assert_close(out_varlen, out_na3d, atol=1e-5, rtol=1e-5)

    def test_mixed_sizes(self):
        """B=2 with different spatial sizes per sample."""
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = torch.randn(B, D_max, H_max, W_max, heads, dim)
        k = torch.randn(B, D_max, H_max, W_max, heads, dim)
        v = torch.randn(B, D_max, H_max, W_max, heads, dim)
        spatial_sizes = torch.tensor([[6, 6, 6], [4, 5, 3]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_minimum_sizes(self):
        """All spatial dims == kernel_size (smallest valid)."""
        B, K, heads, dim = 2, 3, 2, 8
        D_max, H_max, W_max = 6, 6, 6
        q = torch.randn(B, D_max, H_max, W_max, heads, dim)
        k = torch.randn(B, D_max, H_max, W_max, heads, dim)
        v = torch.randn(B, D_max, H_max, W_max, heads, dim)
        spatial_sizes = torch.tensor([[K, K, K], [K, K, K]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_single_batch(self):
        """B=1 should work."""
        B, D, H, W, heads, dim, K = 1, 6, 6, 6, 2, 8, 3
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn(B, D, H, W, heads, dim)
        v = torch.randn(B, D, H, W, heads, dim)
        spatial_sizes = torch.tensor([[4, 5, 3]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_per_sample_parity(self):
        """Each slice must match an independent na3d call."""
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = torch.randn(B, D_max, H_max, W_max, heads, dim)
        k = torch.randn(B, D_max, H_max, W_max, heads, dim)
        v = torch.randn(B, D_max, H_max, W_max, heads, dim)
        spatial_sizes = torch.tensor([[6, 6, 6], [3, 4, 5]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        for b in range(B):
            D_b, H_b, W_b = [int(spatial_sizes[b, d].item()) for d in range(3)]
            expected = na3d(q[b:b+1, :D_b, :H_b, :W_b], k[b:b+1, :D_b, :H_b, :W_b],
                           v[b:b+1, :D_b, :H_b, :W_b], kernel_size=K)
            torch.testing.assert_close(out[b, :D_b, :H_b, :W_b], expected[0], atol=1e-5, rtol=1e-5,
                                       msg=f"Mismatch at batch {b}")

    def test_padding_positions_zero(self):
        """Output beyond spatial_sizes must be zero."""
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = torch.randn(B, D_max, H_max, W_max, heads, dim)
        k = torch.randn(B, D_max, H_max, W_max, heads, dim)
        v = torch.randn(B, D_max, H_max, W_max, heads, dim)
        spatial_sizes = torch.tensor([[4, 3, 5], [3, 3, 3]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_np = out.numpy()
        # Batch 0: D >= 4 should be zero
        assert (out_np[0, 4:] == 0).all(), "Batch 0 depth padding should be zero"
        assert (out_np[0, :4, 3:] == 0).all(), "Batch 0 height padding should be zero"
        assert (out_np[0, :4, :3, 5:] == 0).all(), "Batch 0 width padding should be zero"

    def test_custom_scale(self):
        """Explicit scale parameter should be honored."""
        B, D, H, W, heads, dim, K = 2, 4, 4, 4, 2, 8, 3
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn(B, D, H, W, heads, dim)
        v = torch.randn(B, D, H, W, heads, dim)
        spatial_sizes = torch.tensor([[4, 4, 4], [3, 3, 3]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K, scale=0.1)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, 0.1)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


class TestVarlen3DValidation:
    """Input validation tests."""

    def test_rejects_small_spatial(self):
        """spatial_size < kernel_size should raise ValueError."""
        B, D, H, W, heads, dim, K = 2, 6, 6, 6, 2, 8, 5
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn(B, D, H, W, heads, dim)
        v = torch.randn(B, D, H, W, heads, dim)
        spatial_sizes = torch.tensor([[6, 6, 6], [4, 6, 6]], dtype=torch.int32)

        with pytest.raises(ValueError, match="kernel_size"):
            na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_exceeding_max(self):
        """spatial_size > max should raise ValueError."""
        B, D, H, W, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn(B, D, H, W, heads, dim)
        v = torch.randn(B, D, H, W, heads, dim)
        spatial_sizes = torch.tensor([[6, 6, 6], [6, 7, 6]], dtype=torch.int32)

        with pytest.raises(ValueError, match="max_spatial"):
            na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_wrong_shape(self):
        """spatial_sizes shape must be (B, 3)."""
        B, D, H, W, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = torch.randn(B, D, H, W, heads, dim)
        k = torch.randn(B, D, H, W, heads, dim)
        v = torch.randn(B, D, H, W, heads, dim)
        spatial_sizes = torch.tensor([[6, 6], [6, 6]], dtype=torch.int32)  # [B,2] instead of [B,3]

        with pytest.raises(ValueError):
            na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)


class TestVarlen3DBackward:
    """Backward gradient tests."""

    def test_backward_gradients(self):
        """Backward gradients match per-sample reference."""
        B, D_max, H_max, W_max, heads, dim, K = 2, 4, 4, 4, 2, 8, 3
        q = torch.randn(B, D_max, H_max, W_max, heads, dim, requires_grad=True)
        k = torch.randn(B, D_max, H_max, W_max, heads, dim, requires_grad=True)
        v = torch.randn(B, D_max, H_max, W_max, heads, dim, requires_grad=True)
        spatial_sizes = torch.tensor([[4, 4, 4], [3, 3, 3]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        loss = out.sum()
        loss.backward()

        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

        for b in range(B):
            D_b, H_b, W_b = [int(spatial_sizes[b, d].item()) for d in range(3)]
            q_b = q.data[b:b+1, :D_b, :H_b, :W_b].detach().requires_grad_(True)
            k_b = k.data[b:b+1, :D_b, :H_b, :W_b].detach().requires_grad_(True)
            v_b = v.data[b:b+1, :D_b, :H_b, :W_b].detach().requires_grad_(True)
            out_b = na3d(q_b, k_b, v_b, kernel_size=K)
            out_b.sum().backward()
            torch.testing.assert_close(dq[b, :D_b, :H_b, :W_b], q_b.grad[0], atol=1e-4, rtol=1e-4,
                                       msg=f"dq mismatch at batch {b}")
            torch.testing.assert_close(dk[b, :D_b, :H_b, :W_b], k_b.grad[0], atol=1e-4, rtol=1e-4,
                                       msg=f"dk mismatch at batch {b}")
            torch.testing.assert_close(dv[b, :D_b, :H_b, :W_b], v_b.grad[0], atol=1e-4, rtol=1e-4,
                                       msg=f"dv mismatch at batch {b}")

    def test_backward_padding_zero(self):
        """Gradients at padding positions must be zero."""
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = torch.randn(B, D_max, H_max, W_max, heads, dim, requires_grad=True)
        k = torch.randn(B, D_max, H_max, W_max, heads, dim, requires_grad=True)
        v = torch.randn(B, D_max, H_max, W_max, heads, dim, requires_grad=True)
        spatial_sizes = torch.tensor([[4, 3, 5], [3, 3, 3]], dtype=torch.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out.sum().backward()

        assert (q.grad[0, 4:] == 0).all(), "dq depth padding should be zero"
        assert (q.grad[0, :4, 3:] == 0).all(), "dq height padding should be zero"
        assert (v.grad[1, 3:] == 0).all(), "dv depth padding should be zero"
