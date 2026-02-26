"""Tests for variable-length 2D neighborhood attention."""

import pytest
import torch
import numpy as np

from natten_mps.functional import na2d, na2d_varlen


def _varlen_reference(q, k, v, spatial_sizes, kernel_size, dilation, scale):
    """Per-sample reference: slice each batch element and run na2d independently."""
    B = q.shape[0]
    H_max, W_max, heads, D = q.shape[1], q.shape[2], q.shape[3], q.shape[4]
    out = torch.zeros_like(q)
    for b in range(B):
        H_b = int(spatial_sizes[b, 0].item())
        W_b = int(spatial_sizes[b, 1].item())
        out_b = na2d(
            q[b:b+1, :H_b, :W_b], k[b:b+1, :H_b, :W_b], v[b:b+1, :H_b, :W_b],
            kernel_size=kernel_size, dilation=dilation, scale=scale,
        )
        out[b, :H_b, :W_b] = out_b[0]
    return out


class TestVarlen2DForward:
    """Forward-pass correctness tests."""

    def test_uniform_sizes(self):
        """All spatial_sizes == (H_max, W_max) must match na2d exactly."""
        B, H, W, heads, D, K = 2, 8, 8, 2, 16, 3
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([[H, W], [H, W]], dtype=torch.int32)

        out_varlen = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_na2d = na2d(q, k, v, kernel_size=K)
        torch.testing.assert_close(out_varlen, out_na2d, atol=1e-5, rtol=1e-5)

    def test_mixed_sizes(self):
        """B=3 with different spatial sizes per sample."""
        B, H_max, W_max, heads, D, K = 3, 12, 12, 2, 16, 3
        q = torch.randn(B, H_max, W_max, heads, D)
        k = torch.randn(B, H_max, W_max, heads, D)
        v = torch.randn(B, H_max, W_max, heads, D)
        spatial_sizes = torch.tensor([[12, 12], [8, 10], [6, 6]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_minimum_sizes(self):
        """All spatial dims == kernel_size (smallest valid)."""
        B, K, heads, D = 2, 3, 2, 8
        H_max, W_max = 12, 12
        q = torch.randn(B, H_max, W_max, heads, D)
        k = torch.randn(B, H_max, W_max, heads, D)
        v = torch.randn(B, H_max, W_max, heads, D)
        spatial_sizes = torch.tensor([[K, K], [K, K]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_single_batch(self):
        """B=1 should work."""
        B, H, W, heads, D, K = 1, 16, 16, 2, 16, 3
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([[10, 12]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_per_sample_parity(self):
        """Each slice must match an independent na2d call."""
        B, H_max, W_max, heads, D, K = 3, 16, 16, 2, 16, 3
        q = torch.randn(B, H_max, W_max, heads, D)
        k = torch.randn(B, H_max, W_max, heads, D)
        v = torch.randn(B, H_max, W_max, heads, D)
        spatial_sizes = torch.tensor([[16, 16], [10, 12], [8, 8]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        for b in range(B):
            H_b = int(spatial_sizes[b, 0].item())
            W_b = int(spatial_sizes[b, 1].item())
            expected = na2d(q[b:b+1, :H_b, :W_b], k[b:b+1, :H_b, :W_b], v[b:b+1, :H_b, :W_b], kernel_size=K)
            torch.testing.assert_close(out[b, :H_b, :W_b], expected[0], atol=1e-5, rtol=1e-5,
                                       msg=f"Mismatch at batch {b}")

    def test_dilation(self):
        """dilation=2 must produce correct results."""
        B, H_max, W_max, heads, D, K, dil = 2, 16, 16, 2, 16, 3, 2
        q = torch.randn(B, H_max, W_max, heads, D)
        k = torch.randn(B, H_max, W_max, heads, D)
        v = torch.randn(B, H_max, W_max, heads, D)
        spatial_sizes = torch.tensor([[16, 16], [10, 12]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K, dilation=dil)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, dil, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_asymmetric_sizes(self):
        """Different H and W per sample."""
        B, H_max, W_max, heads, D, K = 2, 16, 16, 2, 16, 3
        q = torch.randn(B, H_max, W_max, heads, D)
        k = torch.randn(B, H_max, W_max, heads, D)
        v = torch.randn(B, H_max, W_max, heads, D)
        spatial_sizes = torch.tensor([[16, 8], [6, 14]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_padding_positions_zero(self):
        """Output beyond spatial_sizes must be zero."""
        B, H_max, W_max, heads, D, K = 2, 12, 12, 2, 16, 3
        q = torch.randn(B, H_max, W_max, heads, D)
        k = torch.randn(B, H_max, W_max, heads, D)
        v = torch.randn(B, H_max, W_max, heads, D)
        spatial_sizes = torch.tensor([[8, 6], [4, 4]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_np = out.numpy()
        # Batch 0: rows >= 8 should be zero, cols >= 6 should be zero
        assert (out_np[0, 8:] == 0).all(), "Batch 0 row padding should be zero"
        assert (out_np[0, :8, 6:] == 0).all(), "Batch 0 col padding should be zero"
        # Batch 1
        assert (out_np[1, 4:] == 0).all(), "Batch 1 row padding should be zero"
        assert (out_np[1, :4, 4:] == 0).all(), "Batch 1 col padding should be zero"

    def test_custom_scale(self):
        """Explicit scale parameter should be honored."""
        B, H, W, heads, D, K = 2, 8, 8, 2, 16, 3
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([[8, 6], [6, 8]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K, scale=0.1)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, 0.1)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


class TestVarlen2DValidation:
    """Input validation tests."""

    def test_rejects_small_spatial(self):
        """spatial_size < kernel_size should raise ValueError."""
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 5
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([[12, 12], [4, 12]], dtype=torch.int32)  # 4 < K=5

        with pytest.raises(ValueError, match="kernel_size"):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_exceeding_max(self):
        """spatial_size > max should raise ValueError."""
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([[12, 12], [12, 13]], dtype=torch.int32)  # 13 > W_max=12

        with pytest.raises(ValueError, match="max_spatial"):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_wrong_shape(self):
        """spatial_sizes shape must be (B, 2)."""
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([8, 8], dtype=torch.int32)  # 1D instead of [B, 2]

        with pytest.raises(ValueError):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_float_spatial_sizes(self):
        """spatial_sizes must be integer dtype."""
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([[8.0, 8.0], [6.0, 6.0]], dtype=torch.float32)

        with pytest.raises(ValueError, match="int32 or int64"):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_mismatched_shapes(self):
        """Q/K/V must have identical shapes."""
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = torch.randn(B, H, W, heads, D)
        k = torch.randn(B, H, W, heads, D + 1)
        v = torch.randn(B, H, W, heads, D)
        spatial_sizes = torch.tensor([[12, 12], [8, 8]], dtype=torch.int32)

        with pytest.raises(ValueError):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)


class TestVarlen2DBackward:
    """Backward gradient tests."""

    def test_backward_gradients(self):
        """Backward gradients match per-sample reference."""
        B, H_max, W_max, heads, D, K = 2, 8, 8, 2, 16, 3
        q = torch.randn(B, H_max, W_max, heads, D, requires_grad=True)
        k = torch.randn(B, H_max, W_max, heads, D, requires_grad=True)
        v = torch.randn(B, H_max, W_max, heads, D, requires_grad=True)
        spatial_sizes = torch.tensor([[8, 8], [6, 6]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        loss = out.sum()
        loss.backward()

        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

        # Reference: per-sample grads
        for b in range(B):
            H_b = int(spatial_sizes[b, 0].item())
            W_b = int(spatial_sizes[b, 1].item())
            q_b = q.data[b:b+1, :H_b, :W_b].detach().requires_grad_(True)
            k_b = k.data[b:b+1, :H_b, :W_b].detach().requires_grad_(True)
            v_b = v.data[b:b+1, :H_b, :W_b].detach().requires_grad_(True)
            out_b = na2d(q_b, k_b, v_b, kernel_size=K)
            out_b.sum().backward()
            torch.testing.assert_close(dq[b, :H_b, :W_b], q_b.grad[0], atol=1e-4, rtol=1e-4,
                                       msg=f"dq mismatch at batch {b}")
            torch.testing.assert_close(dk[b, :H_b, :W_b], k_b.grad[0], atol=1e-4, rtol=1e-4,
                                       msg=f"dk mismatch at batch {b}")
            torch.testing.assert_close(dv[b, :H_b, :W_b], v_b.grad[0], atol=1e-4, rtol=1e-4,
                                       msg=f"dv mismatch at batch {b}")

    def test_backward_padding_zero(self):
        """Gradients at padding positions must be zero."""
        B, H_max, W_max, heads, D, K = 2, 12, 12, 2, 16, 3
        q = torch.randn(B, H_max, W_max, heads, D, requires_grad=True)
        k = torch.randn(B, H_max, W_max, heads, D, requires_grad=True)
        v = torch.randn(B, H_max, W_max, heads, D, requires_grad=True)
        spatial_sizes = torch.tensor([[8, 6], [4, 4]], dtype=torch.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out.sum().backward()

        # Batch 0: padding rows (8:) should be zero, padding cols (:8, 6:) should be zero
        assert (q.grad[0, 8:] == 0).all(), "dq row padding should be zero"
        assert (q.grad[0, :8, 6:] == 0).all(), "dq col padding should be zero"
        assert (v.grad[0, 8:] == 0).all(), "dv row padding should be zero"
        assert (v.grad[0, :8, 6:] == 0).all(), "dv col padding should be zero"
