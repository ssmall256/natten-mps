"""Tests for variable-length 1D neighborhood attention."""

import pytest
import torch

from natten_mps.functional import na1d, na1d_varlen


def _varlen_reference(q, k, v, seq_lens, kernel_size, dilation, scale):
    """Per-sample reference: slice each batch element and run na1d independently."""
    B = q.shape[0]
    out = torch.zeros_like(q)
    for b in range(B):
        L = int(seq_lens[b])
        out[b, :L] = na1d(
            q[b : b + 1, :L],
            k[b : b + 1, :L],
            v[b : b + 1, :L],
            kernel_size=kernel_size,
            dilation=dilation,
            scale=scale,
        )[0]
    return out


DEVICE = "cpu"


class TestVarlenForward:
    """Forward-pass correctness tests."""

    def test_uniform_lengths(self):
        """All seq_lens == L_max must match na1d exactly."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
        seq_lens = torch.full((B,), L, dtype=torch.int32)

        out_varlen = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        out_na1d = na1d(q, k, v, kernel_size=K)
        torch.testing.assert_close(out_varlen, out_na1d, atol=1e-5, rtol=1e-5)

    def test_mixed_lengths(self):
        """B=4 with different lengths per sample."""
        B, L_max, H, D, K = 4, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device=DEVICE)
        k = torch.randn(B, L_max, H, D, device=DEVICE)
        v = torch.randn(B, L_max, H, D, device=DEVICE)
        seq_lens = torch.tensor([32, 24, 16, 8], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_minimum_lengths(self):
        """All seq_lens == kernel_size (smallest valid)."""
        B, K, H, D = 3, 7, 2, 8
        L_max = 32
        q = torch.randn(B, L_max, H, D, device=DEVICE)
        k = torch.randn(B, L_max, H, D, device=DEVICE)
        v = torch.randn(B, L_max, H, D, device=DEVICE)
        seq_lens = torch.full((B,), K, dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_single_batch(self):
        """B=1 should work."""
        B, L, H, D, K = 1, 64, 4, 16, 7
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
        seq_lens = torch.tensor([48], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_per_sample_parity(self):
        """Each slice must match an independent na1d call."""
        B, L_max, H, D, K = 3, 64, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device=DEVICE)
        k = torch.randn(B, L_max, H, D, device=DEVICE)
        v = torch.randn(B, L_max, H, D, device=DEVICE)
        seq_lens = torch.tensor([64, 32, 16], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        for b in range(B):
            L = int(seq_lens[b])
            expected = na1d(
                q[b : b + 1, :L], k[b : b + 1, :L], v[b : b + 1, :L],
                kernel_size=K,
            )
            torch.testing.assert_close(
                out[b, :L], expected[0], atol=1e-5, rtol=1e-5,
                msg=f"Mismatch at batch {b}",
            )

    def test_dilation(self):
        """dilation=2 must produce correct results."""
        B, L_max, H, D, K, dil = 2, 64, 4, 16, 7, 2
        q = torch.randn(B, L_max, H, D, device=DEVICE)
        k = torch.randn(B, L_max, H, D, device=DEVICE)
        v = torch.randn(B, L_max, H, D, device=DEVICE)
        seq_lens = torch.tensor([64, 32], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K, dilation=dil)
        ref = _varlen_reference(q, k, v, seq_lens, K, dil, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_large_kernel(self):
        """K=15, L_max=16 — kernel nearly covers entire sequence."""
        B, L_max, H, D, K = 2, 16, 2, 8, 15
        q = torch.randn(B, L_max, H, D, device=DEVICE)
        k = torch.randn(B, L_max, H, D, device=DEVICE)
        v = torch.randn(B, L_max, H, D, device=DEVICE)
        seq_lens = torch.tensor([16, 15], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_padding_positions_zero(self):
        """Output beyond seq_lens[b] must be zero."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device=DEVICE)
        k = torch.randn(B, L_max, H, D, device=DEVICE)
        v = torch.randn(B, L_max, H, D, device=DEVICE)
        seq_lens = torch.tensor([16, 8], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        assert (out[0, 16:] == 0).all(), "Batch 0 padding should be zero"
        assert (out[1, 8:] == 0).all(), "Batch 1 padding should be zero"

    def test_custom_scale(self):
        """Explicit scale parameter should be honored."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
        seq_lens = torch.tensor([24, 16], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K, scale=0.1)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, 0.1)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


class TestVarlenValidation:
    """Input validation tests."""

    def test_rejects_short_seqlens(self):
        """seq_len < kernel_size should raise ValueError."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        seq_lens = torch.tensor([6, 32], dtype=torch.int32)  # 6 < K=7

        with pytest.raises(ValueError, match="kernel_size"):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_exceeding_lmax(self):
        """seq_len > L_max should raise ValueError."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        seq_lens = torch.tensor([32, 33], dtype=torch.int32)  # 33 > L_max=32

        with pytest.raises(ValueError, match="L_max"):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_wrong_seq_lens_shape(self):
        """seq_lens shape must be (B,)."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        seq_lens = torch.tensor([[32, 32]], dtype=torch.int32)

        with pytest.raises(ValueError):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_float_seq_lens(self):
        """seq_lens must be integer dtype."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        seq_lens = torch.tensor([32.0, 16.0], dtype=torch.float32)

        with pytest.raises(ValueError, match="int32 or int64"):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_mismatched_shapes(self):
        """Q/K/V must have identical shapes."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D + 1)
        v = torch.randn(B, L, H, D)
        seq_lens = torch.tensor([32, 16], dtype=torch.int32)

        with pytest.raises(ValueError):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
class TestVarlenMPS:
    """Tests on MPS device (when available)."""

    def test_forward_mps(self):
        """Basic forward on MPS device."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device="mps")
        k = torch.randn(B, L_max, H, D, device="mps")
        v = torch.randn(B, L_max, H, D, device="mps")
        seq_lens = torch.tensor([32, 16], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_fp16_mps(self):
        """Half-precision on MPS."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device="mps", dtype=torch.float16)
        k = torch.randn(B, L_max, H, D, device="mps", dtype=torch.float16)
        v = torch.randn(B, L_max, H, D, device="mps", dtype=torch.float16)
        seq_lens = torch.tensor([32, 16], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(
            q.float(), k.float(), v.float(), seq_lens, K, 1, q.shape[-1] ** -0.5,
        ).half()
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_metal_matches_pure(self):
        """Metal varlen kernels must match pure backend exactly."""
        import natten_mps
        B, L_max, H, D, K = 3, 64, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device="mps")
        k = torch.randn(B, L_max, H, D, device="mps")
        v = torch.randn(B, L_max, H, D, device="mps")
        seq_lens = torch.tensor([64, 32, 16], dtype=torch.int32)

        # Metal path
        out_metal = na1d_varlen(q, k, v, seq_lens, kernel_size=K)

        # Pure path (force CPU)
        q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()
        natten_mps.set_backend("pure")
        out_pure = na1d_varlen(q_cpu, k_cpu, v_cpu, seq_lens, kernel_size=K)
        natten_mps.set_backend("auto")

        torch.testing.assert_close(out_metal.cpu(), out_pure, atol=1e-4, rtol=1e-4)

    def test_dilation_mps(self):
        """Dilation on MPS."""
        B, L_max, H, D, K, dil = 2, 64, 4, 16, 7, 2
        q = torch.randn(B, L_max, H, D, device="mps")
        k = torch.randn(B, L_max, H, D, device="mps")
        v = torch.randn(B, L_max, H, D, device="mps")
        seq_lens = torch.tensor([64, 32], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K, dilation=dil)
        ref = _varlen_reference(q, k, v, seq_lens, K, dil, q.shape[-1] ** -0.5)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_padding_zero_mps(self):
        """Padding positions are zero on MPS."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device="mps")
        k = torch.randn(B, L_max, H, D, device="mps")
        v = torch.randn(B, L_max, H, D, device="mps")
        seq_lens = torch.tensor([16, 8], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        assert (out[0, 16:].cpu() == 0).all(), "Batch 0 padding should be zero"
        assert (out[1, 8:].cpu() == 0).all(), "Batch 1 padding should be zero"


class TestVarlenBackward:
    """Backward gradient tests."""

    def _reference_grads(self, q, k, v, seq_lens, kernel_size, dilation, scale):
        """Compute reference grads via per-sample autograd."""
        B = q.shape[0]
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        for b in range(B):
            L = int(seq_lens[b])
            q_b = q[b:b+1, :L].detach().requires_grad_(True)
            k_b = k[b:b+1, :L].detach().requires_grad_(True)
            v_b = v[b:b+1, :L].detach().requires_grad_(True)
            out_b = na1d(q_b, k_b, v_b, kernel_size=kernel_size, dilation=dilation, scale=scale)
            out_b.sum().backward()
            dq[b, :L] = q_b.grad[0]
            dk[b, :L] = k_b.grad[0]
            dv[b, :L] = v_b.grad[0]
        return dq, dk, dv

    def test_backward_cpu(self):
        """Backward gradients on CPU match per-sample reference."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, requires_grad=True)
        k = torch.randn(B, L_max, H, D, requires_grad=True)
        v = torch.randn(B, L_max, H, D, requires_grad=True)
        seq_lens = torch.tensor([32, 16], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        out.sum().backward()

        ref_dq, ref_dk, ref_dv = self._reference_grads(
            q.detach(), k.detach(), v.detach(), seq_lens, K, 1, q.shape[-1] ** -0.5,
        )
        torch.testing.assert_close(q.grad, ref_dq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad, ref_dk, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad, ref_dv, atol=1e-4, rtol=1e-4)

    def test_backward_padding_zero(self):
        """Gradients at padding positions must be zero."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, requires_grad=True)
        k = torch.randn(B, L_max, H, D, requires_grad=True)
        v = torch.randn(B, L_max, H, D, requires_grad=True)
        seq_lens = torch.tensor([16, 8], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        out.sum().backward()

        assert (q.grad[0, 16:] == 0).all(), "dq padding should be zero"
        assert (q.grad[1, 8:] == 0).all(), "dq padding should be zero"
        assert (v.grad[0, 16:] == 0).all(), "dv padding should be zero"
        assert (v.grad[1, 8:] == 0).all(), "dv padding should be zero"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_backward_mps(self):
        """Backward gradients on MPS match per-sample reference."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = torch.randn(B, L_max, H, D, device="mps", requires_grad=True)
        k = torch.randn(B, L_max, H, D, device="mps", requires_grad=True)
        v = torch.randn(B, L_max, H, D, device="mps", requires_grad=True)
        seq_lens = torch.tensor([32, 16], dtype=torch.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        out.sum().backward()

        ref_dq, ref_dk, ref_dv = self._reference_grads(
            q.detach(), k.detach(), v.detach(), seq_lens, K, 1, q.shape[-1] ** -0.5,
        )
        torch.testing.assert_close(q.grad, ref_dq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad, ref_dk, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad, ref_dv, atol=1e-4, rtol=1e-4)
