"""Tests for the Metal (Tier-1) backend.

Verifies that Metal kernel results match the pure backend within float32 tolerance.
Tests use MPS tensors to exercise the actual Metal kernels.
"""

import pytest
import torch

import natten_mps
from natten_mps._core import metal
from natten_mps.functional import (
    na1d, na1d_av, na1d_qk,
    na2d, na2d_av, na2d_qk,
    na3d, na3d_av, na3d_qk,
)

MPS = torch.device("mps")


@pytest.fixture(autouse=True)
def _use_metal_backend():
    """Switch to metal backend for all tests, restore after."""
    prev = natten_mps.get_backend()
    natten_mps.set_backend("metal")
    yield
    natten_mps.set_backend(prev)


def _pure_ref(fn, *args, **kwargs):
    """Run a function using the pure backend on CPU copies."""
    prev = natten_mps.get_backend()
    natten_mps.set_backend("pure")
    try:
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        cpu_kwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        return fn(*cpu_args, **cpu_kwargs)
    finally:
        natten_mps.set_backend(prev)


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------


def test_metal_is_available():
    assert metal.is_available() is True


def test_metal_auto_selected():
    natten_mps.set_backend("auto")
    assert natten_mps.get_backend() == "metal"


def test_cpu_tensors_fall_back_to_pure():
    """CPU tensors should work transparently via pure fallback."""
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 8, 2, 4)
    v = torch.randn(1, 8, 2, 4)
    out = na1d(q, k, v, kernel_size=3)
    assert out.device.type == "cpu"
    assert out.shape == (1, 8, 2, 4)


# ------------------------------------------------------------------
# 1D: Metal vs Pure parity (MPS tensors)
# ------------------------------------------------------------------


class TestMetal1D:
    def test_na1d_matches_pure(self):
        q = torch.randn(2, 12, 4, 8, device=MPS)
        k = torch.randn(2, 12, 4, 8, device=MPS)
        v = torch.randn(2, 12, 4, 8, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=3).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=3)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_with_dilation(self):
        q = torch.randn(1, 16, 2, 8, device=MPS)
        k = torch.randn(1, 16, 2, 8, device=MPS)
        v = torch.randn(1, 16, 2, 8, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=3, dilation=2).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=3, dilation=2)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_larger_kernel(self):
        q = torch.randn(1, 20, 4, 16, device=MPS)
        k = torch.randn(1, 20, 4, 16, device=MPS)
        v = torch.randn(1, 20, 4, 16, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=7).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=7)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_qk_matches_pure(self):
        q = torch.randn(2, 10, 4, 8, device=MPS)
        k = torch.randn(2, 10, 4, 8, device=MPS)

        logits_metal = na1d_qk(q, k, kernel_size=3).cpu()
        logits_pure = _pure_ref(na1d_qk, q, k, kernel_size=3)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-6, rtol=1e-6)

    def test_na1d_av_matches_pure(self):
        attn = torch.softmax(torch.randn(2, 10, 4, 5, device=MPS), dim=-1)
        v = torch.randn(2, 10, 4, 8, device=MPS)

        out_metal = na1d_av(attn, v, kernel_size=5).cpu()
        out_pure = _pure_ref(na1d_av, attn, v, kernel_size=5)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_causal_matches_pure(self):
        q = torch.randn(2, 12, 4, 8, device=MPS)
        k = torch.randn(2, 12, 4, 8, device=MPS)
        v = torch.randn(2, 12, 4, 8, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=3, is_causal=True).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=3, is_causal=True)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_causal_stays_on_mps(self):
        q = torch.randn(1, 8, 2, 4, device=MPS)
        k = torch.randn(1, 8, 2, 4, device=MPS)
        v = torch.randn(1, 8, 2, 4, device=MPS)
        out = na1d(q, k, v, kernel_size=3, is_causal=True)
        assert out.device.type == "mps"

    def test_na1d_qk_causal_matches_pure(self):
        q = torch.randn(1, 10, 4, 8, device=MPS)
        k = torch.randn(1, 10, 4, 8, device=MPS)

        logits_metal = na1d_qk(q, k, kernel_size=5, is_causal=True).cpu()
        logits_pure = _pure_ref(na1d_qk, q, k, kernel_size=5, is_causal=True)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-6, rtol=1e-6)

    def test_na1d_av_causal_matches_pure(self):
        attn = torch.softmax(torch.randn(1, 10, 4, 5, device=MPS), dim=-1)
        v = torch.randn(1, 10, 4, 8, device=MPS)

        out_metal = na1d_av(attn, v, kernel_size=5, is_causal=True).cpu()
        out_pure = _pure_ref(na1d_av, attn, v, kernel_size=5, is_causal=True)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_stride2_matches_pure(self):
        q = torch.randn(2, 12, 4, 8, device=MPS)
        k = torch.randn(2, 12, 4, 8, device=MPS)
        v = torch.randn(2, 12, 4, 8, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=3, stride=2).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=3, stride=2)
        assert out_metal.shape == (2, 6, 4, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_stride2_stays_on_mps(self):
        q = torch.randn(1, 8, 2, 4, device=MPS)
        k = torch.randn(1, 8, 2, 4, device=MPS)
        v = torch.randn(1, 8, 2, 4, device=MPS)
        out = na1d(q, k, v, kernel_size=3, stride=2)
        assert out.device.type == "mps"

    def test_na1d_stride3_matches_pure(self):
        q = torch.randn(1, 18, 2, 8, device=MPS)
        k = torch.randn(1, 18, 2, 8, device=MPS)
        v = torch.randn(1, 18, 2, 8, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=5, stride=3).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=5, stride=3)
        assert out_metal.shape == (1, 6, 2, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_qk_strided_matches_pure(self):
        q = torch.randn(1, 12, 4, 8, device=MPS)
        k = torch.randn(1, 12, 4, 8, device=MPS)

        logits_metal = na1d_qk(q, k, kernel_size=3, stride=2).cpu()
        logits_pure = _pure_ref(na1d_qk, q, k, kernel_size=3, stride=2)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-6, rtol=1e-6)

    def test_na1d_av_strided_matches_pure(self):
        attn = torch.softmax(torch.randn(1, 6, 4, 3, device=MPS), dim=-1)
        v = torch.randn(1, 12, 4, 8, device=MPS)

        out_metal = na1d_av(attn, v, kernel_size=3, stride=2).cpu()
        out_pure = _pure_ref(na1d_av, attn, v, kernel_size=3, stride=2)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------
# 2D: Metal vs Pure parity (MPS tensors)
# ------------------------------------------------------------------


class TestMetal2D:
    def test_na2d_matches_pure(self):
        q = torch.randn(1, 8, 8, 4, 16, device=MPS)
        k = torch.randn(1, 8, 8, 4, 16, device=MPS)
        v = torch.randn(1, 8, 8, 4, 16, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_with_dilation(self):
        q = torch.randn(1, 10, 10, 2, 8, device=MPS)
        k = torch.randn(1, 10, 10, 2, 8, device=MPS)
        v = torch.randn(1, 10, 10, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, dilation=2).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, dilation=2)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_larger_kernel(self):
        q = torch.randn(1, 14, 14, 4, 8, device=MPS)
        k = torch.randn(1, 14, 14, 4, 8, device=MPS)
        v = torch.randn(1, 14, 14, 4, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=5).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=5)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_qk_matches_pure(self):
        q = torch.randn(1, 8, 8, 4, 16, device=MPS)
        k = torch.randn(1, 8, 8, 4, 16, device=MPS)

        logits_metal = na2d_qk(q, k, kernel_size=3).cpu()
        logits_pure = _pure_ref(na2d_qk, q, k, kernel_size=3)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-6, rtol=1e-6)

    def test_na2d_av_matches_pure(self):
        attn = torch.softmax(torch.randn(1, 8, 8, 4, 9, device=MPS), dim=-1)
        v = torch.randn(1, 8, 8, 4, 16, device=MPS)

        out_metal = na2d_av(attn, v, kernel_size=3).cpu()
        out_pure = _pure_ref(na2d_av, attn, v, kernel_size=3)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_causal_both_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, is_causal=(True, True)).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, is_causal=(True, True))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_causal_h_only_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, is_causal=(True, False)).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, is_causal=(True, False))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_causal_w_only_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, is_causal=(False, True)).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, is_causal=(False, True))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_qk_causal_matches_pure(self):
        q = torch.randn(1, 8, 8, 4, 16, device=MPS)
        k = torch.randn(1, 8, 8, 4, 16, device=MPS)

        logits_metal = na2d_qk(q, k, kernel_size=3, is_causal=(True, True)).cpu()
        logits_pure = _pure_ref(na2d_qk, q, k, kernel_size=3, is_causal=(True, True))
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-6, rtol=1e-6)

    def test_na2d_stride2_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, stride=2).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, stride=2)
        assert out_metal.shape == (1, 4, 4, 2, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_stride2_stays_on_mps(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)
        out = na2d(q, k, v, kernel_size=3, stride=2)
        assert out.device.type == "mps"

    def test_na2d_qk_strided_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)

        logits_metal = na2d_qk(q, k, kernel_size=3, stride=2).cpu()
        logits_pure = _pure_ref(na2d_qk, q, k, kernel_size=3, stride=2)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-6, rtol=1e-6)

    def test_na2d_nonsquare_kernel_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=(3, 5)).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=(3, 5))
        assert out_metal.shape == (1, 8, 8, 2, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_nonsquare_kernel_stays_on_mps(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)
        out = na2d(q, k, v, kernel_size=(3, 5))
        assert out.device.type == "mps"

    def test_na2d_nonsquare_dilation_matches_pure(self):
        q = torch.randn(1, 10, 10, 2, 8, device=MPS)
        k = torch.randn(1, 10, 10, 2, 8, device=MPS)
        v = torch.randn(1, 10, 10, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, dilation=(1, 2)).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, dilation=(1, 2))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_qk_nonsquare_kernel_matches_pure(self):
        q = torch.randn(1, 8, 8, 4, 16, device=MPS)
        k = torch.randn(1, 8, 8, 4, 16, device=MPS)

        logits_metal = na2d_qk(q, k, kernel_size=(3, 5)).cpu()
        logits_pure = _pure_ref(na2d_qk, q, k, kernel_size=(3, 5))
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------
# 3D: Metal vs Pure parity
# ------------------------------------------------------------------


class TestMetal3D:
    def test_na3d_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=3).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=3)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_causal_all_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=3, is_causal=(True, True, True)).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=3, is_causal=(True, True, True))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_causal_d_only_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=3, is_causal=(True, False, False)).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=3, is_causal=(True, False, False))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_qk_causal_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        logits_metal = na3d_qk(q, k, kernel_size=3, is_causal=(True, True, True)).cpu()
        logits_pure = _pure_ref(na3d_qk, q, k, kernel_size=3, is_causal=(True, True, True))
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_stride2_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=3, stride=2).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=3, stride=2)
        assert out_metal.shape == (1, 2, 2, 2, 2, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_stride2_stays_on_mps(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        out = na3d(q, k, v, kernel_size=3, stride=2)
        assert out.device.type == "mps"

    def test_na3d_nonuniform_kernel_matches_pure(self):
        q = torch.randn(1, 4, 6, 6, 2, 8, device=MPS)
        k = torch.randn(1, 4, 6, 6, 2, 8, device=MPS)
        v = torch.randn(1, 4, 6, 6, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=(3, 5, 3)).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=(3, 5, 3))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_nonuniform_kernel_stays_on_mps(self):
        q = torch.randn(1, 4, 6, 6, 2, 8, device=MPS)
        k = torch.randn(1, 4, 6, 6, 2, 8, device=MPS)
        v = torch.randn(1, 4, 6, 6, 2, 8, device=MPS)
        out = na3d(q, k, v, kernel_size=(3, 5, 3))
        assert out.device.type == "mps"

    def test_na3d_nonuniform_dilation_matches_pure(self):
        q = torch.randn(1, 6, 6, 6, 2, 8, device=MPS)
        k = torch.randn(1, 6, 6, 6, 2, 8, device=MPS)
        v = torch.randn(1, 6, 6, 6, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=3, dilation=(1, 2, 1)).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=3, dilation=(1, 2, 1))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------
# Vec4 parity: verify vec4-aligned and unaligned dims produce same results
# ------------------------------------------------------------------


class TestMetalVec4:
    @pytest.mark.parametrize("dim", [16, 17])
    def test_na1d_qk_vec4_parity(self, dim):
        q = torch.randn(1, 12, 2, dim, device=MPS)
        k = torch.randn(1, 12, 2, dim, device=MPS)

        logits_metal = na1d_qk(q, k, kernel_size=3).cpu()
        logits_pure = _pure_ref(na1d_qk, q, k, kernel_size=3)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dim", [16, 17])
    def test_na2d_qk_vec4_parity(self, dim):
        q = torch.randn(1, 8, 8, 2, dim, device=MPS)
        k = torch.randn(1, 8, 8, 2, dim, device=MPS)

        logits_metal = na2d_qk(q, k, kernel_size=3).cpu()
        logits_pure = _pure_ref(na2d_qk, q, k, kernel_size=3)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dim", [16, 17])
    def test_na1d_fused_vec4_parity(self, dim):
        q = torch.randn(1, 12, 2, dim, device=MPS)
        k = torch.randn(1, 12, 2, dim, device=MPS)
        v = torch.randn(1, 12, 2, dim, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=3).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=3)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------
# Autograd through Metal backend (CPU tensors, uses pure fallback)
# ------------------------------------------------------------------


class TestMetalAutograd:
    def test_na1d_gradient_flows(self):
        q = torch.randn(1, 8, 2, 4, requires_grad=True)
        k = torch.randn(1, 8, 2, 4, requires_grad=True)
        v = torch.randn(1, 8, 2, 4, requires_grad=True)

        out = na1d(q, k, v, kernel_size=3)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_na2d_gradient_flows(self):
        q = torch.randn(1, 6, 6, 2, 8, requires_grad=True)
        k = torch.randn(1, 6, 6, 2, 8, requires_grad=True)
        v = torch.randn(1, 6, 6, 2, 8, requires_grad=True)

        out = na2d(q, k, v, kernel_size=3)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


# ------------------------------------------------------------------
# Backward: Metal backward kernels vs re-differentiation (pure)
# ------------------------------------------------------------------


def _grad_ref(fn, inputs, kernel_size, **kwargs):
    """Get reference gradients using pure backend re-differentiation."""
    prev = natten_mps.get_backend()
    natten_mps.set_backend("pure")
    try:
        cpu_inputs = [t.detach().cpu().requires_grad_(True) for t in inputs]
        out = fn(*cpu_inputs, kernel_size=kernel_size, **kwargs)
        out.sum().backward()
        return [t.grad for t in cpu_inputs]
    finally:
        natten_mps.set_backend(prev)


class TestMetalBackward1D:
    def test_na1d_qk_backward_matches_pure(self):
        q = torch.randn(2, 10, 4, 8, device=MPS, requires_grad=True)
        k = torch.randn(2, 10, 4, 8, device=MPS, requires_grad=True)

        out = na1d_qk(q, k, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na1d_qk, [q, k], kernel_size=3)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na1d_av_backward_matches_pure(self):
        attn = torch.softmax(torch.randn(2, 10, 4, 5, device=MPS), dim=-1).requires_grad_(True)
        v = torch.randn(2, 10, 4, 8, device=MPS, requires_grad=True)

        out = na1d_av(attn, v, kernel_size=5)
        out.sum().backward()

        ref_grads = _grad_ref(na1d_av, [attn, v], kernel_size=5)
        torch.testing.assert_close(attn.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na1d_fused_backward_matches_pure(self):
        q = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        k = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        v = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)

        out = na1d(q, k, v, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na1d, [q, k, v], kernel_size=3)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na1d_backward_stays_on_mps(self):
        q = torch.randn(1, 8, 2, 4, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 2, 4, device=MPS, requires_grad=True)
        v = torch.randn(1, 8, 2, 4, device=MPS, requires_grad=True)

        out = na1d(q, k, v, kernel_size=3)
        out.sum().backward()

        assert q.grad.device.type == "mps"
        assert k.grad.device.type == "mps"
        assert v.grad.device.type == "mps"


class TestMetalBackward2D:
    def test_na2d_qk_backward_matches_pure(self):
        q = torch.randn(1, 8, 8, 4, 16, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 8, 4, 16, device=MPS, requires_grad=True)

        out = na2d_qk(q, k, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na2d_qk, [q, k], kernel_size=3)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na2d_av_backward_matches_pure(self):
        attn = torch.softmax(torch.randn(1, 8, 8, 4, 9, device=MPS), dim=-1).requires_grad_(True)
        v = torch.randn(1, 8, 8, 4, 16, device=MPS, requires_grad=True)

        out = na2d_av(attn, v, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na2d_av, [attn, v], kernel_size=3)
        torch.testing.assert_close(attn.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na2d_fused_backward_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)

        out = na2d(q, k, v, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na2d, [q, k, v], kernel_size=3)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na2d_backward_stays_on_mps(self):
        q = torch.randn(1, 6, 6, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 6, 6, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 6, 6, 2, 8, device=MPS, requires_grad=True)

        out = na2d(q, k, v, kernel_size=3)
        out.sum().backward()

        assert q.grad.device.type == "mps"
        assert k.grad.device.type == "mps"
        assert v.grad.device.type == "mps"


class TestMetalBackward3D:
    def test_na3d_qk_backward_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d_qk(q, k, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na3d_qk, [q, k], kernel_size=3)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na3d_av_backward_matches_pure(self):
        attn = torch.softmax(torch.randn(1, 4, 4, 4, 2, 27, device=MPS), dim=-1).requires_grad_(True)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d_av(attn, v, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na3d_av, [attn, v], kernel_size=3)
        torch.testing.assert_close(attn.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na3d_fused_backward_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d(q, k, v, kernel_size=3)
        out.sum().backward()

        ref_grads = _grad_ref(na3d, [q, k, v], kernel_size=3)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na3d_backward_stays_on_mps(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d(q, k, v, kernel_size=3)
        out.sum().backward()

        assert q.grad.device.type == "mps"
        assert k.grad.device.type == "mps"
        assert v.grad.device.type == "mps"


# ------------------------------------------------------------------
# Causal + strided combined (previously fell back to pure)
# ------------------------------------------------------------------


class TestCausalStrided1D:
    def test_na1d_causal_strided_matches_pure(self):
        q = torch.randn(2, 12, 4, 8, device=MPS)
        k = torch.randn(2, 12, 4, 8, device=MPS)
        v = torch.randn(2, 12, 4, 8, device=MPS)

        out_metal = na1d(q, k, v, kernel_size=3, stride=2, is_causal=True).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=3, stride=2, is_causal=True)
        assert out_metal.shape == (2, 6, 4, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_causal_strided_stays_on_mps(self):
        q = torch.randn(1, 8, 2, 4, device=MPS)
        k = torch.randn(1, 8, 2, 4, device=MPS)
        v = torch.randn(1, 8, 2, 4, device=MPS)
        out = na1d(q, k, v, kernel_size=3, stride=2, is_causal=True)
        assert out.device.type == "mps"

    def test_na1d_qk_causal_strided_matches_pure(self):
        q = torch.randn(1, 12, 4, 8, device=MPS)
        k = torch.randn(1, 12, 4, 8, device=MPS)

        logits_metal = na1d_qk(q, k, kernel_size=3, stride=2, is_causal=True).cpu()
        logits_pure = _pure_ref(na1d_qk, q, k, kernel_size=3, stride=2, is_causal=True)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_av_causal_strided_matches_pure(self):
        attn = torch.softmax(torch.randn(1, 6, 4, 3, device=MPS), dim=-1)
        v = torch.randn(1, 12, 4, 8, device=MPS)

        out_metal = na1d_av(attn, v, kernel_size=3, stride=2, is_causal=True).cpu()
        out_pure = _pure_ref(na1d_av, attn, v, kernel_size=3, stride=2, is_causal=True)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)


class TestCausalStrided2D:
    def test_na2d_causal_strided_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, stride=2, is_causal=(True, True)).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, stride=2, is_causal=(True, True))
        assert out_metal.shape == (1, 4, 4, 2, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_causal_strided_stays_on_mps(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)
        out = na2d(q, k, v, kernel_size=3, stride=2, is_causal=(True, True))
        assert out.device.type == "mps"

    def test_na2d_causal_h_strided_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS)

        out_metal = na2d(q, k, v, kernel_size=3, stride=2, is_causal=(True, False)).cpu()
        out_pure = _pure_ref(na2d, q, k, v, kernel_size=3, stride=2, is_causal=(True, False))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na2d_qk_causal_strided_matches_pure(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS)

        logits_metal = na2d_qk(q, k, kernel_size=3, stride=2, is_causal=(True, True)).cpu()
        logits_pure = _pure_ref(na2d_qk, q, k, kernel_size=3, stride=2, is_causal=(True, True))
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)


class TestCausalStrided3D:
    def test_na3d_causal_strided_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=3, stride=2, is_causal=(True, True, True)).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=3, stride=2, is_causal=(True, True, True))
        assert out_metal.shape == (1, 2, 2, 2, 2, 8)
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_causal_strided_stays_on_mps(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        out = na3d(q, k, v, kernel_size=3, stride=2, is_causal=(True, True, True))
        assert out.device.type == "mps"

    def test_na3d_causal_d_strided_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        out_metal = na3d(q, k, v, kernel_size=3, stride=2, is_causal=(True, False, False)).cpu()
        out_pure = _pure_ref(na3d, q, k, v, kernel_size=3, stride=2, is_causal=(True, False, False))
        torch.testing.assert_close(out_metal, out_pure, atol=1e-5, rtol=1e-5)

    def test_na3d_qk_causal_strided_matches_pure(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS)

        logits_metal = na3d_qk(q, k, kernel_size=3, stride=2, is_causal=(True, True, True)).cpu()
        logits_pure = _pure_ref(na3d_qk, q, k, kernel_size=3, stride=2, is_causal=(True, True, True))
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------
# Vec4 parity for causal QK kernels
# ------------------------------------------------------------------


class TestMetalVec4Causal:
    @pytest.mark.parametrize("dim", [16, 17])
    def test_na1d_qk_causal_vec4_parity(self, dim):
        q = torch.randn(1, 12, 2, dim, device=MPS)
        k = torch.randn(1, 12, 2, dim, device=MPS)

        logits_metal = na1d_qk(q, k, kernel_size=3, is_causal=True).cpu()
        logits_pure = _pure_ref(na1d_qk, q, k, kernel_size=3, is_causal=True)
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dim", [16, 17])
    def test_na2d_qk_causal_vec4_parity(self, dim):
        q = torch.randn(1, 8, 8, 2, dim, device=MPS)
        k = torch.randn(1, 8, 8, 2, dim, device=MPS)

        logits_metal = na2d_qk(q, k, kernel_size=3, is_causal=(True, True)).cpu()
        logits_pure = _pure_ref(na2d_qk, q, k, kernel_size=3, is_causal=(True, True))
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dim", [16, 17])
    def test_na3d_qk_causal_vec4_parity(self, dim):
        q = torch.randn(1, 4, 4, 4, 2, dim, device=MPS)
        k = torch.randn(1, 4, 4, 4, 2, dim, device=MPS)

        logits_metal = na3d_qk(q, k, kernel_size=3, is_causal=(True, True, True)).cpu()
        logits_pure = _pure_ref(na3d_qk, q, k, kernel_size=3, is_causal=(True, True, True))
        torch.testing.assert_close(logits_metal, logits_pure, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------
# Backward: causal, strided, causal+strided, non-uniform
# (These use the pure-backend re-differentiation fallback)
# ------------------------------------------------------------------


class TestBackwardCausal1D:
    def test_na1d_fused_causal_backward(self):
        q = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        k = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        v = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)

        out = na1d(q, k, v, kernel_size=3, is_causal=True)
        out.sum().backward()

        ref_grads = _grad_ref(na1d, [q, k, v], kernel_size=3, is_causal=True)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na1d_qk_causal_backward(self):
        q = torch.randn(1, 10, 4, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 10, 4, 8, device=MPS, requires_grad=True)

        out = na1d_qk(q, k, kernel_size=5, is_causal=True)
        out.sum().backward()

        ref_grads = _grad_ref(na1d_qk, [q, k], kernel_size=5, is_causal=True)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na1d_av_causal_backward(self):
        attn = torch.softmax(torch.randn(1, 10, 4, 5, device=MPS), dim=-1).requires_grad_(True)
        v = torch.randn(1, 10, 4, 8, device=MPS, requires_grad=True)

        out = na1d_av(attn, v, kernel_size=5, is_causal=True)
        out.sum().backward()

        ref_grads = _grad_ref(na1d_av, [attn, v], kernel_size=5, is_causal=True)
        torch.testing.assert_close(attn.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)


class TestBackwardStrided1D:
    def test_na1d_fused_strided_backward(self):
        q = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        k = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        v = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)

        out = na1d(q, k, v, kernel_size=3, stride=2)
        out.sum().backward()

        ref_grads = _grad_ref(na1d, [q, k, v], kernel_size=3, stride=2)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na1d_qk_strided_backward(self):
        q = torch.randn(1, 12, 4, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 12, 4, 8, device=MPS, requires_grad=True)

        out = na1d_qk(q, k, kernel_size=3, stride=2)
        out.sum().backward()

        ref_grads = _grad_ref(na1d_qk, [q, k], kernel_size=3, stride=2)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)

    def test_na1d_av_strided_backward(self):
        attn = torch.softmax(torch.randn(1, 6, 4, 3, device=MPS), dim=-1).requires_grad_(True)
        v = torch.randn(1, 12, 4, 8, device=MPS, requires_grad=True)

        out = na1d_av(attn, v, kernel_size=3, stride=2)
        out.sum().backward()

        ref_grads = _grad_ref(na1d_av, [attn, v], kernel_size=3, stride=2)
        torch.testing.assert_close(attn.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)


class TestBackwardCausalStrided1D:
    def test_na1d_fused_causal_strided_backward(self):
        q = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        k = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)
        v = torch.randn(2, 12, 4, 8, device=MPS, requires_grad=True)

        out = na1d(q, k, v, kernel_size=3, stride=2, is_causal=True)
        out.sum().backward()

        ref_grads = _grad_ref(na1d, [q, k, v], kernel_size=3, stride=2, is_causal=True)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)


class TestBackwardCausal2D:
    def test_na2d_fused_causal_backward(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)

        out = na2d(q, k, v, kernel_size=3, is_causal=(True, True))
        out.sum().backward()

        ref_grads = _grad_ref(na2d, [q, k, v], kernel_size=3, is_causal=(True, True))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na2d_qk_causal_backward(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)

        out = na2d_qk(q, k, kernel_size=3, is_causal=(True, True))
        out.sum().backward()

        ref_grads = _grad_ref(na2d_qk, [q, k], kernel_size=3, is_causal=(True, True))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)


class TestBackwardStrided2D:
    def test_na2d_fused_strided_backward(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)

        out = na2d(q, k, v, kernel_size=3, stride=2)
        out.sum().backward()

        ref_grads = _grad_ref(na2d, [q, k, v], kernel_size=3, stride=2)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)


class TestBackwardCausalStrided2D:
    def test_na2d_fused_causal_strided_backward(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)

        out = na2d(q, k, v, kernel_size=3, stride=2, is_causal=(True, True))
        out.sum().backward()

        ref_grads = _grad_ref(na2d, [q, k, v], kernel_size=3, stride=2, is_causal=(True, True))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)


class TestBackwardNonUniform2D:
    def test_na2d_nonsquare_kernel_backward(self):
        q = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 8, 8, 2, 8, device=MPS, requires_grad=True)

        out = na2d(q, k, v, kernel_size=(3, 5))
        out.sum().backward()

        ref_grads = _grad_ref(na2d, [q, k, v], kernel_size=(3, 5))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na2d_nonsquare_dilation_backward(self):
        q = torch.randn(1, 10, 10, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 10, 10, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 10, 10, 2, 8, device=MPS, requires_grad=True)

        out = na2d(q, k, v, kernel_size=3, dilation=(1, 2))
        out.sum().backward()

        ref_grads = _grad_ref(na2d, [q, k, v], kernel_size=3, dilation=(1, 2))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)


class TestBackwardCausal3D:
    def test_na3d_fused_causal_backward(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d(q, k, v, kernel_size=3, is_causal=(True, True, True))
        out.sum().backward()

        ref_grads = _grad_ref(na3d, [q, k, v], kernel_size=3, is_causal=(True, True, True))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na3d_qk_causal_backward(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d_qk(q, k, kernel_size=3, is_causal=(True, True, True))
        out.sum().backward()

        ref_grads = _grad_ref(na3d_qk, [q, k], kernel_size=3, is_causal=(True, True, True))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)


class TestBackwardStrided3D:
    def test_na3d_fused_strided_backward(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d(q, k, v, kernel_size=3, stride=2)
        out.sum().backward()

        ref_grads = _grad_ref(na3d, [q, k, v], kernel_size=3, stride=2)
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)


class TestBackwardCausalStrided3D:
    def test_na3d_fused_causal_strided_backward(self):
        q = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 4, 4, 4, 2, 8, device=MPS, requires_grad=True)

        out = na3d(q, k, v, kernel_size=3, stride=2, is_causal=(True, True, True))
        out.sum().backward()

        ref_grads = _grad_ref(na3d, [q, k, v], kernel_size=3, stride=2, is_causal=(True, True, True))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)


class TestBackwardNonUniform3D:
    def test_na3d_nonuniform_kernel_backward(self):
        q = torch.randn(1, 4, 6, 6, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 4, 6, 6, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 4, 6, 6, 2, 8, device=MPS, requires_grad=True)

        out = na3d(q, k, v, kernel_size=(3, 5, 3))
        out.sum().backward()

        ref_grads = _grad_ref(na3d, [q, k, v], kernel_size=(3, 5, 3))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)

    def test_na3d_nonuniform_dilation_backward(self):
        q = torch.randn(1, 6, 6, 6, 2, 8, device=MPS, requires_grad=True)
        k = torch.randn(1, 6, 6, 6, 2, 8, device=MPS, requires_grad=True)
        v = torch.randn(1, 6, 6, 6, 2, 8, device=MPS, requires_grad=True)

        out = na3d(q, k, v, kernel_size=3, dilation=(1, 2, 1))
        out.sum().backward()

        ref_grads = _grad_ref(na3d, [q, k, v], kernel_size=3, dilation=(1, 2, 1))
        torch.testing.assert_close(q.grad.cpu(), ref_grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad.cpu(), ref_grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad.cpu(), ref_grads[2], atol=1e-4, rtol=1e-4)
