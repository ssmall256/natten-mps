"""Tests for the Nanobind (Tier-2) compiled Metal extension backend.

Skipped unless the nanobind extension is installed and available.
When available, verifies parity with the pure backend.
"""

import pytest
import torch

# Skip entire module if nanobind extension not built
nb_ext = pytest.importorskip(
    "natten_mps._core._nanobind_ext",
    reason="nanobind extension not built — install with pip install -e '.[nanobind]'",
)

import natten_mps
from natten_mps.functional import na1d, na1d_av, na1d_qk, na2d, na3d

MPS = torch.device("mps")


@pytest.fixture(autouse=True)
def _use_nanobind_backend():
    prev = natten_mps.get_backend()
    natten_mps.set_backend("nanobind")
    yield
    natten_mps.set_backend(prev)


def _pure_ref(fn, *args, **kwargs):
    prev = natten_mps.get_backend()
    natten_mps.set_backend("pure")
    try:
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        cpu_kwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        return fn(*cpu_args, **cpu_kwargs)
    finally:
        natten_mps.set_backend(prev)


class TestNanobindAvailability:
    def test_extension_loads(self):
        assert nb_ext.is_available()


class TestNanobind1D:
    def test_na1d_qk_matches_pure(self):
        q = torch.randn(2, 12, 4, 8, device=MPS)
        k = torch.randn(2, 12, 4, 8, device=MPS)

        logits_nb = na1d_qk(q, k, kernel_size=3).cpu()
        logits_pure = _pure_ref(na1d_qk, q, k, kernel_size=3)
        torch.testing.assert_close(logits_nb, logits_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_av_matches_pure(self):
        attn = torch.softmax(torch.randn(2, 10, 4, 5, device=MPS), dim=-1)
        v = torch.randn(2, 10, 4, 8, device=MPS)

        out_nb = na1d_av(attn, v, kernel_size=5).cpu()
        out_pure = _pure_ref(na1d_av, attn, v, kernel_size=5)
        torch.testing.assert_close(out_nb, out_pure, atol=1e-5, rtol=1e-5)

    def test_na1d_fused_matches_pure(self):
        q = torch.randn(1, 16, 2, 8, device=MPS)
        k = torch.randn(1, 16, 2, 8, device=MPS)
        v = torch.randn(1, 16, 2, 8, device=MPS)

        out_nb = na1d(q, k, v, kernel_size=5).cpu()
        out_pure = _pure_ref(na1d, q, k, v, kernel_size=5)
        torch.testing.assert_close(out_nb, out_pure, atol=1e-5, rtol=1e-5)
