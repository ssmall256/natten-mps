"""Tests for extras/allin1 backward pass (Metal-accelerated gradients).

Tests gradient correctness via:
1. torch.autograd.gradcheck (numerical Jacobian vs analytical)
2. Metal vs pure parity (same gradients from both backends)
"""

import pytest
import torch

from natten_mps.extras.allin1 import na1d_av_fused, na1d_qk_rpb, na2d_av_fused, na2d_qk_rpb

# Use MPS if available, otherwise CPU (pure fallback)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Small sizes for gradcheck (numerical Jacobian is O(n) forward passes)
KERNEL_SIZES_1D = [3, 5, 7]
KERNEL_SIZES_2D = [3, 5, 7]


# ---------------------------------------------------------------------------
# 1D QK+RPB gradient tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", KERNEL_SIZES_1D)
def test_1d_qkrpb_gradcheck(K):
    """Numerical gradient check for 1D QK+RPB."""
    B, L, H, D = 1, max(K * 2, 8), 2, 4
    q = torch.randn(B, L, H, D, dtype=torch.float64, device="cpu", requires_grad=True)
    k = torch.randn(B, L, H, D, dtype=torch.float64, device="cpu", requires_grad=True)
    rpb = torch.randn(H, 2 * K - 1, dtype=torch.float64, device="cpu", requires_grad=True)

    def fn(q_, k_, rpb_):
        return na1d_qk_rpb(q_, k_, rpb_, kernel_size=K, dilation=1)

    assert torch.autograd.gradcheck(fn, (q, k, rpb), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("K", KERNEL_SIZES_1D)
def test_1d_qkrpb_grad_shapes(K):
    """Verify gradient shapes match input shapes for 1D QK+RPB."""
    B, L, H, D = 2, 16, 4, 8
    q = torch.randn(B, L, H, D, device=DEVICE, requires_grad=True)
    k = torch.randn(B, L, H, D, device=DEVICE, requires_grad=True)
    rpb = torch.randn(H, 2 * K - 1, device=DEVICE, requires_grad=True)

    out = na1d_qk_rpb(q, k, rpb, kernel_size=K, dilation=1)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert rpb.grad is not None and rpb.grad.shape == rpb.shape


@pytest.mark.parametrize("K", KERNEL_SIZES_1D)
def test_1d_qkrpb_dilation(K):
    """Gradient check with dilation > 1."""
    B, L, H, D = 1, max(K * 4, 16), 2, 4
    dilation = 2
    q = torch.randn(B, L, H, D, dtype=torch.float64, device="cpu", requires_grad=True)
    k = torch.randn(B, L, H, D, dtype=torch.float64, device="cpu", requires_grad=True)
    rpb = torch.randn(H, 2 * K - 1, dtype=torch.float64, device="cpu", requires_grad=True)

    def fn(q_, k_, rpb_):
        return na1d_qk_rpb(q_, k_, rpb_, kernel_size=K, dilation=dilation)

    assert torch.autograd.gradcheck(fn, (q, k, rpb), eps=1e-6, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 1D AV gradient tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", KERNEL_SIZES_1D)
def test_1d_av_gradcheck(K):
    """Numerical gradient check for 1D AV."""
    B, L, H, D = 1, max(K * 2, 8), 2, 4
    attn = torch.softmax(torch.randn(B, L, H, K, dtype=torch.float64, device="cpu"), dim=-1).requires_grad_(True)
    v = torch.randn(B, L, H, D, dtype=torch.float64, device="cpu", requires_grad=True)

    def fn(a_, v_):
        return na1d_av_fused(a_, v_, kernel_size=K, dilation=1)

    assert torch.autograd.gradcheck(fn, (attn, v), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("K", KERNEL_SIZES_1D)
def test_1d_av_grad_shapes(K):
    """Verify gradient shapes match input shapes for 1D AV."""
    B, L, H, D = 2, 16, 4, 8
    attn = torch.softmax(torch.randn(B, L, H, K, device=DEVICE), dim=-1).requires_grad_(True)
    v = torch.randn(B, L, H, D, device=DEVICE, requires_grad=True)

    out = na1d_av_fused(attn, v, kernel_size=K, dilation=1)
    loss = out.sum()
    loss.backward()

    assert attn.grad is not None and attn.grad.shape == attn.shape
    assert v.grad is not None and v.grad.shape == v.shape


# ---------------------------------------------------------------------------
# 2D QK+RPB gradient tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", KERNEL_SIZES_2D)
def test_2d_qkrpb_gradcheck(K):
    """Numerical gradient check for 2D QK+RPB."""
    B, Hh, Hw, H, D = 1, max(K * 2, 8), max(K * 2, 8), 2, 4
    q = torch.randn(B, Hh, Hw, H, D, dtype=torch.float64, device="cpu", requires_grad=True)
    k = torch.randn(B, Hh, Hw, H, D, dtype=torch.float64, device="cpu", requires_grad=True)
    rpb = torch.randn(H, 2 * K - 1, 2 * K - 1, dtype=torch.float64, device="cpu", requires_grad=True)

    def fn(q_, k_, rpb_):
        return na2d_qk_rpb(q_, k_, rpb_, kernel_size=K, dilation=1)

    assert torch.autograd.gradcheck(fn, (q, k, rpb), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("K", KERNEL_SIZES_2D)
def test_2d_qkrpb_grad_shapes(K):
    """Verify gradient shapes match input shapes for 2D QK+RPB."""
    B, Hh, Hw, H, D = 2, 8, 8, 4, 8
    q = torch.randn(B, Hh, Hw, H, D, device=DEVICE, requires_grad=True)
    k = torch.randn(B, Hh, Hw, H, D, device=DEVICE, requires_grad=True)
    rpb = torch.randn(H, 2 * K - 1, 2 * K - 1, device=DEVICE, requires_grad=True)

    out = na2d_qk_rpb(q, k, rpb, kernel_size=K, dilation=1)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert rpb.grad is not None and rpb.grad.shape == rpb.shape


# ---------------------------------------------------------------------------
# 2D AV gradient tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", KERNEL_SIZES_2D)
def test_2d_av_gradcheck(K):
    """Numerical gradient check for 2D AV."""
    B, Hh, Hw, H, D = 1, max(K * 2, 8), max(K * 2, 8), 2, 4
    attn = torch.softmax(
        torch.randn(B, Hh, Hw, H, K * K, dtype=torch.float64, device="cpu"), dim=-1
    ).requires_grad_(True)
    v = torch.randn(B, Hh, Hw, H, D, dtype=torch.float64, device="cpu", requires_grad=True)

    def fn(a_, v_):
        return na2d_av_fused(a_, v_, kernel_size=K, dilation=1)

    assert torch.autograd.gradcheck(fn, (attn, v), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("K", KERNEL_SIZES_2D)
def test_2d_av_grad_shapes(K):
    """Verify gradient shapes match input shapes for 2D AV."""
    B, Hh, Hw, H, D = 2, 8, 8, 4, 8
    attn = torch.softmax(
        torch.randn(B, Hh, Hw, H, K * K, device=DEVICE), dim=-1
    ).requires_grad_(True)
    v = torch.randn(B, Hh, Hw, H, D, device=DEVICE, requires_grad=True)

    out = na2d_av_fused(attn, v, kernel_size=K, dilation=1)
    loss = out.sum()
    loss.backward()

    assert attn.grad is not None and attn.grad.shape == attn.shape
    assert v.grad is not None and v.grad.shape == v.shape


# ---------------------------------------------------------------------------
# Metal vs pure parity tests (MPS only)
# ---------------------------------------------------------------------------


def _run_with_backend(fn, backend_env):
    """Run fn with a specific backend by temporarily setting env var."""
    import os
    old = os.environ.get("NATTEN_BACKEND")
    os.environ["NATTEN_BACKEND"] = backend_env
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("NATTEN_BACKEND", None)
        else:
            os.environ["NATTEN_BACKEND"] = old


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("K", [3, 5, 7])
def test_1d_qkrpb_metal_vs_pure_grad(K):
    """Compare Metal backward vs pure PyTorch backward for 1D QK+RPB."""
    B, L, H, D = 1, 16, 2, 8
    torch.manual_seed(42)

    # Metal path (on MPS)
    q_m = torch.randn(B, L, H, D, device="mps", requires_grad=True)
    k_m = torch.randn(B, L, H, D, device="mps", requires_grad=True)
    rpb_m = torch.randn(H, 2 * K - 1, device="mps", requires_grad=True)

    out_m = na1d_qk_rpb(q_m, k_m, rpb_m, kernel_size=K, dilation=1)
    out_m.sum().backward()

    # Pure path (on CPU)
    q_c = q_m.detach().cpu().requires_grad_(True)
    k_c = k_m.detach().cpu().requires_grad_(True)
    rpb_c = rpb_m.detach().cpu().requires_grad_(True)

    out_c = na1d_qk_rpb(q_c, k_c, rpb_c, kernel_size=K, dilation=1)
    out_c.sum().backward()

    torch.testing.assert_close(q_m.grad.cpu(), q_c.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(k_m.grad.cpu(), k_c.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(rpb_m.grad.cpu(), rpb_c.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("K", [3, 5, 7])
def test_1d_av_metal_vs_pure_grad(K):
    """Compare Metal backward vs pure PyTorch backward for 1D AV."""
    B, L, H, D = 1, 16, 2, 8
    torch.manual_seed(42)

    attn_m = torch.softmax(torch.randn(B, L, H, K, device="mps"), dim=-1).requires_grad_(True)
    v_m = torch.randn(B, L, H, D, device="mps", requires_grad=True)

    out_m = na1d_av_fused(attn_m, v_m, kernel_size=K, dilation=1)
    out_m.sum().backward()

    attn_c = attn_m.detach().cpu().requires_grad_(True)
    v_c = v_m.detach().cpu().requires_grad_(True)

    out_c = na1d_av_fused(attn_c, v_c, kernel_size=K, dilation=1)
    out_c.sum().backward()

    torch.testing.assert_close(attn_m.grad.cpu(), attn_c.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(v_m.grad.cpu(), v_c.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("K", [3, 5, 7])
def test_2d_qkrpb_metal_vs_pure_grad(K):
    """Compare Metal backward vs pure PyTorch backward for 2D QK+RPB."""
    B, Hh, Hw, H, D = 1, 8, 8, 2, 8
    torch.manual_seed(42)

    q_m = torch.randn(B, Hh, Hw, H, D, device="mps", requires_grad=True)
    k_m = torch.randn(B, Hh, Hw, H, D, device="mps", requires_grad=True)
    rpb_m = torch.randn(H, 2 * K - 1, 2 * K - 1, device="mps", requires_grad=True)

    out_m = na2d_qk_rpb(q_m, k_m, rpb_m, kernel_size=K, dilation=1)
    out_m.sum().backward()

    q_c = q_m.detach().cpu().requires_grad_(True)
    k_c = k_m.detach().cpu().requires_grad_(True)
    rpb_c = rpb_m.detach().cpu().requires_grad_(True)

    out_c = na2d_qk_rpb(q_c, k_c, rpb_c, kernel_size=K, dilation=1)
    out_c.sum().backward()

    torch.testing.assert_close(q_m.grad.cpu(), q_c.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(k_m.grad.cpu(), k_c.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(rpb_m.grad.cpu(), rpb_c.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("K", [3, 5, 7])
def test_2d_av_metal_vs_pure_grad(K):
    """Compare Metal backward vs pure PyTorch backward for 2D AV."""
    B, Hh, Hw, H, D = 1, 8, 8, 2, 8
    torch.manual_seed(42)

    attn_m = torch.softmax(
        torch.randn(B, Hh, Hw, H, K * K, device="mps"), dim=-1
    ).requires_grad_(True)
    v_m = torch.randn(B, Hh, Hw, H, D, device="mps", requires_grad=True)

    out_m = na2d_av_fused(attn_m, v_m, kernel_size=K, dilation=1)
    out_m.sum().backward()

    attn_c = attn_m.detach().cpu().requires_grad_(True)
    v_c = v_m.detach().cpu().requires_grad_(True)

    out_c = na2d_av_fused(attn_c, v_c, kernel_size=K, dilation=1)
    out_c.sum().backward()

    torch.testing.assert_close(attn_m.grad.cpu(), attn_c.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(v_m.grad.cpu(), v_c.grad, atol=1e-5, rtol=1e-4)
