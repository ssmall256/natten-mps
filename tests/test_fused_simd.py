"""Tests for fused SIMD cooperative kernels — numerical parity against split path."""

import torch
import pytest

from natten_mps._core import metal


def _skip_if_no_mps():
    if not metal.is_available():
        pytest.skip("MPS Metal not available")


# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_1d_basic(D, dtype):
    """Compare fused vs split path for basic 1D NA."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, L, H, K = 2, 32, 4, 7
    q = torch.randn(B, L, H, D, device="mps", dtype=dtype)
    k = torch.randn(B, L, H, D, device="mps", dtype=dtype)
    v = torch.randn(B, L, H, D, device="mps", dtype=dtype)

    scale = float(D ** -0.5)
    ks = (K,)
    dil = (1,)
    stride = (1,)
    causal = (False,)

    # Split path
    [qf, kf, vf], _ = metal._upcast_bf16(q, k, v)
    logits = metal.na1d_qk_forward(qf, kf, ks, dil, stride=stride, is_causal=causal)
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_split = metal.na1d_av_forward(attn, vf, ks, dil, stride=stride, is_causal=causal)

    # Fused path
    out_fused, lse = metal.na1d_fused_forward(qf, kf, vf, ks, dil, stride, causal, scale)

    atol = 1e-4 if dtype == torch.float32 else 5e-3
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    assert torch.allclose(out_split.float(), out_fused.float(), atol=atol, rtol=rtol), \
        f"Max diff: {(out_split.float() - out_fused.float()).abs().max().item()}"


@pytest.mark.parametrize("D", [32, 64])
def test_1d_causal(D):
    """Test fused kernel with causal masking."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, L, H, K = 1, 16, 2, 5
    q = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)

    scale = float(D ** -0.5)
    ks = (K,)
    dil = (1,)
    stride = (1,)
    causal = (True,)

    logits = metal.na1d_qk_forward(q, k, ks, dil, stride=stride, is_causal=causal)
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_split = metal.na1d_av_forward(attn, v, ks, dil, stride=stride, is_causal=causal)

    out_fused, lse = metal.na1d_fused_forward(q, k, v, ks, dil, stride, causal, scale)

    assert torch.allclose(out_split, out_fused, atol=1e-4, rtol=1e-3), \
        f"Max diff: {(out_split - out_fused).abs().max().item()}"


@pytest.mark.parametrize("stride_val", [2, 4])
def test_1d_strided(stride_val):
    """Test fused kernel with stride."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, L, H, D, K = 1, 32, 2, 64, 7
    q = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)

    scale = float(D ** -0.5)
    ks = (K,)
    dil = (1,)
    stride = (stride_val,)
    causal = (False,)

    logits = metal.na1d_qk_forward(q, k, ks, dil, stride=stride, is_causal=causal)
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_split = metal.na1d_av_forward(attn, v, ks, dil, stride=stride, is_causal=causal)

    out_fused, lse = metal.na1d_fused_forward(q, k, v, ks, dil, stride, causal, scale)

    assert torch.allclose(out_split, out_fused, atol=1e-4, rtol=1e-3), \
        f"Max diff: {(out_split - out_fused).abs().max().item()}"


def test_1d_dilation():
    """Test fused kernel with dilation > 1."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, L, H, D, K = 1, 32, 2, 64, 5
    q = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)

    scale = float(D ** -0.5)
    ks = (K,)
    dil = (2,)
    stride = (1,)
    causal = (False,)

    logits = metal.na1d_qk_forward(q, k, ks, dil, stride=stride, is_causal=causal)
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_split = metal.na1d_av_forward(attn, v, ks, dil, stride=stride, is_causal=causal)

    out_fused, lse = metal.na1d_fused_forward(q, k, v, ks, dil, stride, causal, scale)

    assert torch.allclose(out_split, out_fused, atol=1e-4, rtol=1e-3), \
        f"Max diff: {(out_split - out_fused).abs().max().item()}"


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_2d_basic(D, dtype):
    """Compare fused vs split path for basic 2D NA."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, Hi, Wi, H = 1, 8, 8, 2
    Kh, Kw = 3, 3
    q = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=dtype)
    k = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=dtype)
    v = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=dtype)

    scale = float(D ** -0.5)
    ks = (Kh, Kw)
    dil = (1, 1)
    stride = (1, 1)
    causal = (False, False)

    [qf, kf, vf], _ = metal._upcast_bf16(q, k, v)
    logits = metal.na2d_qk_forward(qf, kf, ks, dil, stride=stride, is_causal=causal)
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_split = metal.na2d_av_forward(attn, vf, ks, dil, stride=stride, is_causal=causal)

    out_fused, lse = metal.na2d_fused_forward(qf, kf, vf, ks, dil, stride, causal, scale)

    atol = 1e-4 if dtype == torch.float32 else 5e-3
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    assert torch.allclose(out_split.float(), out_fused.float(), atol=atol, rtol=rtol), \
        f"Max diff: {(out_split.float() - out_fused.float()).abs().max().item()}"


def test_2d_causal():
    """Test 2D fused kernel with causal masking."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, Hi, Wi, H, D = 1, 8, 8, 2, 64
    Kh, Kw = 3, 3
    q = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=torch.float32)
    v = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=torch.float32)

    scale = float(D ** -0.5)
    ks = (Kh, Kw)
    dil = (1, 1)
    stride = (1, 1)
    causal = (True, True)

    logits = metal.na2d_qk_forward(q, k, ks, dil, stride=stride, is_causal=causal)
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_split = metal.na2d_av_forward(attn, v, ks, dil, stride=stride, is_causal=causal)

    out_fused, lse = metal.na2d_fused_forward(q, k, v, ks, dil, stride, causal, scale)

    assert torch.allclose(out_split, out_fused, atol=1e-4, rtol=1e-3), \
        f"Max diff: {(out_split - out_fused).abs().max().item()}"


# ---------------------------------------------------------------------------
# 3D tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [32, 64])
def test_3d_basic(D):
    """Compare fused vs split path for basic 3D NA."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, Dp, Hi, Wi, H = 1, 4, 4, 4, 2
    Kd, Kh, Kw = 3, 3, 3
    q = torch.randn(B, Dp, Hi, Wi, H, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, Dp, Hi, Wi, H, D, device="mps", dtype=torch.float32)
    v = torch.randn(B, Dp, Hi, Wi, H, D, device="mps", dtype=torch.float32)

    scale = float(D ** -0.5)
    ks = (Kd, Kh, Kw)
    dil = (1, 1, 1)
    stride = (1, 1, 1)
    causal = (False, False, False)

    logits = metal.na3d_qk_forward(q, k, ks, dil, stride=stride, is_causal=causal)
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_split = metal.na3d_av_forward(attn, v, ks, dil, stride=stride, is_causal=causal)

    out_fused, lse = metal.na3d_fused_forward(q, k, v, ks, dil, stride, causal, scale)

    assert torch.allclose(out_split, out_fused, atol=1e-4, rtol=1e-3), \
        f"Max diff: {(out_split - out_fused).abs().max().item()}"


# ---------------------------------------------------------------------------
# LSE correctness
# ---------------------------------------------------------------------------


def test_1d_lse():
    """Verify LSE matches torch.logsumexp of scaled logits."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, L, H, D, K = 1, 16, 2, 64, 5
    q = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)

    scale = float(D ** -0.5)
    ks = (K,)
    dil = (1,)
    stride = (1,)
    causal = (False,)

    logits = metal.na1d_qk_forward(q, k, ks, dil, stride=stride, is_causal=causal)
    logits_scaled = logits * scale
    lse_ref = torch.logsumexp(logits_scaled, dim=-1)  # [B, L, H]

    _, lse_fused = metal.na1d_fused_forward(q, k, v, ks, dil, stride, causal, scale)
    # lse_fused is [B, H, L] (heads-first), need to transpose
    lse_fused_spatial = lse_fused.permute(0, 2, 1)  # [B, L, H]

    assert torch.allclose(lse_ref, lse_fused_spatial, atol=1e-4, rtol=1e-3), \
        f"Max diff: {(lse_ref - lse_fused_spatial).abs().max().item()}"


# ---------------------------------------------------------------------------
# Integration: na*d_forward auto-selects fused path
# ---------------------------------------------------------------------------


def test_1d_forward_integration():
    """Verify na1d_forward uses fused path and produces correct results."""
    _skip_if_no_mps()
    torch.manual_seed(42)
    B, L, H, D, K = 2, 32, 4, 64, 7
    q = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="mps", dtype=torch.float32)

    # Full forward via integration point (should auto-select fused)
    out = metal.na1d_forward(q, k, v, (K,), (1,), (1,), (False,), None)

    # Reference via split path
    scale = float(D ** -0.5)
    logits = metal.na1d_qk_forward(q, k, (K,), (1,), stride=(1,), is_causal=(False,))
    logits = logits * scale
    attn = torch.softmax(logits, dim=-1)
    out_ref = metal.na1d_av_forward(attn, v, (K,), (1,), stride=(1,), is_causal=(False,))

    assert torch.allclose(out_ref, out, atol=1e-4, rtol=1e-3), \
        f"Max diff: {(out_ref - out).abs().max().item()}"


# ---------------------------------------------------------------------------
# Autograd: gradients through fused forward path
# ---------------------------------------------------------------------------


def test_1d_autograd_fused():
    """Verify gradients work through the fused forward path."""
    _skip_if_no_mps()
    from natten_mps.functional import na1d

    torch.manual_seed(42)
    B, L, H, D, K = 1, 16, 2, 64, 5
    q = torch.randn(B, L, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, L, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, L, H, D, device="mps", dtype=torch.float32, requires_grad=True)

    # D=64 hits fused path (D%32==0, D>=32)
    assert metal._can_use_fused(D)

    out = na1d(q, k, v, kernel_size=K)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any(), "q.grad has NaN"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "k.grad has NaN"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "v.grad has NaN"


def test_2d_autograd_fused():
    """Verify gradients work through the 2D fused forward path."""
    _skip_if_no_mps()
    from natten_mps.functional import na2d

    torch.manual_seed(42)
    B, Hi, Wi, H, D = 1, 8, 8, 2, 64
    Kh, Kw = 3, 3
    q = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, Hi, Wi, H, D, device="mps", dtype=torch.float32, requires_grad=True)

    assert metal._can_use_fused(D)

    out = na2d(q, k, v, kernel_size=(Kh, Kw))
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any(), "q.grad has NaN"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "k.grad has NaN"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "v.grad has NaN"


def test_3d_autograd_fused():
    """Verify gradients work through the 3D fused forward path."""
    _skip_if_no_mps()
    from natten_mps.functional import na3d

    torch.manual_seed(42)
    B, Dp, Hi, Wi, H, D = 1, 4, 4, 4, 2, 32
    Kd, Kh, Kw = 3, 3, 3
    q = torch.randn(B, Dp, Hi, Wi, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, Dp, Hi, Wi, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, Dp, Hi, Wi, H, D, device="mps", dtype=torch.float32, requires_grad=True)

    assert metal._can_use_fused(D)

    out = na3d(q, k, v, kernel_size=(Kd, Kh, Kw))
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any(), "q.grad has NaN"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "k.grad has NaN"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "v.grad has NaN"


def test_1d_autograd_causal_fused():
    """Verify gradients with causal masking through fused path."""
    _skip_if_no_mps()
    from natten_mps.functional import na1d

    torch.manual_seed(42)
    B, L, H, D, K = 1, 16, 2, 64, 5
    q = torch.randn(B, L, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, L, H, D, device="mps", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, L, H, D, device="mps", dtype=torch.float32, requires_grad=True)

    out = na1d(q, k, v, kernel_size=K, is_causal=True)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any(), "q.grad has NaN"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "k.grad has NaN"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "v.grad has NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
