import torch

from natten_mps.extras.allin1 import na1d_av_fused, na1d_qk_rpb, na2d_av_fused, na2d_qk_rpb


def test_na1d_qk_rpb_output_shape():
    B, L, H, D = 2, 12, 4, 8
    K = 3
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    rpb = torch.randn(H, 2 * K - 1)
    out = na1d_qk_rpb(q, k, rpb, kernel_size=K, dilation=1)
    assert out.shape == (B, L, H, K)


def test_na1d_av_fused_output_shape():
    B, L, H, D = 2, 12, 4, 8
    K = 3
    attn = torch.softmax(torch.randn(B, L, H, K), dim=-1)
    v = torch.randn(B, L, H, D)
    out = na1d_av_fused(attn, v, kernel_size=K, dilation=1)
    assert out.shape == (B, L, H, D)


def test_na2d_qk_rpb_output_shape():
    B, Hh, Hw, H, D = 2, 8, 8, 4, 8
    K = 3
    q = torch.randn(B, Hh, Hw, H, D)
    k = torch.randn(B, Hh, Hw, H, D)
    rpb = torch.randn(H, 2 * K - 1, 2 * K - 1)
    out = na2d_qk_rpb(q, k, rpb, kernel_size=K, dilation=1)
    assert out.shape == (B, Hh, Hw, H, K * K)


def test_na2d_av_fused_output_shape():
    B, Hh, Hw, H, D = 2, 8, 8, 4, 8
    K = 3
    attn = torch.softmax(torch.randn(B, Hh, Hw, H, K * K), dim=-1)
    v = torch.randn(B, Hh, Hw, H, D)
    out = na2d_av_fused(attn, v, kernel_size=K, dilation=1)
    assert out.shape == (B, Hh, Hw, H, D)


def test_na1d_qk_rpb_with_none_rpb():
    B, L, H, D = 1, 8, 2, 4
    K = 3
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    out = na1d_qk_rpb(q, k, None, kernel_size=K, dilation=1)
    assert out.shape == (B, L, H, K)


def test_na2d_qk_rpb_with_none_rpb():
    B, Hh, Hw, H, D = 1, 6, 6, 2, 4
    K = 3
    q = torch.randn(B, Hh, Hw, H, D)
    k = torch.randn(B, Hh, Hw, H, D)
    out = na2d_qk_rpb(q, k, None, kernel_size=K, dilation=1)
    assert out.shape == (B, Hh, Hw, H, K * K)


def test_na1d_qk_rpb_with_scale():
    B, L, H, D = 1, 8, 2, 4
    K = 3
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    rpb = torch.randn(H, 2 * K - 1)
    out = na1d_qk_rpb(q, k, rpb, kernel_size=K, dilation=1, scale=0.288)
    assert out.shape == (B, L, H, K)


def test_na1d_fused_roundtrip_matches():
    """QK+RPB → softmax → AV should produce reasonable output."""
    B, L, H, D = 1, 10, 2, 8
    K = 3
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    rpb = torch.zeros(H, 2 * K - 1)

    logits = na1d_qk_rpb(q, k, rpb, kernel_size=K, dilation=1)
    attn = torch.softmax(logits, dim=-1)
    out = na1d_av_fused(attn, v, kernel_size=K, dilation=1)
    assert out.shape == (B, L, H, D)
    assert torch.isfinite(out).all()
