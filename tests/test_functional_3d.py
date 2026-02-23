import pytest
import torch

from natten_mps.functional import na3d, na3d_av, na3d_qk


def test_na3d_output_shape_basic():
    q = torch.randn(2, 5, 6, 7, 2, 4)
    k = torch.randn(2, 5, 6, 7, 2, 4)
    v = torch.randn(2, 5, 6, 7, 2, 4)
    out = na3d(q, k, v, kernel_size=3)
    assert out.shape == (2, 5, 6, 7, 2, 4)


def test_na3d_output_shape_with_stride():
    q = torch.randn(2, 9, 8, 7, 2, 4)
    k = torch.randn(2, 9, 8, 7, 2, 4)
    v = torch.randn(2, 9, 8, 7, 2, 4)
    out = na3d(q, k, v, kernel_size=3, stride=(2, 3, 2))
    assert out.shape == (2, 5, 3, 4, 2, 4)


def test_na3d_with_dilation():
    q = torch.randn(1, 7, 7, 7, 2, 4)
    k = torch.randn(1, 7, 7, 7, 2, 4)
    v = torch.randn(1, 7, 7, 7, 2, 4)
    out = na3d(q, k, v, kernel_size=3, dilation=2)
    assert out.shape == (1, 7, 7, 7, 2, 4)


def test_na3d_with_causal():
    q = torch.randn(1, 5, 5, 5, 2, 4)
    k = torch.randn(1, 5, 5, 5, 2, 4)
    v = torch.randn(1, 5, 5, 5, 2, 4)
    out = na3d(q, k, v, kernel_size=3, is_causal=True)
    assert out.shape == (1, 5, 5, 5, 2, 4)


def test_na3d_equals_global_attention_when_kernel_covers_volume():
    """When kernel covers the full volume, na3d should match global attention."""
    B, D, H, W, heads, dim = 1, 3, 3, 3, 1, 4
    q = torch.randn(B, D, H, W, heads, dim)
    k = torch.randn(B, D, H, W, heads, dim)
    v = torch.randn(B, D, H, W, heads, dim)

    na_out = na3d(q, k, v, kernel_size=3)

    q_flat = q.reshape(B, D * H * W, heads, dim)
    k_flat = k.reshape(B, D * H * W, heads, dim)
    v_flat = v.reshape(B, D * H * W, heads, dim)
    scale = dim ** -0.5
    logits = torch.einsum("blhd,bmhd->blhm", q_flat, k_flat) * scale
    attn = torch.softmax(logits, dim=-1)
    global_out = torch.einsum("blhm,bmhd->blhd", attn, v_flat).reshape(B, D, H, W, heads, dim)

    torch.testing.assert_close(na_out, global_out, atol=1e-5, rtol=1e-5)


def test_na3d_split_qk_av_matches_fused():
    """Split QK returns unscaled logits; apply scale manually to match fused path."""
    B, D, H, W, heads, dim = 1, 5, 5, 5, 2, 4
    q = torch.randn(B, D, H, W, heads, dim)
    k = torch.randn(B, D, H, W, heads, dim)
    v = torch.randn(B, D, H, W, heads, dim)

    fused = na3d(q, k, v, kernel_size=3)
    logits = na3d_qk(q, k, kernel_size=3)
    scale = dim ** -0.5
    attn = torch.softmax(logits * scale, dim=-1)
    split_out = na3d_av(attn, v, kernel_size=3)

    torch.testing.assert_close(fused, split_out, atol=1e-5, rtol=1e-5)


def test_na3d_qk_output_shape():
    q = torch.randn(1, 5, 6, 7, 2, 4)
    k = torch.randn(1, 5, 6, 7, 2, 4)
    logits = na3d_qk(q, k, kernel_size=3)
    assert logits.shape == (1, 5, 6, 7, 2, 27)


def test_na3d_av_output_shape():
    attn = torch.randn(1, 5, 6, 7, 2, 27)
    attn = torch.softmax(attn, dim=-1)
    v = torch.randn(1, 5, 6, 7, 2, 4)
    out = na3d_av(attn, v, kernel_size=3)
    assert out.shape == (1, 5, 6, 7, 2, 4)


def test_na3d_int_and_tuple_kernel_equivalence():
    q = torch.randn(1, 5, 5, 5, 1, 4)
    k = torch.randn(1, 5, 5, 5, 1, 4)
    v = torch.randn(1, 5, 5, 5, 1, 4)
    out_int = na3d(q, k, v, kernel_size=3)
    out_tuple = na3d(q, k, v, kernel_size=(3, 3, 3))
    torch.testing.assert_close(out_int, out_tuple)


def test_na3d_kernel_larger_than_input_raises():
    q = torch.randn(1, 3, 3, 3, 1, 4)
    k = torch.randn(1, 3, 3, 3, 1, 4)
    v = torch.randn(1, 3, 3, 3, 1, 4)
    with pytest.raises(ValueError):
        na3d(q, k, v, kernel_size=5)


def test_na3d_stride_larger_than_kernel_raises():
    q = torch.randn(1, 9, 9, 9, 1, 4)
    k = torch.randn(1, 9, 9, 9, 1, 4)
    v = torch.randn(1, 9, 9, 9, 1, 4)
    with pytest.raises(ValueError):
        na3d(q, k, v, kernel_size=3, stride=5)


def test_na3d_wrong_ndim_raises():
    q = torch.randn(1, 5, 5, 1, 4)
    k = torch.randn(1, 5, 5, 1, 4)
    v = torch.randn(1, 5, 5, 1, 4)
    with pytest.raises(ValueError, match="na3d"):
        na3d(q, k, v, kernel_size=3)
