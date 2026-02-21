import pytest
import torch

from natten_mps.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk
from natten_mps.utils.window import get_window_start_vectorized


def _global_attention_1d(q, k, v, scale):
    scores = torch.einsum("blhd,bshd->blhs", q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("blhs,bshd->blhd", attn, v)


def _global_attention_2d(q, k, v, scale):
    bsz, height, width, heads, dim = q.shape
    qf = q.reshape(bsz, height * width, heads, dim)
    kf = k.reshape(bsz, height * width, heads, dim)
    vf = v.reshape(bsz, height * width, heads, dim)

    scores = torch.einsum("bshd,bthd->bsht", qf, kf) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bsht,bthd->bshd", attn, vf)
    return out.reshape(bsz, height, width, heads, dim)


def _na1d_reference_shifted(q, k, v, kernel_size, dilation):
    bsz, length, heads, dim = q.shape
    scale = dim ** -0.5
    starts = get_window_start_vectorized(
        torch.arange(length, device=q.device, dtype=torch.long),
        length=length,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    offsets = torch.arange(kernel_size, device=q.device, dtype=torch.long) * dilation
    key_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)

    out = torch.empty_like(q)
    for i in range(length):
        k_neighborhood = k[:, key_idx[i]]  # [B, K, H, D]
        v_neighborhood = v[:, key_idx[i]]
        logits = torch.einsum("bhd,bkhd->bhk", q[:, i], k_neighborhood) * scale
        attn = torch.softmax(logits, dim=-1)
        out[:, i] = torch.einsum("bhk,bkhd->bhd", attn, v_neighborhood)
    return out


def _na2d_reference_shifted(q, k, v, kernel_size, dilation):
    bsz, height, width, heads, dim = q.shape
    kh, kw = kernel_size
    dh, dw = dilation
    scale = dim ** -0.5

    starts_h = get_window_start_vectorized(
        torch.arange(height, device=q.device, dtype=torch.long),
        length=height,
        kernel_size=kh,
        dilation=dh,
    )
    starts_w = get_window_start_vectorized(
        torch.arange(width, device=q.device, dtype=torch.long),
        length=width,
        kernel_size=kw,
        dilation=dw,
    )
    offs_h = torch.arange(kh, device=q.device, dtype=torch.long) * dh
    offs_w = torch.arange(kw, device=q.device, dtype=torch.long) * dw

    out = torch.empty_like(q)
    for i in range(height):
        h_idx = starts_h[i] + offs_h
        for j in range(width):
            w_idx = starts_w[j] + offs_w
            k_neighborhood = []
            v_neighborhood = []
            for hh in h_idx:
                for ww in w_idx:
                    k_neighborhood.append(k[:, hh, ww])  # [B, heads, dim]
                    v_neighborhood.append(v[:, hh, ww])
            k_neighborhood = torch.stack(k_neighborhood, dim=1)  # [B, Kh*Kw, heads, dim]
            v_neighborhood = torch.stack(v_neighborhood, dim=1)
            logits = torch.einsum("bhd,bkhd->bhk", q[:, i, j], k_neighborhood) * scale
            attn = torch.softmax(logits, dim=-1)
            out[:, i, j] = torch.einsum("bhk,bkhd->bhd", attn, v_neighborhood)
    return out


def _na1d_qk_reference_unscaled(q, k, kernel_size, dilation):
    bsz, length, heads, _ = q.shape
    starts = get_window_start_vectorized(
        torch.arange(length, device=q.device, dtype=torch.long),
        length=length,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    offsets = torch.arange(kernel_size, device=q.device, dtype=torch.long) * dilation
    key_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)

    out = torch.empty((bsz, length, heads, kernel_size), device=q.device, dtype=q.dtype)
    for i in range(length):
        k_neighborhood = k[:, key_idx[i]]
        out[:, i] = torch.einsum("bhd,bkhd->bhk", q[:, i], k_neighborhood)
    return out


def test_na1d_basic_shape():
    q = torch.randn(2, 16, 4, 8)
    k = torch.randn(2, 16, 4, 8)
    v = torch.randn(2, 16, 4, 8)

    out = na1d(q, k, v, kernel_size=7)
    assert out.shape == (2, 16, 4, 8)


def test_na2d_basic_shape():
    q = torch.randn(2, 8, 8, 4, 8)
    k = torch.randn(2, 8, 8, 4, 8)
    v = torch.randn(2, 8, 8, 4, 8)

    out = na2d(q, k, v, kernel_size=(3, 3))
    assert out.shape == (2, 8, 8, 4, 8)


def test_na1d_stride_and_dilation_shape():
    q = torch.randn(2, 17, 3, 8)
    k = torch.randn(2, 17, 3, 8)
    v = torch.randn(2, 17, 3, 8)

    out = na1d(q, k, v, kernel_size=5, stride=2, dilation=1)
    assert out.shape == (2, 9, 3, 8)


def test_na2d_stride_and_dilation_shape():
    q = torch.randn(2, 9, 10, 3, 8)
    k = torch.randn(2, 9, 10, 3, 8)
    v = torch.randn(2, 9, 10, 3, 8)

    out = na2d(q, k, v, kernel_size=(3, 3), stride=(2, 3), dilation=(1, 1))
    assert out.shape == (2, 5, 4, 3, 8)


def test_na1d_equals_global_attention_when_kernel_covers_sequence():
    q = torch.randn(1, 7, 2, 4)
    k = torch.randn(1, 7, 2, 4)
    v = torch.randn(1, 7, 2, 4)
    scale = q.shape[-1] ** -0.5

    out = na1d(q, k, v, kernel_size=7)
    ref = _global_attention_1d(q, k, v, scale)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na2d_equals_global_attention_when_kernel_covers_spatial_extent():
    q = torch.randn(1, 4, 5, 2, 4)
    k = torch.randn(1, 4, 5, 2, 4)
    v = torch.randn(1, 4, 5, 2, 4)
    scale = q.shape[-1] ** -0.5

    out = na2d(q, k, v, kernel_size=(4, 5))
    ref = _global_attention_2d(q, k, v, scale)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na1d_matches_shifted_boundary_reference_with_dilation():
    q = torch.randn(1, 7, 2, 3)
    k = torch.randn(1, 7, 2, 3)
    v = torch.randn(1, 7, 2, 3)

    out = na1d(q, k, v, kernel_size=3, dilation=2)
    ref = _na1d_reference_shifted(q, k, v, kernel_size=3, dilation=2)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na2d_matches_shifted_boundary_reference_with_dilation():
    q = torch.randn(1, 7, 8, 2, 3)
    k = torch.randn(1, 7, 8, 2, 3)
    v = torch.randn(1, 7, 8, 2, 3)

    out = na2d(q, k, v, kernel_size=(3, 3), dilation=(2, 2))
    ref = _na2d_reference_shifted(q, k, v, kernel_size=(3, 3), dilation=(2, 2))
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na1d_qk_returns_unscaled_logits():
    q = torch.randn(1, 7, 2, 3)
    k = torch.randn(1, 7, 2, 3)

    logits = na1d_qk(q, k, kernel_size=3, dilation=2)
    ref = _na1d_qk_reference_unscaled(q, k, kernel_size=3, dilation=2)
    assert torch.allclose(logits, ref, atol=1e-5, rtol=1e-5)


def test_na2d_qk_returns_unscaled_logits():
    q = torch.randn(1, 7, 8, 2, 3)
    k = torch.randn(1, 7, 8, 2, 3)

    logits = na2d_qk(q, k, kernel_size=(3, 3), dilation=(2, 2))

    bsz, height, width, heads, _ = q.shape
    ref = torch.empty((bsz, height, width, heads, 9), dtype=q.dtype, device=q.device)

    starts_h = get_window_start_vectorized(
        torch.arange(height, device=q.device, dtype=torch.long),
        length=height,
        kernel_size=3,
        dilation=2,
    )
    starts_w = get_window_start_vectorized(
        torch.arange(width, device=q.device, dtype=torch.long),
        length=width,
        kernel_size=3,
        dilation=2,
    )
    offs = torch.arange(3, device=q.device, dtype=torch.long) * 2

    for i in range(height):
        h_idx = starts_h[i] + offs
        for j in range(width):
            w_idx = starts_w[j] + offs
            k_neighborhood = []
            for hh in h_idx:
                for ww in w_idx:
                    k_neighborhood.append(k[:, hh, ww])
            k_neighborhood = torch.stack(k_neighborhood, dim=1)
            ref[:, i, j] = torch.einsum("bhd,bkhd->bhk", q[:, i, j], k_neighborhood)

    assert torch.allclose(logits, ref, atol=1e-5, rtol=1e-5)


def test_na1d_causal_first_position_attends_only_to_itself():
    q = torch.randn(2, 8, 3, 4)
    k = torch.randn(2, 8, 3, 4)
    v = torch.randn(2, 8, 3, 4)

    out = na1d(q, k, v, kernel_size=5, is_causal=True)
    assert torch.allclose(out[:, 0], v[:, 0], atol=1e-5, rtol=1e-5)


def test_int_vs_tuple_parameter_normalization_equivalence():
    q = torch.randn(1, 10, 2, 4)
    k = torch.randn(1, 10, 2, 4)
    v = torch.randn(1, 10, 2, 4)

    out_int = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False)
    out_tuple = na1d(q, k, v, kernel_size=(3,), stride=(1,), dilation=(1,), is_causal=(False,))
    assert torch.allclose(out_int, out_tuple, atol=1e-6, rtol=1e-6)


def test_invalid_parameters_raise_value_error():
    q1 = torch.randn(1, 6, 2, 4)
    k1 = torch.randn(1, 6, 2, 4)
    v1 = torch.randn(1, 6, 2, 4)

    with pytest.raises(ValueError):
        na1d(q1, k1, v1, kernel_size=7)

    with pytest.raises(ValueError):
        na1d(q1, k1, v1, kernel_size=3, stride=4)

    with pytest.raises(ValueError):
        na1d(q1, k1, v1, kernel_size=4, dilation=2)

    q3 = torch.randn(1, 5, 2, 4)
    k3 = torch.randn(1, 5, 2, 4)
    v3 = torch.randn(1, 5, 2, 4)
    with pytest.raises(ValueError):
        na1d(q3, k3, v3, kernel_size=3, dilation=2)

    q2 = torch.randn(1, 5, 5, 2, 4)
    k2 = torch.randn(1, 5, 5, 2, 4)
    v2 = torch.randn(1, 5, 5, 2, 4)

    with pytest.raises(ValueError):
        na2d(q2, k2, v2, kernel_size=(6, 3))

    q4 = torch.randn(1, 6, 5, 2, 4)
    k4 = torch.randn(1, 6, 5, 2, 4)
    v4 = torch.randn(1, 6, 5, 2, 4)
    with pytest.raises(ValueError):
        na2d(q4, k4, v4, kernel_size=(3, 3), dilation=(2, 2))


def test_na1d_av_validates_shapes_before_backend_dispatch():
    attn = torch.randn(1, 6, 2, 3)
    value = torch.randn(1, 6, 3, 4)
    with pytest.raises(ValueError):
        na1d_av(attn, value, kernel_size=3)

    attn_bad_kernel = torch.randn(1, 6, 2, 5)
    value_ok = torch.randn(1, 6, 2, 4)
    with pytest.raises(ValueError):
        na1d_av(attn_bad_kernel, value_ok, kernel_size=3)


def test_na2d_av_validates_shapes_before_backend_dispatch():
    attn = torch.randn(1, 4, 5, 2, 9)
    value = torch.randn(1, 4, 5, 3, 4)
    with pytest.raises(ValueError):
        na2d_av(attn, value, kernel_size=(3, 3))

    attn_bad_kernel = torch.randn(1, 4, 5, 2, 8)
    value_ok = torch.randn(1, 4, 5, 2, 4)
    with pytest.raises(ValueError):
        na2d_av(attn_bad_kernel, value_ok, kernel_size=(3, 3))
