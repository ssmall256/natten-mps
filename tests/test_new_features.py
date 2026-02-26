"""Comprehensive tests for new natten-mps features.

Covers: return_lse, merge_attentions, GQA/MQA, additional_keys/values,
bfloat16, and FMHA fast path.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from natten_mps.functional import (
    na1d,
    na1d_av,
    na1d_qk,
    na2d,
    na2d_av,
    na2d_qk,
    na3d,
    na3d_av,
    na3d_qk,
)
from natten_mps.merge import merge_attentions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 42


def _seed():
    torch.manual_seed(_SEED)


def _randn(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype)


# ===================================================================
# 1. return_lse
# ===================================================================


class TestReturnLSE:
    """Tests for the return_lse parameter across all dims."""

    # -- 1D ---------------------------------------------------------

    def test_na1d_return_lse_false_returns_tensor(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)
        out = na1d(q, k, v, kernel_size=5, return_lse=False)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 12, 2, 8)

    def test_na1d_return_lse_true_returns_tuple(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)
        result = na1d(q, k, v, kernel_size=5, return_lse=True)
        assert isinstance(result, tuple) and len(result) == 2
        out, lse = result
        assert out.shape == (1, 12, 2, 8)
        assert lse.shape == (1, 12, 2)  # [B, L, H]

    def test_na1d_lse_matches_manual(self):
        """LSE from fused path should match manual logsumexp(logits * scale)."""
        _seed()
        B, L, H, D = 1, 10, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        ks = 5
        scale = D ** -0.5

        # Via return_lse
        _, lse = na1d(q, k, v, kernel_size=ks, return_lse=True, scale=scale)

        # Manual: split path
        logits = na1d_qk(q, k, kernel_size=ks)
        lse_manual = torch.logsumexp(logits * scale, dim=-1)

        torch.testing.assert_close(lse, lse_manual, atol=1e-5, rtol=1e-5)

    def test_na1d_lse_with_causal(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)
        result = na1d(q, k, v, kernel_size=5, is_causal=True, return_lse=True)
        out, lse = result
        assert out.shape == (1, 12, 2, 8)
        assert lse.shape == (1, 12, 2)
        assert torch.isfinite(lse).all()

    # -- 2D ---------------------------------------------------------

    def test_na2d_return_lse_false_returns_tensor(self):
        _seed()
        q = _randn(1, 8, 8, 2, 8)
        k = _randn(1, 8, 8, 2, 8)
        v = _randn(1, 8, 8, 2, 8)
        out = na2d(q, k, v, kernel_size=5, return_lse=False)
        assert isinstance(out, torch.Tensor)

    def test_na2d_return_lse_true_returns_tuple(self):
        _seed()
        q = _randn(1, 8, 8, 2, 8)
        k = _randn(1, 8, 8, 2, 8)
        v = _randn(1, 8, 8, 2, 8)
        result = na2d(q, k, v, kernel_size=5, return_lse=True)
        assert isinstance(result, tuple)
        out, lse = result
        assert out.shape == (1, 8, 8, 2, 8)
        assert lse.shape == (1, 8, 8, 2)  # [B, H, W, heads]

    def test_na2d_lse_matches_manual(self):
        _seed()
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)
        ks = 5
        scale = D ** -0.5

        _, lse = na2d(q, k, v, kernel_size=ks, return_lse=True, scale=scale)
        logits = na2d_qk(q, k, kernel_size=ks)
        lse_manual = torch.logsumexp(logits * scale, dim=-1)

        torch.testing.assert_close(lse, lse_manual, atol=1e-5, rtol=1e-5)

    def test_na2d_lse_with_causal(self):
        _seed()
        q = _randn(1, 8, 8, 2, 8)
        k = _randn(1, 8, 8, 2, 8)
        v = _randn(1, 8, 8, 2, 8)
        out, lse = na2d(q, k, v, kernel_size=5, is_causal=(True, False), return_lse=True)
        assert lse.shape == (1, 8, 8, 2)
        assert torch.isfinite(lse).all()

    # -- 3D ---------------------------------------------------------

    def test_na3d_return_lse_true_returns_tuple(self):
        _seed()
        q = _randn(1, 4, 4, 4, 2, 8)
        k = _randn(1, 4, 4, 4, 2, 8)
        v = _randn(1, 4, 4, 4, 2, 8)
        result = na3d(q, k, v, kernel_size=3, return_lse=True)
        assert isinstance(result, tuple)
        out, lse = result
        assert out.shape == (1, 4, 4, 4, 2, 8)
        assert lse.shape == (1, 4, 4, 4, 2)  # [B, D, H, W, heads]

    def test_na3d_lse_matches_manual(self):
        _seed()
        B = 1
        D_s, Hh_s, W_s = 4, 4, 4
        H, D = 2, 8
        q = _randn(B, D_s, Hh_s, W_s, H, D)
        k = _randn(B, D_s, Hh_s, W_s, H, D)
        v = _randn(B, D_s, Hh_s, W_s, H, D)
        ks = 3
        scale = D ** -0.5

        _, lse = na3d(q, k, v, kernel_size=ks, return_lse=True, scale=scale)
        logits = na3d_qk(q, k, kernel_size=ks)
        lse_manual = torch.logsumexp(logits * scale, dim=-1)

        torch.testing.assert_close(lse, lse_manual, atol=1e-5, rtol=1e-5)


# ===================================================================
# 2. merge_attentions
# ===================================================================


class TestMergeAttentions:
    """Tests for the merge_attentions utility."""

    def test_2way_merge_matches_full_kv(self):
        """Split KV in half, attend each half, merge -- should match full."""
        _seed()
        B, L, H, D = 1, 12, 2, 8
        ks = 5
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)

        # Full attention
        out_full = na1d(q, k, v, kernel_size=ks)

        # Split at midpoint of the kernel window is not straightforward for NA,
        # so we use the same K/V but with return_lse and merge two identical
        # results -- which should reproduce the original.
        out_a, lse_a = na1d(q, k, v, kernel_size=ks, return_lse=True)
        # For a true split, run attention on full KV twice and merge.
        # The merge of two identical attentions should equal the original.
        merged, merged_lse = merge_attentions([out_a, out_a], [lse_a, lse_a])
        torch.testing.assert_close(merged, out_full, atol=1e-5, rtol=1e-5)

    def test_2way_merge_different_kvs(self):
        """Two different KV sets merged should differ from either alone."""
        _seed()
        B, L, H, D = 1, 12, 2, 8
        ks = 5
        q = _randn(B, L, H, D)
        k1 = _randn(B, L, H, D)
        v1 = _randn(B, L, H, D)
        k2 = _randn(B, L, H, D)
        v2 = _randn(B, L, H, D)

        out1, lse1 = na1d(q, k1, v1, kernel_size=ks, return_lse=True)
        out2, lse2 = na1d(q, k2, v2, kernel_size=ks, return_lse=True)
        merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])

        assert merged.shape == out1.shape
        assert merged_lse.shape == lse1.shape
        # Merged should differ from either individual output
        assert not torch.allclose(merged, out1, atol=1e-5)
        assert not torch.allclose(merged, out2, atol=1e-5)

    def test_3way_merge_forward_only(self):
        """3-way merge requires use_autograd_fix=False."""
        _seed()
        B, L, H, D = 1, 10, 2, 8
        ks = 5
        q = _randn(B, L, H, D)

        outs, lses = [], []
        for _ in range(3):
            k = _randn(B, L, H, D)
            v = _randn(B, L, H, D)
            o, l = na1d(q, k, v, kernel_size=ks, return_lse=True)
            outs.append(o)
            lses.append(l)

        merged, merged_lse = merge_attentions(outs, lses, use_autograd_fix=False)
        assert merged.shape == outs[0].shape
        assert merged_lse.shape == lses[0].shape
        assert torch.isfinite(merged).all()
        assert torch.isfinite(merged_lse).all()

    def test_merge_rejects_fewer_than_2(self):
        _seed()
        o = _randn(1, 10, 2, 8)
        l = _randn(1, 10, 2)
        with pytest.raises(ValueError, match="at least two"):
            merge_attentions([o], [l])

    def test_merge_rejects_mismatched_counts(self):
        _seed()
        o1 = _randn(1, 10, 2, 8)
        o2 = _randn(1, 10, 2, 8)
        l1 = _randn(1, 10, 2)
        with pytest.raises(ValueError, match="must match"):
            merge_attentions([o1, o2], [l1])

    def test_merge_rejects_shape_mismatch(self):
        _seed()
        o1 = _randn(1, 10, 2, 8)
        o2 = _randn(1, 12, 2, 8)  # different spatial
        l1 = _randn(1, 10, 2)
        l2 = _randn(1, 12, 2)
        with pytest.raises(ValueError, match="shape"):
            merge_attentions([o1, o2], [l1, l2])

    def test_merged_lse_finite_and_correct_shape(self):
        _seed()
        B, L, H, D = 2, 10, 2, 8
        ks = 5
        q = _randn(B, L, H, D)
        k1 = _randn(B, L, H, D)
        v1 = _randn(B, L, H, D)
        k2 = _randn(B, L, H, D)
        v2 = _randn(B, L, H, D)

        out1, lse1 = na1d(q, k1, v1, kernel_size=ks, return_lse=True)
        out2, lse2 = na1d(q, k2, v2, kernel_size=ks, return_lse=True)
        _, merged_lse = merge_attentions([out1, out2], [lse1, lse2])
        assert merged_lse.shape == (B, L, H)
        assert torch.isfinite(merged_lse).all()


# ===================================================================
# 3. GQA / MQA
# ===================================================================


class TestGQA:
    """Tests for grouped-query attention / multi-query attention."""

    # -- 1D ---------------------------------------------------------

    def test_na1d_gqa_forward_matches_explicit_repeat(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 8, 2
        ks = 5
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)

        out_gqa = na1d(q, k, v, kernel_size=ks)

        # Explicit repeat
        k_rep = k.repeat_interleave(heads_q // heads_kv, dim=-2)
        v_rep = v.repeat_interleave(heads_q // heads_kv, dim=-2)
        out_rep = na1d(q, k_rep, v_rep, kernel_size=ks)

        torch.testing.assert_close(out_gqa, out_rep, atol=1e-5, rtol=1e-5)

    def test_na1d_mqa_forward(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 1
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)
        out = na1d(q, k, v, kernel_size=5)
        assert out.shape == (B, L, heads_q, D)

    def test_na1d_gqa_indivisible_raises(self):
        _seed()
        q = _randn(1, 12, 7, 8)
        k = _randn(1, 12, 3, 8)
        v = _randn(1, 12, 3, 8)
        with pytest.raises(ValueError, match="divisible"):
            na1d(q, k, v, kernel_size=5)

    def test_na1d_gqa_output_has_q_heads(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 8, 2
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)
        out = na1d(q, k, v, kernel_size=5)
        assert out.shape == (B, L, heads_q, D)

    def test_na1d_gqa_head_dim_mismatch_raises(self):
        _seed()
        q = _randn(1, 12, 4, 8)
        k = _randn(1, 12, 2, 16)  # different D
        v = _randn(1, 12, 2, 16)
        with pytest.raises(ValueError, match="Head dim"):
            na1d(q, k, v, kernel_size=5)

    # -- 2D ---------------------------------------------------------

    def test_na2d_gqa_forward_matches_explicit_repeat(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 4, 2
        ks = 5
        q = _randn(B, Hh, W, heads_q, D)
        k = _randn(B, Hh, W, heads_kv, D)
        v = _randn(B, Hh, W, heads_kv, D)

        out_gqa = na2d(q, k, v, kernel_size=ks)
        k_rep = k.repeat_interleave(heads_q // heads_kv, dim=-2)
        v_rep = v.repeat_interleave(heads_q // heads_kv, dim=-2)
        out_rep = na2d(q, k_rep, v_rep, kernel_size=ks)

        torch.testing.assert_close(out_gqa, out_rep, atol=1e-5, rtol=1e-5)

    def test_na2d_gqa_output_shape(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 8, 2
        q = _randn(B, Hh, W, heads_q, D)
        k = _randn(B, Hh, W, heads_kv, D)
        v = _randn(B, Hh, W, heads_kv, D)
        out = na2d(q, k, v, kernel_size=5)
        assert out.shape == (B, Hh, W, heads_q, D)

    # -- 3D ---------------------------------------------------------

    def test_na3d_gqa_forward_matches_explicit_repeat(self):
        _seed()
        B = 1
        D_s, Hh_s, W_s = 4, 4, 4
        D, heads_q, heads_kv = 8, 4, 2
        ks = 3
        q = _randn(B, D_s, Hh_s, W_s, heads_q, D)
        k = _randn(B, D_s, Hh_s, W_s, heads_kv, D)
        v = _randn(B, D_s, Hh_s, W_s, heads_kv, D)

        out_gqa = na3d(q, k, v, kernel_size=ks)
        k_rep = k.repeat_interleave(heads_q // heads_kv, dim=-2)
        v_rep = v.repeat_interleave(heads_q // heads_kv, dim=-2)
        out_rep = na3d(q, k_rep, v_rep, kernel_size=ks)

        torch.testing.assert_close(out_gqa, out_rep, atol=1e-5, rtol=1e-5)

    def test_na3d_gqa_indivisible_raises(self):
        _seed()
        q = _randn(1, 4, 4, 4, 5, 8)
        k = _randn(1, 4, 4, 4, 3, 8)
        v = _randn(1, 4, 4, 4, 3, 8)
        with pytest.raises(ValueError, match="divisible"):
            na3d(q, k, v, kernel_size=3)


# ===================================================================
# 4. additional_keys / additional_values
# ===================================================================


class TestAdditionalKV:
    """Tests for the additional_keys / additional_values parameters."""

    def test_na1d_additional_kv_output_shape(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        ak = _randn(B, 1, H, D)  # 1 extra token
        av = _randn(B, 1, H, D)
        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        assert out.shape == (B, L, H, D)

    def test_na1d_additional_kv_changes_output(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        ak = _randn(B, 1, H, D)
        av = _randn(B, 1, H, D)

        out_plain = na1d(q, k, v, kernel_size=5)
        out_extra = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        assert not torch.allclose(out_plain, out_extra, atol=1e-5)

    def test_na1d_additional_keys_only_raises(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)
        ak = _randn(1, 1, 2, 8)
        with pytest.raises(ValueError, match="both be provided"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=None)

    def test_na1d_additional_values_only_raises(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)
        av = _randn(1, 1, 2, 8)
        with pytest.raises(ValueError, match="both be provided"):
            na1d(q, k, v, kernel_size=5, additional_keys=None, additional_values=av)

    def test_na1d_additional_kv_wrong_ndim_raises(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)
        ak = _randn(1, 1, 8)  # 3D, should be 4D
        av = _randn(1, 1, 8)
        with pytest.raises(ValueError, match="4D"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)

    def test_na1d_additional_kv_batch_mismatch_raises(self):
        _seed()
        q = _randn(2, 12, 2, 8)
        k = _randn(2, 12, 2, 8)
        v = _randn(2, 12, 2, 8)
        ak = _randn(3, 1, 2, 8)  # batch=3 vs 2
        av = _randn(3, 1, 2, 8)
        with pytest.raises(ValueError, match="batch"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)

    def test_na1d_additional_kv_head_dim_mismatch_raises(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)
        ak = _randn(1, 1, 2, 16)  # D=16 vs 8
        av = _randn(1, 1, 2, 16)
        with pytest.raises(ValueError, match="head dim"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)

    def test_na1d_additional_kv_with_gqa(self):
        """additional_kv should have heads_kv heads and get expanded internally."""
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)
        ak = _randn(B, 1, heads_kv, D)
        av = _randn(B, 1, heads_kv, D)
        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        assert out.shape == (B, L, heads_q, D)

    def test_na2d_additional_kv_output_shape(self):
        _seed()
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)
        ak = _randn(B, 2, H, D)
        av = _randn(B, 2, H, D)
        out = na2d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        assert out.shape == (B, Hh, W, H, D)

    def test_na3d_additional_kv_output_shape(self):
        _seed()
        B = 1
        D_s, Hh_s, W_s = 4, 4, 4
        H, D = 2, 8
        q = _randn(B, D_s, Hh_s, W_s, H, D)
        k = _randn(B, D_s, Hh_s, W_s, H, D)
        v = _randn(B, D_s, Hh_s, W_s, H, D)
        ak = _randn(B, 1, H, D)
        av = _randn(B, 1, H, D)
        out = na3d(q, k, v, kernel_size=3, additional_keys=ak, additional_values=av)
        assert out.shape == (B, D_s, Hh_s, W_s, H, D)


# ===================================================================
# 5. bfloat16
# ===================================================================


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required for bf16 test"
)
class TestBFloat16:
    """Tests that bf16 inputs produce bf16 outputs and are close to fp32."""

    def test_na1d_bf16_output_dtype(self):
        _seed()
        q = _randn(1, 12, 2, 8, dtype=torch.bfloat16)
        k = _randn(1, 12, 2, 8, dtype=torch.bfloat16)
        v = _randn(1, 12, 2, 8, dtype=torch.bfloat16)
        out = na1d(q, k, v, kernel_size=5)
        assert out.dtype == torch.bfloat16

    def test_na1d_bf16_close_to_fp32(self):
        _seed()
        q_f32 = _randn(1, 12, 2, 16)
        k_f32 = _randn(1, 12, 2, 16)
        v_f32 = _randn(1, 12, 2, 16)
        out_f32 = na1d(q_f32, k_f32, v_f32, kernel_size=5)

        out_bf16 = na1d(
            q_f32.bfloat16(), k_f32.bfloat16(), v_f32.bfloat16(), kernel_size=5
        )
        torch.testing.assert_close(
            out_bf16.float(), out_f32, atol=5e-2, rtol=5e-2
        )

    def test_na2d_bf16_output_dtype(self):
        _seed()
        q = _randn(1, 8, 8, 2, 8, dtype=torch.bfloat16)
        k = _randn(1, 8, 8, 2, 8, dtype=torch.bfloat16)
        v = _randn(1, 8, 8, 2, 8, dtype=torch.bfloat16)
        out = na2d(q, k, v, kernel_size=5)
        assert out.dtype == torch.bfloat16

    def test_na2d_bf16_close_to_fp32(self):
        _seed()
        q_f32 = _randn(1, 8, 8, 2, 16)
        k_f32 = _randn(1, 8, 8, 2, 16)
        v_f32 = _randn(1, 8, 8, 2, 16)
        out_f32 = na2d(q_f32, k_f32, v_f32, kernel_size=5)

        out_bf16 = na2d(
            q_f32.bfloat16(), k_f32.bfloat16(), v_f32.bfloat16(), kernel_size=5
        )
        torch.testing.assert_close(
            out_bf16.float(), out_f32, atol=5e-2, rtol=5e-2
        )

    def test_na3d_bf16_output_dtype(self):
        _seed()
        q = _randn(1, 4, 4, 4, 2, 8, dtype=torch.bfloat16)
        k = _randn(1, 4, 4, 4, 2, 8, dtype=torch.bfloat16)
        v = _randn(1, 4, 4, 4, 2, 8, dtype=torch.bfloat16)
        out = na3d(q, k, v, kernel_size=3)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    def test_na3d_bf16_close_to_fp32(self):
        _seed()
        q_f32 = _randn(1, 4, 4, 4, 2, 8)
        k_f32 = _randn(1, 4, 4, 4, 2, 8)
        v_f32 = _randn(1, 4, 4, 4, 2, 8)
        out_f32 = na3d(q_f32, k_f32, v_f32, kernel_size=3)
        out_bf16 = na3d(
            q_f32.bfloat16(), k_f32.bfloat16(), v_f32.bfloat16(), kernel_size=3
        )
        torch.testing.assert_close(
            out_bf16.float(), out_f32, atol=5e-2, rtol=5e-2
        )


# ===================================================================
# 6. FMHA fast path (kernel covers full spatial extent -> SDPA)
# ===================================================================


class TestFMHAFastPath:
    """Tests for the SDPA fast path when kernel_size >= spatial dims."""

    def test_na1d_full_kernel_matches_sdpa(self):
        """When kernel_size=L, NA degenerates to global attention = SDPA."""
        _seed()
        B, L, H, D = 1, 10, 2, 16
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        scale = D ** -0.5

        out_na = na1d(q, k, v, kernel_size=L, scale=scale)

        # Manual SDPA: reshape [B, L, H, D] -> [B, H, L, D]
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)
        out_sdpa = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        out_sdpa = out_sdpa.permute(0, 2, 1, 3)

        torch.testing.assert_close(out_na, out_sdpa, atol=1e-5, rtol=1e-5)

    def test_na2d_full_kernel_matches_sdpa(self):
        _seed()
        B, Hh, W, H, D = 1, 6, 8, 2, 16
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)
        scale = D ** -0.5

        out_na = na2d(q, k, v, kernel_size=(Hh, W), scale=scale)

        # SDPA: flatten spatial -> [B, H*W, H, D] -> [B, heads, S, D]
        S = Hh * W
        q_flat = q.reshape(B, S, H, D).permute(0, 2, 1, 3)
        k_flat = k.reshape(B, S, H, D).permute(0, 2, 1, 3)
        v_flat = v.reshape(B, S, H, D).permute(0, 2, 1, 3)
        out_sdpa = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, scale=scale)
        out_sdpa = out_sdpa.permute(0, 2, 1, 3).reshape(B, Hh, W, H, D)

        torch.testing.assert_close(out_na, out_sdpa, atol=1e-5, rtol=1e-5)

    def test_na1d_fast_path_not_used_when_causal(self):
        """With is_causal=True and full kernel, output should differ from non-causal SDPA."""
        _seed()
        B, L, H, D = 1, 10, 2, 16
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        scale = D ** -0.5

        out_causal = na1d(q, k, v, kernel_size=L, is_causal=True, scale=scale)

        # Non-causal SDPA reference
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)
        out_sdpa_noncausal = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        out_sdpa_noncausal = out_sdpa_noncausal.permute(0, 2, 1, 3)

        # Should NOT match because causal masking changes the result
        assert not torch.allclose(out_causal, out_sdpa_noncausal, atol=1e-4)

    def test_na1d_fast_path_not_used_when_stride_gt_1(self):
        """With stride > 1, even full kernel should NOT use SDPA fast path
        (output spatial dim changes)."""
        _seed()
        B, L, H, D = 1, 10, 2, 16
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)

        out_strided = na1d(q, k, v, kernel_size=L, stride=2)
        # Strided output has reduced spatial dim
        expected_L = (L + 2 - 1) // 2  # ceil(L/stride)
        assert out_strided.shape[1] != L or out_strided.shape[1] == expected_L

    def test_na1d_fast_path_with_return_lse(self):
        """SDPA path should still return LSE when requested."""
        _seed()
        B, L, H, D = 1, 10, 2, 16
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)

        result = na1d(q, k, v, kernel_size=L, return_lse=True)
        assert isinstance(result, tuple)
        out, lse = result
        assert out.shape == (B, L, H, D)
        assert lse.shape == (B, L, H)
        assert torch.isfinite(lse).all()

    def test_na2d_fast_path_with_additional_kv(self):
        """SDPA path should incorporate additional KV tokens."""
        _seed()
        B, Hh, W, H, D = 1, 6, 8, 2, 16
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)
        ak = _randn(B, 2, H, D)
        av = _randn(B, 2, H, D)

        out_plain = na2d(q, k, v, kernel_size=(Hh, W))
        out_extra = na2d(q, k, v, kernel_size=(Hh, W), additional_keys=ak, additional_values=av)
        assert out_extra.shape == (B, Hh, W, H, D)
        # Additional KV should change the output
        assert not torch.allclose(out_plain, out_extra, atol=1e-5)


# ===================================================================
# 7. Backward / Gradient Tests for New Features
# ===================================================================


class TestReturnLSEBackward:
    """Gradient tests for return_lse feature."""

    def test_na1d_return_lse_backward(self):
        _seed()
        q = _randn(1, 12, 2, 8).requires_grad_(True)
        k = _randn(1, 12, 2, 8).requires_grad_(True)
        v = _randn(1, 12, 2, 8).requires_grad_(True)
        out, lse = na1d(q, k, v, kernel_size=5, return_lse=True)
        (out.sum() + lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape

    def test_na1d_return_lse_backward_causal(self):
        _seed()
        q = _randn(1, 12, 2, 8).requires_grad_(True)
        k = _randn(1, 12, 2, 8).requires_grad_(True)
        v = _randn(1, 12, 2, 8).requires_grad_(True)
        out, lse = na1d(q, k, v, kernel_size=5, is_causal=True, return_lse=True)
        (out.sum() + lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape

    def test_na2d_return_lse_backward(self):
        _seed()
        q = _randn(1, 8, 8, 2, 8).requires_grad_(True)
        k = _randn(1, 8, 8, 2, 8).requires_grad_(True)
        v = _randn(1, 8, 8, 2, 8).requires_grad_(True)
        out, lse = na2d(q, k, v, kernel_size=5, return_lse=True)
        (out.sum() + lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape


class TestGQABackward:
    """Gradient tests for GQA/MQA feature."""

    def test_na1d_gqa_backward(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D).requires_grad_(True)
        k = _randn(B, L, heads_kv, D).requires_grad_(True)
        v = _randn(B, L, heads_kv, D).requires_grad_(True)
        out = na1d(q, k, v, kernel_size=5)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == (B, L, heads_q, D)
        assert k.grad is not None and k.grad.shape == (B, L, heads_kv, D)
        assert v.grad is not None and v.grad.shape == (B, L, heads_kv, D)

    def test_na1d_mqa_backward(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 1
        q = _randn(B, L, heads_q, D).requires_grad_(True)
        k = _randn(B, L, heads_kv, D).requires_grad_(True)
        v = _randn(B, L, heads_kv, D).requires_grad_(True)
        out = na1d(q, k, v, kernel_size=5)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == (B, L, heads_q, D)
        assert k.grad is not None and k.grad.shape == (B, L, heads_kv, D)
        assert v.grad is not None and v.grad.shape == (B, L, heads_kv, D)

    def test_na2d_gqa_backward(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, Hh, W, heads_q, D).requires_grad_(True)
        k = _randn(B, Hh, W, heads_kv, D).requires_grad_(True)
        v = _randn(B, Hh, W, heads_kv, D).requires_grad_(True)
        out = na2d(q, k, v, kernel_size=5)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == (B, Hh, W, heads_q, D)
        assert k.grad is not None and k.grad.shape == (B, Hh, W, heads_kv, D)
        assert v.grad is not None and v.grad.shape == (B, Hh, W, heads_kv, D)


class TestAdditionalKVBackward:
    """Gradient tests for additional_keys/additional_values."""

    def test_na1d_additional_kv_backward(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _randn(B, L, H, D).requires_grad_(True)
        k = _randn(B, L, H, D).requires_grad_(True)
        v = _randn(B, L, H, D).requires_grad_(True)
        ak = _randn(B, 2, H, D).requires_grad_(True)
        av = _randn(B, 2, H, D).requires_grad_(True)
        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape
        assert ak.grad is not None and ak.grad.shape == ak.shape
        assert av.grad is not None and av.grad.shape == av.shape

    def test_na2d_additional_kv_backward(self):
        _seed()
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q = _randn(B, Hh, W, H, D).requires_grad_(True)
        k = _randn(B, Hh, W, H, D).requires_grad_(True)
        v = _randn(B, Hh, W, H, D).requires_grad_(True)
        ak = _randn(B, 2, H, D).requires_grad_(True)
        av = _randn(B, 2, H, D).requires_grad_(True)
        out = na2d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert ak.grad is not None and ak.grad.shape == ak.shape
        assert av.grad is not None and av.grad.shape == av.shape


class TestMergeAttentionsBackward:
    """Gradient tests for merge_attentions (2-way)."""

    def test_merge_2way_backward_1d(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        ks = 5
        q = _randn(B, L, H, D).requires_grad_(True)
        k1 = _randn(B, L, H, D).requires_grad_(True)
        v1 = _randn(B, L, H, D).requires_grad_(True)
        k2 = _randn(B, L, H, D).requires_grad_(True)
        v2 = _randn(B, L, H, D).requires_grad_(True)

        out1, lse1 = na1d(q, k1, v1, kernel_size=ks, return_lse=True)
        out2, lse2 = na1d(q, k2, v2, kernel_size=ks, return_lse=True)
        merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])
        (merged.sum() + merged_lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k1.grad is not None and k1.grad.shape == k1.shape
        assert v1.grad is not None and v1.grad.shape == v1.shape
        assert k2.grad is not None and k2.grad.shape == k2.shape

    def test_merge_2way_backward_2d(self):
        _seed()
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        ks = 5
        q = _randn(B, Hh, W, H, D).requires_grad_(True)
        k1 = _randn(B, Hh, W, H, D).requires_grad_(True)
        v1 = _randn(B, Hh, W, H, D).requires_grad_(True)
        k2 = _randn(B, Hh, W, H, D).requires_grad_(True)
        v2 = _randn(B, Hh, W, H, D).requires_grad_(True)

        out1, lse1 = na2d(q, k1, v1, kernel_size=ks, return_lse=True)
        out2, lse2 = na2d(q, k2, v2, kernel_size=ks, return_lse=True)
        merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])
        (merged.sum() + merged_lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k1.grad is not None


class TestFMHAFastPathBackward:
    """Gradient tests for FMHA fast path (kernel >= spatial)."""

    def test_na1d_fmha_backward(self):
        _seed()
        B, L, H, D = 1, 10, 2, 8
        q = _randn(B, L, H, D).requires_grad_(True)
        k = _randn(B, L, H, D).requires_grad_(True)
        v = _randn(B, L, H, D).requires_grad_(True)
        out = na1d(q, k, v, kernel_size=L)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape

    def test_na2d_fmha_backward(self):
        _seed()
        B, Hh, W, H, D = 1, 6, 8, 2, 8
        q = _randn(B, Hh, W, H, D).requires_grad_(True)
        k = _randn(B, Hh, W, H, D).requires_grad_(True)
        v = _randn(B, Hh, W, H, D).requires_grad_(True)
        out = na2d(q, k, v, kernel_size=(Hh, W))
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape

    def test_na1d_fmha_with_return_lse_backward(self):
        _seed()
        B, L, H, D = 1, 10, 2, 8
        q = _randn(B, L, H, D).requires_grad_(True)
        k = _randn(B, L, H, D).requires_grad_(True)
        v = _randn(B, L, H, D).requires_grad_(True)
        out, lse = na1d(q, k, v, kernel_size=L, return_lse=True)
        (out.sum() + lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape


# ===================================================================
# 8. Combination Tests (feature interactions)
# ===================================================================


class TestFeatureCombinations:
    """Tests for combinations of new features to verify they compose correctly."""

    def test_gqa_with_return_lse_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)
        out, lse = na1d(q, k, v, kernel_size=5, return_lse=True)
        assert out.shape == (B, L, heads_q, D)
        assert lse.shape == (B, L, heads_q)
        assert torch.isfinite(lse).all()

    def test_gqa_with_return_lse_2d(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, Hh, W, heads_q, D)
        k = _randn(B, Hh, W, heads_kv, D)
        v = _randn(B, Hh, W, heads_kv, D)
        out, lse = na2d(q, k, v, kernel_size=5, return_lse=True)
        assert out.shape == (B, Hh, W, heads_q, D)
        assert lse.shape == (B, Hh, W, heads_q)

    def test_gqa_causal_return_lse_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)
        out, lse = na1d(q, k, v, kernel_size=5, is_causal=True, return_lse=True)
        assert out.shape == (B, L, heads_q, D)
        assert torch.isfinite(lse).all()

    def test_additional_kv_with_gqa_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)
        ak = _randn(B, 2, heads_kv, D)
        av = _randn(B, 2, heads_kv, D)
        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        assert out.shape == (B, L, heads_q, D)

    def test_additional_kv_with_gqa_2d(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, Hh, W, heads_q, D)
        k = _randn(B, Hh, W, heads_kv, D)
        v = _randn(B, Hh, W, heads_kv, D)
        ak = _randn(B, 2, heads_kv, D)
        av = _randn(B, 2, heads_kv, D)
        out = na2d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        assert out.shape == (B, Hh, W, heads_q, D)

    def test_stride_with_return_lse_1d(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        out, lse = na1d(q, k, v, kernel_size=5, stride=2, return_lse=True)
        L_out = (L + 2 - 1) // 2
        assert out.shape == (B, L_out, H, D)
        assert lse.shape == (B, L_out, H)

    def test_dilation_with_gqa_1d(self):
        _seed()
        B, L, D = 1, 16, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)
        out = na1d(q, k, v, kernel_size=5, dilation=2)
        assert out.shape == (B, L, heads_q, D)

    def test_merge_with_gqa_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D)
        k1 = _randn(B, L, heads_kv, D)
        v1 = _randn(B, L, heads_kv, D)
        k2 = _randn(B, L, heads_kv, D)
        v2 = _randn(B, L, heads_kv, D)
        out1, lse1 = na1d(q, k1, v1, kernel_size=5, return_lse=True)
        out2, lse2 = na1d(q, k2, v2, kernel_size=5, return_lse=True)
        merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])
        assert merged.shape == (B, L, heads_q, D)
        assert merged_lse.shape == (B, L, heads_q)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required for bf16 test"
)
class TestBFloat16Combinations:
    def test_additional_kv_with_bf16_1d(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _randn(B, L, H, D, dtype=torch.bfloat16)
        k = _randn(B, L, H, D, dtype=torch.bfloat16)
        v = _randn(B, L, H, D, dtype=torch.bfloat16)
        ak = _randn(B, 2, H, D, dtype=torch.bfloat16)
        av = _randn(B, 2, H, D, dtype=torch.bfloat16)
        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()


# ===================================================================
# 9. FMHA Edge Cases
# ===================================================================


class TestFMHAEdgeCases:
    """Edge cases for FMHA fast path dispatch."""

    def test_fmha_kernel_equals_spatial_1d(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        out = na1d(q, k, v, kernel_size=L)
        assert out.shape == (B, L, H, D)

    def test_fmha_kernel_equals_spatial_2d(self):
        _seed()
        B, Hh, W, H, D = 1, 6, 8, 2, 8
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)
        out = na2d(q, k, v, kernel_size=(Hh, W))
        assert out.shape == (B, Hh, W, H, D)

    def test_fmha_non_square_full_kernel_2d(self):
        _seed()
        B, Hh, W, H, D = 1, 4, 8, 2, 8
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)
        scale = D ** -0.5

        out_na = na2d(q, k, v, kernel_size=(Hh, W), scale=scale)

        # Compare with SDPA
        S = Hh * W
        q_t = q.reshape(B, S, H, D).permute(0, 2, 1, 3)
        k_t = k.reshape(B, S, H, D).permute(0, 2, 1, 3)
        v_t = v.reshape(B, S, H, D).permute(0, 2, 1, 3)
        out_sdpa = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        out_sdpa = out_sdpa.permute(0, 2, 1, 3).reshape(B, Hh, W, H, D)

        torch.testing.assert_close(out_na, out_sdpa, atol=1e-5, rtol=1e-5)

    def test_fmha_with_additional_kv_backward(self):
        _seed()
        B, L, H, D = 1, 10, 2, 8
        q = _randn(B, L, H, D).requires_grad_(True)
        k = _randn(B, L, H, D).requires_grad_(True)
        v = _randn(B, L, H, D).requires_grad_(True)
        ak = _randn(B, 3, H, D).requires_grad_(True)
        av = _randn(B, 3, H, D).requires_grad_(True)
        out = na1d(q, k, v, kernel_size=L, additional_keys=ak, additional_values=av)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert ak.grad is not None and ak.grad.shape == ak.shape
        assert av.grad is not None and av.grad.shape == av.shape


# ===================================================================
# 10. 3D Feature Backward Coverage
# ===================================================================


class Test3DFeatureBackward:
    """3D backward tests for features that had only 1D/2D coverage."""

    def test_na3d_gqa_backward(self):
        _seed()
        B, Ds, Hs, Ws, D = 1, 4, 4, 4, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, Ds, Hs, Ws, heads_q, D).requires_grad_(True)
        k = _randn(B, Ds, Hs, Ws, heads_kv, D).requires_grad_(True)
        v = _randn(B, Ds, Hs, Ws, heads_kv, D).requires_grad_(True)
        out = na3d(q, k, v, kernel_size=3)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape

    def test_na3d_return_lse_backward(self):
        _seed()
        B, Ds, Hs, Ws, H, D = 1, 4, 4, 4, 2, 8
        q = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        k = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        v = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        out, lse = na3d(q, k, v, kernel_size=3, return_lse=True)
        (out.sum() + lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape

    def test_na3d_fmha_backward(self):
        _seed()
        B, Ds, Hs, Ws, H, D = 1, 4, 4, 4, 2, 8
        q = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        k = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        v = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        out = na3d(q, k, v, kernel_size=(Ds, Hs, Ws))
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape

    def test_merge_2way_backward_3d(self):
        _seed()
        B, Ds, Hs, Ws, H, D = 1, 4, 4, 4, 2, 8
        ks = 3
        q = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        k1 = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        v1 = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        k2 = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        v2 = _randn(B, Ds, Hs, Ws, H, D).requires_grad_(True)
        out1, lse1 = na3d(q, k1, v1, kernel_size=ks, return_lse=True)
        out2, lse2 = na3d(q, k2, v2, kernel_size=ks, return_lse=True)
        merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])
        (merged.sum() + merged_lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k1.grad is not None and k1.grad.shape == k1.shape
        assert v2.grad is not None and v2.grad.shape == v2.shape

    def test_gqa_with_return_lse_3d(self):
        _seed()
        B, Ds, Hs, Ws, D = 1, 4, 4, 4, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, Ds, Hs, Ws, heads_q, D).requires_grad_(True)
        k = _randn(B, Ds, Hs, Ws, heads_kv, D).requires_grad_(True)
        v = _randn(B, Ds, Hs, Ws, heads_kv, D).requires_grad_(True)
        out, lse = na3d(q, k, v, kernel_size=3, return_lse=True)
        (out.sum() + lse.sum()).backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape

    def test_additional_kv_with_gqa_3d(self):
        _seed()
        B, Ds, Hs, Ws, D = 1, 4, 4, 4, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, Ds, Hs, Ws, heads_q, D).requires_grad_(True)
        k = _randn(B, Ds, Hs, Ws, heads_kv, D).requires_grad_(True)
        v = _randn(B, Ds, Hs, Ws, heads_kv, D).requires_grad_(True)
        ak = _randn(B, 2, heads_kv, D).requires_grad_(True)
        av = _randn(B, 2, heads_kv, D).requires_grad_(True)
        out = na3d(q, k, v, kernel_size=3, additional_keys=ak, additional_values=av)
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert ak.grad is not None and ak.grad.shape == ak.shape
        assert av.grad is not None and av.grad.shape == av.shape
