"""Attention merging for natten-mps.

Merges multiple attention outputs that share the same query, as if their
key/value contexts had been concatenated. Uses a numerically stable
sigmoid-based formulation from ring-flash-attention.

Typical usage:
    out_a, lse_a = na1d(q, k_a, v_a, kernel_size=7, return_lse=True)
    out_b, lse_b = na1d(q, k_b, v_b, kernel_size=7, return_lse=True)
    merged_out, merged_lse = merge_attentions([out_a, out_b], [lse_a, lse_b])
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from torch.autograd import Function


def _merge_attentions_fn(
    outputs: List[Tensor], lse_tensors: List[Tensor]
) -> Tuple[Tensor, Tensor]:
    """Core merge implementation (no autograd wrapper)."""
    assert len(outputs) >= 2 and len(outputs) == len(lse_tensors)

    accum_dtype = torch.float32
    out_dtype = outputs[0].dtype

    lse_list = [lse.to(accum_dtype).unsqueeze(-1) for lse in lse_tensors]
    out_list = [o.to(accum_dtype) for o in outputs]

    # Sigmoid-based merge (ref: ring-flash-attention)
    output = out_list[0] - torch.sigmoid(lse_list[1] - lse_list[0]) * (
        out_list[0] - out_list[1]
    )
    logsumexp = lse_list[0] - torch.nn.functional.logsigmoid(
        lse_list[0] - lse_list[1]
    )

    for i in range(2, len(out_list)):
        output = output - torch.sigmoid(lse_list[i] - logsumexp) * (
            output - out_list[i]
        )
        logsumexp = logsumexp - torch.nn.functional.logsigmoid(
            logsumexp - lse_list[i]
        )

    return output.to(out_dtype), logsumexp.squeeze(-1)


class _MergeAttentionsAutogradFn(Function):
    """Custom autograd for 2-way merge with analytical backward.

    Forward:
        sig = sigmoid(lse_1 - lse_0)          (unsqueezed to [..., 1] for output)
        output = (1 - sig) * out_0 + sig * out_1
        lse_out = lse_0 + softplus(lse_1 - lse_0)
    """

    @staticmethod
    def forward(ctx, out_0: Tensor, out_1: Tensor, lse_0: Tensor, lse_1: Tensor):
        accum_dtype = torch.float32
        lse_0_f = lse_0.to(accum_dtype).unsqueeze(-1)
        lse_1_f = lse_1.to(accum_dtype).unsqueeze(-1)
        out_0_f = out_0.to(accum_dtype)
        out_1_f = out_1.to(accum_dtype)

        sig = torch.sigmoid(lse_1_f - lse_0_f)  # [..., 1]
        merged_out = (1 - sig) * out_0_f + sig * out_1_f
        # lse_out = lse_0 - logsigmoid(lse_0 - lse_1) = lse_0 + softplus(lse_1 - lse_0)
        merged_lse = lse_0_f + torch.nn.functional.softplus(lse_1_f - lse_0_f)

        ctx.save_for_backward(out_0_f, out_1_f, sig)
        return merged_out.to(out_0.dtype), merged_lse.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_out: Tensor, grad_lse: Tensor):
        out_0, out_1, sig = ctx.saved_tensors
        grad_out_f = grad_out.to(torch.float32)
        grad_lse_f = grad_lse.to(torch.float32).unsqueeze(-1)

        # d output / d out_0 = (1 - sig),  d output / d out_1 = sig
        d_out_0 = (1 - sig) * grad_out_f
        d_out_1 = sig * grad_out_f

        # d output / d sig = (out_1 - out_0), then chain through sig = sigmoid(lse_1 - lse_0)
        # d sig / d lse_0 = -sig*(1-sig),  d sig / d lse_1 = sig*(1-sig)
        diff = out_1 - out_0  # [..., D]
        dsig_from_out = (grad_out_f * diff).sum(dim=-1, keepdim=True)
        sig_deriv = sig * (1 - sig)

        # d lse_out / d lse_0 = (1 - sig),  d lse_out / d lse_1 = sig
        d_lse_0 = (1 - sig) * grad_lse_f - dsig_from_out * sig_deriv
        d_lse_1 = sig * grad_lse_f + dsig_from_out * sig_deriv

        return (
            d_out_0.to(grad_out.dtype),
            d_out_1.to(grad_out.dtype),
            d_lse_0.squeeze(-1),
            d_lse_1.squeeze(-1),
        )


def merge_attentions(
    outputs: List[Tensor],
    lse_tensors: List[Tensor],
    use_autograd_fix: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Merge multiple attention outputs sharing the same query.

    Takes attention outputs and their logsumexp tensors and merges them
    as if their key/value contexts had been concatenated.

    Args:
        outputs: List of attention output tensors, each ``[B, ..spatial.., H, D]``.
        lse_tensors: List of logsumexp tensors, each ``[B, ..spatial.., H]``.
        use_autograd_fix: Use custom autograd for correct backward (2-way only).

    Returns:
        ``(merged_output, merged_lse)``
    """
    if len(outputs) < 2:
        raise ValueError("merge_attentions expects at least two outputs.")
    if len(outputs) != len(lse_tensors):
        raise ValueError("Number of outputs and LSE tensors must match.")

    ref_shape = outputs[0].shape
    for i, (o, l) in enumerate(zip(outputs, lse_tensors)):
        if o.shape != ref_shape:
            raise ValueError(
                f"Output {i} shape {o.shape} does not match output 0 shape {ref_shape}."
            )
        expected_lse_shape = ref_shape[:-1]  # all dims except head_dim
        if l.shape != expected_lse_shape:
            raise ValueError(
                f"LSE {i} shape {l.shape} does not match expected {expected_lse_shape}."
            )

    requires_grad = outputs[0].requires_grad

    if use_autograd_fix and requires_grad and len(outputs) == 2:
        return _MergeAttentionsAutogradFn.apply(
            outputs[0], outputs[1], lse_tensors[0], lse_tensors[1]
        )

    if use_autograd_fix and requires_grad and len(outputs) > 2:
        raise NotImplementedError(
            "merge_attentions backward only supports 2-way merge. "
            "Set use_autograd_fix=False for N-way forward-only merge."
        )

    return _merge_attentions_fn(
        [o.contiguous() for o in outputs],
        [l.contiguous() for l in lse_tensors],
    )


__all__ = ["merge_attentions"]
