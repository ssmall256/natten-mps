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
    """Custom autograd for 2-way merge with correct backward pass."""

    @staticmethod
    def forward(ctx, out_0: Tensor, out_1: Tensor, lse_0: Tensor, lse_1: Tensor):
        merged_out, merged_lse = _merge_attentions_fn(
            [out_0.contiguous(), out_1.contiguous()],
            [lse_0.contiguous(), lse_1.contiguous()],
        )
        ctx.save_for_backward(out_0, out_1, lse_0, lse_1, merged_out, merged_lse)
        return merged_out, merged_lse

    @staticmethod
    def backward(ctx, grad_out: Tensor, grad_lse: Tensor):
        out_0, out_1, lse_0, lse_1, merged_out, merged_lse = ctx.saved_tensors
        # Replace originating outputs/LSEs with merged values in-place
        # so upstream attention backward sees the correct context.
        out_0.data.copy_(merged_out.data.reshape(out_0.shape))
        out_1.data.copy_(merged_out.data.reshape(out_1.shape))
        lse_0.data.copy_(merged_lse.data.reshape(lse_0.shape))
        lse_1.data.copy_(merged_lse.data.reshape(lse_1.shape))
        return grad_out, grad_out, grad_lse, grad_lse


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
