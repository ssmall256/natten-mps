from __future__ import annotations

import torch

from natten_mps._core import ops
from natten_mps._core import pure as _pure


class NeighborhoodAttention1DFunction(torch.autograd.Function):
    """Fused 1D neighborhood attention with autograd support."""

    @staticmethod
    def forward(ctx, q, k, v, kernel_size, stride, dilation, is_causal, scale):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.save_for_backward(q, k, v)
        return ops.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        scale_value = float(q.shape[-1] ** -0.5 if ctx.scale is None else ctx.scale)

        # Try Metal backward path: AV backward -> softmax backward -> QK backward
        logits = ops.na1d_qk_forward(q, k, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
        attn_weights = torch.softmax(logits * scale_value, dim=-1)

        av_result = ops.na1d_av_backward(
            grad_output, attn_weights, v, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
        if av_result is not None:
            d_attn, d_v = av_result
            # Softmax backward: d_logits = attn * (d_attn - (attn * d_attn).sum(-1, keepdim=True)) * scale
            d_logits = attn_weights * (d_attn - (attn_weights * d_attn).sum(-1, keepdim=True))
            d_logits = d_logits * scale_value

            qk_result = ops.na1d_qk_backward(
                d_logits, q, k, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
            if qk_result is not None:
                return qk_result[0], qk_result[1], d_v, None, None, None, None, None

        # Fallback: re-differentiation via pure backend (builds autograd graph)
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(ctx.needs_input_grad[0])
            k_ = k.detach().requires_grad_(ctx.needs_input_grad[1])
            v_ = v.detach().requires_grad_(ctx.needs_input_grad[2])
            out = _pure.na1d_forward(
                q_, k_, v_, ctx.kernel_size, ctx.stride, ctx.dilation, ctx.is_causal, ctx.scale)
            grads = torch.autograd.grad(out, (q_, k_, v_), grad_output, allow_unused=True)
        return grads[0], grads[1], grads[2], None, None, None, None, None


class NA1DQKFunction(torch.autograd.Function):
    """Separate QK operation with autograd. Returns attention logits."""

    @staticmethod
    def forward(ctx, q, k, kernel_size, dilation, stride, is_causal, scale):
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.stride = stride
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.save_for_backward(q, k)
        return ops.na1d_qk_forward(q, k, kernel_size, dilation, stride, is_causal, scale)

    @staticmethod
    def backward(ctx, grad_output):
        q, k = ctx.saved_tensors

        result = ops.na1d_qk_backward(
            grad_output, q, k, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
        if result is not None:
            return result[0], result[1], None, None, None, None, None

        # Fallback: re-differentiation via pure backend (builds autograd graph)
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(ctx.needs_input_grad[0])
            k_ = k.detach().requires_grad_(ctx.needs_input_grad[1])
            out = _pure.na1d_qk_forward(q_, k_, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal, ctx.scale)
            grads = torch.autograd.grad(out, (q_, k_), grad_output, allow_unused=True)
        return grads[0], grads[1], None, None, None, None, None


class NA1DAVFunction(torch.autograd.Function):
    """Separate AV operation with autograd. Returns output."""

    @staticmethod
    def forward(ctx, attn, v, kernel_size, dilation, stride, is_causal):
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.stride = stride
        ctx.is_causal = is_causal
        ctx.save_for_backward(attn, v)
        return ops.na1d_av_forward(attn, v, kernel_size, dilation, stride, is_causal)

    @staticmethod
    def backward(ctx, grad_output):
        attn, v = ctx.saved_tensors

        result = ops.na1d_av_backward(
            grad_output, attn, v, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
        if result is not None:
            return result[0], result[1], None, None, None, None

        # Fallback: re-differentiation via pure backend (builds autograd graph)
        with torch.enable_grad():
            attn_ = attn.detach().requires_grad_(ctx.needs_input_grad[0])
            v_ = v.detach().requires_grad_(ctx.needs_input_grad[1])
            out = _pure.na1d_av_forward(attn_, v_, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
            grads = torch.autograd.grad(out, (attn_, v_), grad_output, allow_unused=True)
        return grads[0], grads[1], None, None, None, None
