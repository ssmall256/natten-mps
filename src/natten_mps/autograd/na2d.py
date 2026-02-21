from __future__ import annotations

import torch

from natten_mps._core import ops


class NeighborhoodAttention2DFunction(torch.autograd.Function):
    """Fused 2D neighborhood attention with autograd support."""

    @staticmethod
    def forward(ctx, q, k, v, kernel_size, stride, dilation, is_causal, scale):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.save_for_backward(q, k, v)
        return ops.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors

        with torch.enable_grad():
            q_ = q.detach().requires_grad_(ctx.needs_input_grad[0])
            k_ = k.detach().requires_grad_(ctx.needs_input_grad[1])
            v_ = v.detach().requires_grad_(ctx.needs_input_grad[2])
            out = ops.na2d_forward(
                q_,
                k_,
                v_,
                ctx.kernel_size,
                ctx.stride,
                ctx.dilation,
                ctx.is_causal,
                ctx.scale,
            )
            grads = torch.autograd.grad(
                out,
                (q_, k_, v_),
                grad_output,
                allow_unused=True,
            )

        return grads[0], grads[1], grads[2], None, None, None, None, None


class NA2DQKFunction(torch.autograd.Function):
    """Separate QK operation with autograd. Returns attention logits."""

    @staticmethod
    def forward(ctx, q, k, kernel_size, dilation):
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.save_for_backward(q, k)
        return ops.na2d_qk_forward(q, k, kernel_size, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        q, k = ctx.saved_tensors

        with torch.enable_grad():
            q_ = q.detach().requires_grad_(ctx.needs_input_grad[0])
            k_ = k.detach().requires_grad_(ctx.needs_input_grad[1])
            out = ops.na2d_qk_forward(q_, k_, ctx.kernel_size, ctx.dilation)
            grads = torch.autograd.grad(out, (q_, k_), grad_output, allow_unused=True)

        return grads[0], grads[1], None, None


class NA2DAVFunction(torch.autograd.Function):
    """Separate AV operation with autograd. Returns output."""

    @staticmethod
    def forward(ctx, attn, v, kernel_size, dilation):
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.save_for_backward(attn, v)
        return ops.na2d_av_forward(attn, v, kernel_size, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        attn, v = ctx.saved_tensors

        with torch.enable_grad():
            attn_ = attn.detach().requires_grad_(ctx.needs_input_grad[0])
            v_ = v.detach().requires_grad_(ctx.needs_input_grad[1])
            out = ops.na2d_av_forward(attn_, v_, ctx.kernel_size, ctx.dilation)
            grads = torch.autograd.grad(out, (attn_, v_), grad_output, allow_unused=True)

        return grads[0], grads[1], None, None
