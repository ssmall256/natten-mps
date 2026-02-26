"""Factory functions for dimension-generic neighborhood attention autograd classes.

Each factory creates a ``torch.autograd.Function`` subclass parameterised by
rank (1, 2, or 3).  The generated classes are functionally identical to the
hand-written ones they replace — same class names, same public API.
"""

from __future__ import annotations

import torch

from natten_mps._core import ops
from natten_mps._core import pure as _pure


def create_na_autograd_fn(rank: int) -> type:
    """Create a fused NA autograd function for the given rank."""

    qk_fwd = getattr(ops, f"na{rank}d_qk_forward")
    av_bwd = getattr(ops, f"na{rank}d_av_backward")
    qk_bwd = getattr(ops, f"na{rank}d_qk_backward")
    pure_fwd = getattr(_pure, f"na{rank}d_forward")

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, kernel_size, stride, dilation, is_causal, scale):
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.is_causal = is_causal
            ctx.scale = scale

            # Try to get LSE for optimized fused backward
            lse = None
            out = None
            try:
                from natten_mps._core import metal as _metal
                fwd_with_lse = getattr(_metal, f'na{rank}d_forward_with_lse')
                out, lse = fwd_with_lse(q, k, v, kernel_size, stride, dilation, is_causal, scale)
            except (ImportError, AttributeError):
                pass

            if lse is not None:
                ctx.save_for_backward(q, k, v, lse, out)
                ctx.has_lse = True
                return out

            ctx.has_lse = False
            fwd = getattr(ops, f"na{rank}d_forward")
            out = fwd(q, k, v, kernel_size, stride, dilation, is_causal, scale)
            ctx.save_for_backward(q, k, v)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            scale_value = float(
                ctx.saved_tensors[0].shape[-1] ** -0.5
                if ctx.scale is None
                else ctx.scale
            )

            # Try fused SIMD backward (uses saved LSE + forward output)
            if ctx.has_lse:
                q, k, v, lse, fwd_out = ctx.saved_tensors
                try:
                    from natten_mps._core import metal as _metal
                    fused_bwd = getattr(_metal, f'na{rank}d_fused_backward')
                    dq, dk, dv = fused_bwd(
                        grad_output, q, k, v, fwd_out, lse,
                        ctx.kernel_size, ctx.dilation, ctx.stride,
                        ctx.is_causal, scale_value)
                    return dq, dk, dv, None, None, None, None, None
                except Exception:
                    pass
                # Fall through to split backward
            else:
                q, k, v = ctx.saved_tensors

            # Split backward path (original)
            logits = qk_fwd(q, k, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
            attn_weights = torch.softmax(logits * scale_value, dim=-1)

            av_result = av_bwd(
                grad_output, attn_weights, v, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
            if av_result is not None:
                d_attn, d_v = av_result
                d_logits = attn_weights * (d_attn - (attn_weights * d_attn).sum(-1, keepdim=True))
                d_logits = d_logits * scale_value

                qk_result = qk_bwd(
                    d_logits, q, k, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
                if qk_result is not None:
                    return qk_result[0], qk_result[1], d_v, None, None, None, None, None

            with torch.enable_grad():
                q_ = q.detach().requires_grad_(ctx.needs_input_grad[0])
                k_ = k.detach().requires_grad_(ctx.needs_input_grad[1])
                v_ = v.detach().requires_grad_(ctx.needs_input_grad[2])
                out = pure_fwd(
                    q_, k_, v_, ctx.kernel_size, ctx.stride, ctx.dilation, ctx.is_causal, ctx.scale)
                grads = torch.autograd.grad(out, (q_, k_, v_), grad_output, allow_unused=True)
            return grads[0], grads[1], grads[2], None, None, None, None, None

    _Fn.__name__ = f"NeighborhoodAttention{rank}DFunction"
    _Fn.__qualname__ = _Fn.__name__
    _Fn.__doc__ = f"Fused {rank}D neighborhood attention with autograd support."
    return _Fn


def create_na_qk_autograd_fn(rank: int) -> type:
    """Create a separate QK autograd function for the given rank."""

    qk_fwd = getattr(ops, f"na{rank}d_qk_forward")
    qk_bwd = getattr(ops, f"na{rank}d_qk_backward")
    pure_qk_fwd = getattr(_pure, f"na{rank}d_qk_forward")

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, kernel_size, dilation, stride, is_causal, scale):
            ctx.kernel_size = kernel_size
            ctx.dilation = dilation
            ctx.stride = stride
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.save_for_backward(q, k)
            return qk_fwd(q, k, kernel_size, dilation, stride, is_causal, scale)

        @staticmethod
        def backward(ctx, grad_output):
            q, k = ctx.saved_tensors

            result = qk_bwd(
                grad_output, q, k, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
            if result is not None:
                return result[0], result[1], None, None, None, None, None

            with torch.enable_grad():
                q_ = q.detach().requires_grad_(ctx.needs_input_grad[0])
                k_ = k.detach().requires_grad_(ctx.needs_input_grad[1])
                out = pure_qk_fwd(q_, k_, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal, ctx.scale)
                grads = torch.autograd.grad(out, (q_, k_), grad_output, allow_unused=True)
            return grads[0], grads[1], None, None, None, None, None

    _Fn.__name__ = f"NA{rank}DQKFunction"
    _Fn.__qualname__ = _Fn.__name__
    _Fn.__doc__ = f"Separate {rank}D QK operation with autograd. Returns attention logits."
    return _Fn


def create_na_av_autograd_fn(rank: int) -> type:
    """Create a separate AV autograd function for the given rank."""

    av_fwd = getattr(ops, f"na{rank}d_av_forward")
    av_bwd = getattr(ops, f"na{rank}d_av_backward")
    pure_av_fwd = getattr(_pure, f"na{rank}d_av_forward")

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, attn, v, kernel_size, dilation, stride, is_causal):
            ctx.kernel_size = kernel_size
            ctx.dilation = dilation
            ctx.stride = stride
            ctx.is_causal = is_causal
            ctx.save_for_backward(attn, v)
            return av_fwd(attn, v, kernel_size, dilation, stride, is_causal)

        @staticmethod
        def backward(ctx, grad_output):
            attn, v = ctx.saved_tensors

            result = av_bwd(
                grad_output, attn, v, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
            if result is not None:
                return result[0], result[1], None, None, None, None

            with torch.enable_grad():
                attn_ = attn.detach().requires_grad_(ctx.needs_input_grad[0])
                v_ = v.detach().requires_grad_(ctx.needs_input_grad[1])
                out = pure_av_fwd(attn_, v_, ctx.kernel_size, ctx.dilation, ctx.stride, ctx.is_causal)
                grads = torch.autograd.grad(out, (attn_, v_), grad_output, allow_unused=True)
            return grads[0], grads[1], None, None, None, None

    _Fn.__name__ = f"NA{rank}DAVFunction"
    _Fn.__qualname__ = _Fn.__name__
    _Fn.__doc__ = f"Separate {rank}D AV operation with autograd. Returns output."
    return _Fn
