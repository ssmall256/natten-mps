"""Verify fused-path gradients match split-path (QK -> softmax -> AV) gradients."""

import pytest
import torch

from natten_mps.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk, na3d, na3d_av, na3d_qk


def _split_path_1d(q, k, v, kernel_size, dilation=1):
    scale = q.shape[-1] ** -0.5
    logits = na1d_qk(q, k, kernel_size, dilation=dilation)
    attn = torch.softmax(logits * scale, dim=-1)
    return na1d_av(attn, v, kernel_size, dilation=dilation)


def _split_path_2d(q, k, v, kernel_size, dilation=1):
    scale = q.shape[-1] ** -0.5
    logits = na2d_qk(q, k, kernel_size, dilation=dilation)
    attn = torch.softmax(logits * scale, dim=-1)
    return na2d_av(attn, v, kernel_size, dilation=dilation)


def _split_path_3d(q, k, v, kernel_size, dilation=1):
    scale = q.shape[-1] ** -0.5
    logits = na3d_qk(q, k, kernel_size, dilation=dilation)
    attn = torch.softmax(logits * scale, dim=-1)
    return na3d_av(attn, v, kernel_size, dilation=dilation)


@pytest.mark.parametrize("kernel_size,dilation", [(3, 1), (5, 1), (3, 2)])
class TestBackwardParity1D:
    def test_output_matches(self, kernel_size, dilation):
        q = torch.randn(1, 12, 2, 8)
        k = q.clone()
        v = torch.randn_like(q)

        out_fused = na1d(q, k, v, kernel_size, dilation=dilation)
        out_split = _split_path_1d(q, k, v, kernel_size, dilation=dilation)
        torch.testing.assert_close(out_fused, out_split, atol=1e-6, rtol=1e-6)

    def test_gradients_match(self, kernel_size, dilation):
        q1 = torch.randn(1, 12, 2, 8, requires_grad=True)
        k1 = torch.randn_like(q1, requires_grad=True)
        v1 = torch.randn_like(q1, requires_grad=True)
        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)

        na1d(q1, k1, v1, kernel_size, dilation=dilation).sum().backward()
        _split_path_1d(q2, k2, v2, kernel_size, dilation=dilation).sum().backward()

        torch.testing.assert_close(q1.grad, q2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k1.grad, k2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v1.grad, v2.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kernel_size", [3, 5])
class TestBackwardParity2D:
    def test_gradients_match(self, kernel_size):
        q1 = torch.randn(1, 8, 8, 2, 8, requires_grad=True)
        k1 = torch.randn_like(q1, requires_grad=True)
        v1 = torch.randn_like(q1, requires_grad=True)
        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)

        na2d(q1, k1, v1, kernel_size).sum().backward()
        _split_path_2d(q2, k2, v2, kernel_size).sum().backward()

        torch.testing.assert_close(q1.grad, q2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k1.grad, k2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v1.grad, v2.grad, atol=1e-5, rtol=1e-5)


class TestBackwardParity3D:
    def test_gradients_match(self):
        q1 = torch.randn(1, 4, 4, 4, 2, 8, requires_grad=True)
        k1 = torch.randn_like(q1, requires_grad=True)
        v1 = torch.randn_like(q1, requires_grad=True)
        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)

        na3d(q1, k1, v1, 3).sum().backward()
        _split_path_3d(q2, k2, v2, 3).sum().backward()

        torch.testing.assert_close(q1.grad, q2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k1.grad, k2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v1.grad, v2.grad, atol=1e-5, rtol=1e-5)
