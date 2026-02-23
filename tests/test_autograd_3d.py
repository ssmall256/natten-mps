import torch

from natten_mps.functional import na3d, na3d_av, na3d_qk


def test_na3d_gradcheck():
    q = torch.randn(1, 3, 3, 3, 1, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 3, 3, 3, 1, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 3, 3, 3, 1, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda q_, k_, v_: na3d(q_, k_, v_, kernel_size=3),
        (q, k, v),
        atol=1e-4,
        rtol=1e-3,
    )


def test_na3d_qk_gradcheck():
    q = torch.randn(1, 3, 3, 3, 1, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 3, 3, 3, 1, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda q_, k_: na3d_qk(q_, k_, kernel_size=3),
        (q, k),
        atol=1e-4,
        rtol=1e-3,
    )


def test_na3d_av_gradcheck():
    attn = torch.randn(1, 3, 3, 3, 1, 27, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 3, 3, 3, 1, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a_, v_: na3d_av(a_, v_, kernel_size=3),
        (attn, v),
        atol=1e-4,
        rtol=1e-3,
    )
