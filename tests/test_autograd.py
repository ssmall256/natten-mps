import torch

from natten_mps.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk


def test_na1d_gradcheck():
    torch.manual_seed(0)
    q = torch.randn(1, 5, 2, 3, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 5, 2, 3, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 5, 2, 3, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b, c: na1d(a, b, c, kernel_size=3),
        (q, k, v),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_na2d_gradcheck():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 4, 2, 2, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 4, 4, 2, 2, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 4, 4, 2, 2, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b, c: na2d(a, b, c, kernel_size=(3, 3)),
        (q, k, v),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_na1d_qk_and_av_gradcheck():
    torch.manual_seed(2)
    q = torch.randn(1, 5, 2, 3, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 5, 2, 3, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b: na1d_qk(a, b, kernel_size=3),
        (q, k),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )

    attn = torch.randn(1, 5, 2, 3, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 5, 2, 3, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b: na1d_av(a, b, kernel_size=3),
        (attn, v),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_na2d_qk_and_av_gradcheck():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 4, 2, 2, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 4, 4, 2, 2, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b: na2d_qk(a, b, kernel_size=(3, 3)),
        (q, k),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )

    attn = torch.randn(1, 4, 4, 2, 9, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 4, 4, 2, 2, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b: na2d_av(a, b, kernel_size=(3, 3)),
        (attn, v),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
