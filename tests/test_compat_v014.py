import torch

import natten_mps.compat.v014 as natten
from natten_mps.utils.window import get_pb_start_vectorized


def test_v014_neighborhood_attention_1d_module_smoke():
    layer = natten.NeighborhoodAttention1D(dim=128, kernel_size=7, num_heads=4)
    x = torch.randn(2, 16, 128)
    y = layer(x)
    assert y.shape == (2, 16, 128)


def test_v014_1d_functionals_shapes():
    q = torch.randn(2, 4, 16, 8)
    k = torch.randn(2, 4, 16, 8)
    v = torch.randn(2, 4, 16, 8)
    rpb = torch.randn(4, 13)

    logits = natten.natten1dqkrpb(q, k, rpb, kernel_size=7, dilation=1)
    out = natten.natten1dav(torch.softmax(logits, dim=-1), v, kernel_size=7, dilation=1)

    assert logits.shape == (2, 4, 16, 7)
    assert out.shape == (2, 4, 16, 8)


def test_v014_2d_functionals_shapes():
    q = torch.randn(2, 4, 6, 6, 8)
    k = torch.randn(2, 4, 6, 6, 8)
    v = torch.randn(2, 4, 6, 6, 8)
    rpb = torch.randn(4, 5, 5)

    logits = natten.natten2dqkrpb(q, k, rpb, kernel_size=(3, 3), dilation=(1, 1))
    out = natten.natten2dav(torch.softmax(logits, dim=-1), v, kernel_size=(3, 3), dilation=(1, 1))

    assert logits.shape == (2, 4, 6, 6, 9)
    assert out.shape == (2, 4, 6, 6, 8)


def test_v014_rpb_is_added_linearly():
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)

    rpb_a = torch.randn(2, 9)
    rpb_b = torch.randn(2, 9)
    rpb_0 = torch.zeros(2, 9)

    base = natten.natten1dqkrpb(q, k, rpb_0, kernel_size=5, dilation=1)
    out_a = natten.natten1dqkrpb(q, k, rpb_a, kernel_size=5, dilation=1)
    out_b = natten.natten1dqkrpb(q, k, rpb_b, kernel_size=5, dilation=1)
    out_ab = natten.natten1dqkrpb(q, k, rpb_a + rpb_b, kernel_size=5, dilation=1)

    assert torch.allclose(out_ab - base, (out_a - base) + (out_b - base), atol=1e-6, rtol=1e-6)


def test_v014_1d_rpb_indexing_matches_natten_pb_start_for_dilation():
    q = torch.zeros(1, 1, 7, 4)
    k = torch.zeros(1, 1, 7, 4)
    rpb = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])

    logits = natten.natten1dqkrpb(q, k, rpb, kernel_size=3, dilation=2)

    pb_start = get_pb_start_vectorized(
        torch.arange(7, dtype=torch.long),
        length=7,
        kernel_size=3,
        dilation=2,
    )
    pb_indices = pb_start.unsqueeze(1) + torch.arange(3, dtype=torch.long).unsqueeze(0)
    expected = rpb[0, pb_indices]

    assert torch.allclose(logits[0, 0], expected, atol=1e-6, rtol=1e-6)


def test_v014_2d_rpb_indexing_matches_natten_pb_start_for_dilation():
    q = torch.zeros(1, 1, 7, 8, 4)
    k = torch.zeros(1, 1, 7, 8, 4)
    rpb = torch.arange(25, dtype=torch.float32).view(1, 5, 5)

    logits = natten.natten2dqkrpb(q, k, rpb, kernel_size=(3, 3), dilation=(2, 2))

    pb_h_start = get_pb_start_vectorized(
        torch.arange(7, dtype=torch.long),
        length=7,
        kernel_size=3,
        dilation=2,
    )
    pb_w_start = get_pb_start_vectorized(
        torch.arange(8, dtype=torch.long),
        length=8,
        kernel_size=3,
        dilation=2,
    )
    pb_h = pb_h_start.unsqueeze(1) + torch.arange(3, dtype=torch.long).unsqueeze(0)
    pb_w = pb_w_start.unsqueeze(1) + torch.arange(3, dtype=torch.long).unsqueeze(0)

    expected = torch.empty(7, 8, 9, dtype=torch.float32)
    for i in range(7):
        for j in range(8):
            patch = rpb[0, pb_h[i][:, None], pb_w[j][None, :]]
            expected[i, j] = patch.reshape(-1)

    assert torch.allclose(logits[0, 0], expected, atol=1e-6, rtol=1e-6)
