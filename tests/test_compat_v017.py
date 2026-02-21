import torch

import natten_mps.compat.v017 as natten


def test_v017_reexported_surface_smoke():
    q = torch.randn(1, 10, 2, 4)
    k = torch.randn(1, 10, 2, 4)
    v = torch.randn(1, 10, 2, 4)

    out = natten.na1d(q, k, v, kernel_size=3)
    assert out.shape == (1, 10, 2, 4)
