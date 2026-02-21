import torch

import natten_mps.compat.v020 as natten


def test_v020_module_signature_with_stride():
    layer = natten.NeighborhoodAttention1D(embed_dim=128, num_heads=4, kernel_size=7, stride=2)
    x = torch.randn(2, 17, 128)
    y = layer(x)
    assert y.shape == (2, 9, 128)


def test_v020_functionals_with_stride():
    q = torch.randn(1, 11, 2, 8)
    k = torch.randn(1, 11, 2, 8)
    v = torch.randn(1, 11, 2, 8)

    out = natten.na1d(q, k, v, kernel_size=5, stride=2)
    assert out.shape == (1, 6, 2, 8)


def test_v020_feature_detection_functions_return_bool():
    assert isinstance(natten.has_cuda(), bool)
    assert isinstance(natten.has_mps(), bool)
    assert isinstance(natten.has_fna(), bool)
