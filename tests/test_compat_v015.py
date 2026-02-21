import pytest
import torch

import natten_mps.compat.v015 as natten


def test_v015_fused_apis_work():
    q = torch.randn(2, 12, 4, 8)
    k = torch.randn(2, 12, 4, 8)
    v = torch.randn(2, 12, 4, 8)

    out = natten.na1d(q, k, v, kernel_size=5, dilation=1, is_causal=False)
    assert out.shape == (2, 12, 4, 8)


def test_v015_neighborhood_attention3d_not_supported():
    with pytest.raises(NotImplementedError):
        natten.NeighborhoodAttention3D(dim=64, kernel_size=3, num_heads=4)


def test_v015_feature_detection_functions_return_bool():
    assert isinstance(natten.has_cuda(), bool)
    assert isinstance(natten.has_mps(), bool)
    assert isinstance(natten.has_gemm(), bool)
    assert isinstance(natten.has_fna(), bool)
