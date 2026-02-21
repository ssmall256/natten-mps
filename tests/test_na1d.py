import torch

from natten_mps.nn import NeighborhoodAttention1D


def test_na1d_module_forward_shape():
    layer = NeighborhoodAttention1D(embed_dim=32, num_heads=4, kernel_size=7)
    x = torch.randn(2, 16, 32)
    y = layer(x)
    assert y.shape == (2, 16, 32)


def test_na1d_module_stride_reduces_length():
    layer = NeighborhoodAttention1D(embed_dim=32, num_heads=4, kernel_size=5, stride=2)
    x = torch.randn(2, 17, 32)
    y = layer(x)
    assert y.shape == (2, 9, 32)


def test_na1d_module_attn_drop_fallback_path_runs():
    layer = NeighborhoodAttention1D(embed_dim=32, num_heads=4, kernel_size=5, attn_drop=0.1)
    x = torch.randn(2, 12, 32)
    y = layer(x)
    assert y.shape == (2, 12, 32)
