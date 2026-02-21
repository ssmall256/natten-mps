import torch

from natten_mps.nn import NeighborhoodAttention2D


def test_na2d_module_forward_shape():
    layer = NeighborhoodAttention2D(embed_dim=32, num_heads=4, kernel_size=(5, 5))
    x = torch.randn(2, 8, 8, 32)
    y = layer(x)
    assert y.shape == (2, 8, 8, 32)


def test_na2d_module_stride_reduces_spatial_shape():
    layer = NeighborhoodAttention2D(embed_dim=32, num_heads=4, kernel_size=(3, 3), stride=(2, 2))
    x = torch.randn(2, 9, 9, 32)
    y = layer(x)
    assert y.shape == (2, 5, 5, 32)


def test_na2d_module_attn_drop_fallback_path_runs():
    layer = NeighborhoodAttention2D(embed_dim=32, num_heads=4, kernel_size=(3, 3), attn_drop=0.1)
    x = torch.randn(2, 8, 8, 32)
    y = layer(x)
    assert y.shape == (2, 8, 8, 32)
