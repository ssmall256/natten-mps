import pytest
import torch

from natten_mps.nn import NeighborhoodAttention3D


def test_neighborhood_attention_3d_module_shape():
    mod = NeighborhoodAttention3D(embed_dim=32, num_heads=4, kernel_size=3)
    x = torch.randn(2, 6, 6, 6, 32)
    out = mod(x)
    assert out.shape == (2, 6, 6, 6, 32)


def test_neighborhood_attention_3d_module_stride_downsamples():
    mod = NeighborhoodAttention3D(embed_dim=32, num_heads=4, kernel_size=3, stride=2)
    x = torch.randn(1, 8, 8, 8, 32)
    out = mod(x)
    assert out.shape == (1, 4, 4, 4, 32)


def test_neighborhood_attention_3d_module_with_attn_drop():
    mod = NeighborhoodAttention3D(embed_dim=32, num_heads=4, kernel_size=3, attn_drop=0.1)
    mod.eval()
    x = torch.randn(1, 5, 5, 5, 32)
    out = mod(x)
    assert out.shape == (1, 5, 5, 5, 32)


def test_neighborhood_attention_3d_module_with_dilation():
    mod = NeighborhoodAttention3D(embed_dim=16, num_heads=2, kernel_size=3, dilation=2)
    x = torch.randn(1, 7, 7, 7, 16)
    out = mod(x)
    assert out.shape == (1, 7, 7, 7, 16)


def test_neighborhood_attention_3d_wrong_input_ndim():
    mod = NeighborhoodAttention3D(embed_dim=32, num_heads=4, kernel_size=3)
    with pytest.raises(ValueError, match="expects input shape"):
        mod(torch.randn(2, 6, 6, 32))


def test_neighborhood_attention_3d_wrong_channels():
    mod = NeighborhoodAttention3D(embed_dim=32, num_heads=4, kernel_size=3)
    with pytest.raises(ValueError, match="embed_dim"):
        mod(torch.randn(1, 5, 5, 5, 16))
