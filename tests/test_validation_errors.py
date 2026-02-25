import pytest
import torch

from natten_mps.functional import na1d, na2d, na3d


class TestNa1dValidation:
    def test_wrong_ndim(self):
        q = torch.randn(2, 5, 5, 2, 4)
        with pytest.raises(ValueError, match="na1d"):
            na1d(q, q, q, kernel_size=3)

    def test_shape_mismatch(self):
        q = torch.randn(2, 10, 2, 4)
        k = torch.randn(2, 8, 2, 4)
        with pytest.raises(ValueError, match="Spatial dimensions must match"):
            na1d(q, k, q, kernel_size=3)

    def test_kernel_larger_than_input(self):
        q = torch.randn(1, 4, 1, 4)
        with pytest.raises(ValueError):
            na1d(q, q, q, kernel_size=7)

    def test_stride_larger_than_kernel(self):
        q = torch.randn(1, 10, 1, 4)
        with pytest.raises(ValueError):
            na1d(q, q, q, kernel_size=3, stride=5)

    def test_dilation_kernel_exceeds_input(self):
        q = torch.randn(1, 6, 1, 4)
        with pytest.raises(ValueError):
            na1d(q, q, q, kernel_size=3, dilation=3)


class TestNa2dValidation:
    def test_wrong_ndim(self):
        q = torch.randn(2, 5, 2, 4)
        with pytest.raises(ValueError, match="na2d"):
            na2d(q, q, q, kernel_size=3)

    def test_shape_mismatch(self):
        q = torch.randn(2, 5, 5, 2, 4)
        k = torch.randn(2, 5, 4, 2, 4)
        with pytest.raises(ValueError, match="Spatial dimensions must match"):
            na2d(q, k, q, kernel_size=3)

    def test_kernel_larger_than_input(self):
        q = torch.randn(1, 4, 4, 1, 4)
        with pytest.raises(ValueError):
            na2d(q, q, q, kernel_size=7)

    def test_stride_larger_than_kernel(self):
        q = torch.randn(1, 10, 10, 1, 4)
        with pytest.raises(ValueError):
            na2d(q, q, q, kernel_size=3, stride=5)


class TestNa3dValidation:
    def test_wrong_ndim(self):
        q = torch.randn(2, 5, 5, 2, 4)
        with pytest.raises(ValueError, match="na3d"):
            na3d(q, q, q, kernel_size=3)

    def test_shape_mismatch(self):
        q = torch.randn(2, 5, 5, 5, 2, 4)
        k = torch.randn(2, 5, 5, 4, 2, 4)
        with pytest.raises(ValueError, match="Spatial dimensions must match"):
            na3d(q, k, q, kernel_size=3)

    def test_kernel_larger_than_input(self):
        q = torch.randn(1, 4, 4, 4, 1, 4)
        with pytest.raises(ValueError):
            na3d(q, q, q, kernel_size=7)

    def test_stride_larger_than_kernel(self):
        q = torch.randn(1, 9, 9, 9, 1, 4)
        with pytest.raises(ValueError):
            na3d(q, q, q, kernel_size=3, stride=5)

    def test_dilation_kernel_exceeds_input(self):
        q = torch.randn(1, 5, 5, 5, 1, 4)
        with pytest.raises(ValueError):
            na3d(q, q, q, kernel_size=3, dilation=3)
