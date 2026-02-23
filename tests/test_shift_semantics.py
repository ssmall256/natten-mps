"""Verify NATTEN windowing semantics: boundary behavior, dilation effects."""

import torch

from natten_mps.utils.window import (
    get_pb_start,
    get_pb_start_vectorized,
    get_window_end,
    get_window_start,
    get_window_start_vectorized,
)


class TestScalarMatchesVectorized:
    def _check_window_start(self, length, kernel_size, dilation):
        indices = torch.arange(length)
        vec = get_window_start_vectorized(indices, length, kernel_size, dilation)
        for i in range(length):
            scalar = get_window_start(i, length, kernel_size, dilation)
            assert scalar == vec[i].item(), (
                f"Mismatch at index={i}, length={length}, ks={kernel_size}, dil={dilation}: "
                f"scalar={scalar}, vec={vec[i].item()}"
            )

    def _check_pb_start(self, length, kernel_size, dilation):
        indices = torch.arange(length)
        vec = get_pb_start_vectorized(indices, length, kernel_size, dilation)
        for i in range(length):
            scalar = get_pb_start(i, length, kernel_size, dilation)
            assert scalar == vec[i].item(), (
                f"pb_start mismatch at index={i}, length={length}, ks={kernel_size}, dil={dilation}: "
                f"scalar={scalar}, vec={vec[i].item()}"
            )

    def test_window_start_basic(self):
        self._check_window_start(10, 3, 1)

    def test_window_start_large_kernel(self):
        self._check_window_start(10, 7, 1)

    def test_window_start_dilation(self):
        self._check_window_start(12, 3, 2)

    def test_window_start_dilation_large(self):
        self._check_window_start(15, 5, 3)

    def test_pb_start_basic(self):
        self._check_pb_start(10, 3, 1)

    def test_pb_start_dilation(self):
        self._check_pb_start(12, 3, 2)


class TestWindowEnd:
    def test_end_equals_start_plus_range(self):
        for ks in [3, 5, 7]:
            for dil in [1, 2]:
                for length in [10, 15, 20]:
                    for i in range(length):
                        start = get_window_start(i, length, ks, dil)
                        end = get_window_end(i, length, ks, dil)
                        assert end == start + (ks - 1) * dil


class TestBoundaryBehavior:
    def test_center_positions_centered(self):
        """Interior positions should produce centered windows."""
        length, ks, dil = 20, 3, 1
        neighborhood = ks // 2
        for i in range(neighborhood, length - neighborhood):
            start = get_window_start(i, length, ks, dil)
            assert start == i - neighborhood

    def test_left_boundary_clamps(self):
        """At the left edge, window starts at 0."""
        start = get_window_start(0, 10, 3, 1)
        assert start == 0

    def test_right_boundary_shifts(self):
        """At the right edge, window shifts left to stay in bounds."""
        start = get_window_start(9, 10, 3, 1)
        assert start == 7  # 10 - 3 = 7

    def test_dilated_positions_in_bounds(self):
        """All dilated positions should be within [0, length)."""
        length, ks, dil = 12, 3, 2
        for i in range(length):
            start = get_window_start(i, length, ks, dil)
            for ki in range(ks):
                pos = start + ki * dil
                assert 0 <= pos < length, f"Out of bounds: pos={pos} at index={i}"

    def test_window_covers_kernel_size_positions(self):
        """Each window should touch exactly kernel_size distinct positions."""
        length, ks, dil = 15, 5, 1
        for i in range(length):
            start = get_window_start(i, length, ks, dil)
            positions = [start + ki * dil for ki in range(ks)]
            assert len(set(positions)) == ks
