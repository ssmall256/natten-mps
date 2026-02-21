from __future__ import annotations

import torch


def get_window_start_vectorized(
    indices: torch.Tensor,
    length: int,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """
    Vectorized form of NATTEN's get_window_start from CPU kernels.

    This preserves boundary behavior for dilated neighborhoods, including
    sequence-length / dilation remainder handling.
    """
    neighborhood_size = kernel_size // 2
    indices = indices.to(dtype=torch.long)

    if dilation <= 1:
        start = torch.clamp(indices - neighborhood_size, min=0)
        right_boundary = indices + neighborhood_size >= length
        start = start + right_boundary.to(dtype=start.dtype) * (
            length - indices - neighborhood_size - 1
        )
        return start

    start = indices - neighborhood_size * dilation
    left_boundary = start < 0
    left_result = torch.remainder(indices, dilation)

    right_boundary = indices + neighborhood_size * dilation >= length
    imodd = torch.remainder(indices, dilation)
    a = (length // dilation) * dilation
    b = length - a
    right_result = torch.where(
        imodd < b,
        length - b + imodd - 2 * neighborhood_size * dilation,
        a + imodd - kernel_size * dilation,
    )

    result = torch.where(left_boundary, left_result, start)
    right_only = (~left_boundary) & right_boundary
    result = torch.where(right_only, right_result, result)
    return result


def get_pb_start_vectorized(
    indices: torch.Tensor,
    length: int,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """
    Vectorized form of NATTEN's get_pb_start from CPU kernels.
    """
    neighborhood_size = kernel_size // 2
    indices = indices.to(dtype=torch.long)

    if dilation <= 1:
        pb_start = torch.full_like(indices, neighborhood_size)
        left_boundary = indices < neighborhood_size
        pb_start = pb_start + left_boundary.to(dtype=pb_start.dtype) * (
            neighborhood_size - indices
        )
        right_boundary = indices + neighborhood_size >= length
        pb_start = pb_start + right_boundary.to(dtype=pb_start.dtype) * (
            length - indices - 1 - neighborhood_size
        )
        return pb_start

    left_boundary = indices - neighborhood_size * dilation < 0
    left_result = kernel_size - 1 - (indices // dilation)

    right_boundary = indices + neighborhood_size * dilation >= length
    right_result = (length - indices - 1) // dilation

    default_result = torch.full_like(indices, neighborhood_size)
    result = torch.where(left_boundary, left_result, default_result)
    right_only = (~left_boundary) & right_boundary
    result = torch.where(right_only, right_result, result)
    return result
