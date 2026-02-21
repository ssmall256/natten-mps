from __future__ import annotations

from typing import Sequence


def _to_tuple(param, rank: int, name: str):
    if isinstance(param, (int, bool)):
        values = (param,) * rank
    elif isinstance(param, Sequence) and not isinstance(param, (str, bytes)):
        values = tuple(param)
    else:
        raise ValueError(f"{name} must be an int, bool, or tuple/list of length {rank}.")

    if len(values) != rank:
        raise ValueError(f"{name} must have length {rank}, got {len(values)}.")
    return values


def normalize_tuple_param(param, rank, name):
    """Generic: int/bool/tuple -> tuple of length `rank`, with name for error messages."""
    values = _to_tuple(param, rank, name)

    if all(isinstance(v, bool) for v in values):
        return tuple(bool(v) for v in values)

    for v in values:
        if isinstance(v, bool):
            raise ValueError(f"{name} cannot mix bool and int values.")
        if not isinstance(v, int):
            raise ValueError(f"{name} values must be int or bool.")
        if v <= 0:
            raise ValueError(f"{name} values must be > 0, got {v}.")

    return tuple(int(v) for v in values)


def normalize_kernel_size(kernel_size, rank):
    """Convert int -> tuple of length `rank`. Pass-through tuples after validation."""
    values = normalize_tuple_param(kernel_size, rank, "kernel_size")
    for v in values:
        if isinstance(v, bool):
            raise ValueError("kernel_size must contain integers, not bools.")
    return values


def check_kernel_size_vs_input(kernel_size, input_spatial_shape):
    """Validate kernel_size <= spatial dims."""
    if len(kernel_size) != len(input_spatial_shape):
        raise ValueError(
            f"kernel_size rank ({len(kernel_size)}) must match input rank ({len(input_spatial_shape)})."
        )
    for idx, (k, n) in enumerate(zip(kernel_size, input_spatial_shape)):
        if k > n:
            raise ValueError(
                f"kernel_size[{idx}]={k} cannot exceed input spatial size {n}."
            )


def check_stride_vs_kernel(stride, kernel_size):
    """Validate stride <= kernel_size per dimension."""
    if len(stride) != len(kernel_size):
        raise ValueError(
            f"stride rank ({len(stride)}) must match kernel_size rank ({len(kernel_size)})."
        )
    for idx, (s, k) in enumerate(zip(stride, kernel_size)):
        if s > k:
            raise ValueError(
                f"stride[{idx}]={s} must be <= kernel_size[{idx}]={k}."
            )


def check_dilation_kernel_vs_input(dilation, kernel_size, input_spatial_shape):
    """Validate dilation * kernel_size <= spatial dims per dimension."""
    if not (len(dilation) == len(kernel_size) == len(input_spatial_shape)):
        raise ValueError("dilation, kernel_size, and input_spatial_shape must share the same rank.")

    for idx, (d, k, n) in enumerate(zip(dilation, kernel_size, input_spatial_shape)):
        required = d * k
        if required > n:
            raise ValueError(
                f"dilation[{idx}] * kernel_size[{idx}] = {required} exceeds input spatial size {n}."
            )
