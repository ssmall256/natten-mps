from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from natten_mps._core import ops
from natten_mps.autograd.na1d import NA1DAVFunction, NA1DQKFunction, NeighborhoodAttention1DFunction
from natten_mps.autograd.na2d import NA2DAVFunction, NA2DQKFunction, NeighborhoodAttention2DFunction
from natten_mps.utils.params import (
    check_dilation_kernel_vs_input,
    check_kernel_size_vs_input,
    check_stride_vs_kernel,
    normalize_kernel_size,
    normalize_tuple_param,
)


def _normalize_is_causal(is_causal, rank: int) -> tuple[bool, ...]:
    values = normalize_tuple_param(is_causal, rank, "is_causal")
    return tuple(bool(v) for v in values)


def _require_same_shape_1d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("na1d expects query/key/value with shape [B, L, H, D].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have the same shape for na1d.")


def _require_same_shape_2d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if query.ndim != 5 or key.ndim != 5 or value.ndim != 5:
        raise ValueError("na2d expects query/key/value with shape [B, H, W, heads, dim].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have the same shape for na2d.")


def _using_pure_backend() -> bool:
    return ops.get_backend() == "pure"


def na1d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int]] = 1,
    dilation: Union[int, Tuple[int]] = 1,
    is_causal: Union[bool, Tuple[bool]] = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    _require_same_shape_1d(query, key, value)

    ks = normalize_kernel_size(kernel_size, 1)
    st = normalize_tuple_param(stride, 1, "stride")
    dil = normalize_tuple_param(dilation, 1, "dilation")
    causal = _normalize_is_causal(is_causal, 1)

    check_kernel_size_vs_input(ks, (query.shape[1],))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (query.shape[1],))

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    if _using_pure_backend():
        return ops.na1d_forward(query, key, value, ks, st, dil, causal, scale_value)
    return NeighborhoodAttention1DFunction.apply(query, key, value, ks, st, dil, causal, scale_value)


def na2d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool]] = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    _require_same_shape_2d(query, key, value)

    ks = normalize_kernel_size(kernel_size, 2)
    st = normalize_tuple_param(stride, 2, "stride")
    dil = normalize_tuple_param(dilation, 2, "dilation")
    causal = _normalize_is_causal(is_causal, 2)

    spatial_shape = (query.shape[1], query.shape[2])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    if _using_pure_backend():
        return ops.na2d_forward(query, key, value, ks, st, dil, causal, scale_value)
    return NeighborhoodAttention2DFunction.apply(query, key, value, ks, st, dil, causal, scale_value)


def na1d_qk(query, key, kernel_size, dilation=1):
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("na1d_qk expects query/key with shape [B, L, H, D].")
    if query.shape != key.shape:
        raise ValueError("query and key must share the same shape in na1d_qk.")

    ks = normalize_kernel_size(kernel_size, 1)
    dil = normalize_tuple_param(dilation, 1, "dilation")
    check_kernel_size_vs_input(ks, (query.shape[1],))
    check_dilation_kernel_vs_input(dil, ks, (query.shape[1],))

    if _using_pure_backend():
        return ops.na1d_qk_forward(query, key, ks, dil)
    return NA1DQKFunction.apply(query, key, ks, dil)


def na1d_av(attn, value, kernel_size, dilation=1):
    if attn.ndim != 4:
        raise ValueError("na1d_av expects attn with shape [B, L, H, K].")
    if value.ndim != 4:
        raise ValueError("na1d_av expects value with shape [B, L, H, D].")
    if attn.shape[0] != value.shape[0] or attn.shape[2] != value.shape[2]:
        raise ValueError("na1d_av requires attn/value to match on batch and heads dimensions.")
    if attn.shape[1] > value.shape[1]:
        raise ValueError("na1d_av attn sequence length cannot exceed value sequence length.")

    ks = normalize_kernel_size(kernel_size, 1)
    dil = normalize_tuple_param(dilation, 1, "dilation")
    if attn.shape[-1] != ks[0]:
        raise ValueError(
            f"na1d_av attn last dim ({attn.shape[-1]}) must match kernel_size ({ks[0]})."
        )
    check_kernel_size_vs_input(ks, (value.shape[1],))
    check_dilation_kernel_vs_input(dil, ks, (value.shape[1],))

    if _using_pure_backend():
        return ops.na1d_av_forward(attn, value, ks, dil)
    return NA1DAVFunction.apply(attn, value, ks, dil)


def na2d_qk(query, key, kernel_size, dilation=1):
    if query.ndim != 5 or key.ndim != 5:
        raise ValueError("na2d_qk expects query/key with shape [B, H, W, heads, dim].")
    if query.shape != key.shape:
        raise ValueError("query and key must share the same shape in na2d_qk.")

    ks = normalize_kernel_size(kernel_size, 2)
    dil = normalize_tuple_param(dilation, 2, "dilation")
    spatial_shape = (query.shape[1], query.shape[2])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    if _using_pure_backend():
        return ops.na2d_qk_forward(query, key, ks, dil)
    return NA2DQKFunction.apply(query, key, ks, dil)


def na2d_av(attn, value, kernel_size, dilation=1):
    if attn.ndim != 5:
        raise ValueError("na2d_av expects attn with shape [B, H, W, heads, K].")
    if value.ndim != 5:
        raise ValueError("na2d_av expects value with shape [B, H, W, heads, D].")
    if attn.shape[0] != value.shape[0] or attn.shape[3] != value.shape[3]:
        raise ValueError("na2d_av requires attn/value to match on batch and heads dimensions.")
    if attn.shape[1] > value.shape[1] or attn.shape[2] > value.shape[2]:
        raise ValueError("na2d_av attn spatial size cannot exceed value spatial size.")

    ks = normalize_kernel_size(kernel_size, 2)
    dil = normalize_tuple_param(dilation, 2, "dilation")
    kernel_area = ks[0] * ks[1]
    if attn.shape[-1] != kernel_area:
        raise ValueError(
            f"na2d_av attn last dim ({attn.shape[-1]}) must match kernel area ({kernel_area})."
        )
    spatial_shape = (value.shape[1], value.shape[2])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    if _using_pure_backend():
        return ops.na2d_av_forward(attn, value, ks, dil)
    return NA2DAVFunction.apply(attn, value, ks, dil)


__all__ = ["na1d", "na2d", "na1d_qk", "na1d_av", "na2d_qk", "na2d_av"]
