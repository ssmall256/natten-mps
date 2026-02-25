from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

import torch.nn.functional as F

from natten_mps._core import ops
from natten_mps.autograd.na1d import NA1DAVFunction, NA1DQKFunction, NeighborhoodAttention1DFunction
from natten_mps.autograd.na2d import NA2DAVFunction, NA2DQKFunction, NeighborhoodAttention2DFunction
from natten_mps.autograd.na3d import NA3DAVFunction, NA3DQKFunction, NeighborhoodAttention3DFunction
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


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for GQA/MQA."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=-2)


def _validate_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, rank: int
) -> int:
    """Validate Q/K/V shapes with GQA support. Returns kv_repeat factor."""
    expected_ndim = rank + 3  # batch + spatial dims + heads + dim
    if query.ndim != expected_ndim or key.ndim != expected_ndim or value.ndim != expected_ndim:
        layouts = {1: "[B, L, H, D]", 2: "[B, H, W, heads, dim]", 3: "[B, D, H, W, heads, dim]"}
        raise ValueError(f"na{rank}d expects query/key/value with shape {layouts[rank]}.")

    # Spatial dims must match
    spatial_q = query.shape[1:-2]
    spatial_k = key.shape[1:-2]
    spatial_v = value.shape[1:-2]
    if spatial_q != spatial_k or spatial_q != spatial_v:
        raise ValueError(
            f"Spatial dimensions must match: Q={spatial_q}, K={spatial_k}, V={spatial_v}."
        )

    # Batch must match
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch dimensions must match.")

    # Head dim must match between Q and K; K and V must have same heads
    heads_q, dim_q = query.shape[-2], query.shape[-1]
    heads_kv, dim_k = key.shape[-2], key.shape[-1]
    if dim_q != dim_k:
        raise ValueError(f"Head dim must match: Q has {dim_q}, K has {dim_k}.")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(f"K and V must have same number of heads: K={key.shape[-2]}, V={value.shape[-2]}.")
    if dim_k != value.shape[-1]:
        raise ValueError(f"Head dim must match: K has {dim_k}, V has {value.shape[-1]}.")

    # GQA: heads_q must be divisible by heads_kv
    if heads_q % heads_kv != 0:
        raise ValueError(
            f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv}) for GQA."
        )

    return heads_q // heads_kv


def _full_attn_with_lse(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full (non-neighborhood) attention with LSE return.

    Q: [B, ..spatial.., H, D], K/V: [B, N_extra, H, D].
    Returns (output, lse) where output has Q's spatial shape and lse has shape
    [B, ..spatial.., H].
    """
    spatial_dims = query.shape[1:-2]
    B, H, D = query.shape[0], query.shape[-2], query.shape[-1]
    N_extra = key.shape[1]

    # Flatten spatial dims for matmul: [B, S, H, D]
    S = 1
    for s in spatial_dims:
        S *= s
    q_flat = query.reshape(B, S, H, D)

    # Compute logits: [B, S, H, N_extra]
    # q_flat: [B, S, H, D], key: [B, N_extra, H, D]
    # Transpose to [B, H, S, D] and [B, H, N_extra, D] for bmm
    q_t = q_flat.permute(0, 2, 1, 3)  # [B, H, S, D]
    k_t = key.permute(0, 2, 1, 3)     # [B, H, N_extra, D]
    logits = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, H, S, N_extra]

    lse = torch.logsumexp(logits, dim=-1)  # [B, H, S]
    attn = torch.softmax(logits, dim=-1)   # [B, H, S, N_extra]

    v_t = value.permute(0, 2, 1, 3)  # [B, H, N_extra, D]
    out = torch.matmul(attn, v_t)     # [B, H, S, D]

    # Back to [B, ..spatial.., H, D]
    out = out.permute(0, 2, 1, 3).reshape(*([B] + list(spatial_dims) + [H, D]))
    lse = lse.permute(0, 2, 1).reshape(*([B] + list(spatial_dims) + [H]))

    return out, lse


def _validate_additional_kv(
    query: torch.Tensor,
    additional_keys: Optional[torch.Tensor],
    additional_values: Optional[torch.Tensor],
    kv_repeat: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Validate and optionally expand additional K/V for GQA."""
    if additional_keys is None and additional_values is None:
        return None, None
    if additional_keys is None or additional_values is None:
        raise ValueError("additional_keys and additional_values must both be provided or both None.")

    if additional_keys.ndim != 4 or additional_values.ndim != 4:
        raise ValueError(
            "additional_keys/additional_values must be 4D: [B, N_extra, heads_kv, dim]."
        )
    if additional_keys.shape[0] != query.shape[0]:
        raise ValueError("additional_keys batch must match query batch.")
    if additional_values.shape[0] != query.shape[0]:
        raise ValueError("additional_values batch must match query batch.")
    if additional_keys.shape[-1] != query.shape[-1]:
        raise ValueError("additional_keys head dim must match query head dim.")
    if additional_values.shape[-1] != query.shape[-1]:
        raise ValueError("additional_values head dim must match query head dim.")
    if additional_keys.shape[-2] != additional_values.shape[-2]:
        raise ValueError("additional_keys and additional_values must have same number of heads.")
    if additional_keys.shape[1] != additional_values.shape[1]:
        raise ValueError("additional_keys and additional_values must have same N_extra.")

    # Expand for GQA if needed
    ak = _repeat_kv(additional_keys, kv_repeat)
    av = _repeat_kv(additional_values, kv_repeat)
    return ak, av


def _is_full_attention(kernel_size: tuple, spatial_shape: tuple) -> bool:
    """Check if kernel covers all spatial dims (NA degenerates to global attention)."""
    return all(ks >= s for ks, s in zip(kernel_size, spatial_shape))


def _sdpa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    spatial_shape: tuple,
    return_lse: bool = False,
    additional_keys: Optional[torch.Tensor] = None,
    additional_values: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Full-attention fast path via SDPA when kernel covers entire spatial extent.

    Reshapes from [B, ..spatial.., H, D] to [B, H, S, D] for SDPA, then back.
    """
    B, H, D = query.shape[0], query.shape[-2], query.shape[-1]

    # Concatenate additional KV if provided
    if additional_keys is not None:
        # Flatten spatial dims of K/V
        S = 1
        for s in spatial_shape:
            S *= s
        k_flat = key.reshape(B, S, H, D)
        v_flat = value.reshape(B, S, H, D)
        # Concat additional tokens
        k_all = torch.cat([k_flat, additional_keys], dim=1)
        v_all = torch.cat([v_flat, additional_values], dim=1)
    else:
        S = 1
        for s in spatial_shape:
            S *= s
        k_all = key.reshape(B, S, H, D)
        v_all = value.reshape(B, S, H, D)

    q_flat = query.reshape(B, -1, H, D)

    # SDPA expects [B, H, S, D]
    q_t = q_flat.permute(0, 2, 1, 3)
    k_t = k_all.permute(0, 2, 1, 3)
    v_t = v_all.permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
    # Back to [B, ..spatial.., H, D]
    out = out.permute(0, 2, 1, 3).reshape(*([B] + list(spatial_shape) + [H, D]))

    if return_lse:
        # Compute LSE for the full-attention path
        logits = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        lse = torch.logsumexp(logits, dim=-1)  # [B, H, S_q]
        S_q = q_flat.shape[1]
        lse = lse.permute(0, 2, 1).reshape(*([B] + list(spatial_shape) + [H]))
        return out, lse

    return out


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
    return_lse: bool = False,
    additional_keys: Optional[torch.Tensor] = None,
    additional_values: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """1D neighborhood attention.

    Each query attends to a local window of ``kernel_size`` neighbors along
    the sequence dimension.  Supports GQA/MQA when K/V have fewer heads
    than Q, and optional extra global tokens via ``additional_keys``/
    ``additional_values``.

    Args:
        query: ``[B, L, H_q, D]``.
        key: ``[B, L, H_kv, D]``.  ``H_q`` must be divisible by ``H_kv``.
        value: ``[B, L, H_kv, D]``.
        kernel_size: Neighborhood window size (scalar or 1-tuple).
        stride: Output stride for downsampling.  Default ``1``.
        dilation: Gap between attended positions.  Default ``1``.
        is_causal: Causal masking (attend only to past/current).
        scale: Logit scaling factor.  Default ``D ** -0.5``.
        return_lse: If ``True``, return ``(output, lse)`` where ``lse``
            has shape ``[B, L_out, H_q]``.
        additional_keys: ``[B, N_extra, H_kv, D]`` — extra tokens every
            query attends to (global attention).  Requires
            ``additional_values``.
        additional_values: ``[B, N_extra, H_kv, D]``.

    Returns:
        ``[B, L_out, H_q, D]``, or ``(output, lse)`` when
        ``return_lse=True``.

    Note:
        When ``kernel_size >= L`` the call is dispatched to
        ``F.scaled_dot_product_attention`` automatically.
    """
    kv_repeat = _validate_qkv(query, key, value, 1)
    add_k, add_v = _validate_additional_kv(query, additional_keys, additional_values, kv_repeat)

    ks = normalize_kernel_size(kernel_size, 1)
    st = normalize_tuple_param(stride, 1, "stride")
    dil = normalize_tuple_param(dilation, 1, "dilation")
    causal = _normalize_is_causal(is_causal, 1)

    spatial_shape = (query.shape[1],)
    check_kernel_size_vs_input(ks, spatial_shape)
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    # Expand KV heads for GQA
    k_expanded = _repeat_kv(key, kv_repeat)
    v_expanded = _repeat_kv(value, kv_repeat)

    # FMHA fast path: kernel covers all spatial dims → use SDPA
    if _is_full_attention(ks, spatial_shape) and all(s == 1 for s in st) and not any(causal):
        return _sdpa_forward(
            query, k_expanded, v_expanded, scale_value, spatial_shape,
            return_lse=return_lse, additional_keys=add_k, additional_values=add_v,
        )

    has_additional = add_k is not None

    if has_additional or return_lse:
        logits = na1d_qk(query, k_expanded, kernel_size=ks, dilation=dil, stride=st, is_causal=causal)
        logits_scaled = logits * scale_value
        lse = torch.logsumexp(logits_scaled, dim=-1)
        attn = torch.softmax(logits_scaled, dim=-1)
        out = na1d_av(attn, v_expanded, kernel_size=ks, dilation=dil, stride=st, is_causal=causal)

        if has_additional:
            from natten_mps.merge import merge_attentions
            out_extra, lse_extra = _full_attn_with_lse(query, add_k, add_v, scale_value)
            out, lse = merge_attentions([out, out_extra], [lse, lse_extra])

        if return_lse:
            return out, lse
        return out

    if _using_pure_backend():
        return ops.na1d_forward(query, k_expanded, v_expanded, ks, st, dil, causal, scale_value)
    return NeighborhoodAttention1DFunction.apply(query, k_expanded, v_expanded, ks, st, dil, causal, scale_value)


def na2d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool]] = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    additional_keys: Optional[torch.Tensor] = None,
    additional_values: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """2D neighborhood attention.

    Each query attends to a local ``kernel_size × kernel_size`` window on the
    spatial grid.  Supports GQA/MQA when K/V have fewer heads than Q, and
    optional extra global tokens via ``additional_keys``/``additional_values``.

    Args:
        query: ``[B, H, W, H_q, D]``.
        key: ``[B, H, W, H_kv, D]``.  ``H_q`` must be divisible by ``H_kv``.
        value: ``[B, H, W, H_kv, D]``.
        kernel_size: Neighborhood window size (scalar or ``(kH, kW)``).
        stride: Output stride for downsampling.  Default ``1``.
        dilation: Gap between attended positions.  Default ``1``.
        is_causal: Causal masking per axis, e.g. ``(False, True)``.
        scale: Logit scaling factor.  Default ``D ** -0.5``.
        return_lse: If ``True``, return ``(output, lse)`` where ``lse``
            has shape ``[B, H_out, W_out, H_q]``.
        additional_keys: ``[B, N_extra, H_kv, D]`` — extra tokens every
            query attends to (global attention).  Requires
            ``additional_values``.
        additional_values: ``[B, N_extra, H_kv, D]``.

    Returns:
        ``[B, H_out, W_out, H_q, D]``, or ``(output, lse)`` when
        ``return_lse=True``.

    Note:
        When ``kernel_size >= (H, W)`` the call is dispatched to
        ``F.scaled_dot_product_attention`` automatically.
    """
    kv_repeat = _validate_qkv(query, key, value, 2)
    add_k, add_v = _validate_additional_kv(query, additional_keys, additional_values, kv_repeat)

    ks = normalize_kernel_size(kernel_size, 2)
    st = normalize_tuple_param(stride, 2, "stride")
    dil = normalize_tuple_param(dilation, 2, "dilation")
    causal = _normalize_is_causal(is_causal, 2)

    spatial_shape = (query.shape[1], query.shape[2])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    k_expanded = _repeat_kv(key, kv_repeat)
    v_expanded = _repeat_kv(value, kv_repeat)

    # FMHA fast path: kernel covers all spatial dims → use SDPA
    if _is_full_attention(ks, spatial_shape) and all(s == 1 for s in st) and not any(causal):
        return _sdpa_forward(
            query, k_expanded, v_expanded, scale_value, spatial_shape,
            return_lse=return_lse, additional_keys=add_k, additional_values=add_v,
        )

    has_additional = add_k is not None

    if has_additional or return_lse:
        logits = na2d_qk(query, k_expanded, kernel_size=ks, dilation=dil, stride=st, is_causal=causal)
        logits_scaled = logits * scale_value
        lse = torch.logsumexp(logits_scaled, dim=-1)
        attn = torch.softmax(logits_scaled, dim=-1)
        out = na2d_av(attn, v_expanded, kernel_size=ks, dilation=dil, stride=st, is_causal=causal)

        if has_additional:
            from natten_mps.merge import merge_attentions
            out_extra, lse_extra = _full_attn_with_lse(query, add_k, add_v, scale_value)
            out, lse = merge_attentions([out, out_extra], [lse, lse_extra])

        if return_lse:
            return out, lse
        return out

    if _using_pure_backend():
        return ops.na2d_forward(query, k_expanded, v_expanded, ks, st, dil, causal, scale_value)
    return NeighborhoodAttention2DFunction.apply(query, k_expanded, v_expanded, ks, st, dil, causal, scale_value)


def na1d_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Union[int, Tuple[int]],
    dilation: Union[int, Tuple[int]] = 1,
    *,
    stride: Union[int, Tuple[int]] = 1,
    is_causal: Union[bool, Tuple[bool]] = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("na1d_qk expects query/key with shape [B, L, H, D].")
    if query.shape != key.shape:
        raise ValueError("query and key must share the same shape in na1d_qk.")

    ks = normalize_kernel_size(kernel_size, 1)
    dil = normalize_tuple_param(dilation, 1, "dilation")
    st = normalize_tuple_param(stride, 1, "stride")
    causal = _normalize_is_causal(is_causal, 1)
    check_kernel_size_vs_input(ks, (query.shape[1],))
    check_dilation_kernel_vs_input(dil, ks, (query.shape[1],))

    if _using_pure_backend():
        return ops.na1d_qk_forward(query, key, ks, dil, st, causal, scale)
    return NA1DQKFunction.apply(query, key, ks, dil, st, causal, scale)


def na1d_av(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int]],
    dilation: Union[int, Tuple[int]] = 1,
    *,
    stride: Union[int, Tuple[int]] = 1,
    is_causal: Union[bool, Tuple[bool]] = False,
) -> torch.Tensor:
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
    st = normalize_tuple_param(stride, 1, "stride")
    causal = _normalize_is_causal(is_causal, 1)
    if attn.shape[-1] != ks[0]:
        raise ValueError(
            f"na1d_av attn last dim ({attn.shape[-1]}) must match kernel_size ({ks[0]})."
        )
    check_kernel_size_vs_input(ks, (value.shape[1],))
    check_dilation_kernel_vs_input(dil, ks, (value.shape[1],))

    if _using_pure_backend():
        return ops.na1d_av_forward(attn, value, ks, dil, st, causal)
    return NA1DAVFunction.apply(attn, value, ks, dil, st, causal)


def na2d_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    *,
    stride: Union[int, Tuple[int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool]] = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if query.ndim != 5 or key.ndim != 5:
        raise ValueError("na2d_qk expects query/key with shape [B, H, W, heads, dim].")
    if query.shape != key.shape:
        raise ValueError("query and key must share the same shape in na2d_qk.")

    ks = normalize_kernel_size(kernel_size, 2)
    dil = normalize_tuple_param(dilation, 2, "dilation")
    st = normalize_tuple_param(stride, 2, "stride")
    causal = _normalize_is_causal(is_causal, 2)
    spatial_shape = (query.shape[1], query.shape[2])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    if _using_pure_backend():
        return ops.na2d_qk_forward(query, key, ks, dil, st, causal, scale)
    return NA2DQKFunction.apply(query, key, ks, dil, st, causal, scale)


def na2d_av(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    *,
    stride: Union[int, Tuple[int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool]] = False,
) -> torch.Tensor:
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
    st = normalize_tuple_param(stride, 2, "stride")
    causal = _normalize_is_causal(is_causal, 2)
    kernel_area = ks[0] * ks[1]
    if attn.shape[-1] != kernel_area:
        raise ValueError(
            f"na2d_av attn last dim ({attn.shape[-1]}) must match kernel area ({kernel_area})."
        )
    spatial_shape = (value.shape[1], value.shape[2])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    if _using_pure_backend():
        return ops.na2d_av_forward(attn, value, ks, dil, st, causal)
    return NA2DAVFunction.apply(attn, value, ks, dil, st, causal)


def na3d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool, bool]] = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    additional_keys: Optional[torch.Tensor] = None,
    additional_values: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """3D neighborhood attention.

    Each query attends to a local ``kernel_size³`` window in the volumetric
    spatial grid.  Supports GQA/MQA when K/V have fewer heads than Q, and
    optional extra global tokens via ``additional_keys``/``additional_values``.

    Args:
        query: ``[B, D1, D2, D3, H_q, D]``.
        key: ``[B, D1, D2, D3, H_kv, D]``.  ``H_q`` must be divisible by
            ``H_kv``.
        value: ``[B, D1, D2, D3, H_kv, D]``.
        kernel_size: Neighborhood window size (scalar or ``(k1, k2, k3)``).
        stride: Output stride for downsampling.  Default ``1``.
        dilation: Gap between attended positions.  Default ``1``.
        is_causal: Causal masking per axis.
        scale: Logit scaling factor.  Default ``D ** -0.5``.
        return_lse: If ``True``, return ``(output, lse)`` where ``lse``
            has shape ``[B, D1_out, D2_out, D3_out, H_q]``.
        additional_keys: ``[B, N_extra, H_kv, D]`` — extra tokens every
            query attends to (global attention).  Requires
            ``additional_values``.
        additional_values: ``[B, N_extra, H_kv, D]``.

    Returns:
        ``[B, D1_out, D2_out, D3_out, H_q, D]``, or ``(output, lse)``
        when ``return_lse=True``.

    Note:
        When ``kernel_size >= (D1, D2, D3)`` the call is dispatched to
        ``F.scaled_dot_product_attention`` automatically.
    """
    kv_repeat = _validate_qkv(query, key, value, 3)
    add_k, add_v = _validate_additional_kv(query, additional_keys, additional_values, kv_repeat)

    ks = normalize_kernel_size(kernel_size, 3)
    st = normalize_tuple_param(stride, 3, "stride")
    dil = normalize_tuple_param(dilation, 3, "dilation")
    causal = _normalize_is_causal(is_causal, 3)

    spatial_shape = (query.shape[1], query.shape[2], query.shape[3])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    k_expanded = _repeat_kv(key, kv_repeat)
    v_expanded = _repeat_kv(value, kv_repeat)

    # FMHA fast path: kernel covers all spatial dims → use SDPA
    if _is_full_attention(ks, spatial_shape) and all(s == 1 for s in st) and not any(causal):
        return _sdpa_forward(
            query, k_expanded, v_expanded, scale_value, spatial_shape,
            return_lse=return_lse, additional_keys=add_k, additional_values=add_v,
        )

    has_additional = add_k is not None

    if has_additional or return_lse:
        logits = na3d_qk(query, k_expanded, kernel_size=ks, dilation=dil, stride=st, is_causal=causal)
        logits_scaled = logits * scale_value
        lse = torch.logsumexp(logits_scaled, dim=-1)
        attn = torch.softmax(logits_scaled, dim=-1)
        out = na3d_av(attn, v_expanded, kernel_size=ks, dilation=dil, stride=st, is_causal=causal)

        if has_additional:
            from natten_mps.merge import merge_attentions
            out_extra, lse_extra = _full_attn_with_lse(query, add_k, add_v, scale_value)
            out, lse = merge_attentions([out, out_extra], [lse, lse_extra])

        if return_lse:
            return out, lse
        return out

    if _using_pure_backend():
        return ops.na3d_forward(query, k_expanded, v_expanded, ks, st, dil, causal, scale_value)
    return NeighborhoodAttention3DFunction.apply(query, k_expanded, v_expanded, ks, st, dil, causal, scale_value)


def na3d_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    dilation: Union[int, Tuple[int, int, int]] = 1,
    *,
    stride: Union[int, Tuple[int, int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool, bool]] = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if query.ndim != 6 or key.ndim != 6:
        raise ValueError("na3d_qk expects query/key with shape [B, D, H, W, heads, dim].")
    if query.shape != key.shape:
        raise ValueError("query and key must share the same shape in na3d_qk.")

    ks = normalize_kernel_size(kernel_size, 3)
    dil = normalize_tuple_param(dilation, 3, "dilation")
    st = normalize_tuple_param(stride, 3, "stride")
    causal = _normalize_is_causal(is_causal, 3)
    spatial_shape = (query.shape[1], query.shape[2], query.shape[3])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    if _using_pure_backend():
        return ops.na3d_qk_forward(query, key, ks, dil, st, causal, scale)
    return NA3DQKFunction.apply(query, key, ks, dil, st, causal, scale)


def na3d_av(
    attn: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    dilation: Union[int, Tuple[int, int, int]] = 1,
    *,
    stride: Union[int, Tuple[int, int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool, bool]] = False,
) -> torch.Tensor:
    if attn.ndim != 6:
        raise ValueError("na3d_av expects attn with shape [B, D, H, W, heads, K].")
    if value.ndim != 6:
        raise ValueError("na3d_av expects value with shape [B, D, H, W, heads, D].")
    if attn.shape[0] != value.shape[0] or attn.shape[4] != value.shape[4]:
        raise ValueError("na3d_av requires attn/value to match on batch and heads dimensions.")
    if attn.shape[1] > value.shape[1] or attn.shape[2] > value.shape[2] or attn.shape[3] > value.shape[3]:
        raise ValueError("na3d_av attn spatial size cannot exceed value spatial size.")

    ks = normalize_kernel_size(kernel_size, 3)
    dil = normalize_tuple_param(dilation, 3, "dilation")
    st = normalize_tuple_param(stride, 3, "stride")
    causal = _normalize_is_causal(is_causal, 3)
    kernel_volume = ks[0] * ks[1] * ks[2]
    if attn.shape[-1] != kernel_volume:
        raise ValueError(
            f"na3d_av attn last dim ({attn.shape[-1]}) must match kernel volume ({kernel_volume})."
        )
    spatial_shape = (value.shape[1], value.shape[2], value.shape[3])
    check_kernel_size_vs_input(ks, spatial_shape)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    if _using_pure_backend():
        return ops.na3d_av_forward(attn, value, ks, dil, st, causal)
    return NA3DAVFunction.apply(attn, value, ks, dil, st, causal)


__all__ = ["na1d", "na2d", "na3d", "na1d_qk", "na1d_av", "na2d_qk", "na2d_av", "na3d_qk", "na3d_av"]
