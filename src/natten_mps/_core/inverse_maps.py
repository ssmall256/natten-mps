"""Precomputed inverse index maps for Metal backward kernels.

For k_backward and v_backward, each thread needs to know which output positions
reference a given key/value position. Brute-force iteration is O(L_out * K) per
thread. Inverse maps precompute this as a CSR (Compressed Sparse Row) structure
so each thread only touches its actual referencing positions.

The maps are computed on CPU via numpy and cached by configuration parameters.
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Caches (keyed by spatial config tuple)
# ---------------------------------------------------------------------------

_INV_MAP_1D_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_INV_MAP_1D_QK_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_INV_MAP_2D_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_INV_MAP_2D_QK_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_INV_MAP_3D_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_INV_MAP_3D_QK_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


# ---------------------------------------------------------------------------
# Window helpers (numpy, matching Metal get_window_start / get_causal_window_start)
# ---------------------------------------------------------------------------

def _get_window_start(index: int, length: int, kernel_size: int, dilation: int) -> int:
    """Non-causal window start for a single position (matches Metal get_window_start exactly)."""
    neighborhood_size = kernel_size // 2
    if dilation <= 1:
        start = max(index - neighborhood_size, 0)
        if index + neighborhood_size >= length:
            start += (length - index - neighborhood_size - 1)
        return start
    ni = index - neighborhood_size * dilation
    if ni < 0:
        return index % dilation
    if index + neighborhood_size * dilation >= length:
        imodd = index % dilation
        a = (length // dilation) * dilation
        b = length - a
        if imodd < b:
            return length - b + imodd - 2 * neighborhood_size * dilation
        return a + imodd - kernel_size * dilation
    return ni


def _compute_axis_indices(
    query_positions: np.ndarray,
    spatial_size: int,
    kernel_size: int,
    dilation: int,
    is_causal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute neighbor indices and validity mask for an axis.

    Returns:
        idx: [num_queries, kernel_size] - neighbor positions (clipped)
        valid: [num_queries, kernel_size] - boolean validity mask
    """
    qpos = np.asarray(query_positions, dtype=np.int32)
    k_steps = np.arange(kernel_size, dtype=np.int32)

    if is_causal:
        start = qpos - (kernel_size - 1) * dilation
        raw = start[:, None] + k_steps[None, :] * dilation
        valid = (raw >= 0) & (raw <= qpos[:, None]) & (raw < spatial_size)
        clipped = np.clip(raw, 0, spatial_size - 1).astype(np.int32)
        return clipped, valid.astype(np.bool_)

    # Non-causal: vectorized window start computation
    starts = np.array(
        [_get_window_start(int(p), spatial_size, kernel_size, dilation) for p in qpos],
        dtype=np.int32,
    )
    raw = starts[:, None] + k_steps[None, :] * dilation
    valid = (raw >= 0) & (raw < spatial_size)
    clipped = np.clip(raw, 0, spatial_size - 1).astype(np.int32)
    return clipped, valid.astype(np.bool_)


# ---------------------------------------------------------------------------
# CSR builder
# ---------------------------------------------------------------------------

def _build_inverse_csr(
    *,
    value_indices: np.ndarray,
    out_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    num_values: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR-format inverse map from forward-direction indices.

    Args:
        value_indices: which input position each edge references
        out_indices: which output position each edge comes from
        neighbor_indices: which neighbor index within the kernel window
        num_values: total number of input positions

    Returns:
        offsets: [num_values + 1] - CSR row offsets
        out_ids: sorted output position indices
        nbr_ids: sorted neighbor indices
    """
    if value_indices.size == 0:
        offsets = np.zeros((num_values + 1,), dtype=np.int32)
        return offsets, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    order = np.argsort(value_indices, kind="stable")
    vals = value_indices[order].astype(np.int32, copy=False)
    out_ids = out_indices[order].astype(np.int32, copy=False)
    nbr_ids = neighbor_indices[order].astype(np.int32, copy=False)

    counts = np.bincount(vals, minlength=num_values).astype(np.int32, copy=False)
    offsets = np.zeros((num_values + 1,), dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32, copy=False)
    return offsets, out_ids, nbr_ids


def _to_mps_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to int32 MPS tensor."""
    return torch.from_numpy(arr.astype(np.int32, copy=False)).to("mps")


# ---------------------------------------------------------------------------
# 1D inverse maps
# ---------------------------------------------------------------------------

def inverse_map_1d(
    length: int, out_len: int, kernel_size: int,
    stride: int, dilation: int, causal: bool, dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse map for 1D AV backward (v_backward).

    Returns (offsets, attn_base, grad_base) as MPS int32 tensors.
    """
    key = (length, out_len, kernel_size, stride, dilation, causal, dim)
    cached = _INV_MAP_1D_CACHE.get(key)
    if cached is not None:
        return cached

    qpos = (np.arange(out_len, dtype=np.int32) * stride).astype(np.int32)
    idx, valid = _compute_axis_indices(qpos, length, kernel_size, dilation, causal)
    flat_valid = valid.reshape(-1)
    value_indices = idx.reshape(-1)[flat_valid].astype(np.int32, copy=False)
    out_indices = np.repeat(np.arange(out_len, dtype=np.int32), kernel_size)[flat_valid]
    neighbor_indices = np.tile(np.arange(kernel_size, dtype=np.int32), out_len)[flat_valid]

    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=value_indices,
        out_indices=out_indices,
        neighbor_indices=neighbor_indices,
        num_values=length,
    )
    attn_base = (out_ids * kernel_size + nbr_ids).astype(np.int32, copy=False)
    grad_base = (out_ids * dim).astype(np.int32, copy=False)
    result = (_to_mps_tensor(offsets), _to_mps_tensor(attn_base), _to_mps_tensor(grad_base))
    _INV_MAP_1D_CACHE[key] = result
    return result


def inverse_map_1d_qk(
    length: int, out_len: int, kernel_size: int,
    stride: int, dilation: int, causal: bool, dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse map for 1D QK backward (k_backward).

    Returns (offsets, attn_base, query_base) as MPS int32 tensors.
    """
    key = (length, out_len, kernel_size, stride, dilation, causal, dim)
    cached = _INV_MAP_1D_QK_CACHE.get(key)
    if cached is not None:
        return cached

    qpos = (np.arange(out_len, dtype=np.int32) * stride).astype(np.int32)
    idx, valid = _compute_axis_indices(qpos, length, kernel_size, dilation, causal)
    flat_valid = valid.reshape(-1)
    value_indices = idx.reshape(-1)[flat_valid].astype(np.int32, copy=False)
    out_indices = np.repeat(np.arange(out_len, dtype=np.int32), kernel_size)[flat_valid]
    neighbor_indices = np.tile(np.arange(kernel_size, dtype=np.int32), out_len)[flat_valid]

    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=value_indices,
        out_indices=out_indices,
        neighbor_indices=neighbor_indices,
        num_values=length,
    )
    attn_base = (out_ids * kernel_size + nbr_ids).astype(np.int32, copy=False)
    query_base = (out_ids * stride * dim).astype(np.int32, copy=False)
    result = (_to_mps_tensor(offsets), _to_mps_tensor(attn_base), _to_mps_tensor(query_base))
    _INV_MAP_1D_QK_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# 2D inverse maps
# ---------------------------------------------------------------------------

def inverse_map_2d(
    height: int, width: int, out_h: int, out_w: int,
    kernel_h: int, kernel_w: int,
    stride_h: int, stride_w: int,
    dilation_h: int, dilation_w: int,
    causal_h: bool, causal_w: bool, dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse map for 2D AV backward (v_backward)."""
    key = (height, width, out_h, out_w, kernel_h, kernel_w,
           stride_h, stride_w, dilation_h, dilation_w, causal_h, causal_w, dim)
    cached = _INV_MAP_2D_CACHE.get(key)
    if cached is not None:
        return cached

    qh = (np.arange(out_h, dtype=np.int32) * stride_h).astype(np.int32)
    qw = (np.arange(out_w, dtype=np.int32) * stride_w).astype(np.int32)
    h_idx, h_valid = _compute_axis_indices(qh, height, kernel_h, dilation_h, causal_h)
    w_idx, w_valid = _compute_axis_indices(qw, width, kernel_w, dilation_w, causal_w)

    k_area = kernel_h * kernel_w
    # Outer product of h and w indices -> linearized 2D positions
    lin = (
        h_idx[:, None, :, None].astype(np.int32) * width
        + w_idx[None, :, None, :].astype(np.int32)
    ).reshape(out_h, out_w, k_area)
    valid = (h_valid[:, None, :, None] & w_valid[None, :, None, :]).reshape(out_h, out_w, k_area)

    out_flat = np.arange(out_h * out_w, dtype=np.int32).reshape(out_h, out_w, 1)
    out_indices = np.broadcast_to(out_flat, (out_h, out_w, k_area)).reshape(-1)
    neighbor_indices = np.broadcast_to(
        np.arange(k_area, dtype=np.int32), (out_h, out_w, k_area)
    ).reshape(-1)
    mask = valid.reshape(-1)

    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=lin.reshape(-1)[mask].astype(np.int32, copy=False),
        out_indices=out_indices[mask],
        neighbor_indices=neighbor_indices[mask],
        num_values=height * width,
    )
    attn_base = (out_ids * k_area + nbr_ids).astype(np.int32, copy=False)
    grad_base = (out_ids * dim).astype(np.int32, copy=False)
    result = (_to_mps_tensor(offsets), _to_mps_tensor(attn_base), _to_mps_tensor(grad_base))
    _INV_MAP_2D_CACHE[key] = result
    return result


def inverse_map_2d_qk(
    height: int, width: int, out_h: int, out_w: int,
    kernel_h: int, kernel_w: int,
    stride_h: int, stride_w: int,
    dilation_h: int, dilation_w: int,
    causal_h: bool, causal_w: bool, dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse map for 2D QK backward (k_backward)."""
    key = (height, width, out_h, out_w, kernel_h, kernel_w,
           stride_h, stride_w, dilation_h, dilation_w, causal_h, causal_w, dim)
    cached = _INV_MAP_2D_QK_CACHE.get(key)
    if cached is not None:
        return cached

    qh = (np.arange(out_h, dtype=np.int32) * stride_h).astype(np.int32)
    qw = (np.arange(out_w, dtype=np.int32) * stride_w).astype(np.int32)
    h_idx, h_valid = _compute_axis_indices(qh, height, kernel_h, dilation_h, causal_h)
    w_idx, w_valid = _compute_axis_indices(qw, width, kernel_w, dilation_w, causal_w)

    k_area = kernel_h * kernel_w
    lin = (
        h_idx[:, None, :, None].astype(np.int32) * width
        + w_idx[None, :, None, :].astype(np.int32)
    ).reshape(out_h, out_w, k_area)
    valid = (h_valid[:, None, :, None] & w_valid[None, :, None, :]).reshape(out_h, out_w, k_area)

    out_flat = np.arange(out_h * out_w, dtype=np.int32).reshape(out_h, out_w, 1)
    out_indices = np.broadcast_to(out_flat, (out_h, out_w, k_area)).reshape(-1)
    neighbor_indices = np.broadcast_to(
        np.arange(k_area, dtype=np.int32), (out_h, out_w, k_area)
    ).reshape(-1)
    mask = valid.reshape(-1)

    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=lin.reshape(-1)[mask].astype(np.int32, copy=False),
        out_indices=out_indices[mask],
        neighbor_indices=neighbor_indices[mask],
        num_values=height * width,
    )
    attn_base = (out_ids * k_area + nbr_ids).astype(np.int32, copy=False)
    # query_base: linearized output position * dim -> base index into query tensor
    out_h_ids = out_ids // out_w
    out_w_ids = out_ids % out_w
    query_base = (
        (out_h_ids * stride_h * width + out_w_ids * stride_w) * dim
    ).astype(np.int32, copy=False)
    result = (_to_mps_tensor(offsets), _to_mps_tensor(attn_base), _to_mps_tensor(query_base))
    _INV_MAP_2D_QK_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# 3D inverse maps
# ---------------------------------------------------------------------------

def inverse_map_3d(
    depth: int, height: int, width: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    dilation_d: int, dilation_h: int, dilation_w: int,
    causal_d: bool, causal_h: bool, causal_w: bool, dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse map for 3D AV backward (v_backward)."""
    key = (depth, height, width, out_d, out_h, out_w,
           kernel_d, kernel_h, kernel_w,
           stride_d, stride_h, stride_w,
           dilation_d, dilation_h, dilation_w,
           causal_d, causal_h, causal_w, dim)
    cached = _INV_MAP_3D_CACHE.get(key)
    if cached is not None:
        return cached

    qd = (np.arange(out_d, dtype=np.int32) * stride_d).astype(np.int32)
    qh = (np.arange(out_h, dtype=np.int32) * stride_h).astype(np.int32)
    qw = (np.arange(out_w, dtype=np.int32) * stride_w).astype(np.int32)
    d_idx, d_valid = _compute_axis_indices(qd, depth, kernel_d, dilation_d, causal_d)
    h_idx, h_valid = _compute_axis_indices(qh, height, kernel_h, dilation_h, causal_h)
    w_idx, w_valid = _compute_axis_indices(qw, width, kernel_w, dilation_w, causal_w)

    k_vol = kernel_d * kernel_h * kernel_w
    # 3D outer product -> linearized positions
    lin = (
        d_idx[:, None, None, :, None, None].astype(np.int32) * height * width
        + h_idx[None, :, None, None, :, None].astype(np.int32) * width
        + w_idx[None, None, :, None, None, :].astype(np.int32)
    ).reshape(out_d, out_h, out_w, k_vol)
    valid = (
        d_valid[:, None, None, :, None, None]
        & h_valid[None, :, None, None, :, None]
        & w_valid[None, None, :, None, None, :]
    ).reshape(out_d, out_h, out_w, k_vol)

    out_flat = np.arange(out_d * out_h * out_w, dtype=np.int32).reshape(out_d, out_h, out_w, 1)
    out_indices = np.broadcast_to(out_flat, (out_d, out_h, out_w, k_vol)).reshape(-1)
    neighbor_indices = np.broadcast_to(
        np.arange(k_vol, dtype=np.int32), (out_d, out_h, out_w, k_vol)
    ).reshape(-1)
    mask = valid.reshape(-1)

    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=lin.reshape(-1)[mask].astype(np.int32, copy=False),
        out_indices=out_indices[mask],
        neighbor_indices=neighbor_indices[mask],
        num_values=depth * height * width,
    )
    attn_base = (out_ids * k_vol + nbr_ids).astype(np.int32, copy=False)
    grad_base = (out_ids * dim).astype(np.int32, copy=False)
    result = (_to_mps_tensor(offsets), _to_mps_tensor(attn_base), _to_mps_tensor(grad_base))
    _INV_MAP_3D_CACHE[key] = result
    return result


def inverse_map_3d_qk(
    depth: int, height: int, width: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    dilation_d: int, dilation_h: int, dilation_w: int,
    causal_d: bool, causal_h: bool, causal_w: bool, dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse map for 3D QK backward (k_backward)."""
    key = (depth, height, width, out_d, out_h, out_w,
           kernel_d, kernel_h, kernel_w,
           stride_d, stride_h, stride_w,
           dilation_d, dilation_h, dilation_w,
           causal_d, causal_h, causal_w, dim)
    cached = _INV_MAP_3D_QK_CACHE.get(key)
    if cached is not None:
        return cached

    qd = (np.arange(out_d, dtype=np.int32) * stride_d).astype(np.int32)
    qh = (np.arange(out_h, dtype=np.int32) * stride_h).astype(np.int32)
    qw = (np.arange(out_w, dtype=np.int32) * stride_w).astype(np.int32)
    d_idx, d_valid = _compute_axis_indices(qd, depth, kernel_d, dilation_d, causal_d)
    h_idx, h_valid = _compute_axis_indices(qh, height, kernel_h, dilation_h, causal_h)
    w_idx, w_valid = _compute_axis_indices(qw, width, kernel_w, dilation_w, causal_w)

    k_vol = kernel_d * kernel_h * kernel_w
    lin = (
        d_idx[:, None, None, :, None, None].astype(np.int32) * height * width
        + h_idx[None, :, None, None, :, None].astype(np.int32) * width
        + w_idx[None, None, :, None, None, :].astype(np.int32)
    ).reshape(out_d, out_h, out_w, k_vol)
    valid = (
        d_valid[:, None, None, :, None, None]
        & h_valid[None, :, None, None, :, None]
        & w_valid[None, None, :, None, None, :]
    ).reshape(out_d, out_h, out_w, k_vol)

    out_flat = np.arange(out_d * out_h * out_w, dtype=np.int32).reshape(out_d, out_h, out_w, 1)
    out_indices = np.broadcast_to(out_flat, (out_d, out_h, out_w, k_vol)).reshape(-1)
    neighbor_indices = np.broadcast_to(
        np.arange(k_vol, dtype=np.int32), (out_d, out_h, out_w, k_vol)
    ).reshape(-1)
    mask = valid.reshape(-1)

    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=lin.reshape(-1)[mask].astype(np.int32, copy=False),
        out_indices=out_indices[mask],
        neighbor_indices=neighbor_indices[mask],
        num_values=depth * height * width,
    )
    attn_base = (out_ids * k_vol + nbr_ids).astype(np.int32, copy=False)
    # query_base: linearized output -> input position * dim
    out_d_ids = out_ids // (out_h * out_w)
    rem = out_ids % (out_h * out_w)
    out_h_ids = rem // out_w
    out_w_ids = rem % out_w
    query_base = (
        (out_d_ids * stride_d * height * width
         + out_h_ids * stride_h * width
         + out_w_ids * stride_w) * dim
    ).astype(np.int32, copy=False)
    result = (_to_mps_tensor(offsets), _to_mps_tensor(attn_base), _to_mps_tensor(query_base))
    _INV_MAP_3D_QK_CACHE[key] = result
    return result
