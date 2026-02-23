from __future__ import annotations

from typing import Optional, Tuple

import torch

from natten_mps.utils.window import get_window_start_vectorized


def is_available() -> bool:
    return True


def _default_scale(t: torch.Tensor) -> float:
    return t.shape[-1] ** -0.5


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _output_positions(length: int, stride: int, device: torch.device) -> torch.Tensor:
    out_len = _ceil_div(length, stride)
    return torch.arange(out_len, device=device, dtype=torch.long) * stride


def _noncausal_1d_indices(
    query_positions: torch.Tensor,
    length: int,
    kernel_size: int,
    dilation: int,
    device: torch.device,
) -> torch.Tensor:
    start = get_window_start_vectorized(
        query_positions,
        length=length,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    offsets = torch.arange(kernel_size, device=device, dtype=torch.long) * dilation
    return start.unsqueeze(-1) + offsets.unsqueeze(0)


def _causal_1d_indices(
    query_positions: torch.Tensor,
    length: int,
    kernel_size: int,
    dilation: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(kernel_size, device=device, dtype=torch.long) * dilation
    start = query_positions - (kernel_size - 1) * dilation
    raw = start.unsqueeze(-1) + offsets.unsqueeze(0)
    valid = (raw >= 0) & (raw <= query_positions.unsqueeze(-1))
    clamped = raw.clamp(0, length - 1)
    return clamped, valid


def _noncausal_2d_dim_indices(
    query_positions: torch.Tensor,
    length: int,
    kernel_size: int,
    dilation: int,
    device: torch.device,
) -> torch.Tensor:
    start = get_window_start_vectorized(
        query_positions,
        length=length,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    offsets = torch.arange(kernel_size, device=device, dtype=torch.long) * dilation
    return start.unsqueeze(-1) + offsets.unsqueeze(0)


def _causal_2d_dim_indices(
    query_positions: torch.Tensor,
    length: int,
    kernel_size: int,
    dilation: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(kernel_size, device=device, dtype=torch.long) * dilation
    start = query_positions - (kernel_size - 1) * dilation
    raw = start.unsqueeze(-1) + offsets.unsqueeze(0)
    valid = (raw >= 0) & (raw <= query_positions.unsqueeze(-1))
    clamped = raw.clamp(0, length - 1)
    return clamped, valid


def _build_2d_neighborhood(
    query_h_positions: torch.Tensor,
    query_w_positions: torch.Tensor,
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    is_causal: Tuple[bool, bool],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    kh, kw = kernel_size
    dh, dw = dilation
    causal_h, causal_w = is_causal

    if causal_h:
        idx_h, valid_h = _causal_2d_dim_indices(query_h_positions, height, kh, dh, device)
    else:
        idx_h = _noncausal_2d_dim_indices(query_h_positions, height, kh, dh, device)
        valid_h = torch.ones_like(idx_h, dtype=torch.bool)

    if causal_w:
        idx_w, valid_w = _causal_2d_dim_indices(query_w_positions, width, kw, dw, device)
    else:
        idx_w = _noncausal_2d_dim_indices(query_w_positions, width, kw, dw, device)
        valid_w = torch.ones_like(idx_w, dtype=torch.bool)

    oh = query_h_positions.numel()
    ow = query_w_positions.numel()

    idx_h_grid = idx_h[:, None, :, None].expand(oh, ow, kh, kw)
    idx_w_grid = idx_w[None, :, None, :].expand(oh, ow, kh, kw)
    flat_idx = (idx_h_grid * width + idx_w_grid).reshape(oh, ow, kh * kw)

    if causal_h or causal_w:
        valid = (valid_h[:, None, :, None] & valid_w[None, :, None, :]).reshape(oh, ow, kh * kw)
    else:
        valid = None

    return flat_idx, valid


def _validate_1d_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("1D inputs must have shape [B, L, H, D].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("query, key, and value must have identical shapes for 1D neighborhood attention.")


def _validate_2d_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 5 or k.ndim != 5 or v.ndim != 5:
        raise ValueError("2D inputs must have shape [B, H, W, heads, dim].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("query, key, and value must have identical shapes for 2D neighborhood attention.")


def na1d_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    stride: Tuple[int],
    dilation: Tuple[int],
    is_causal: Tuple[bool],
    scale: Optional[float],
) -> torch.Tensor:
    _validate_1d_qkv(q, k, v)
    scale_value = float(_default_scale(q) if scale is None else scale)

    length = q.shape[1]
    query_positions = _output_positions(length, stride[0], q.device)
    q_selected = q.index_select(1, query_positions)

    if is_causal[0]:
        key_idx, valid = _causal_1d_indices(query_positions, length, kernel_size[0], dilation[0], q.device)
    else:
        key_idx = _noncausal_1d_indices(query_positions, length, kernel_size[0], dilation[0], q.device)
        valid = None

    k_neighborhood = k[:, key_idx]
    logits = torch.einsum("blhd,blkhd->blhk", q_selected, k_neighborhood) * scale_value

    if valid is not None:
        logits = logits.masked_fill(~valid.view(1, -1, 1, kernel_size[0]), float("-inf"))

    attn = torch.softmax(logits, dim=-1)
    v_neighborhood = v[:, key_idx]
    return torch.einsum("blhk,blkhd->blhd", attn, v_neighborhood)


def na2d_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    is_causal: Tuple[bool, bool],
    scale: Optional[float],
) -> torch.Tensor:
    _validate_2d_qkv(q, k, v)
    scale_value = float(_default_scale(q) if scale is None else scale)

    _, height, width, _, _ = q.shape
    qh = _output_positions(height, stride[0], q.device)
    qw = _output_positions(width, stride[1], q.device)
    q_selected = q.index_select(1, qh).index_select(2, qw)

    flat_idx, valid = _build_2d_neighborhood(
        qh,
        qw,
        height,
        width,
        kernel_size,
        dilation,
        is_causal,
        q.device,
    )

    k_flat = k.reshape(k.shape[0], height * width, k.shape[3], k.shape[4])
    v_flat = v.reshape(v.shape[0], height * width, v.shape[3], v.shape[4])

    k_neighborhood = k_flat[:, flat_idx]
    logits = torch.einsum("bijhd,bijkhd->bijhk", q_selected, k_neighborhood) * scale_value

    if valid is not None:
        logits = logits.masked_fill(~valid.view(1, valid.shape[0], valid.shape[1], 1, valid.shape[2]), float("-inf"))

    attn = torch.softmax(logits, dim=-1)
    v_neighborhood = v_flat[:, flat_idx]
    return torch.einsum("bijhk,bijkhd->bijhd", attn, v_neighborhood)


def na1d_qk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    stride: Tuple[int] = (1,),
    is_causal: Tuple[bool] = (False,),
    scale: Optional[float] = None,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("na1d_qk expects query/key shaped [B, L, H, D].")
    if q.shape != k.shape:
        raise ValueError("query and key must have identical shape for na1d_qk.")

    length = q.shape[1]
    query_positions = _output_positions(length, stride[0], q.device)
    q_selected = q.index_select(1, query_positions) if stride[0] != 1 else q

    if is_causal[0]:
        key_idx, valid = _causal_1d_indices(query_positions, length, kernel_size[0], dilation[0], q.device)
    else:
        key_idx = _noncausal_1d_indices(query_positions, length, kernel_size[0], dilation[0], q.device)
        valid = None

    k_neighborhood = k[:, key_idx]
    logits = torch.einsum("blhd,blkhd->blhk", q_selected, k_neighborhood)

    if valid is not None:
        logits = logits.masked_fill(~valid.view(1, -1, 1, kernel_size[0]), float("-inf"))
    if scale is not None:
        logits = logits * scale
    return logits


def na1d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    stride: Tuple[int] = (1,),
    is_causal: Tuple[bool] = (False,),
) -> torch.Tensor:
    if attn.ndim != 4:
        raise ValueError("attn must have shape [B, L, H, K] for na1d_av.")
    if v.ndim != 4:
        raise ValueError("value must have shape [B, L, H, D] for na1d_av.")

    b1, l_out, h1, k_attn = attn.shape
    b2, length, h2, _ = v.shape
    if b1 != b2 or h1 != h2:
        raise ValueError("Batch and head dimensions of attn/value must match in na1d_av.")
    if k_attn != kernel_size[0]:
        raise ValueError(
            f"attn last dim ({k_attn}) must match kernel_size ({kernel_size[0]})."
        )
    if l_out > length:
        raise ValueError("attn sequence length cannot exceed value sequence length.")

    query_positions = _output_positions(length, stride[0], v.device) if stride[0] != 1 else torch.arange(l_out, device=v.device, dtype=torch.long)

    if is_causal[0]:
        key_idx, valid = _causal_1d_indices(query_positions, length, kernel_size[0], dilation[0], v.device)
        v_neighborhood = v[:, key_idx]
        attn_masked = attn.masked_fill(~valid.view(1, -1, 1, kernel_size[0]), 0.0)
        return torch.einsum("blhk,blkhd->blhd", attn_masked, v_neighborhood)
    else:
        key_idx = _noncausal_1d_indices(query_positions, length, kernel_size[0], dilation[0], v.device)
        v_neighborhood = v[:, key_idx]
        return torch.einsum("blhk,blkhd->blhd", attn, v_neighborhood)


def na2d_qk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    is_causal: Tuple[bool, bool] = (False, False),
    scale: Optional[float] = None,
) -> torch.Tensor:
    if q.ndim != 5 or k.ndim != 5:
        raise ValueError("na2d_qk expects query/key shaped [B, H, W, heads, dim].")
    if q.shape != k.shape:
        raise ValueError("query and key must have identical shape for na2d_qk.")

    _, height, width, _, _ = q.shape
    qh = _output_positions(height, stride[0], q.device) if stride[0] != 1 else torch.arange(height, device=q.device, dtype=torch.long)
    qw = _output_positions(width, stride[1], q.device) if stride[1] != 1 else torch.arange(width, device=q.device, dtype=torch.long)
    q_selected = q.index_select(1, qh).index_select(2, qw) if stride[0] != 1 or stride[1] != 1 else q

    flat_idx, valid = _build_2d_neighborhood(
        qh,
        qw,
        height,
        width,
        kernel_size,
        dilation,
        is_causal,
        q.device,
    )

    k_flat = k.reshape(k.shape[0], height * width, k.shape[3], k.shape[4])
    k_neighborhood = k_flat[:, flat_idx]
    logits = torch.einsum("bijhd,bijkhd->bijhk", q_selected, k_neighborhood)

    if valid is not None:
        logits = logits.masked_fill(~valid.view(1, valid.shape[0], valid.shape[1], 1, valid.shape[2]), float("-inf"))
    if scale is not None:
        logits = logits * scale
    return logits


def _build_3d_neighborhood(
    query_d_positions: torch.Tensor,
    query_h_positions: torch.Tensor,
    query_w_positions: torch.Tensor,
    depth: int,
    height: int,
    width: int,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    kd, kh, kw = kernel_size
    dd, dh, dw = dilation
    causal_d, causal_h, causal_w = is_causal

    if causal_d:
        idx_d, valid_d = _causal_2d_dim_indices(query_d_positions, depth, kd, dd, device)
    else:
        idx_d = _noncausal_2d_dim_indices(query_d_positions, depth, kd, dd, device)
        valid_d = torch.ones_like(idx_d, dtype=torch.bool)

    if causal_h:
        idx_h, valid_h = _causal_2d_dim_indices(query_h_positions, height, kh, dh, device)
    else:
        idx_h = _noncausal_2d_dim_indices(query_h_positions, height, kh, dh, device)
        valid_h = torch.ones_like(idx_h, dtype=torch.bool)

    if causal_w:
        idx_w, valid_w = _causal_2d_dim_indices(query_w_positions, width, kw, dw, device)
    else:
        idx_w = _noncausal_2d_dim_indices(query_w_positions, width, kw, dw, device)
        valid_w = torch.ones_like(idx_w, dtype=torch.bool)

    od = query_d_positions.numel()
    oh = query_h_positions.numel()
    ow = query_w_positions.numel()

    idx_d_grid = idx_d[:, None, None, :, None, None].expand(od, oh, ow, kd, kh, kw)
    idx_h_grid = idx_h[None, :, None, None, :, None].expand(od, oh, ow, kd, kh, kw)
    idx_w_grid = idx_w[None, None, :, None, None, :].expand(od, oh, ow, kd, kh, kw)
    flat_idx = ((idx_d_grid * height + idx_h_grid) * width + idx_w_grid).reshape(od, oh, ow, kd * kh * kw)

    if causal_d or causal_h or causal_w:
        valid = (
            valid_d[:, None, None, :, None, None]
            & valid_h[None, :, None, None, :, None]
            & valid_w[None, None, :, None, None, :]
        ).reshape(od, oh, ow, kd * kh * kw)
    else:
        valid = None

    return flat_idx, valid


def _validate_3d_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 6 or k.ndim != 6 or v.ndim != 6:
        raise ValueError("3D inputs must have shape [B, D, H, W, heads, dim].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("query, key, and value must have identical shapes for 3D neighborhood attention.")


def na3d_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    scale: Optional[float],
) -> torch.Tensor:
    _validate_3d_qkv(q, k, v)
    scale_value = float(_default_scale(q) if scale is None else scale)

    _, depth, height, width, _, _ = q.shape
    qd = _output_positions(depth, stride[0], q.device)
    qh = _output_positions(height, stride[1], q.device)
    qw = _output_positions(width, stride[2], q.device)
    q_selected = q.index_select(1, qd).index_select(2, qh).index_select(3, qw)

    flat_idx, valid = _build_3d_neighborhood(
        qd, qh, qw, depth, height, width, kernel_size, dilation, is_causal, q.device,
    )

    kv_flat = k.reshape(k.shape[0], depth * height * width, k.shape[4], k.shape[5])
    k_neighborhood = kv_flat[:, flat_idx]
    logits = torch.einsum("bdijhf,bdijkhf->bdijhk", q_selected, k_neighborhood) * scale_value

    if valid is not None:
        logits = logits.masked_fill(
            ~valid.view(1, valid.shape[0], valid.shape[1], valid.shape[2], 1, valid.shape[3]),
            float("-inf"),
        )

    attn = torch.softmax(logits, dim=-1)
    v_flat = v.reshape(v.shape[0], depth * height * width, v.shape[4], v.shape[5])
    v_neighborhood = v_flat[:, flat_idx]
    return torch.einsum("bdijhk,bdijkhf->bdijhf", attn, v_neighborhood)


def na3d_qk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    stride: Tuple[int, int, int] = (1, 1, 1),
    is_causal: Tuple[bool, bool, bool] = (False, False, False),
    scale: Optional[float] = None,
) -> torch.Tensor:
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("na3d_qk expects query/key shaped [B, D, H, W, heads, dim].")
    if q.shape != k.shape:
        raise ValueError("query and key must have identical shape for na3d_qk.")

    _, depth, height, width, _, _ = q.shape
    qd = _output_positions(depth, stride[0], q.device) if stride[0] != 1 else torch.arange(depth, device=q.device, dtype=torch.long)
    qh = _output_positions(height, stride[1], q.device) if stride[1] != 1 else torch.arange(height, device=q.device, dtype=torch.long)
    qw = _output_positions(width, stride[2], q.device) if stride[2] != 1 else torch.arange(width, device=q.device, dtype=torch.long)
    q_selected = q.index_select(1, qd).index_select(2, qh).index_select(3, qw) if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 else q

    flat_idx, valid = _build_3d_neighborhood(
        qd, qh, qw, depth, height, width, kernel_size, dilation, is_causal, q.device,
    )

    k_flat = k.reshape(k.shape[0], depth * height * width, k.shape[4], k.shape[5])
    k_neighborhood = k_flat[:, flat_idx]
    logits = torch.einsum("bdijhf,bdijkhf->bdijhk", q_selected, k_neighborhood)

    if valid is not None:
        logits = logits.masked_fill(
            ~valid.view(1, valid.shape[0], valid.shape[1], valid.shape[2], 1, valid.shape[3]),
            float("-inf"),
        )
    if scale is not None:
        logits = logits * scale
    return logits


def na3d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    stride: Tuple[int, int, int] = (1, 1, 1),
    is_causal: Tuple[bool, bool, bool] = (False, False, False),
) -> torch.Tensor:
    if attn.ndim != 6:
        raise ValueError("attn must have shape [B, D, H, W, heads, K] for na3d_av.")
    if v.ndim != 6:
        raise ValueError("value must have shape [B, D, H, W, heads, D] for na3d_av.")

    b1, d_out, h_out, w_out, heads1, k_attn = attn.shape
    b2, depth, height, width, heads2, _ = v.shape

    if b1 != b2 or heads1 != heads2:
        raise ValueError("Batch and head dimensions of attn/value must match in na3d_av.")
    kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if k_attn != kernel_volume:
        raise ValueError(
            f"attn last dim ({k_attn}) must match kernel volume ({kernel_volume})."
        )
    if d_out > depth or h_out > height or w_out > width:
        raise ValueError("attn spatial dimensions cannot exceed value spatial dimensions.")

    qd = _output_positions(depth, stride[0], v.device) if stride[0] != 1 else torch.arange(d_out, device=v.device, dtype=torch.long)
    qh = _output_positions(height, stride[1], v.device) if stride[1] != 1 else torch.arange(h_out, device=v.device, dtype=torch.long)
    qw = _output_positions(width, stride[2], v.device) if stride[2] != 1 else torch.arange(w_out, device=v.device, dtype=torch.long)

    flat_idx, valid = _build_3d_neighborhood(
        qd, qh, qw, depth, height, width, kernel_size, dilation, is_causal, v.device,
    )

    v_flat = v.reshape(v.shape[0], depth * height * width, v.shape[4], v.shape[5])
    v_neighborhood = v_flat[:, flat_idx]

    if valid is not None:
        attn_masked = attn.masked_fill(
            ~valid.view(1, valid.shape[0], valid.shape[1], valid.shape[2], 1, valid.shape[3]),
            0.0,
        )
        return torch.einsum("bdijhk,bdijkhf->bdijhf", attn_masked, v_neighborhood)
    return torch.einsum("bdijhk,bdijkhf->bdijhf", attn, v_neighborhood)


def na2d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    is_causal: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    if attn.ndim != 5:
        raise ValueError("attn must have shape [B, H, W, heads, K] for na2d_av.")
    if v.ndim != 5:
        raise ValueError("value must have shape [B, H, W, heads, D] for na2d_av.")

    b1, h_out, w_out, heads1, k_attn = attn.shape
    b2, height, width, heads2, _ = v.shape

    if b1 != b2 or heads1 != heads2:
        raise ValueError("Batch and head dimensions of attn/value must match in na2d_av.")
    if k_attn != kernel_size[0] * kernel_size[1]:
        raise ValueError(
            f"attn last dim ({k_attn}) must match kernel area ({kernel_size[0] * kernel_size[1]})."
        )
    if h_out > height or w_out > width:
        raise ValueError("attn spatial dimensions cannot exceed value spatial dimensions.")

    qh = _output_positions(height, stride[0], v.device) if stride[0] != 1 else torch.arange(h_out, device=v.device, dtype=torch.long)
    qw = _output_positions(width, stride[1], v.device) if stride[1] != 1 else torch.arange(w_out, device=v.device, dtype=torch.long)

    flat_idx, valid = _build_2d_neighborhood(
        qh,
        qw,
        height,
        width,
        kernel_size,
        dilation,
        is_causal,
        v.device,
    )

    v_flat = v.reshape(v.shape[0], height * width, v.shape[3], v.shape[4])
    v_neighborhood = v_flat[:, flat_idx]

    if valid is not None:
        attn_masked = attn.masked_fill(~valid.view(1, valid.shape[0], valid.shape[1], 1, valid.shape[2]), 0.0)
        return torch.einsum("bijhk,bijkhd->bijhd", attn_masked, v_neighborhood)
    return torch.einsum("bijhk,bijkhd->bijhd", attn, v_neighborhood)


# ---------------------------------------------------------------------------
# Backward stubs — pure backend always returns None (use re-differentiation)
# ---------------------------------------------------------------------------

def na1d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return None

def na1d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1,), is_causal=(False,)):
    return None

def na2d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return None

def na2d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1), is_causal=(False, False)):
    return None

def na3d_qk_backward(d_attn, q, k, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return None

def na3d_av_backward(d_out, attn, v, kernel_size, dilation, stride=(1, 1, 1), is_causal=(False, False, False)):
    return None
