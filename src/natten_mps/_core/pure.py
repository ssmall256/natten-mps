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
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("na1d_qk expects query/key shaped [B, L, H, D].")
    if q.shape != k.shape:
        raise ValueError("query and key must have identical shape for na1d_qk.")

    length = q.shape[1]
    query_positions = torch.arange(length, device=q.device, dtype=torch.long)
    key_idx = _noncausal_1d_indices(query_positions, length, kernel_size[0], dilation[0], q.device)
    k_neighborhood = k[:, key_idx]
    scale_value = _default_scale(q)
    return torch.einsum("blhd,blkhd->blhk", q, k_neighborhood) * scale_value


def na1d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int],
    dilation: Tuple[int],
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

    query_positions = torch.arange(l_out, device=v.device, dtype=torch.long)
    key_idx = _noncausal_1d_indices(query_positions, length, kernel_size[0], dilation[0], v.device)
    v_neighborhood = v[:, key_idx]
    return torch.einsum("blhk,blkhd->blhd", attn, v_neighborhood)


def na2d_qk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
) -> torch.Tensor:
    if q.ndim != 5 or k.ndim != 5:
        raise ValueError("na2d_qk expects query/key shaped [B, H, W, heads, dim].")
    if q.shape != k.shape:
        raise ValueError("query and key must have identical shape for na2d_qk.")

    _, height, width, _, _ = q.shape
    qh = torch.arange(height, device=q.device, dtype=torch.long)
    qw = torch.arange(width, device=q.device, dtype=torch.long)

    flat_idx, _ = _build_2d_neighborhood(
        qh,
        qw,
        height,
        width,
        kernel_size,
        dilation,
        (False, False),
        q.device,
    )

    k_flat = k.reshape(k.shape[0], height * width, k.shape[3], k.shape[4])
    k_neighborhood = k_flat[:, flat_idx]
    scale_value = _default_scale(q)
    return torch.einsum("bijhd,bijkhd->bijhk", q, k_neighborhood) * scale_value


def na2d_av_forward(
    attn: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
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

    qh = torch.arange(h_out, device=v.device, dtype=torch.long)
    qw = torch.arange(w_out, device=v.device, dtype=torch.long)

    flat_idx, _ = _build_2d_neighborhood(
        qh,
        qw,
        height,
        width,
        kernel_size,
        dilation,
        (False, False),
        v.device,
    )

    v_flat = v.reshape(v.shape[0], height * width, v.shape[3], v.shape[4])
    v_neighborhood = v_flat[:, flat_idx]
    return torch.einsum("bijhk,bijkhd->bijhd", attn, v_neighborhood)
