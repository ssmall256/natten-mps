"""
Pure PyTorch implementations of fused QK+RPB and AV operations
for DiNAT-style neighborhood attention.

Exposes intermediate stages (QK, Softmax, AV) separately for API compatibility.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .reference_impl import get_pb_start, get_window_end, get_window_start


def _check_args_against_dim(length: int, kernel_size: int, dilation: int, axis_name: str) -> None:
    if kernel_size * dilation > length:
        raise ValueError(
            f"Invalid NATTEN args on {axis_name}: kernel_size * dilation must be <= axis length. "
            f"Got kernel_size={kernel_size}, dilation={dilation}, {axis_name}={length}."
        )


def _natten1dqkrpb_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    rpb: Optional[torch.Tensor],
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """1D QK+RPB with shifted window semantics. Heads-first layout [B, H, L, D]."""
    B, H, L, D = query.shape
    K = int(kernel_size)
    nh = K // 2
    dil = int(dilation)
    rpb_size = 2 * K - 1

    ni_list = [get_window_start(i, L, K, nh, dil) for i in range(L)]
    ei_list = [get_window_end(ni_list[i], L, K, dil) for i in range(L)]
    pi_list = [get_pb_start(i, L, K, nh, dil) for i in range(L)]

    ni = torch.tensor(ni_list, dtype=torch.long, device=query.device)
    ei = torch.tensor(ei_list, dtype=torch.long, device=query.device)
    pi = torch.tensor(pi_list, dtype=torch.long, device=query.device)

    min_ni = int(ni.min().item())
    pad_before = max(0, -min_ni)
    max_ei = int(ei.max().item())
    pad_after = max(0, max_ei - L)

    key_pad = torch.nn.functional.pad(key, (0, 0, pad_before, pad_after))
    Lp = L + pad_before + pad_after

    attn_scores = []
    for ki in range(K):
        pos = ni + ki * dil
        valid = (pos >= 0) & (pos < ei)
        mask = torch.where(valid, torch.tensor(0.0, device=query.device), torch.tensor(float("-inf"), device=query.device))
        mask = mask.reshape(1, 1, L)

        idx = (pos + pad_before).clamp(0, Lp - 1)
        k_shifted = key_pad[:, :, idx, :]

        score = (query * k_shifted).sum(dim=-1)

        if rpb is not None:
            rpb_idx = (pi + ki).clamp(0, rpb_size - 1)
            rpb_val = rpb[:, rpb_idx]
            rpb_val = rpb_val.reshape(1, H, L)
            score = score + rpb_val

        score = score + mask
        attn_scores.append(score)

    return torch.stack(attn_scores, dim=-1)


def _natten1dav_torch(
    attention_probs: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """1D AV with shifted window semantics. Heads-first layout [B, H, L, K]."""
    B, H, L, _ = attention_probs.shape
    _, _, _, D = value.shape
    K = int(kernel_size)
    nh = K // 2
    dil = int(dilation)

    ni_list = [get_window_start(i, L, K, nh, dil) for i in range(L)]
    ei_list = [get_window_end(ni_list[i], L, K, dil) for i in range(L)]

    ni = torch.tensor(ni_list, dtype=torch.long, device=value.device)
    ei = torch.tensor(ei_list, dtype=torch.long, device=value.device)

    min_ni = int(ni.min().item())
    pad_before = max(0, -min_ni)
    max_ei = int(ei.max().item())
    pad_after = max(0, max_ei - L)

    value_pad = torch.nn.functional.pad(value, (0, 0, pad_before, pad_after))
    Lp = L + pad_before + pad_after

    output = torch.zeros(B, H, L, D, dtype=value.dtype, device=value.device)

    for ki in range(K):
        pos = ni + ki * dil
        valid = (pos >= 0) & (pos < ei)
        valid_reshaped = valid.reshape(1, 1, L, 1)

        idx = (pos + pad_before).clamp(0, Lp - 1)
        v_shifted = value_pad[:, :, idx, :]

        attn_weight = attention_probs[:, :, :, ki : ki + 1]
        attn_weight = torch.where(valid_reshaped, attn_weight, torch.zeros_like(attn_weight))
        output = output + attn_weight * v_shifted

    return output


def _natten2dqkrpb_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    rpb: Optional[torch.Tensor],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """2D QK+RPB with shifted window semantics. Heads-first layout [B, H, Ht, W, D]."""
    if isinstance(kernel_size, int):
        ks_h = ks_w = kernel_size
    else:
        ks_h, ks_w = kernel_size

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    if ks_h != ks_w or dil_h != dil_w:
        raise ValueError("Only square kernels and uniform dilation are supported for shifted semantics.")

    B, H, height, width, D = query.shape
    K = int(ks_h)
    nh = K // 2
    rpb_size = 2 * K - 1

    ni_list = [get_window_start(i, height, K, nh, dil_h) for i in range(height)]
    ei_list = [get_window_end(ni_list[i], height, K, dil_h) for i in range(height)]
    pi_list = [get_pb_start(i, height, K, nh, dil_h) for i in range(height)]

    nj_list = [get_window_start(j, width, K, nh, dil_w) for j in range(width)]
    ej_list = [get_window_end(nj_list[j], width, K, dil_w) for j in range(width)]
    pj_list = [get_pb_start(j, width, K, nh, dil_w) for j in range(width)]

    ni = torch.tensor(ni_list, dtype=torch.long, device=query.device)
    ei = torch.tensor(ei_list, dtype=torch.long, device=query.device)
    pi = torch.tensor(pi_list, dtype=torch.long, device=query.device)
    nj = torch.tensor(nj_list, dtype=torch.long, device=query.device)
    ej = torch.tensor(ej_list, dtype=torch.long, device=query.device)
    pj = torch.tensor(pj_list, dtype=torch.long, device=query.device)

    min_ni = int(ni.min().item())
    min_nj = int(nj.min().item())
    pad_before_i = max(0, -min_ni)
    pad_before_j = max(0, -min_nj)

    max_ei = int(ei.max().item())
    max_ej = int(ej.max().item())
    pad_after_i = max(0, max_ei - height)
    pad_after_j = max(0, max_ej - width)

    key_pad = torch.nn.functional.pad(key, (0, 0, pad_before_j, pad_after_j, pad_before_i, pad_after_i))
    Hp = height + pad_before_i + pad_after_i
    Wp = width + pad_before_j + pad_after_j

    attn_scores = []

    for ki in range(K):
        pos_i = ni + ki * dil_h
        valid_i = (pos_i >= 0) & (pos_i < ei)

        h_idx = (pos_i + pad_before_i).clamp(0, Hp - 1)
        k_h = key_pad[:, :, h_idx, :, :]

        if rpb is not None:
            bias_i = (pi + ki).clamp(0, rpb_size - 1)
            rpb_rows = rpb[:, bias_i, :]
        else:
            rpb_rows = None

        for kj in range(K):
            pos_j = nj + kj * dil_w
            valid_j = (pos_j >= 0) & (pos_j < ej)

            w_idx = (pos_j + pad_before_j).clamp(0, Wp - 1)
            k_shifted = k_h[:, :, :, w_idx, :]

            valid = valid_i.reshape(height, 1) & valid_j.reshape(1, width)
            mask = torch.where(valid, torch.tensor(0.0, device=query.device), torch.tensor(float("-inf"), device=query.device))
            mask = mask.reshape(1, 1, height, width)

            score = (query * k_shifted).sum(dim=-1)

            if rpb_rows is not None:
                bias_j = (pj + kj).clamp(0, rpb_size - 1)
                rpb_ij = rpb_rows[:, :, bias_j]
                rpb_ij = rpb_ij.reshape(1, H, height, width)
                score = score + rpb_ij

            score = score + mask
            attn_scores.append(score)

    return torch.stack(attn_scores, dim=-1)


def _natten2dav_torch(
    attention_probs: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """2D AV with shifted window semantics. Heads-first layout [B, H, Ht, W, K*K]."""
    if isinstance(kernel_size, int):
        ks_h = ks_w = kernel_size
    else:
        ks_h, ks_w = kernel_size

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    if ks_h != ks_w or dil_h != dil_w:
        raise ValueError("Only square kernels and uniform dilation are supported for shifted semantics.")

    B, H, height, width, _ = attention_probs.shape
    _, _, _, _, D = value.shape
    K = int(ks_h)
    nh = K // 2

    ni_list = [get_window_start(i, height, K, nh, dil_h) for i in range(height)]
    ei_list = [get_window_end(ni_list[i], height, K, dil_h) for i in range(height)]
    nj_list = [get_window_start(j, width, K, nh, dil_w) for j in range(width)]
    ej_list = [get_window_end(nj_list[j], width, K, dil_w) for j in range(width)]

    ni = torch.tensor(ni_list, dtype=torch.long, device=value.device)
    ei = torch.tensor(ei_list, dtype=torch.long, device=value.device)
    nj = torch.tensor(nj_list, dtype=torch.long, device=value.device)
    ej = torch.tensor(ej_list, dtype=torch.long, device=value.device)

    min_ni = int(ni.min().item())
    min_nj = int(nj.min().item())
    pad_before_i = max(0, -min_ni)
    pad_before_j = max(0, -min_nj)

    max_ei = int(ei.max().item())
    max_ej = int(ej.max().item())
    pad_after_i = max(0, max_ei - height)
    pad_after_j = max(0, max_ej - width)

    value_pad = torch.nn.functional.pad(value, (0, 0, pad_before_j, pad_after_j, pad_before_i, pad_after_i))
    Hp = height + pad_before_i + pad_after_i
    Wp = width + pad_before_j + pad_after_j

    output = torch.zeros(B, H, height, width, D, dtype=value.dtype, device=value.device)
    idx = 0

    for ki in range(K):
        pos_i = ni + ki * dil_h
        valid_i = (pos_i >= 0) & (pos_i < ei)

        h_idx = (pos_i + pad_before_i).clamp(0, Hp - 1)
        v_h = value_pad[:, :, h_idx, :, :]

        for kj in range(K):
            pos_j = nj + kj * dil_w
            valid_j = (pos_j >= 0) & (pos_j < ej)

            w_idx = (pos_j + pad_before_j).clamp(0, Wp - 1)
            v_shifted = v_h[:, :, :, w_idx, :]

            valid = valid_i.reshape(height, 1) & valid_j.reshape(1, width)
            valid_reshaped = valid.reshape(1, 1, height, width, 1)

            attn_weight = attention_probs[:, :, :, :, idx : idx + 1]
            attn_weight = torch.where(valid_reshaped, attn_weight, torch.zeros_like(attn_weight))

            output = output + attn_weight * v_shifted
            idx += 1

    return output


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """1D NATTEN QK+RPB (returns scores BEFORE softmax).

    Layout: spatial-first [B, L, H, D].
    RPB: [H, 2*kernel_size - 1] or None.
    Returns: [B, L, H, K].
    """
    _check_args_against_dim(int(query.shape[1]), int(kernel_size), int(dilation), "length")

    q_hf = query.permute(0, 2, 1, 3)
    k_hf = key.permute(0, 2, 1, 3)
    out_hf = _natten1dqkrpb_torch(q_hf, k_hf, rpb, kernel_size, dilation)
    return out_hf.permute(0, 2, 1, 3)


def natten1dav(attention_probs, value, kernel_size, dilation):
    """1D NATTEN AV (applies softmaxed attention to values).

    Layout: spatial-first [B, L, H, K] for attn, [B, L, H, D] for value.
    Returns: [B, L, H, D].
    """
    _check_args_against_dim(int(value.shape[1]), int(kernel_size), int(dilation), "length")

    attn_hf = attention_probs.permute(0, 2, 1, 3)
    v_hf = value.permute(0, 2, 1, 3)
    out_hf = _natten1dav_torch(attn_hf, v_hf, kernel_size, dilation)
    return out_hf.permute(0, 2, 1, 3)


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """2D NATTEN QK+RPB (returns scores BEFORE softmax).

    Layout: spatial-first [B, Hh, Hw, H, D].
    RPB: [H, 2*K-1, 2*K-1] or None.
    Returns: [B, Hh, Hw, H, K*K].
    """
    k = int(kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size)
    d = int(dilation[0] if isinstance(dilation, tuple) else dilation)
    _check_args_against_dim(int(query.shape[1]), k, d, "height")
    _check_args_against_dim(int(query.shape[2]), k, d, "width")

    q_hf = query.permute(0, 3, 1, 2, 4)
    k_hf = key.permute(0, 3, 1, 2, 4)
    out_hf = _natten2dqkrpb_torch(q_hf, k_hf, rpb, kernel_size, dilation)
    return out_hf.permute(0, 2, 3, 1, 4)


def natten2dav(attention_probs, value, kernel_size, dilation):
    """2D NATTEN AV (applies softmaxed attention to values).

    Layout: spatial-first [B, Hh, Hw, H, K*K] for attn, [B, Hh, Hw, H, D] for value.
    Returns: [B, Hh, Hw, H, D].
    """
    k = int(kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size)
    d = int(dilation[0] if isinstance(dilation, tuple) else dilation)
    _check_args_against_dim(int(value.shape[1]), k, d, "height")
    _check_args_against_dim(int(value.shape[2]), k, d, "width")

    attn_hf = attention_probs.permute(0, 3, 1, 2, 4)
    v_hf = value.permute(0, 3, 1, 2, 4)
    out_hf = _natten2dav_torch(attn_hf, v_hf, kernel_size, dilation)
    return out_hf.permute(0, 2, 3, 1, 4)


__all__ = ["natten1dqkrpb", "natten1dav", "natten2dqkrpb", "natten2dav"]
