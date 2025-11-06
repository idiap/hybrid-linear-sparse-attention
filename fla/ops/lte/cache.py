#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import torch
from typing import TYPE_CHECKING, Optional, Tuple, List

torch._dynamo.config.capture_dynamic_output_shape_ops = True

@torch.compile
def compact_front(x: torch.Tensor, mask: torch.Tensor, out: Optional[torch.Tensor]):
    pos = mask.cumsum(dim=1) - 1
    nz = mask.nonzero(as_tuple=False)

    if nz.numel() > 0:
        b_idx, l_idx, h_idx = nz.unbind(1)
        p_idx = pos[mask]
        out[b_idx, p_idx, h_idx] = x[b_idx, l_idx, h_idx]

@torch.compile
def prepare_cache(k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor], score: torch.Tensor,
                  margin: int, buffer_size: int, window_size: int, sink_size: int):
    # k, v: [B, L, H, D]
    # attention_mask: [B, L]
    # selected, score: [B, L, H]
    device = k.device
    B, L, H, D = k.shape

    if attention_mask is None:
        offsets = torch.zeros(B, device=device, dtype=torch.long)  # [B]
    else:
        offsets = torch.argmax(attention_mask, dim=1) # [B]
        score[~attention_mask.bool()] = float('-inf')

    cache_k = torch.zeros(B, buffer_size + window_size, H, D, device=device, dtype=k.dtype)
    cache_v = torch.zeros(B, buffer_size + window_size, H, D, device=device, dtype=k.dtype)
    cache_score = torch.full((B, buffer_size + window_size, H), device=device, dtype=score.dtype, fill_value=float('-inf'))

    # Copy those inside the recent window
    window_copy_size = min(L, window_size)
    idx = torch.arange(L, device=score.device, dtype=score.dtype)[None, :]
    # TODO: fix the case when L < sink_size, otherwise sink tokens can be evicted when L grows over window_size
    sink_mask = (offsets[:, None] <= idx) & (idx < (offsets[:, None] + sink_size))
    score[sink_mask] = float('inf')

    selected_indices = torch.full((B, buffer_size, H), L, device=score.device, dtype=torch.long)
    n_sel = min(buffer_size, L - window_copy_size)
    top_vals, top_idx = torch.topk(score[:, :L - window_copy_size], k=n_sel, dim=1, sorted=False)
    mask = top_vals > 0

    selected_indices[:, :n_sel].copy_(torch.where(mask, top_idx, L))
    selected_indices, _ = torch.sort(selected_indices, dim=1)
    past_lengths = mask.sum(1).to(torch.int32)  # [B]

    recent_lengths = window_copy_size - (offsets - (L - window_copy_size)).clamp(min=0)  # [B]
    blank_lengths = window_size - recent_lengths

    if window_copy_size > 0:
        cache_k[:, window_size - window_copy_size: window_size] = k[:, -window_copy_size:]
        cache_v[:, window_size - window_copy_size: window_size] = v[:, -window_copy_size:]
        cache_score[:, window_size - window_copy_size: window_size] = score[:, -window_copy_size:]

    # Aggregate and copy those outside the recent window but selected
    base_start = L - window_copy_size  # scalar
    selected_indices_ = selected_indices.clamp(max=L - 1)
    if buffer_size > 0 and base_start > 0:
        cache_k[:, window_size:] = torch.gather(k, dim=1, index=selected_indices_[:, :, :, None].expand(-1, -1, -1, D))
        cache_v[:, window_size:] = torch.gather(v, dim=1, index=selected_indices_[:, :, :, None].expand(-1, -1, -1, D))
        cache_score[:, window_size:] = torch.gather(score, dim=1, index=selected_indices_)

    return cache_k, cache_v, cache_score, offsets, past_lengths, blank_lengths, selected_indices

@torch.compile
def rotate_recent(cache_k, cache_v, cache_score, next_recent_pos, window_size):
    # next_recent_pos = next_recent_pos.item()
    buffer = cache_k[:, :next_recent_pos].clone()
    cache_k[:, :window_size - next_recent_pos] = \
        cache_k[:, next_recent_pos:window_size]
    cache_k[:, window_size - next_recent_pos: window_size] = buffer

    buffer = cache_v[:, :next_recent_pos].clone()
    cache_v[:, :window_size - next_recent_pos] = \
        cache_v[:, next_recent_pos: window_size]
    cache_v[:, window_size - next_recent_pos: window_size] = buffer

    buffer = cache_score[:, :next_recent_pos].clone()
    cache_score[:, :window_size - next_recent_pos] = \
        cache_score[:, next_recent_pos: window_size]
    cache_score[:, window_size - next_recent_pos: window_size] = buffer

@torch.compile
def replace_at_idx(target: torch.Tensor,
                           source: torch.Tensor,
                           idx:    torch.Tensor,
                           mask:   torch.Tensor
                   ) -> torch.Tensor:
    B, L, H, D = target.shape
    idx_exp = idx[:, None, :, None].expand(B, 1, H, D)       # [B, 1, H, D]
    cur = target.gather(dim=1, index=idx_exp)                        # [B, 1, H, D]
    src = torch.where(mask[:, None, :, None], source[:, None, :, :], cur)
    return idx_exp, src

@torch.compile
def replace_at_idx_1d(target: torch.Tensor,
                           source: torch.Tensor,
                           idx:    torch.Tensor,
                           mask:   torch.Tensor
                   ) -> torch.Tensor:
    idx_exp = idx[:, None, :]                                          # [B,1,H]
    cur = target.gather(dim=1, index=idx_exp)                          # [B,1,H]
    src = torch.where(mask[:, None, :], source[:, None, :], cur)
    return idx_exp, src


@torch.compile
def swap_recent_cache(cache_k, cache_v, cache_score, blank_lengths, past_lengths, next_recent_pos, k, v, buffer_size, window_size):
    # cache_k, cache_v: [B, L, H, D]
    # cache_score: [B, L, H]
    # blank_lengths: [B], int32
    # past_lengths: [B, H], int32
    # next_recent_pos: scalar int
    # k, v: [B, 1, H, D]
    # buffer_size, window_size: constant int
    next_recent_pos = next_recent_pos.item()
    out_score = cache_score[:, next_recent_pos]
    swap_mask = out_score > 0
    if (past_lengths < buffer_size).all():
        target_idx = past_lengths.to(torch.int64)
    else:
        vals, target_idx = torch.min(cache_score[:, window_size:], dim=1) # [B, H]
        swap_mask &= vals < out_score
    if swap_mask.any():
        target_idx += window_size
        idx, val = replace_at_idx(cache_k, cache_k[:, next_recent_pos], target_idx, swap_mask)
        cache_k.scatter_(dim=1, index=idx, src=val)
        idx, val = replace_at_idx(cache_v, cache_v[:, next_recent_pos], target_idx, swap_mask)
        cache_v.scatter_(dim=1, index=idx, src=val)
        idx, val = replace_at_idx_1d(cache_score, out_score, target_idx, swap_mask)
        cache_score.scatter_(dim=1, index=idx, src=val)
        torch.where(swap_mask, (past_lengths + 1).clamp_max(buffer_size), past_lengths, out=past_lengths)
    # Swap-in the new token
    cache_k[:, next_recent_pos] = k[:, 0]
    cache_v[:, next_recent_pos] = v[:, 0]
    # cache_score[:, next_recent_pos].fill_(float('nan'))
    blank_lengths.add_(-1).clamp_(0)
