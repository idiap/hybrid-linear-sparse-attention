#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import torch
from typing import Optional, Tuple, List
import math

from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks, or_masks, create_block_mask, create_mask, flex_attention
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

@torch.compile
def adjust_selected(selected: torch.BoolTensor, batch_size: int, length: int, sink_size: int,
                    attention_mask: Optional[torch.Tensor] = None, offset: int = 0):
    sink_span = max(0, sink_size - offset)
    if attention_mask is None or attention_mask[:, 0].sum() == batch_size:
        selected[:, :sink_span] = True
    else:
        positions = torch.argmax(attention_mask, dim=1)
        seq_range = torch.arange(length, device=positions.device).unsqueeze(0)
        sink_mask = (seq_range >= positions.unsqueeze(1)) & (seq_range < (positions + sink_span).unsqueeze(1))  # [B, L]
        if len(selected.shape) == 3:
            sink_mask = sink_mask.unsqueeze(-1)
        selected |= sink_mask
    if attention_mask is not None:
        if len(selected.shape) == 3:
            attention_mask = attention_mask.unsqueeze(-1)
        selected &= attention_mask.bool()
    return selected


@torch.compile
def naive_enforce_buffer_size(
    score: torch.Tensor,
    selected: torch.BoolTensor,
    buffer_size: int,
    window_size: int,
    sink_size: int,
    attention_mask=None,
) -> torch.BoolTensor:
    B, L, H = score.shape
    sink_size = min(sink_size, L)
    window_start = max(L - window_size - 1, 0)

    if window_start <= sink_size:
        return selected

    if (selected[:, :window_start, :].sum(1) <= buffer_size).all():
        return selected

    pos = torch.arange(L, device=score.device).view(1, L, 1)  # (1,L,1)
    if attention_mask is not None:
        offsets = torch.argmax(attention_mask, dim=1)
        sink_mask = ((pos >= offsets) & (pos < (offsets + sink_size))).transpose(0, 2)
    else:
        sink_mask = pos < sink_size  # keep as-is
    recent_window_mask = pos >= window_start  # never pruned
    prune_region_mask = (~sink_mask) & (~recent_window_mask)  # subject to limit
    buffer_size = buffer_size - sink_size

    candidate_mask = selected & prune_region_mask
    if buffer_size <= 0:
        return selected & (~prune_region_mask)

    neg_inf = torch.tensor(-float("inf"), device=score.device)
    cand_scores = torch.where(candidate_mask, score, neg_inf)

    k = min(buffer_size, L)  # top-k w.r.t. sequence length
    top_vals, top_idx = torch.topk(cand_scores, k=k, dim=1)

    keep_mask = torch.zeros_like(selected, dtype=torch.bool)  # (B,L,H)
    keep_mask.scatter_(1, top_idx, top_vals > neg_inf)  # survivors

    new_selected = (selected & ~prune_region_mask) | keep_mask

    return new_selected

@torch.compile
def get_buffer_indices(
    score: torch.Tensor,
    buffer_size: int,
    window_size: int,
    sink_size: int,
    attention_mask=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, H = score.shape
    sink_size = min(sink_size, L)
    pos = torch.arange(L, device=score.device).view(1, L, 1)  # (1, L, 1)
    if attention_mask is not None:
        offsets = torch.argmax(attention_mask, dim=1)
        sink_mask = ((pos >= offsets) & (pos < (offsets + sink_size))).transpose(0, 2).squeeze(-1)
        score[~attention_mask.bool()] = float('-inf')
    else:
        sink_mask = pos.squeeze(-1) < sink_size
    score[sink_mask] = float('inf')

    selected_idx = torch.full((B, buffer_size, H), L, device=score.device, dtype=torch.long)
    k = min(buffer_size, max(0, L - window_size))
    top_vals, top_idx = torch.topk(score[:, :max(0, L - window_size)], k=k, dim=1)
    mask = top_vals > 0

    selected_idx[:, :k].copy_(torch.where(mask, top_idx, L))
    lengths = mask.sum(1)
    return selected_idx, lengths

@torch.compile
def prepare_lte_mask(selected, window_size, attention_mask=None):
    def lte(b, h, q_idx, kv_idx): # Selected (sink incl.)
        return selected[b, h, kv_idx]

    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    lte_mask = lte
    if window_size > 0:
        lte_mask = or_masks(lte_mask, sliding_window)
    lte_mask = and_masks(lte_mask, causal_mask)
    if attention_mask is not None:
        def attention_mask_func(b, h, q_idx, kv_idx):
            return attention_mask[b, kv_idx] > 0

        lte_mask = and_masks(lte_mask, attention_mask_func)
    lte_mask.__name__ = f"lte_{window_size}"
    return lte_mask

@torch.compile
def get_mask_mod_w_offset(mask_mod: _mask_mod_signature, _offset: int):
    @torch.compile
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + _offset, kv)
    return _mask_mod


@torch.compile
def lte_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, selected: torch.Tensor,
                  window_size: int, attention_mask: Optional[torch.tensor] = None, q_offset: int=0) -> torch.Tensor:
    B, H, S, D = k.shape
    Q = q.shape[-2]
    if k.shape[1] != q.shape[1]:
        selected = selected.repeat_interleave(q.size(-3) // k.size(-3), -2)
        H = q.shape[1]

    mask_fn = prepare_lte_mask(selected, window_size, attention_mask)
    if q_offset != 0:
        mask_fn = get_mask_mod_w_offset(mask_fn, q_offset)
    block_mask = create_block_mask(mask_fn, B, H, Q, S, device=k.device, _compile=True)

    o = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=k.shape[1] != q.shape[1],
                       kernel_options={
                           "BLOCK_M": 64,
                           "BLOCK_N": 64,
                       }
                       )
    return o

def lte_attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, selected: torch.Tensor,
                        window_size: int, attention_mask: Optional[torch.tensor] = None,
                        q_offset: int=0) -> torch.Tensor:
    B, H, S, D = k.shape
    Q = q.shape[-2]

    if k.shape[1] != q.shape[1]:
        selected = selected.repeat_interleave(q.size(-3) // k.size(-3), -2)
        H = q.shape[1]

        k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
        v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)

    mask_fn = prepare_lte_mask(selected, window_size, attention_mask)
    if q_offset != 0:
        mask_fn = get_mask_mod_w_offset(mask_fn, q_offset)
    attention_mask = create_mask(mask_fn, B, H, Q, S, device=k.device)

    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    return o

def lte_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, selected: torch.Tensor,
                        window_size: int, attention_mask: Optional[torch.tensor] = None,
                        q_offset: int=0) -> torch.Tensor:
    B, H, S, D = k.shape
    Q = q.shape[-2]

    if k.shape[1] != q.shape[1]:
        selected = selected.repeat_interleave(q.size(-3) // k.size(-3), -2)
        H = q.shape[1]

        k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
        v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)

    mask_fn = prepare_lte_mask(selected, window_size, attention_mask)
    if q_offset != 0:
        mask_fn = get_mask_mod_w_offset(mask_fn, q_offset)
    attention_mask = create_mask(mask_fn, B, H, Q, S, device=k.device)

    o, attn_weight = scaled_dot_product_attention_torch(q, k, v, is_causal=True, attn_mask=attention_mask)
    return o, attn_weight


def scaled_dot_product_attention_torch(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    B, H = query.size(0), query.size(1)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=attn_bias.device).tril(diagonal=0)[None, None, :, :].repeat((B, H, 1, 1))
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        causal_bias = attn_bias.clone()

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    c_attn_weight = attn_weight
    if is_causal:
        c_attn_weight = c_attn_weight + causal_bias
    c_attn_weight = torch.softmax(c_attn_weight, dim=-1)
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

@torch.compile
def prepare_compact_lte_mask(blank_lengths, prefix_length, past_lengths, window_size: int):
    def lte_mask(b, h, q_idx, kv_idx):
        return (kv_idx < window_size + past_lengths[b, h]) & (~((kv_idx >= prefix_length) & (kv_idx < prefix_length + blank_lengths[b])))
    return lte_mask

@torch.compile
def compact_lte_attention(q, k, v, blank_lengths, prefix_length, past_lengths, window_size):
    B, H, S, D = k.shape
    Q = q.shape[-2]
    if k.shape[1] != q.shape[1]:
        past_lengths = past_lengths.repeat_interleave(q.size(1) // k.size(1), -1)
        H = q.shape[1]

    mask_fn = prepare_compact_lte_mask(blank_lengths, prefix_length, past_lengths, window_size)
    block_mask = create_block_mask(mask_fn, B, H, Q, S, device=k.device, _compile=True)

    o = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=k.shape[1] != q.shape[1])
    return o

@torch.compile
def prepare_flash_compact_lte(q, k, v, past_lengths, window_size, g):
    # k, v -> (B * Hk, L, 1, D)
    # q -> (B * Hk, Lq, g, D)
    k = k.transpose(1, 2).reshape(-1, k.shape[1], 1, k.shape[3])
    v = v.transpose(1, 2).reshape(-1, v.shape[1], 1, v.shape[3])
    q = q.transpose(1, 2).reshape(-1, q.shape[1], g, q.shape[3])
    past_lengths = past_lengths.view(-1) + window_size # [B, Hk] -> [B * Hk]
    return q, k, v, past_lengths

def flash_compact_lte_attention(q, k, v, past_lengths, window_size):
    # k, v: (B, L, Hk, D), q: (B, Lq, Hq, D)
    g = q.shape[2] // k.shape[2]
    B, Hk = k.shape[0], k.shape[2]
    q, k, v, past_lengths = prepare_flash_compact_lte(q, k, v, past_lengths, window_size, g)

    o = flash_attn_with_kvcache(
        q=q, k_cache=k, v_cache=v,
        cache_seqlens=past_lengths,
        window_size=(-1, -1),
    )
    # o: (B * Hk, Lq, g, D) -> (B, Lq, Hq * D)
    o = o.view(B, Hk, o.shape[1], o.shape[2], o.shape[3]).transpose(1, 2).reshape(B, o.shape[1], Hk * o.shape[2] * o.shape[3])
    return o