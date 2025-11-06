#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import torch
import triton
import triton.language as tl

# Not using autotune as there is in-place mutation of inputs
@triton.jit
def _swap_recent_cache_kernel(
        cache_k_ptr, cache_v_ptr, cache_score_ptr,
        past_lengths_ptr, blank_lengths_ptr,
        k_ptr, v_ptr,
        L, H, D,
        next_recent_pos,
        buffer_size: tl.constexpr,
        window_size: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SWAP_LEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Base pointers for the current (batch, head)
    # [B, L, H, D]
    cache_k_head_ptr = cache_k_ptr + pid_b * L * H * D + pid_h * D
    cache_v_head_ptr = cache_v_ptr + pid_b * L * H * D + pid_h * D
    # [B, L, H]
    cache_score_head_ptr = cache_score_ptr + pid_b * L * H + pid_h
    # [B, H]
    past_lengths_head_ptr = past_lengths_ptr + pid_b * H + pid_h
    # [B, H, D]
    k_head_ptr = k_ptr + pid_b * H * D + pid_h * D
    v_head_ptr = v_ptr + pid_b * H * D + pid_h * D

    past_length = tl.load(past_lengths_head_ptr)
    out_score = tl.load(cache_score_head_ptr + next_recent_pos * H)

    swap_mask = out_score > 0

    is_full = past_length >= buffer_size

    target_idx_not_full = window_size + past_length

    offsets_to_search = tl.arange(0, SWAP_LEN)
    search_mask = offsets_to_search < (L - window_size)
    scores_to_search = tl.load(
        cache_score_head_ptr + (window_size + offsets_to_search) * H,
        mask=search_mask,
        other=float('inf')
    )
    target_idx_full = window_size + tl.argmin(scores_to_search, axis=0)
    min_score = tl.load(cache_score_head_ptr + target_idx_full * H)

    target_idx = tl.where(is_full, target_idx_full, target_idx_not_full)
    swap_mask_full = swap_mask & (min_score < out_score)
    swap_mask = tl.where(is_full, swap_mask_full, swap_mask)

    target_k_ptr = cache_k_head_ptr + target_idx * H * D # [B, L, H, D]
    target_v_ptr = cache_v_head_ptr + target_idx * H * D
    target_score_ptr = cache_score_head_ptr + target_idx * H # [B, L, H]

    recent_k_ptr = cache_k_head_ptr + next_recent_pos * H * D
    recent_v_ptr = cache_v_head_ptr + next_recent_pos * H * D

    d_offsets = tl.arange(0, BLOCK_D)
    for i in range(0, tl.cdiv(D, BLOCK_D)):
        d_ptr = i * BLOCK_D + d_offsets
        d_mask = d_ptr < D
        full_mask = d_mask & swap_mask

        recent_k = tl.load(recent_k_ptr + d_ptr, mask=d_mask)
        recent_v = tl.load(recent_v_ptr + d_ptr, mask=d_mask)

        tl.store(target_k_ptr + d_ptr, recent_k, mask=full_mask)
        tl.store(target_v_ptr + d_ptr, recent_v, mask=full_mask)

    tl.store(target_score_ptr, out_score, mask=swap_mask)

    new_past_length = tl.minimum(past_length + 1, buffer_size)
    final_past_length = tl.where(swap_mask, new_past_length, past_length)
    tl.store(past_lengths_head_ptr, final_past_length)

    dest_k_ptr = cache_k_head_ptr + next_recent_pos * H * D
    dest_v_ptr = cache_v_head_ptr + next_recent_pos * H * D
    # dest_score_ptr = cache_score_head_ptr + next_recent_pos * H

    for i in range(0, tl.cdiv(D, BLOCK_D)):
        d_ptr = i * BLOCK_D + d_offsets
        d_mask = d_ptr < D
        new_k = tl.load(k_head_ptr + d_ptr, mask=d_mask)
        new_v = tl.load(v_head_ptr + d_ptr, mask=d_mask)

        # Store them into the recent position
        tl.store(dest_k_ptr + d_ptr, new_k, mask=d_mask)
        tl.store(dest_v_ptr + d_ptr, new_v, mask=d_mask)

    # tl.store(dest_score_ptr, float('nan'))

    if pid_h == 0:
        blank_lengths_b_ptr = blank_lengths_ptr + pid_b
        current_blank = tl.load(blank_lengths_b_ptr)

        new_blank = tl.maximum(current_blank - 1, 0)
        tl.store(blank_lengths_b_ptr, new_blank)

def swap_recent_cache(
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_score: torch.Tensor,
        blank_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        next_recent_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
        buffer_size: int,
        window_size: int
):
    """
    Fused Triton kernel for swapping tokens in a key-value cache based on scores.
    This function replaces the original PyTorch implementation with a single kernel launch.

    Args:
        cache_k, cache_v: The key/value caches. Shape: [B, L, H, D]
        cache_score: The score for each token in the cache. Shape: [B, L, H]
        blank_lengths: Number of blank tokens. Shape: [B]
        past_lengths: Current length of the cache for each head. Shape: [B, H]
        next_recent_pos: The index of the recent token to consider swapping. Scalar tensor.
        k, v: The new key/value tensors to swap in. Shape: [B, 1, H, D]
        buffer_size: The total size of the buffer.
        window_size: The size of the recent window that is not swappable.
    """
    assert cache_k.is_contiguous()
    assert cache_v.is_contiguous()
    assert cache_score.is_contiguous()
    assert past_lengths.is_contiguous()
    assert blank_lengths.is_contiguous()

    B, L, H, D = cache_k.shape

    # Kernel launch grid
    grid = (B, H)

    if D >= 128:
        BLOCK_D = 128
    elif D >= 64:
        BLOCK_D = 64
    else:
        BLOCK_D = 32

    SWAP_LEN = triton.next_power_of_2(L - window_size)

    _swap_recent_cache_kernel[grid](
        cache_k, cache_v, cache_score,
        past_lengths, blank_lengths,
        k, v,
        L, H, D,
        next_recent_pos=next_recent_pos,
        buffer_size=buffer_size,
        window_size=window_size,
        BLOCK_D=BLOCK_D,
        SWAP_LEN=SWAP_LEN,
        num_warps=2, num_stages=3
    )
    # blank_lengths.add_(-1).clamp_(0)