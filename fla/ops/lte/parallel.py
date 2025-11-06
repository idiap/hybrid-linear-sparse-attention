# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

#
# SPDX-FileCopyrightText: Copyright © 2023-2025, Songlin Yang, Yu Zhang
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import warnings
from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2, log2
from fla.utils import check_shared_mem, contiguous


@torch.compile
def upper_bound_index(selected_idx, max_length):
    # selected_idx: [B, M, H]
    # selected_lengths: [B, H] <= M
    B, M, H = selected_idx.shape
    upper = torch.zeros(B, max_length + 1, H, dtype=torch.int64, device=selected_idx.device)
    upper.scatter_add_(1, selected_idx, torch.ones_like(selected_idx, dtype=torch.int64))
    upper = torch.cumsum(upper, dim=1)
    return upper[:, :-1]

@triton.jit
def parallel_attn_fwd_kernel_inner(lo, hi, b_q, k, v, bos, i_h, i_v, o_q, b_o, b_acc, b_m, k_idx, T, n_lp,
                                   H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                                   BK: tl.constexpr, BS: tl.constexpr, BV: tl.constexpr, scale: tl.constexpr, stage: tl.constexpr):
    for i_s in range(lo, hi, BS):
        # p_k: [BK, BS] from [i_b, i_s, i_h, 0]
        # p_v: [BS, BV] from [i_b, i_s, i_h, i_v * BV]
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        # [BS]
        if stage == 0:
            # Recall: For past KV, we index from [i_b, past_offset]
            # bos = i_b * M + past_offset, T = M - W
            # k_idx: [B, buffer_size, H], from [i_b, 0, i_h]
            # Note: [lo, hi) is the valid range of the index of k_idx for the current Q, hi <= M - W
            # So we can use i_s < hi as the boundary mask
            p_ik = k_idx + (i_s + tl.arange(0, BS)) * H
            m_k = (i_s + tl.arange(0, BS)) < hi # No need to check left padding since k/v is aggregated
            o_k = tl.load(p_ik, mask=m_k, other=-1) # [BS], real indices of the block of kv entries, in [0, N)
            m_k = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        elif stage == 1:
            o_k = i_s + tl.arange(0, BS) # k offsets, [BS]
            m_k = (o_q[:, None] < o_k[None, :]) & (o_k >= n_lp)[None, :] # Anti-causal, left edge of SWA
        elif stage == 2:
            o_k = i_s + tl.arange(0, BS) # k offsets, [BS]
            m_k = (o_k >= n_lp)[None, :] # Inside SWA, no masking
        elif stage == 3:
            o_k = i_s + tl.arange(0, BS) # k offsets, [BS]
            m_k = (o_q[:, None] >= o_k[None, :]) & ((o_k < T) & (o_k >= n_lp))[None, :] # Causal edge

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale

        b_s = tl.where(m_k, b_s, float('-inf'))

        # [BT]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)

        m_none = b_m == float('-inf') # Handle the case when all scores are -inf => -inf - (-inf) = nan
        b_r = tl.where(m_none, 1.0, b_r)
        b_m_ = tl.where(m_none, 0.0, b_m)

        # [BT, BS]
        b_p = exp2(b_s - b_m_[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)


    return b_o, b_acc, b_m

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_PAST': lambda args: args['k_past'] is not None,
    'HAS_LEFT_OFFSETS': lambda args: args['left_offsets'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4]
        # for BS in [32, 64]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'IS_VARLEN', 'layer_idx'],
)
@triton.jit
def parallel_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    cu_seqlens,
    chunk_indices,
    k_past,
    v_past,
    kv_past_indices,
    kv_index_ub,
    left_offsets,
    T,
    NT,
    past_offset: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_PAST: tl.constexpr,
    HAS_LEFT_OFFSETS: tl.constexpr,
    layer_idx: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        n_lp = 0
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
        if HAS_LEFT_OFFSETS:
            n_lp = tl.load(left_offsets + i_n).to(tl.int32)
        else:
            n_lp = 0
    scale = scale * 1.4426950216

    # q, k, v: [B, T, HQ or H, K or V]
    # p_q: [BT, BK] from [i_b, i_t * BT, i_hq, 0]
    # p_o: [BT, BV] from [i_b, i_t * BT, i_hq, i_v * BV]
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    n_lp_c = (n_lp // BT) * BT # Start of the block of the left edge
    i_t_c = i_t * BT
    o_q = i_t_c + tl.arange(0, BT) # [BT]

    if i_t_c >= n_lp_c:
        if USE_PAST:
            # kv_past_indices: [B, buffer_size, H], from [i_b, 0, i_h]; buffer_size = M - W
            # k_past: [B, M, H, K]; v_past: [B, M, H, V]; start from past_offset
            # kv_index_ub: [B, NT, H], from [i_b, 0, i_h]
            # So for this stage we replace T with M
            hi_i = tl.load(kv_index_ub + (i_n * NT + i_t) * H + i_h).to(tl.int32) # kv_index_ub[i_b, i_t, i_h]
            bos_p = i_n * M + past_offset # For past KV, we index from [i_b, past_offset], and set T to M-W
            p_ok = kv_past_indices + i_n * (M - W) * H + i_h # shifted to kv_past_indices[i_b, 0, i_h]
            if W == -1:
                o_qw = o_q - float('inf')
            else:
                o_qw = o_q - W
            b_o, b_acc, b_m = parallel_attn_fwd_kernel_inner(
                0, hi_i, b_q, k_past, v_past, bos_p, i_h, i_v, o_qw, b_o, b_acc, b_m, p_ok,
                M - W, 0, H, K, V, BK, BS, BV, scale, 0)
        if W == -1:
            # Blocks without masking
            b_o, b_acc, b_m = parallel_attn_fwd_kernel_inner(
                n_lp_c, i_t_c, b_q, k, v, bos, i_h, i_v, o_q, b_o, b_acc, b_m, None,
                T, n_lp, H, K, V, BK, BS, BV, scale, 2)
        else:
            # Blocks on the left edge of the sliding window, with partial anti-causal masking
            if i_t_c >= W:
                o_qw = o_q - W
                b_o, b_acc, b_m = parallel_attn_fwd_kernel_inner(
                    max(n_lp_c, i_t_c - W), i_t_c - W + BT, b_q, k, v, bos, i_h, i_v, o_qw, b_o, b_acc, b_m, None,
                    T, n_lp, H, K, V, BK, BS, BV, scale, 1)
            # Blocks within the sliding window, no masking
            if W > BT:
                b_o, b_acc, b_m = parallel_attn_fwd_kernel_inner(
                    max(n_lp_c, i_t_c - W + BT), i_t_c, b_q, k, v, bos, i_h, i_v, o_q, b_o, b_acc, b_m, None,
                    T, n_lp, H, K, V, BK, BS, BV, scale, 2)
        if W == -1 or W >= BT:
            # Blocks on the causal edge
            b_o, b_acc, b_m = parallel_attn_fwd_kernel_inner(
                i_t_c, min((i_t + 1) * BT, T), b_q, k, v, bos, i_h, i_v, o_q, b_o, b_acc, b_m, None,
                T, n_lp, H, K, V, BK, BS, BV, scale, 3)

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    b_o = tl.where((o_q >= n_lp)[:, None], b_o, 0.0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))

@contiguous
def flash_lte_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    window_size: int = -1,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    k_past: Optional[torch.Tensor] = None,
    v_past: Optional[torch.Tensor] = None,
    kv_past_indices: Optional[torch.Tensor] = None,
    left_offsets: Optional[torch.Tensor] = None,
    past_offset: int = 0,
    layer_idx: int = -1, # Just to ensure different autotuning for different layers
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    BS = min(BS, chunk_size)
    if scale is None:
        scale = 1.0 / (K ** 0.5)
    if window_size != -1:
        assert window_size > 0 and window_size % BT == 0, f"window_size ({window_size}) % BT ({BT}) must be 0"
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert NK == 1, "The key dimension can not be larger than 256"

    assert (k_past is None) == (v_past is None) == (kv_past_indices is None), "k_past, v_past and kv_past_indices should be all provided or all None"
    if kv_past_indices is not None:
        kv_index_ub = prepare_kv_index_ub(kv_past_indices, B, q.shape[1], H, BT, window_size, NT)
        M = k_past.shape[1]
        assert past_offset % BT == 0, "past_offset must be multiple of BT"
        assert kv_index_ub.shape[1] == NT
    else:
        kv_index_ub = None
        M = -1

    assert not ((cu_seqlens is not None) and (k_past is not None)), "past KV not supported for varlen yet"

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    grid = (NV, NT, B * HQ)
    parallel_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        k_past=k_past,
        v_past=v_past,
        kv_past_indices=kv_past_indices,
        kv_index_ub=kv_index_ub,
        past_offset=past_offset,
        left_offsets=left_offsets,
        B=B,
        T=T,
        M=M,
        W=window_size,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        NT=NT,
        layer_idx=layer_idx
    )
    return o, lse

@torch.compile
def prepare_kv_index_ub(kv_past_indices, B, L, H, BT, window_size, NT):
    kv_index_ub_ = torch.zeros((B, NT, H), dtype=torch.int64, device=kv_past_indices.device)
    kv_index_ub = upper_bound_index(kv_past_indices, L)[:, BT - 1::BT, :] # Use the upper bound of last item in each block
    if window_size == -1:
        kv_index_ub_[:, :kv_index_ub.shape[1], :] = kv_index_ub
        if NT != kv_index_ub.shape[1]:
            kv_index_ub_[:, -1] = kv_past_indices.shape[1]
    else:
        n_window_shift = window_size // BT
        kv_index_ub_[:, n_window_shift:] = kv_index_ub[:, :-n_window_shift + (NT - kv_index_ub.shape[1]), :]
    return kv_index_ub_