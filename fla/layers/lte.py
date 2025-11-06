# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang;

#
# SPDX-FileCopyrightText: Copyright © 2023-2025, Songlin Yang, Yu Zhang
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.modules import RotaryEmbedding
from fla.modules.activations import ACT2FN
from fla.ops.lte.utils import adjust_selected, naive_enforce_buffer_size, \
    compact_lte_attention, flash_compact_lte_attention
from fla.ops.lte.utils import lte_attention, lte_attention_torch, lte_attention_naive
from fla.ops.lte.parallel import flash_lte_attention

enforce_buffer_size = True
use_fast_decode = True
use_flash_attn = True

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)

class DifferentiableMaskingFunction(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, value, score):
        ctx.save_for_backward(value, score)
        return value

    @staticmethod
    @torch.compile
    def backward(ctx, grad_outputs):
        value, score = ctx.saved_tensors
        grad_score = (grad_outputs * value).sum(dim=-1)
        return grad_outputs, grad_score

class LTEConvolutionStack(nn.Module):
    def __init__(self,
                 hidden_size: int = 2048,
                 kernel_sizes: Optional[List[int]] = None,
                 kernel_dilations: Optional[List[int]] = None,
                 group_size: int = 1,
                 dropout_prob: float = 0.0,
                 activation: Optional[str] = 'silu'):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if kernel_dilations is None:
            kernel_dilations = [1, 1, 1]
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.kernel_dilations = kernel_dilations
        self.convs = nn.ModuleList()
        self.activation = ACT2FN[activation]
        self.group_size = group_size
        size = hidden_size
        for k, d in zip(kernel_sizes, kernel_dilations):
            self.convs.append(nn.Conv2d(size, size//2, kernel_size=(1, k), dilation=(1, d),
                                        groups=group_size, padding='same'))
            size //= 2
        self.proj = nn.Conv2d(size, group_size, kernel_size=(1, 1), groups=group_size)
        self.dropout = nn.Dropout(dropout_prob)

    def reset_parameters(self):
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, a=0.0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj.weight, a=0.0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.proj.bias, 0.5)  # Use a positive initial bias to prevent sparsity at the beginning

    @torch.compile
    def forward(self, x, mask=None):         # x: (N, L, C)
        N, L, _ = x.shape
        # (N, L, C) -> (N, C, 1, L) and make it channels_last so cuDNN runs NHWC kernels directly
        x = x.permute(0, 2, 1).unsqueeze(2).contiguous(memory_format=torch.channels_last)
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x)
            if mask is not None:             # (N, L) -> (N,1,1,L)
                x = x * mask.view(N, 1, 1, L)
            x = self.activation(x)
        x = self.proj(x)                     # (N, G, 1, L)
        x = x.permute(0, 3, 1, 2).squeeze(-1) # (N, L, G)
        return x

    @staticmethod
    def get_conv_receptive_field(kernel_sizes: List[int], kernel_dilations: List[int]) -> int:
        """
        Calculate the receptive field size of the convolution stack.
        Args:
            kernel_sizes: List of kernel sizes for each convolution layer.
            kernel_dilations: List of dilation rates for each convolution layer.
        Returns:
            The total receptive field size.
        """
        assert len(kernel_sizes) == len(kernel_dilations)
        rf = 1
        for k, d in zip(kernel_sizes, kernel_dilations):
            rf += (k - 1) * d
        return rf

shared_conv_stack = None

class LTERouter(nn.Module):
    def __init__(self, hidden_size: int,
                 num_kv_heads: int,
                 kernel_sizes: Optional[List[int]] = None,
                 kernel_dilations: Optional[List[int]] = None,
                 routing_method: str = 'sigmoid',
                 routing_grad: bool = True,
                 routing_shared: str = 'head',
                 buffer_size: int = 512,
                 sink_size: int = 4,
                 dropout_prob: float = 0.0,
                 max_length: int = 4096):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if kernel_dilations is None:
            kernel_dilations = [1, 1, 1]
        assert routing_method in ['relu', 'sigmoid'], "Unsupported routing method: {}".format(routing_method)
        self.hidden_size = hidden_size * 2
        self.num_kv_heads = num_kv_heads
        self.kernel_sizes = kernel_sizes
        self.kernel_dilations = kernel_dilations
        if routing_shared == 'all':
            global shared_conv_stack
            if shared_conv_stack is None:
                shared_conv_stack = LTEConvolutionStack(self.hidden_size, kernel_sizes, kernel_dilations,
                                                        dropout_prob=dropout_prob)
            self.conv_stack = shared_conv_stack
        elif routing_shared == 'layer':
            self.conv_stack = LTEConvolutionStack(self.hidden_size, kernel_sizes, kernel_dilations,
                                                  dropout_prob=dropout_prob)
        elif routing_shared == 'head':
            self.conv_stack = LTEConvolutionStack(self.hidden_size * num_kv_heads, kernel_sizes, kernel_dilations,
                                                  group_size=num_kv_heads, dropout_prob=dropout_prob)
        self.routing_method = routing_method
        self.routing_grad = routing_grad
        self.routing_shared = routing_shared
        self.max_length = max_length

        self.add_noise = False
        self.noise_scale = 1.0

        self.buffer_size = buffer_size
        self.sink_size = sink_size
        self.enabled = True

    @torch.compile
    def forward(self,
                k: torch.Tensor,
                v: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                do_selected_adjust: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # k, v: [B, L, H, D]
        # attention_mask: [B, L]
        # offsets: []
        # Real position of k and v is the index + offsets, as the router may compute the score of a slice of the whole sequence
        x = torch.cat([k, v], dim=-1)
        B, L, H = x.shape[:3] # x: (B, L, H, D)

        if self.routing_grad:
            x_ = x
        else:
            x_ = x.detach()

        if attention_mask is not None:
            x_ = x_ * attention_mask[:, :, None, None]

        if self.routing_shared == 'head':
            x = rearrange(x_, 'b l h d -> b l (h d)')
            x = self.conv_stack(x, attention_mask) # (B, L, H)

            # outputs = [] # x: (B, L, H, D)
            # for i in range(len(self.conv_stack)):
            #     outputs.append(self.conv_stack[i](x_[:, :, i])) # o: [(B, L, 1)]
            # x = torch.cat(outputs, dim=-1) # x: (B, L, H)
        else:
            x = rearrange(x_, 'b l h d -> (b h) l d')
            x = self.conv_stack(x)
            x = rearrange(x, '(b h) l 1 -> b l h', b=B, l=L, h=H)

        if self.routing_method == 'relu':
            score = F.relu(x)
        elif self.routing_method == 'sigmoid':
            score = F.sigmoid(x) - 0.5
        else:
            raise NotImplementedError("Unsupported routing method: {}".format(self.routing_method))

        selected = score > 0
        if do_selected_adjust:
            selected = adjust_selected(selected, B, L, self.sink_size, attention_mask)
            if not self.enabled:
                selected = torch.ones_like(selected, dtype=torch.bool)

        return score, selected, k, v

class LTEAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int = 2048,
            num_heads: int = 32,
            num_kv_heads: Optional[int] = None,
            qkv_bias: bool = False,
            window_size: int = 512,
            rope_theta: Optional[float] = 10000.,
            max_position_embeddings: Optional[int] = None,
            kernel_sizes: Optional[List[int]] = None,
            kernel_dilations: Optional[List[int]] = None,
            dropout_prob: float = 0.0,
            buffer_size: int = 512,
            sink_size: int = 4,
            routing_method: str = 'relu',
            routing_grad: bool = True,
            routing_shared: str = 'head',
            layer_idx: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.buffer_size = buffer_size
        self.sink_size = sink_size
        self.dropout_prob = dropout_prob

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.router = LTERouter(self.head_dim, self.num_kv_heads, kernel_sizes, kernel_dilations,
                                routing_method=routing_method,
                                routing_grad=routing_grad, routing_shared=routing_shared,
                                buffer_size=self.buffer_size, sink_size=self.sink_size,
                                dropout_prob=dropout_prob, max_length=self.max_position_embeddings)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            backend: str = 'flex',
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Cache | None, Optional[Tuple[torch.Tensor]]]:
        if use_fast_decode and past_key_values is not None:
            return self.forward_with_cache(hidden_states, attention_mask, past_key_values,
                                           backend='flash' if use_flash_attn else backend)
        if output_attentions:
            backend = 'torch'
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, -hidden_states.shape[1]:, None]

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens', None)
        assert cu_seqlens is None

        max_seqlen = q_len
        cache_offset = padding_offset = 0
        if past_key_values is not None:
            cache_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + cache_offset

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, _ = self.rotary(q, None, seqlen_offset=cache_offset + padding_offset,
                               max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
        if attention_mask is not None:
            q = q * attention_mask[:, -q_len:, None, None]
            k = k * attention_mask[:, -q_len:, None, None]

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=None)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)


        score, selected, k, v = self.router(k, v, attention_mask=attention_mask)

        if not self.training and enforce_buffer_size:
            selected = naive_enforce_buffer_size(score, selected, self.buffer_size, self.window_size, self.sink_size, attention_mask)
        v = DifferentiableMaskingFunction.apply(v, score)

        k, _ = self.rotary(k, None, seqlen_offset=padding_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        attentions = None

        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        selected_ = rearrange(selected, 'b l h -> b h l')
        if q.dtype == torch.float32:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            cast_16 = True
        else:
            cast_16 = False

        if backend == 'torch':
            o, a_ = lte_attention_torch(q, k, v, selected_, window_size=self.window_size,
                                                       attention_mask=attention_mask,
                                                       q_offset=cache_offset)
            if output_attentions:
                attentions = a_
        elif backend == 'naive' or selected_.shape[-1] <= 128: # Otherwise there is some mysterious bug with FlexAttn
            o = lte_attention_naive(q, k, v, selected_, window_size=self.window_size,
                                         attention_mask=attention_mask, q_offset=cache_offset)
        elif backend == 'flex':
            o = lte_attention(q, k, v, selected_, window_size=self.window_size, attention_mask=attention_mask,
                                   q_offset=cache_offset)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if cast_16:
            o = o.to(torch.float32)

        o = rearrange(o, 'b h l d -> b l (h d)')

        o = self.o_proj(o)

        return o, attentions, past_key_values, (score, selected)

    @torch.inference_mode()
    def forward_with_cache(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor],
            past_key_values: Cache,
            backend: str = 'flex',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Cache, Optional[Tuple[torch.Tensor]]]:
        if not hasattr(self, 'margin'):
            self.margin = self.router.conv_stack.get_conv_receptive_field(
                self.router.kernel_sizes, self.router.kernel_dilations) // 2
        if past_key_values.get_seq_length(self.layer_idx) == 0:
            return self.forward_prefill(hidden_states, past_key_values, attention_mask, backend=backend)
        else:
            assert hidden_states.shape[1] == 1, "Hidden state has shape {}, but should be (B, 1, D)".format(hidden_states.shape)
            return self.forward_decode(hidden_states, past_key_values, attention_mask, backend=backend)

    def forward_prefill(
            self,
            hidden_states: torch.Tensor,
            past_key_values: Cache,
            attention_mask: Optional[torch.LongTensor],
            backend: str = 'flex',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Cache, Optional[Tuple[torch.Tensor]]]:
        from fla.ops.lte.cache import prepare_cache

        assert len(past_key_values.states) == self.layer_idx

        batch_size, seq_len, _ = hidden_states.size()

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, -seq_len:, None]

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        max_seqlen = seq_len

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        score, selected, k, v = self.router(k, v, attention_mask=attention_mask, do_selected_adjust=False)

        if attention_mask is not None:
            q = q * attention_mask[:, :, None, None]
            k = k * attention_mask[:, :, None, None]

        q, k = self.rotary(q, k, max_seqlen=max_seqlen)

        if self.layer_idx == 0:
            past_key_values._seen_tokens += q.shape[1]

        cache_k, cache_v, cache_score, offsets, past_lengths, blank_lengths, selected_indices = (
            prepare_cache(k, v, attention_mask, score, self.margin, self.buffer_size,
                          self.window_size + 1, self.sink_size))

        state = dict(
            attn_state=(cache_k, cache_v, cache_score, offsets, blank_lengths.to(torch.int32), 0, past_lengths),
            recurrent_state=None,
            conv_state=None,
            ffn_state=None
        )
        past_key_values.states.append(state)

        if q.dtype == torch.float32:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            cast_16 = True
        else:
            cast_16 = False

        if backend == 'flash':
            if cast_16:
                cache_k = cache_k.to(torch.float16)
                cache_v = cache_v.to(torch.float16)
            if attention_mask is not None:
                left_offsets = attention_mask.argmax(dim=1)
            else:
                left_offsets = None
            o, _ = flash_lte_attention(q, k, v, k_past=cache_k, v_past=cache_v, kv_past_indices=selected_indices,
                                       past_offset=self.window_size + 1, window_size=self.window_size + 1,
                                       left_offsets=left_offsets, layer_idx=self.layer_idx)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            q = rearrange(q, 'b l h d -> b h l d')
            k = rearrange(k, 'b l h d -> b h l d')
            v = rearrange(v, 'b l h d -> b h l d')
            selected = torch.zeros((batch_size, seq_len + 1, self.num_kv_heads), dtype=torch.bool, device=k.device) # padding in selected_indices is L
            selected = selected.scatter_(dim=1, index=selected_indices, value=True)[:, :-1]
            selected_ = rearrange(selected, 'b l h -> b h l')
            if backend == 'naive' or selected_.shape[-1] <= 128: # Otherwise there is some mysterious bug with FlexAttn
                o = lte_attention_naive(q, k, v, selected_, window_size=self.window_size,
                                             attention_mask=attention_mask)
            elif backend == 'flex':
                o = lte_attention(q, k, v, selected_, window_size=self.window_size, attention_mask=attention_mask)
            else:
                raise NotImplementedError("Unsupported backend for prefill: {}".format(backend))
            o = rearrange(o, 'b h l d -> b l (h d)')

        if cast_16:
            o = o.to(torch.float32)

        o = self.o_proj(o)

        return o, None, past_key_values, (score, selected)


    def forward_decode(
            self,
            hidden_states: torch.Tensor,
            past_key_values: Cache,
            attention_mask: Optional[torch.LongTensor], # Ignored; new q shouldn't have any padding
            backend: str = 'flex',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Cache, Optional[Tuple[torch.Tensor]]]:
        from fla.ops.lte.cache import rotate_recent, swap_recent_cache

        batch_size, q_len, _ = hidden_states.size()
        assert q_len == 1

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        cache_offset = past_key_values.get_seq_length(self.layer_idx)
        if self.layer_idx == len(past_key_values.states) - 1:
            past_key_values._seen_tokens += q_len
        cache_k, cache_v, cache_score, offsets, blank_lengths, next_recent_pos, past_lengths = (
            past_key_values.states)[self.layer_idx]['attn_state']
        max_seqlen = q_len + cache_offset

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q, k = self.rotary(q, k, seqlen_offset=cache_offset, max_seqlen=max_seqlen)

        margin = self.margin
        window_size = self.window_size + 1 # Since in SWA we also include the current token
        buffer_size = self.buffer_size

        # Step 1: Update scores in the SWA buffer
        # [0, next_recent_pos) are prefix tokens in the buffer
        # [next_recent_pos, next_recent_pos + blank_length) are blanks due to padding
        next_center = next_recent_pos + margin
        if next_center == window_size - margin:
            # print("Trigger recomputation at", cache_offset)
            if next_recent_pos > 0:
                rotate_recent(cache_k, cache_v, cache_score,
                              torch.tensor(next_recent_pos, dtype=torch.long, device=cache_k.device),
                              window_size)
                next_recent_pos = 0
            # Now next_recent_pos points to 0
            idx = torch.arange(window_size).to(blank_lengths.device).expand(batch_size, -1)  # [B, L]
            recent_attention_mask = (idx >= blank_lengths.unsqueeze(1))  # [B, L], bool
            cache_k_ = cache_k[:, :window_size]
            cache_v_ = cache_v[:, :window_size]
            cache_k_ = self.rotary.derope(cache_k_, seqlen_offset=max(cache_offset - window_size, 0))
            score, _, _, _ = self.router(cache_k_, cache_v_,
                                         attention_mask=recent_attention_mask, do_selected_adjust=False)
            score.masked_fill_(~recent_attention_mask.unsqueeze(2), float('-inf'))
            cache_score[:, margin: window_size] = score[:, margin: window_size]

        from fla.ops.lte.cache_fast import swap_recent_cache as swap_recent_cache_triton
        # Step 2: Swap the moving-out token in the SWA buffer to the past cache, and put the new token into the SWA buffer
        # swap_recent_cache(cache_k, cache_v, cache_score, blank_lengths, past_lengths,
        #                   torch.tensor(next_recent_pos, dtype=torch.long, device=cache_k.device),
        #                   k, v, buffer_size, window_size)
        swap_recent_cache_triton(cache_k, cache_v, cache_score, blank_lengths, past_lengths,
                          next_recent_pos,
                          k, v, buffer_size, window_size)
        next_recent_pos = (next_recent_pos + 1) % window_size

        past_key_values.states[self.layer_idx]['attn_state'] = (
            cache_k, cache_v, cache_score, offsets, blank_lengths, next_recent_pos, past_lengths)

        k = cache_k
        v = cache_v

        if q.dtype == torch.float32:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            cast_16 = True
        else:
            cast_16 = False

        if backend == 'flex' or (blank_lengths != 0).any():
            q = rearrange(q, 'b l h d -> b h l d')
            k = rearrange(k, 'b l h d -> b h l d')
            v = rearrange(v, 'b l h d -> b h l d')
            o = compact_lte_attention(
                q, k, v, blank_lengths,
                torch.tensor(next_recent_pos, dtype=torch.long, device=k.device), past_lengths, window_size)
            o = rearrange(o, 'b h l d -> b l (h d)')
        elif backend == 'flash':
            o = flash_compact_lte_attention(q, k, v, past_lengths, window_size)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if cast_16:
            o = o.to(torch.float32)
        o = self.o_proj(o)

        return o, None, past_key_values, (None, None)