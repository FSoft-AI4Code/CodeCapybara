# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

# import fairscale.nn.model_parallel.initialize as fs_init
# from fairscale.nn.model_parallel.layers import (
#     ParallelEmbedding,
#     nn.Linear,
#     nn.Linear,
# )

@dataclass
class ModelArgs:
    dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    adapter_len: int=10
    adapter_layers: int=8


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.num_heads#// fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim_size // args.num_heads
        self.wq = nn.Linear(
            args.dim_size,
            args.num_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim_size,
            args.num_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim_size,
            args.num_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.num_heads * self.head_dim,
            args.dim_size,
            bias=False
        )

        # self.cache_k = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()
        # self.gate = torch.nn.Parameter(torch.zeros(1, requires_grad = True))
        # self.gate.register_hook(lambda x: print(x))
            
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, cache = {}):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # if not self.training:
        #     # print('EVAL MODE')
        #     if not hasattr(self, 'cache_k'):
        #         self.cache_k = torch.zeros(
        #         (1, 550, self.n_local_heads, self.head_dim)
        #     ).cuda()
        #         self.cache_v = torch.zeros(
        #             (1, 550, self.n_local_heads, self.head_dim)
        #         ).cuda()

        #     self.cache_k = self.cache_k.to(xq)
        #     self.cache_v = self.cache_v.to(xq)

        #     self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        #     self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        #     keys = self.cache_k[:bsz, : start_pos + seqlen]
        #     values = self.cache_v[:bsz, : start_pos + seqlen]
        # else:
        if not self.training:
            if start_pos == 0:
                cache['cache_k'] = xk
                cache['cache_v'] = xv
            else:
                # print( cache['cache_v'].shape, xk.shape)
                if cache['cache_k'].shape[0] != xk.shape[0]:
                    cache['cache_k'] = torch.tile(cache['cache_k'], (xk.shape[0], 1, 1, 1))
                    cache['cache_v'] = torch.tile(cache['cache_v'], (xv.shape[0], 1, 1, 1))
                xk = torch.cat((cache['cache_k'], xk), dim = 1)
                xv = torch.cat((cache['cache_v'], xv), dim = 1)
                cache['cache_k'] = xk
                cache['cache_v'] = xv

        keys = xk#self.cache_k[:bsz, : start_pos + seqlen]
        values = xv#self.cache_v[:bsz, : start_pos + seqlen]

        if adapter is not None:
           adapter_len = adapter.shape[1]
           adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
           adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
           adapter_k = adapter_k.transpose(1, 2)
           adapter_v = adapter_v.transpose(1, 2)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # print('scores', scores[0, 0])
        # print(torch.topk(scores[:, 0], k = min(scores.shape[-1], 5), dim =-1))
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        if adapter is not None:
            adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            adapter_scores = self.gate * F.softmax(adapter_scores.float(), dim=-1).type_as(xq)
            output = output + torch.matmul(adapter_scores, adapter_v)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_heads
        self.dim_size = args.dim_size
        self.head_dim = args.dim_size // args.num_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim_size, hidden_dim=4 * args.dim_size, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim_size, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim_size, eps=args.norm_eps)
        self.cache = {}

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, cache = self.cache)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLaMA(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.num_layers = params.num_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim_size
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim_size, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim_size, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim_size // self.params.num_heads, self.params.max_seq_len * 2
        )
        # self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layers, params.dim_size)
        self.adapter_len = params.adapter_len
        self.adapter_layers = params.adapter_layers
    def forward(self, tokens: torch.Tensor, start_pos: int, mask: torch.Tensor):
        if self.training:
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
            # prompt = self.adapter_query.weight.reshape(self.params.adapter_layers, self.params.adapter_len, self.params.dim_size).unsqueeze(1)
            # mask = None
            # if seqlen > 1:
            #     mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                # mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            for layer in self.layers[: -1 * self.params.adapter_layers]:
                h = layer(h, start_pos, freqs_cis, mask)
            layer_index = 0
            for layer in self.layers[-1 * self.params.adapter_layers:]:
                h = layer(h, start_pos, freqs_cis, mask)# prompt[layer_index])
                layer_index = layer_index + 1
            h = self.norm(h)
            output = self.output(h)  # only compute last logits
            return output.float().transpose(1, 2)
        else:
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
            # prompt = self.adapter_query.weight.reshape(self.params.adapter_layers, self.params.adapter_len, self.params.dim_size).unsqueeze(1)
            mask = None
            if seqlen > 1:
                mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            for layer in self.layers[: -1 * self.params.adapter_layers]:
                h = layer(h, start_pos, freqs_cis, mask)
            layer_index = 0
            for layer in self.layers[-1 * self.params.adapter_layers:]:
                h = layer(h, start_pos, freqs_cis, mask)# prompt[layer_index])
                layer_index = layer_index + 1
            h = self.norm(h)
            output = self.output(h[:, -1, :])  # only compute last logits
            return output.float()
    def enable_weights(self, m):
        if isinstance(m, Attention):
            for name, w in m.named_parameters():
                if 'gate' in name:
                    # pass
                    w.requires_grad = True
        elif isinstance(m, LLaMA):
            for name, w in m.named_parameters():
                if 'adapter_query' in name:
                     w.requires_grad = True
