import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from dataclasses import dataclass
from typing import Optional


# --------------------------
# activation functions

class FNNSwiGLU(nn.Module):

    def __init__(self, dim, dim_ff) -> None:
        super().__init__()

        # we will receive in xW
        self.V = nn.Linear(dim, dim_ff, bias=False)
        self.W = nn.Linear(dim, dim_ff, bias=False)


    def forward(self, x: Tensor) -> Tensor:
        """ Compute SwiGLU output of x, the output of the first linear layer. i.e.
        FFNSwiGLU(x, W, V, W2) = (Swish1(xW) ⊗ xV )W2.
        NOTE: the transformer linear1 layer must be overwritten to identity. This layer only applies
        the Swish(xW) * xV. The W2 multiplication is done in the main transformer layer
        """
        return F.silu(self.W(x)) * self.V(x)


# ---------------------------------
# padding and position layers

class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Assumes x of shape (bs, seq_len, dim) """
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)


# --------------------------------
# kv cache blocks

class CacheView:
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v

    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))

    def get_view(self, layer_id: int) -> CacheView:
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id])

    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)
        return self


# --------------------------------
# Mistral transformer blocks
# Code for the follow blocks are adapted from 
# https://github.com/mistralai/mistral-src
# Thank you Mistral team!

@dataclass
class ModelArgs:
    vocab_size: int

    dim: int = 1152 # default for mars3 and before: 1024
    n_layers: int = 24
    head_dim: int = 64 # = dim/n_heads
    hidden_dim: int = 3584
    n_heads: int = 16
    n_kv_heads: int = 16 # default: 8
    sliding_window: int = 1792
    norm_eps: float = 1e-5

    max_batch_size: int = 256


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    if repeats == 1: return keys, values
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        
        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            bias=False
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor], cache: Optional[CacheView]
    ) -> torch.Tensor:
        
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # The cache is a rotating buffer
        if cache is not None:
            scatter_pos = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
            scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
            cache.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
            cache.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])

        if positions.shape[0] > 1:
            # prefill
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(cache.cache_k[:bsz, :cur_pos, ...], cache.cache_v[:bsz, :cur_pos, ...], self.repeats)

        # print(f"Internal: {xq.shape}, key: {key.shape}, mask: {mask.shape} | {mask.dtype} | xq: {xq.dtype} | mask: {mask} ")
        # if mask is not None: 
        #     mask = mask[None, None, ...].expand(bsz, self.n_heads, -1, -1)
        #     mask = mask.to(key.dtype)

        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # # scores : [bsz, n_heads, seqlen | 1, seqlen]
        # scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        
        output = F.scaled_dot_product_attention(query, key, value, mask) # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )
        self.w2 = nn.Linear(
            args.hidden_dim,
            args.dim,
            bias=False
        )
        self.w3 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


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


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor], cache: Optional[CacheView]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class MistralTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0

        # self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False
        )

        # self.freqs_cis  
        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000)

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_embeddings.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: Optional[RotatingBufferCache]
    ):
        h = input_ids
        if self.freqs_cis.device != h.device:
            self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[positions]

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full(
                (seqlen, seqlen),
                dtype=h.dtype,
                fill_value=1,
                device=h.device,
            )
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # make the mask banded to account for sliding window
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)

        for layer_id, layer in enumerate(self.layers):
            cache_view = None if cache is None else cache.get_view(layer_id)
            h = layer(h, freqs_cis, positions, mask, cache_view)

        return self.output(self.norm(h))


