import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
if XFORMERS_AVAILABLE:
    from xformers.ops import memory_efficient_attention
else:
    memory_efficient_attention = None

from .. import env
from ..utils import compile_wrapper
from .axial_rope import AxialRoPE


if not env.USE_XFORMERS:
    memory_efficient_attention = None
if env.USE_VANILLA:

    @compile_wrapper
    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0):
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(-1)
        attn = F.dropout(attn, p)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()


class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads=8, head_dim=-1, pos_dim=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim > 0 else dim // n_heads
        self.n_heads = dim // self.head_dim
        assert (
            self.n_heads * self.head_dim == dim
        ), "dim must be divisible by n_heads or head_dim"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.rope = AxialRoPE(self.head_dim, self.n_heads, pos_dim)
        self.attn = memory_efficient_attention or F.scaled_dot_product_attention
        self.xformers = memory_efficient_attention is not None

    def forward(self, x, pos_map=None, mask=None):
        b, n, _, h = *x.shape, self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        if pos_map is not None:
            q = self.rope(q.reshape(b, n, h, -1).transpose(1, 2), pos_map)
            k = self.rope(k.reshape(b, n, h, -1).transpose(1, 2), pos_map)
            v = v.reshape(b, n, h, -1)
            if self.xformers:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
            else:
                v = v.transpose(1, 2)
        else:
            q, k, v = map(lambda t: t.reshape(b, n, h, -1), (q, k, v))
            if not self.xformers:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if n % 8 and self.xformers:
                align_n = math.ceil(n / 8) * 8
                mask_align = torch.empty(
                    *mask.shape[:3], align_n, device=mask.device, dtype=mask.dtype
                )
                mask_align[..., :n] = mask
                mask = mask_align.to(q).expand(b, h, n, align_n)[..., :n]
            else:
                mask = mask.to(q).expand(b, h, n, n)

        attn = self.attn(q, k, v, mask)
        if not self.xformers:
            attn = attn.transpose(1, 2)
        attn = attn.reshape(b, n, h * self.head_dim)
        attn = self.out(attn)
        return attn


class CrossAttention(nn.Module):
    def __init__(self, dim, ctx_dim, n_heads=8, head_dim=-1, pos_dim=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim > 0 else dim // n_heads
        self.n_heads = dim // self.head_dim
        assert (
            self.n_heads * self.head_dim == dim
        ), "dim must be divisible by n_heads or head_dim"

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim)
        self.rope = AxialRoPE(self.head_dim, self.n_heads, pos_dim)
        self.attn = memory_efficient_attention or F.scaled_dot_product_attention
        self.xformers = memory_efficient_attention is not None

    def forward(self, x, ctx, pos_map=None, ctx_pos_map=None, mask=None):
        b, n, _, h = *x.shape, self.n_heads
        ctx_n = ctx.shape[1]
        q = self.q(x)
        k, v = self.kv(ctx).chunk(2, dim=-1)

        if pos_map is not None:
            q = self.rope(q.reshape(b, n, h, -1).transpose(1, 2), pos_map)
            q = q if not self.xformers else q.transpose(1, 2)
        else:
            q = q.reshape(b, n, h, -1)
            q = q if self.xformers else q.transpose(1, 2)
        if ctx_pos_map is not None:
            k = self.rope(k.reshape(b, ctx_n, h, -1).transpose(1, 2), ctx_pos_map)
            k = k if not self.xformers else k.transpose(1, 2)
        else:
            k = k.reshape(b, ctx_n, h, -1)
            k = k if self.xformers else k.transpose(1, 2)
        v = v.reshape(b, ctx_n, h, -1)
        v = v if self.xformers else v.transpose(1, 2)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if ctx_n % 8 and self.xformers:
                align_n = math.ceil(ctx_n / 8) * 8
                mask_align = torch.empty(
                    *mask.shape[:3], align_n, device=mask.device, dtype=mask.dtype
                )
                mask_align[..., :ctx_n] = mask
                mask = mask_align.to(q).expand(b, h, n, align_n)[..., :ctx_n]
            else:
                mask = mask.to(q).expand(b, h, n, ctx_n)

        attn = self.attn(q, k, v, mask)
        if not self.xformers:
            attn = attn.transpose(1, 2)
        attn = attn.reshape(b, n, h * self.head_dim)
        attn = self.out(attn)
        return attn
