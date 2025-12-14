import torch
import torch.nn as nn

from .layers import SwiGLU
from .attention import SelfAttention, CrossAttention
from .norm import RMSNorm
from .adaln import AdaLN


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ctx_dim,
        heads,
        dim_head,
        mlp_dim,
        pos_dim,
        use_adaln=False,
        use_shared_adaln=False,
        ctx_from_self=False,
        self_ctx_mode="xut",
        norm_layer=RMSNorm,
    ):
        super().__init__()
        self.use_adaln = use_adaln
        self.attn = SelfAttention(dim, heads, dim_head, pos_dim)
        self.alpha = self.xattn = self.xattn_pre_norm = self.mixer = self.ctx_proj = (
            None
        )
        self.ctx_from_self = ctx_from_self
        self.self_ctx_mode = self_ctx_mode
        if ctx_dim is not None:
            self.ctx_from_self = ctx_from_self
            if not ctx_from_self or self_ctx_mode == "xut":
                print("XUT self-ctx mode")
                # XUT self-ctx mode or simple xattn
                self.xattn = CrossAttention(dim, ctx_dim, heads, dim_head, pos_dim)
            elif self_ctx_mode == "concat-linear":
                print("Concat linear self-ctx mode")
                self.mixer = nn.Linear(ctx_dim + dim, dim)
            elif self_ctx_mode in {"addition", "interpolation"}:
                if ctx_dim != dim:
                    self.ctx_proj = nn.Linear(ctx_dim, dim)
                else:
                    self.ctx_proj = nn.Identity()
                if self_ctx_mode == "interpolation":
                    print("Interpolation self-ctx mode")
                    self.alpha = nn.Parameter(torch.zeros(1))
                else:
                    print("Addition self-ctx mode")
                    self.alpha = None
            elif self_ctx_mode == "ignore":
                pass
            else:
                raise NotImplementedError(f"Unknown self_ctx_mode: {self_ctx_mode}")
        self.mlp = SwiGLU(dim, mlp_dim, dim)

        if self.use_adaln:
            self.attn_pre_norm = AdaLN(
                dim, dim, norm_layer=norm_layer, shared=use_shared_adaln
            )
            self.mlp_pre_norm = AdaLN(
                dim, dim, norm_layer=norm_layer, shared=use_shared_adaln
            )
            if self.xattn is not None:
                self.xattn_pre_norm = AdaLN(
                    dim, dim, norm_layer=norm_layer, shared=use_shared_adaln
                )
        else:
            self.attn_pre_norm = norm_layer(dim)
            self.mlp_pre_norm = norm_layer(dim)
            if self.xattn is not None:
                self.xattn_pre_norm = norm_layer(dim)

    def forward(
        self,
        x,
        ctx,
        pos_map=None,
        ctx_pos_map=None,
        y=None,
        x_mask=None,
        ctx_mask=None,
        shared_adaln=None,
    ):
        if self.mixer is not None:
            x = self.mixer(torch.cat([x, ctx], dim=-1))
        if self.ctx_proj is not None:
            ctx = self.ctx_proj(ctx)
            if self.alpha is None:
                x = x + ctx
            else:
                x = torch.lerp(x, ctx, self.alpha.to(x))

        y = [y] if y is not None else []
        y = y if shared_adaln is None else [y[0], shared_adaln[0]]
        x, gate = self.attn_pre_norm(x, *y)
        x = x + self.attn(x, pos_map, mask=x_mask) * gate

        if self.xattn is not None:
            if shared_adaln is not None:
                y[1] = shared_adaln[1]
            x, gate = self.xattn_pre_norm(x, *y)
            if self.ctx_from_self:
                ctx_mask = x_mask
            x = x + self.xattn(x, ctx, pos_map, ctx_pos_map, mask=ctx_mask) * gate

        if shared_adaln is not None:
            y[1] = shared_adaln[-1]
        x, gate = self.mlp_pre_norm(x, *y)
        x = x + self.mlp(x) * gate
        return x
