# models/fusion/strategies.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaseFusion, FusionRegistry, IdentityFusion


@FusionRegistry.register("gate")
class GateInjectionFusion(BaseFusion):
    """门控注入融合。

    各模态 token 池化 → 投影到公共空间 → 平均 → 门控注入回各模态。

    Extra kwargs:
        gate_hidden_dim: int = None  # 默认 dim // 2
        dropout: float = 0.0
    """

    def __init__(self, modalities, dim, gate_hidden_dim=None, dropout=0.0, **kwargs):
        super().__init__(modalities, dim)
        h = gate_hidden_dim or max(dim // 2, 32)

        self.to_common = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU())
            for m in modalities
        })
        self.to_injection = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(h, dim), nn.Tanh())
            for m in modalities
        })
        self.gate = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(h, dim), nn.Sigmoid())
            for m in modalities
        })
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, tokens):
        # 各模态池化到 [B, dim]
        pooled = {}
        for m in self.modalities:
            x = tokens[m]
            if x.dim() == 3:
                pooled[m] = x.mean(dim=1)
            elif x.dim() == 2:
                pooled[m] = x
            else:
                pooled[m] = x.flatten(1)

        # 投影到公共空间
        projected = [self.to_common[m](pooled[m]) for m in self.modalities]
        fused = torch.stack(projected, dim=0).mean(dim=0)  # [B, h]
        fused = self.dropout(fused)

        # 门控注入回各模态
        out = {}
        for m in self.modalities:
            inj = self.to_injection[m](fused)  # [B, dim]
            g = self.gate[m](fused)            # [B, dim]
            delta = g * inj

            x = tokens[m]
            if x.dim() == 3:
                delta = delta.unsqueeze(1)
            out[m] = x + delta
        return out


@FusionRegistry.register("cross_attn")
class CrossAttentionFusion(BaseFusion):
    """跨模态注意力融合。

    每个模态用 cross-attention 从其他所有模态拉信息。
    Q = 当前模态, K/V = 所有其他模态 token concat。

    Extra kwargs:
        num_heads: int = 8
        dropout: float = 0.0
    """

    def __init__(self, modalities, dim, num_heads=8, dropout=0.0, **kwargs):
        super().__init__(modalities, dim)
        self.num_heads = num_heads

        self.cross_attns = nn.ModuleDict({
            m: nn.MultiheadAttention(
                dim, num_heads, dropout=dropout,
                batch_first=True,
            )
            for m in modalities
        })
        self.norms = nn.ModuleDict({
            m: nn.LayerNorm(dim) for m in modalities
        })

    def forward(self, tokens):
        out = {}
        for m in self.modalities:
            # Q: 当前模态 token
            q = tokens[m]  # [B, N_m, D]

            # K/V: 所有其他模态 token 拼接
            kv_list = [tokens[o] for o in self.modalities if o != m]
            if not kv_list:
                out[m] = tokens[m]
                continue
            kv = torch.cat(kv_list, dim=1)  # [B, N_other, D]

            attn_out, _ = self.cross_attns[m](
                query=q, key=kv, value=kv,
                need_weights=False,
            )
            out[m] = self.norms[m](tokens[m] + attn_out)
        return out


@FusionRegistry.register("token_mix")
class TokenMixerFusion(BaseFusion):
    """全 Token 混合融合。

    把所有模态 token concat → Self-Attention → split 回各模态。

    Extra kwargs:
        num_heads: int = 8
        dropout: float = 0.0
    """

    def __init__(self, modalities, dim, num_heads=8, dropout=0.0, **kwargs):
        super().__init__(modalities, dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens):
        # Concat 所有模态 token
        token_list = [tokens[m] for m in self.modalities]
        sizes = [t.shape[1] for t in token_list]
        x = torch.cat(token_list, dim=1)  # [B, N_total, D]

        # Self-Attention
        x = x + self.attn(self.norm(x), self.norm(x), self.norm(x))[0]
        x = x + self.mlp(self.norm2(x))

        # Split 回各模态
        out = {}
        start = 0
        for m, n in zip(self.modalities, sizes):
            out[m] = x[:, start:start + n, :]
            start += n
        return out
