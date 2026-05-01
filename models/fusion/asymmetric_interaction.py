"""Asymmetric Interaction — Video 单向查询 Physical tokens。

Video → Physical cross-attention (单向): Video tokens 从 Physical tokens 拉取信息，
Physical tokens 保持不变（不被视频信号污染）。

Evidence Gate: Video tokens → scalar [B, 1] evidence score per sample.
"""
import torch
import torch.nn as nn


class AsymmetricVideoPhysicalBlock(nn.Module):
    """Video → Physical 单向 cross-attention block.

    Q = video, K/V = physical. Physical unchanged.
    """

    def __init__(self, dim=256, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_v = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )

    def forward(self, visual_tokens, physical_tokens):
        attn_out, _ = self.attn(self.norm_v(visual_tokens),
                                self.norm_p(physical_tokens),
                                self.norm_p(physical_tokens), need_weights=False)
        x = visual_tokens + attn_out
        return x + self.mlp(self.norm_out(x)), physical_tokens


class AsymmetricInteraction(nn.Module):
    """可堆叠的 Asymmetric Interaction 模块。

    Args:
        dim: token dimension
        num_blocks: number of stacked AsymmetricVideoPhysicalBlock
        num_heads: attention heads
        dropout: dropout rate
    """

    def __init__(self, dim=256, num_blocks=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            AsymmetricVideoPhysicalBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, visual_tokens, physical_tokens):
        v, p = visual_tokens, physical_tokens
        for block in self.blocks:
            v, p = block(v, p)
        return v, p


class EvidenceGate(nn.Module):
    """Video tokens → scalar evidence score per sample.

    High evidence: clear visual signal (splashes, surface agitation)
    Low evidence: occluded view (fish underwater), rely on physical signals
    """

    def __init__(self, dim=256, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, visual_tokens):
        return self.mlp(visual_tokens.mean(dim=1))
