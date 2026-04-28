"""Cross-Modal Interaction — 可堆叠的 InteractionBlock。

每个 Block = SharedTransform（共享参数，各模态独立应用） + Fusion（跨模态交换）。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .fusion.registry import FusionRegistry, BaseFusion


class TransformerBlock(nn.Module):
    """单个 Transformer Encoder Block，batch_first=True。"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SharedTransform(nn.Module):
    """共享变换 — 所有模态 token 经过同一个网络层。

    约定：对每个模态独立调用同一组参数。这意味着所有模态在同一个特征空间中
    以相同规则被处理，但不直接交换信息（交换信息由 Fusion 完成）。
    """

    def __init__(self, block_type: str, dim: int, **block_kwargs):
        super().__init__()
        if block_type == "transformer":
            self.block = TransformerBlock(dim=dim, **block_kwargs)
        else:
            raise ValueError(f"未知 block_type: {block_type}")

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.block(t) for m, t in tokens.items()}


class InteractionBlock(nn.Module):
    """一个跨模态交互块。

    执行顺序:
        1. SharedTransform: 各模态 token 经过共享网络独立变换
        2. Fusion:         跨模态信息交换

    Config 粒度：每个 block 可独立选择 transform_type 和 fusion_type。
    """

    def __init__(
        self,
        modalities,
        dim: int,
        transform_type: str = "transformer",
        transform_kwargs: Optional[dict] = None,
        fusion_type: str = "none",
        fusion_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.transform = SharedTransform(
            block_type=transform_type,
            dim=dim,
            **(transform_kwargs or {}),
        )
        self.fusion = FusionRegistry.create(
            fusion_type,
            modalities=modalities,
            dim=dim,
            **(fusion_kwargs or {}),
        )

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = self.transform(tokens)
        tokens = self.fusion(tokens)
        return tokens
