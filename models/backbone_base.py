"""Backbone 基类接口。

定义了两种 backbone 类型：
- BaseBackbone: 标准 backbone，只支持完整 forward
- StageableBackbone: 支持逐 stage 前向传播，用于多模态中期融合

State 布局约定（StageFusionAdapter 依赖）:
    image/audio: Tensor [B, C, H, W]
    wave:        Tensor [B, C, T]
    text:        dict {"x": [B, L, C], "attention_mask": [B, L]}
"""

from typing import Any, Dict, List
import torch
import torch.nn as nn


class BaseBackbone(nn.Module):
    """所有 backbone 的公共接口。

    Attributes:
        feature_dim: 输出特征维度
    """

    feature_dim: int

    def forward(self, *args, **inputs) -> torch.Tensor:
        """完整前向传播，返回 [B, feature_dim] 特征向量。"""
        raise NotImplementedError

    def tokenize(self, *args, **inputs) -> Dict[str, torch.Tensor]:
        """可选的 token 化输出。

        默认实现：调用 forward() 得到 [B, D]，包装为单 token [B, 1, D]。
        子类可重写以输出更丰富的 token 序列（如 feature map flatten）。

        Returns:
            Dict with keys:
                "tokens": Tensor [B, N, D] — token 序列
                "layout": str — "1d" (temporal) | "2d" (spatial) | "scalar" (single)
        """
        x = inputs.pop('x', None)
        if x is not None:
            inputs.setdefault('x', x)
        feat = self.forward(**inputs)  # [B, D]
        return {"tokens": feat.unsqueeze(1), "layout": "scalar"}  # [B, 1, D]


class StageableBackbone(BaseBackbone):
    """支持逐 stage 前向传播的 backbone。

    子类需要定义:
        num_stages:  总 stage 数（类属性）
        stage_dims:  每个 stage 输出通道数（实例属性，List[int]）

    子类需要实现:
        init_state(**inputs) -> state
        forward_stage(state, stage_idx) -> state
        forward_head(state) -> [B, feature_dim]

    forward() 已提供默认实现:
        state = init_state(**inputs)
        for i in range(num_stages):
            state = forward_stage(state, i)
        return forward_head(state)
    """

    num_stages: int

    def __init__(self) -> None:
        super().__init__()
        self.stage_dims: List[int] = []

    def init_state(self, **inputs) -> Any:
        """将原始输入转换为初始 state。"""
        raise NotImplementedError

    def forward_stage(self, state: Any, stage_idx: int) -> Any:
        """执行单个 stage，返回新的 state。"""
        raise NotImplementedError

    def forward_head(self, state: Any) -> torch.Tensor:
        """所有 stage 执行完毕后，pool + project 输出最终特征向量 [B, feature_dim]。"""
        raise NotImplementedError

    def forward(self, x=None, **inputs) -> torch.Tensor:
        if x is not None:
            inputs.setdefault('x', x)
        state = self.init_state(**inputs)
        for i in range(self.num_stages):
            state = self.forward_stage(state, i)
        return self.forward_head(state)
