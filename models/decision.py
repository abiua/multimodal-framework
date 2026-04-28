"""Decision 模块 — 预留接口，支持未来证据/置信度融合。

当前提供:
    - BaseDecision: 抽象基类，定义标准接口
    - IdentityDecision: 默认实现，不做任何处理

未来可继承 BaseDecision 实现:
    - EvidenceDecision: Dirichlet evidence fusion
    - UncertaintyDecision: MC Dropout / Deep Ensemble
    - GraphReasoningDecision: 图推理决策
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseDecision(nn.Module, ABC):
    """决策模块抽象基类。

    forward 返回:
        feature: Tensor [B, D] — 决策后的特征
        aux: Optional[Dict] — 辅助信息（如 uncertainty, evidence, attention weights）
    """

    @abstractmethod
    def forward(
        self, fused_feature: torch.Tensor, tokens: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            fused_feature: Mid Fusion 后的特征 [B, D_fused]
            tokens:        各模态 token 序列（可选，供复杂决策使用）
        Returns:
            (decision_feature [B, D_reason], aux dict or None)
        """
        ...


class IdentityDecision(BaseDecision):
    """默认决策模块 — 透传不做任何处理。"""

    def __init__(self, in_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, fused_feature, tokens=None):
        return self.proj(fused_feature), None


# Decision Registry — 为未来扩展准备
class DecisionRegistry:
    _decisions: Dict[str, type] = {"identity": IdentityDecision}

    @classmethod
    def register(cls, name: str):
        def decorator(decision_cls):
            cls._decisions[name] = decision_cls
            return decision_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDecision:
        if name not in cls._decisions:
            raise KeyError(f"未知决策模块: '{name}'，可用: {list(cls._decisions.keys())}")
        return cls._decisions[name](**kwargs)
