# models/fusion/registry.py
from abc import ABC, abstractmethod
from typing import Dict, Type
import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """跨模态融合基类。

    输入: tokens Dict[str, Tensor[B,N,D]]
    输出: tokens Dict[str, Tensor[B,N,D]]

    所有融合策略操作统一 token 空间，不改变 shape。
    """

    def __init__(self, modalities, dim, **kwargs):
        super().__init__()
        self.modalities = list(modalities)
        self.dim = dim

    @abstractmethod
    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...


class IdentityFusion(BaseFusion):
    """无融合 — 各模态独立通过。"""

    def forward(self, tokens):
        return tokens


class FusionRegistry:
    _fusions: Dict[str, Type[BaseFusion]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fusion_cls):
            cls._fusions[name] = fusion_cls
            return fusion_cls
        return decorator

    @classmethod
    def create(cls, name: str, modalities, dim: int, **kwargs) -> BaseFusion:
        if name == "none" or name == "identity":
            return IdentityFusion(modalities, dim)
        if name not in cls._fusions:
            raise KeyError(
                f"未知融合策略: '{name}'，可用: {list(cls._fusions.keys())}"
            )
        return cls._fusions[name](modalities=modalities, dim=dim, **kwargs)

    @classmethod
    def list_all(cls) -> list:
        return list(cls._fusions.keys()) + ["none", "identity"]
