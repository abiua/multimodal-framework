"""Tokenization 层 — 将各模态 stem 特征投影到统一 token 空间 [B, N, D]。

组件:
    ModalityProjection: C_modality → D（统一维度）
    ModalityEmbedding:   可学习模态身份嵌入
    PositionEncoding:    可学习位置编码（时间/空间）
    MultiModalTokenizer: 组合以上组件的完整 Tokenization 层
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ModalityProjection(nn.Module):
    """将各模态异构特征投影到统一维度 D。"""

    def __init__(self, feature_dims: Dict[str, int], unified_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(feature_dims[m]),
                nn.Linear(feature_dims[m], unified_dim),
                nn.GELU(),
            )
            for m in feature_dims
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """features: {modality: Tensor[B, D_m]} → tokens: {modality: Tensor[B, 1, D]}"""
        return {
            m: self.projections[m](feat).unsqueeze(1)  # [B, 1, D]
            for m, feat in features.items()
        }


class ModalityEmbedding(nn.Module):
    """可学习的模态身份嵌入 — 加到每个模态的 token 上。

    用法:
        tokens[m] = tokens[m] + modal_emb(m)  # broadcast 到所有 token
    """

    def __init__(self, modalities: List[str], dim: int):
        super().__init__()
        self.embeddings = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(1, 1, dim))
            for m in modalities
        })
        for p in self.embeddings.values():
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: t + self.embeddings[m] for m, t in tokens.items()}


class PositionEncoding(nn.Module):
    """可学习位置编码。

    按模态配置是否启用和最大长度。

    Config 示例:
        pe_configs = {
            "image": {"enabled": True, "max_len": 197},   # 14x14 patches + 1
            "audio": {"enabled": True, "max_len": 197},
            "wave":  {"enabled": True, "max_len": 512},
        }
    """

    def __init__(
        self,
        modalities: List[str],
        dim: int,
        pe_configs: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        self.pe = nn.ParameterDict()
        pe_configs = pe_configs or {}
        for m in modalities:
            cfg = pe_configs.get(m, {})
            if cfg.get("enabled", True):
                max_len = cfg.get("max_len", 256)
                self.pe[m] = nn.Parameter(torch.zeros(1, max_len, dim))
                nn.init.trunc_normal_(self.pe[m], std=0.02)
            else:
                self.pe[m] = None

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m, t in tokens.items():
            if self.pe.get(m) is not None:
                n = t.shape[1]
                out[m] = t + self.pe[m][:, :n, :]
            else:
                out[m] = t
        return out


class MultiModalTokenizer(nn.Module):
    """完整的 Tokenization 层。

    输入: 各模态 stem 特征 {modality: Tensor[B, D_m]}
    输出: 统一 token 序列   {modality: Tensor[B, N_m, D]}

    流程:
        features → ModalityProjection → ModalityEmbedding → PositionEncoding → tokens
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        unified_dim: int,
        modalities: List[str],
        pe_configs: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        self.projection = ModalityProjection(feature_dims, unified_dim)
        self.modal_emb = ModalityEmbedding(modalities, unified_dim)
        self.pos_enc = PositionEncoding(modalities, unified_dim, pe_configs)
        self.unified_dim = unified_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = self.projection(features)       # [B, 1, D] per modality
        tokens = self.modal_emb(tokens)          # + modality embedding
        tokens = self.pos_enc(tokens)            # + position encoding
        return tokens
