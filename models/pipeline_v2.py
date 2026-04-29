"""MultimodalPipelineV2 — 新多模态流水线。

Flow:
    Input -> Stem(各模态独立) -> Tokenization -> InteractionBlocks
        -> Per-Modal Pooling -> Mid Fusion -> Decision -> Classifier -> logits

Config 驱动，所有组件可插拔。

V2.1: 模态Dropout + 辅助分类头，防止强模态压制弱模态。
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .tokenizer import MultiModalTokenizer
from .interaction import InteractionBlock
from .heads import MultimodalFusion
from .decision import DecisionRegistry, BaseDecision


class MultimodalPipelineV2(nn.Module):
    """新多模态流水线模型。

    V2.1 新增:
        - 模态 Dropout: 训练时随机屏蔽模态，防止强模态压制
        - 辅助分类头: 每模态独立分类头，确保 encoder 学到判别特征
    """

    def __init__(
        self,
        # Stem
        stems: Dict[str, nn.Module],
        # Tokenizer
        tokenizer: MultiModalTokenizer,
        # Interaction
        interaction_blocks: nn.ModuleList,
        # Mid Fusion
        mid_fusion: Optional[MultimodalFusion],
        mid_fusion_output_dim: int,
        # Decision
        decision: BaseDecision,
        decision_output_dim: int,
        # Classifier
        num_classes: int,
        dropout_rate: float = 0.1,
        # 模态平衡
        modality_dropout: float = 0.0,
        aux_classifiers: bool = True,
    ):
        super().__init__()
        self.stems = nn.ModuleDict(stems)
        self.tokenizer = tokenizer
        self.interaction_blocks = interaction_blocks
        self.mid_fusion = mid_fusion
        self.decision = decision
        self.modality_dropout = modality_dropout
        self.modalities = list(stems.keys())

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(decision_output_dim, num_classes),
        )

        # 辅助分类头: 每个模态一个独立分类器
        self.aux_classifiers = None
        if aux_classifiers:
            self.aux_classifiers = nn.ModuleDict({
                m: nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(mid_fusion_output_dim, num_classes),
                )
                for m in self.modalities
            })

        self.mid_fusion_output_dim = mid_fusion_output_dim

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                logits: Tensor [B, num_classes] — 融合logits
                aux_logits: Optional[Dict[str, Tensor]] — 各模态辅助logits
        """
        # 1. Stem: 各模态独立特征提取
        stem_features = {}
        for m, stem in self.stems.items():
            inputs = self._resolve_inputs(batch, m)
            if inputs is not None:
                if isinstance(inputs, torch.Tensor):
                    stem_features[m] = stem(inputs)
                else:
                    stem_features[m] = stem(**inputs)

        if not stem_features:
            raise ValueError("输入批次中没有找到与已注册 stem 匹配的模态数据")

        # 2. Tokenization: 投影到统一空间
        tokens = self.tokenizer(stem_features)

        # 2.5 模态 Dropout: 训练时随机屏蔽模态 token
        dropped_modalities = set()
        if self.training and self.modality_dropout > 0:
            for m in self.modalities:
                if m in tokens and torch.rand(1).item() < self.modality_dropout:
                    tokens[m] = torch.zeros_like(tokens[m])
                    dropped_modalities.add(m)

        # 3. Cross-Modal Interaction
        for block in self.interaction_blocks:
            tokens = block(tokens)

        # 4. Per-Modal Pooling -> [B, D] per modality
        pooled = self._pool_tokens(tokens)

        # 5. Mid Fusion
        if len(pooled) == 1:
            fused = list(pooled.values())[0]
        elif self.mid_fusion is not None:
            fused = self.mid_fusion(list(pooled.values()))
        else:
            fused = torch.cat(list(pooled.values()), dim=1)

        # 6. Decision
        decision_out, aux = self.decision(fused, tokens)

        # 7. Classifier
        logits = self.classifier(decision_out)

        # 8. 辅助分类头 logits (仅训练时有效)
        aux_logits = None
        if self.aux_classifiers is not None and self.training:
            aux_logits = {}
            for m, pooled_m in pooled.items():
                aux_logits[m] = self.aux_classifiers[m](pooled_m)

        return logits, aux_logits

    def _pool_tokens(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """每个模态 token 序列池化为一个特征向量 [B, D]"""
        return {m: t.mean(dim=1) for m, t in tokens.items()}

    @staticmethod
    def _resolve_inputs(batch: Dict[str, Any], modality: str):
        """兼容新旧 batch 格式"""
        if modality in batch:
            return batch[modality]
        prefix = f"{modality}_"
        prefixed = {
            key[len(prefix):]: value
            for key, value in batch.items()
            if key.startswith(prefix)
        }
        return prefixed if prefixed else None
