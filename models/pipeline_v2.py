"""MultimodalPipelineV2 -- 新多模态流水线。

Flow:
    Input -> Stem(各模态独立) -> Tokenization -> InteractionBlocks
        -> Per-Modal Pooling -> Mid Fusion -> Decision -> Classifier -> logits

Config 驱动，所有组件可插拔。
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from .tokenizer import MultiModalTokenizer
from .interaction import InteractionBlock
from .heads import MultimodalFusion
from .decision import DecisionRegistry, BaseDecision


class MultimodalPipelineV2(nn.Module):
    """新多模态流水线模型。

    Config 示例见 configs/default_multimodal_v2.yaml。
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
    ):
        super().__init__()
        self.stems = nn.ModuleDict(stems)
        self.tokenizer = tokenizer
        self.interaction_blocks = interaction_blocks
        self.mid_fusion = mid_fusion
        self.decision = decision

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(decision_output_dim, num_classes),
        )

        self.mid_fusion_output_dim = mid_fusion_output_dim

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                "logits": Tensor [B, num_classes]
                "aux": Optional[Dict] -- 来自 decision 的辅助信息
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

        return {"logits": logits, "aux": aux}

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
