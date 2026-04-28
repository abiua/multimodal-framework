# tests/test_pipeline_v2.py
"""MultimodalPipelineV2 端到端测试"""
import torch
import pytest

from utils.config import load_config
from models.builder import ModelBuilder
from models.pipeline_v2 import MultimodalPipelineV2
from models.tokenizer import MultiModalTokenizer
from models.interaction import InteractionBlock
from models.decision import DecisionRegistry
from models.heads import MultimodalFusion


class TestMultimodalPipelineV2:
    def test_minimal_pipeline_build(self):
        """最小流水线：1模态，无fusion，无interaction"""
        stems = {"image": _DummyStem(512)}
        tokenizer = MultiModalTokenizer(
            feature_dims={"image": 512}, unified_dim=256,
            modalities=["image"],
        )
        model = MultimodalPipelineV2(
            stems=stems, tokenizer=tokenizer,
            interaction_blocks=torch.nn.ModuleList([]),
            mid_fusion=None, mid_fusion_output_dim=256,
            decision=DecisionRegistry.create("identity", in_dim=256),
            decision_output_dim=256, num_classes=3,
        )
        batch = {"image": torch.randn(2, 3, 224, 224)}
        out = model(batch)
        assert out["logits"].shape == (2, 3)
        assert out["aux"] is None

    def test_three_modality_pipeline(self):
        """3模态 + 4个interaction blocks + attention fusion"""
        stems = {
            "image": _DummyStem(512),
            "audio": _DummyStem(512),
            "wave": _DummyStem(256),
        }
        tokenizer = MultiModalTokenizer(
            feature_dims={"image": 512, "audio": 512, "wave": 256},
            unified_dim=256, modalities=["image", "audio", "wave"],
        )
        blocks = torch.nn.ModuleList([
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="none"),
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="gate",
                             fusion_kwargs={"gate_hidden_dim": 128}),
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="cross_attn",
                             fusion_kwargs={"num_heads": 4}),
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="token_mix",
                             fusion_kwargs={"num_heads": 4}),
        ])
        mid_fusion = MultimodalFusion(
            feature_dims=[256, 256, 256], output_dim=256,
            fusion_type="attention",
        )
        model = MultimodalPipelineV2(
            stems=stems, tokenizer=tokenizer, interaction_blocks=blocks,
            mid_fusion=mid_fusion, mid_fusion_output_dim=256,
            decision=DecisionRegistry.create("identity", in_dim=256),
            decision_output_dim=256, num_classes=5,
        )
        batch = {
            "image": torch.randn(4, 3, 224, 224),
            "audio": torch.randn(4, 2, 224, 224),
            "wave": torch.randn(4, 512, 6),
        }
        out = model(batch)
        assert out["logits"].shape == (4, 5)

    def test_forward_backward(self):
        """验证梯度可回传"""
        stems = {
            "image": torch.nn.Linear(512, 512),
            "audio": torch.nn.Linear(512, 512),
        }
        tokenizer = MultiModalTokenizer(
            feature_dims={"image": 512, "audio": 512},
            unified_dim=256, modalities=["image", "audio"],
        )
        blocks = torch.nn.ModuleList([
            InteractionBlock(["image","audio"], dim=256, fusion_type="cross_attn",
                             fusion_kwargs={"num_heads": 4}),
        ])
        model = MultimodalPipelineV2(
            stems=stems, tokenizer=tokenizer, interaction_blocks=blocks,
            mid_fusion=None, mid_fusion_output_dim=512,
            decision=DecisionRegistry.create("identity", in_dim=512),
            decision_output_dim=512, num_classes=3,
        )
        batch = {
            "image": torch.randn(2, 512),
            "audio": torch.randn(2, 512),
        }
        out = model(batch)
        loss = out["logits"].sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no grad"

    def test_old_config_still_works(self):
        """旧配置仍走旧路径"""
        cfg = load_config("configs/fish_feeding_unireplknet.yaml")
        assert cfg.model.unified_pipeline is None
        model = ModelBuilder.build_model(cfg)
        from models.builder import MultimodalClassifier
        assert isinstance(model, MultimodalClassifier)


class _DummyStem(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_dim = dim
    def forward(self, x):
        return torch.randn(x.shape[0], self.feature_dim)
