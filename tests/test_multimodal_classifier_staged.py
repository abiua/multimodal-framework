import unittest
import torch
import torch.nn as nn

from models.builder import MultimodalClassifier
from models.backbone_base import StageableBackbone


class ToyStageableBackbone(StageableBackbone):
    num_stages = 4

    def __init__(self, feature_dim=4):
        super().__init__()
        self.stage_dims = [4, 4, 4, 4]
        self.feature_dim = feature_dim
        self.stage_calls = []

    def init_state(self, x):
        return x.float()

    def forward_stage(self, state, stage_idx: int):
        self.stage_calls.append(stage_idx)
        return state + (stage_idx + 1)

    def forward_head(self, state):
        return state

    def forward(self, x):
        state = self.init_state(x)
        for stage_idx in range(self.num_stages):
            state = self.forward_stage(state, stage_idx)
        return self.forward_head(state)


class TestMultimodalClassifierStaged(unittest.TestCase):
    def test_extract_features_staged_with_stage_fusion(self):
        from types import SimpleNamespace

        backbones = nn.ModuleDict({
            "image": ToyStageableBackbone(feature_dim=4),
            "wave": ToyStageableBackbone(feature_dim=4),
        })

        config = SimpleNamespace(
            model=SimpleNamespace(
                stage_fusion_common_dim=4,
                stage_fusion_mode="mean",
                dropout_rate=0.0,
            )
        )

        model = MultimodalClassifier(
            config=config,
            backbones=backbones,
            classifier_head=nn.Identity(),
            fusion=None,
            use_staged_forward=True,
            fusion_stages=(1, 2),
        )

        batch = {
            "image": torch.zeros(2, 4),
            "wave": torch.ones(2, 4),
        }

        out = model(batch)

        self.assertEqual(out.shape, (2, 8))
        self.assertEqual(set(model.stage_fusions.keys()), {"1", "2"})

    def test_extract_features_staged_concat_without_stage_fusion(self):
        backbones = nn.ModuleDict({
            "image": ToyStageableBackbone(feature_dim=4),
            "wave": ToyStageableBackbone(feature_dim=4),
        })

        model = MultimodalClassifier(
            config=None,
            backbones=backbones,
            classifier_head=nn.Identity(),
            fusion=None,
            use_staged_forward=True,
            fusion_stages=(),
        )

        batch = {
            "image": torch.zeros(2, 4),
            "wave": torch.ones(2, 4),
        }

        feats = model.extract_features_staged(batch)
        self.assertEqual(feats.shape, (2, 8))
        self.assertEqual(backbones["image"].stage_calls, [0, 1, 2, 3])
        self.assertEqual(backbones["wave"].stage_calls, [0, 1, 2, 3])

        backbones["image"].stage_calls.clear()
        backbones["wave"].stage_calls.clear()

        out = model(batch)
        self.assertEqual(out.shape, (2, 8))
        self.assertEqual(backbones["image"].stage_calls, [0, 1, 2, 3])
        self.assertEqual(backbones["wave"].stage_calls, [0, 1, 2, 3])

    def test_extract_features_staged_empty_batch_raises(self):
        backbones = nn.ModuleDict({
            "image": ToyStageableBackbone(feature_dim=4),
        })

        model = MultimodalClassifier(
            config=None,
            backbones=backbones,
            classifier_head=nn.Identity(),
            fusion=None,
            use_staged_forward=True,
            fusion_stages=(),
        )

        with self.assertRaises(ValueError):
            model.extract_features_staged({})

    def test_old_extract_features_still_works(self):
        backbones = nn.ModuleDict({
            "image": ToyStageableBackbone(feature_dim=4),
            "wave": ToyStageableBackbone(feature_dim=4),
        })

        model = MultimodalClassifier(
            config=None,
            backbones=backbones,
            classifier_head=nn.Identity(),
            fusion=None,
            use_staged_forward=False,
        )

        batch = {
            "image": torch.zeros(2, 4),
            "wave": torch.ones(2, 4),
        }

        feats = model.extract_features(batch)
        out = model(batch)

        self.assertEqual(feats.shape, (2, 8))
        self.assertEqual(out.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
