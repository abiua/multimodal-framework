import unittest
import torch

from models.modelzoo.image_models import ResNet18Stageable
from models.modelzoo.audio_models import AudioCNNStageable
from models.modelzoo.wave_models import TCNStageable


class TestStageableBackbones(unittest.TestCase):
    def test_resnet18_stageable_shapes(self):
        model = ResNet18Stageable(feature_dim=128, pretrained=False)
        x = torch.randn(2, 3, 224, 224)

        state = model.init_state(x)
        self.assertEqual(state.dim(), 4)
        self.assertEqual(state.shape[0], 2)

        for stage_idx in range(model.num_stages):
            state = model.forward_stage(state, stage_idx)
            self.assertEqual(state.dim(), 4)
            self.assertEqual(state.shape[0], 2)

        feat = model.forward_head(state)
        out = model(x)

        self.assertEqual(feat.shape, (2, 128))
        self.assertEqual(out.shape, (2, 128))

    def test_audiocnn_stageable_shapes(self):
        model = AudioCNNStageable(feature_dim=96)
        x = torch.randn(2, 64, 128)  # [B, H, W] -> init_state 会补 channel

        state = model.init_state(x)
        self.assertEqual(state.dim(), 4)
        self.assertEqual(state.shape[0], 2)

        for stage_idx in range(model.num_stages):
            state = model.forward_stage(state, stage_idx)
            self.assertEqual(state.dim(), 4)
            self.assertEqual(state.shape[0], 2)

        feat = model.forward_head(state)
        self.assertEqual(feat.shape, (2, 96))

    def test_tcn_stageable_shapes(self):
        model = TCNStageable(
            feature_dim=64,
            in_channels=6,
            hidden_channels=16,
            n_layers=6,
            kernel_size=3,
            dropout=0.1,
        )
        x = torch.randn(2, 100, 6)  # [B, T, C]，init_state 会转成 [B, C, T]

        state = model.init_state(x)
        self.assertEqual(state.dim(), 3)
        self.assertEqual(state.shape[0], 2)

        for stage_idx in range(model.num_stages):
            state = model.forward_stage(state, stage_idx)
            self.assertEqual(state.dim(), 3)
            self.assertEqual(state.shape[0], 2)

        feat = model.forward_head(state)
        out = model(x)

        self.assertEqual(feat.shape, (2, 64))
        self.assertEqual(out.shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
