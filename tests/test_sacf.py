"""Tests for SACF v2 components."""

import torch
from torch import nn
import pytest


class TestAudioTemporalEncoder:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_output_shape(self, device):
        from models.modelzoo.audio_temporal import AudioTemporalEncoder
        enc = AudioTemporalEncoder(in_channels=2, output_dim=256).to(device)
        x = torch.randn(4, 2, 224, 224, device=device)
        out = enc(x)
        assert out.dim() == 3, f"Expected [B,T,D], got {out.shape}"
        assert out.shape[0] == 4
        assert out.shape[1] > 1, "Should preserve time dimension (>1 tokens)"
        assert out.shape[2] == 256

    def test_forward_backward(self, device):
        from models.modelzoo.audio_temporal import AudioTemporalEncoder
        enc = AudioTemporalEncoder(in_channels=2, output_dim=256).to(device)
        x = torch.randn(4, 2, 224, 224, device=device)
        out = enc(x)
        out.sum().backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_different_input_shapes(self, device):
        from models.modelzoo.audio_temporal import AudioTemporalEncoder
        enc = AudioTemporalEncoder(in_channels=1, output_dim=128).to(device)
        x = torch.randn(2, 1, 128, 300, device=device)
        out = enc(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 128


class TestBidirectionalCrossAttention:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_output_shape(self, device):
        from models.fusion.temporal_consensus import BidirectionalCrossAttention
        bca = BidirectionalCrossAttention(dim=256, num_heads=8).to(device)
        wave = torch.randn(2, 32, 256, device=device)
        audio = torch.randn(2, 14, 256, device=device)
        out = bca(wave, audio)
        assert out.shape[0] == 2
        assert out.shape[1] == 32 + 14, f"Expected T_w+T_a tokens, got {out.shape[1]}"
        assert out.shape[2] == 256

    def test_forward_backward(self, device):
        from models.fusion.temporal_consensus import BidirectionalCrossAttention
        bca = BidirectionalCrossAttention(dim=256).to(device)
        wave = torch.randn(2, 32, 256, device=device)
        audio = torch.randn(2, 14, 256, device=device)
        out = bca(wave, audio, wave_masked=True, audio_masked=True)
        out.sum().backward()
        for name, p in bca.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_null_token_fallback(self, device):
        from models.fusion.temporal_consensus import BidirectionalCrossAttention
        bca = BidirectionalCrossAttention(dim=256).to(device)
        wave = torch.randn(2, 32, 256, device=device)
        null_audio = bca.null_audio.expand(2, 14, -1).to(device)
        out = bca(wave, null_audio, audio_masked=True)
        assert out.shape == (2, 32 + 14, 256)

    def test_both_masked(self, device):
        from models.fusion.temporal_consensus import BidirectionalCrossAttention
        bca = BidirectionalCrossAttention(dim=256).to(device)
        null_wave = bca.null_wave.expand(2, 32, -1).to(device)
        null_audio = bca.null_audio.expand(2, 14, -1).to(device)
        out = bca(null_wave, null_audio, wave_masked=True, audio_masked=True)
        assert out.shape[0] == 2


class TestFiLMGate:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_output_shape(self, device):
        from models.fusion.film_gate import FiLMGate
        gate = FiLMGate(image_dim=2048, phys_dim=256, residual_dim=64).to(device)
        z_img = torch.randn(4, 2048, device=device)
        phys = torch.randn(4, 256, device=device)
        out, aux = gate(z_img, phys)
        assert out.shape == (4, 256)
        assert aux['film_scale'].shape == (4, 256)
        assert aux['film_shift'].shape == (4, 256)
        assert aux['gate'].shape == (4, 64)

    def test_film_identity_when_image_zero(self, device):
        from models.fusion.film_gate import FiLMGate
        gate = FiLMGate(image_dim=2048, phys_dim=256, residual_dim=64).to(device)
        z_img = torch.zeros(4, 2048, device=device)
        phys = torch.randn(4, 256, device=device)
        out, aux = gate(z_img, phys)
        assert torch.allclose(aux['film_scale'], torch.ones_like(aux['film_scale']), atol=1e-3)
        assert torch.allclose(aux['film_shift'], torch.zeros_like(aux['film_shift']), atol=5e-2)

    def test_forward_backward(self, device):
        from models.fusion.film_gate import FiLMGate
        gate = FiLMGate(image_dim=2048, phys_dim=256, residual_dim=64).to(device)
        z_img = torch.randn(4, 2048, device=device)
        phys = torch.randn(4, 256, device=device)
        out, _ = gate(z_img, phys)
        out.sum().backward()
        for name, p in gate.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_residual_dropout_training(self, device):
        from models.fusion.film_gate import FiLMGate
        gate = FiLMGate(image_dim=2048, phys_dim=256, residual_dim=64, r_dropout=1.0).to(device)
        gate.train()
        z_img = torch.randn(4, 2048, device=device)
        phys = torch.randn(4, 256, device=device)
        out1, _ = gate(z_img, phys)
        out2, _ = gate(z_img, phys)
        assert torch.allclose(out1, out2)


class MockResNet(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.feature_dim = dim

    def forward(self, x):
        if isinstance(x, dict):
            x = list(x.values())[0]
        return torch.randn(x.shape[0], self.feature_dim, device=x.device)


class MockWaveEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 256

    def forward(self, x):
        if isinstance(x, dict):
            x = list(x.values())[0]
        B = x.shape[0] if x.dim() >= 1 else 1
        return torch.randn(B, 32, 256, device=x.device)


class TestSACFPipeline:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def pipeline(self, device):
        from models.pipeline_sacf import SACFPipeline
        from models.modelzoo.audio_temporal import AudioTemporalEncoder
        from models.fusion.temporal_consensus import BidirectionalCrossAttention
        from models.fusion.film_gate import FiLMGate

        return SACFPipeline(
            wave_encoder=MockWaveEncoder(),
            audio_encoder=AudioTemporalEncoder(in_channels=2, output_dim=256),
            image_backbone=MockResNet(dim=2048),
            image_dim=2048,
            temporal_consensus=BidirectionalCrossAttention(dim=256),
            film_gate=FiLMGate(image_dim=2048, phys_dim=256, residual_dim=64),
            mid_fusion_dim=256,
            num_classes=3,
            modal_dropout=0.0,
        ).to(device)

    def test_forward_shapes(self, pipeline, device):
        batch = {
            'wave': torch.randn(2, 512, 6, device=device),
            'audio': torch.randn(2, 2, 224, 224, device=device),
            'image': torch.randn(2, 3, 224, 224, device=device),
        }
        logits, aux = pipeline(batch)
        assert logits.shape == (2, 3)
        assert 'phys_logits' in aux
        assert 'image_logits' in aux
        assert aux['phys_logits'].shape == (2, 3)
        assert aux['image_logits'].shape == (2, 3)

    def test_backward(self, pipeline, device):
        pipeline.train()
        pipeline.modal_dropout = 1.0
        batch = {
            'wave': torch.randn(2, 512, 6, device=device),
            'audio': torch.randn(2, 2, 224, 224, device=device),
            'image': torch.randn(2, 3, 224, 224, device=device),
        }
        # Multiple forward/backward passes to cover all masking branches
        for _ in range(20):
            logits, aux = pipeline(batch)
            loss = logits.sum() + aux['phys_logits'].sum() + aux['image_logits'].sum()
            loss.backward()
        for name, p in pipeline.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_modal_dropout(self, pipeline, device):
        pipeline.modal_dropout = 1.0
        pipeline.train()
        batch = {
            'wave': torch.randn(4, 512, 6, device=device),
            'audio': torch.randn(4, 2, 224, 224, device=device),
        }
        batch['image'] = torch.randn(4, 3, 224, 224, device=device)
        logits, aux = pipeline(batch)
        assert logits.shape == (4, 3)
        assert aux['wave_masked'] or aux['audio_masked'] or aux['image_masked']

    def test_get_teacher_knowledge(self, pipeline, device):
        batch = {
            'wave': torch.randn(2, 512, 6, device=device),
            'audio': torch.randn(2, 2, 224, 224, device=device),
            'image': torch.randn(2, 3, 224, 224, device=device),
        }
        knowledge = pipeline.get_teacher_knowledge(batch)
        for key in ['logits', 'phys_features', 'phys_logits', 'image_logits', 'final_features']:
            assert key in knowledge, f"Missing key: {key}"
