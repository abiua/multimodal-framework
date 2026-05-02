"""Tests for SACF v2 components."""

import torch
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
