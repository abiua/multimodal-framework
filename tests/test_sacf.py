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
