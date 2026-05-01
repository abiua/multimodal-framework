"""Pipeline V3 integration tests — Physics-First Asymmetric Fusion."""
import torch
import pytest


class TestImuChannelLoader:
    def test_columns(self):
        import tempfile, os
        from datasets.loaders.wave_loaders import ImuChannelLoader
        content = "h1,h2,h3,a1,a2,a3,g1,g2,g3,an1,an2,an3\n"
        content += "0,0,0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0\n"
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        tmp.write(content); tmp.close()
        try:
            for ch in ['accel', 'gyro', 'angle']:
                loader = ImuChannelLoader(channel=ch, max_length=512)
                result = loader.transform(loader.load(tmp.name))
                assert result[ch].shape == (512, 3)
        finally:
            os.unlink(tmp.name)

    def test_invalid_channel(self):
        import pytest as pt
        from datasets.loaders.wave_loaders import ImuChannelLoader
        with pt.raises(ValueError, match="Unknown IMU channel"):
            ImuChannelLoader(channel='invalid')


class TestMultiChannelTCN:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_forward_shape(self, device):
        from models.modelzoo.multichannel_tcn import MultiChannelTCN
        tcn = MultiChannelTCN(output_dim=256).to(device)
        imu = {k: torch.randn(2, 512, 3, device=device) for k in ['accel','gyro','angle']}
        out = tcn(imu)
        assert out.shape == (2, 512, 256)

    def test_tokenize(self, device):
        from models.modelzoo.multichannel_tcn import MultiChannelTCN
        tcn = MultiChannelTCN(output_dim=256).to(device)
        imu = {k: torch.randn(2, 512, 3, device=device) for k in ['accel','gyro','angle']}
        result = tcn.tokenize(imu)
        assert result['tokens'].shape == (2, 512, 256)
        assert result['layout'] == '1d'


class TestPhysicalEncoder:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_forward_shape(self, device):
        from models.fusion.physical_encoder import PhysicalDynamicsEncoder
        enc = PhysicalDynamicsEncoder(dim=256).to(device)
        out = enc(torch.randn(2, 200, 256, device=device),
                  torch.randn(2, 50, 256, device=device))
        assert out.shape == (2, 200, 256)

    def test_temporal_alignment(self, device):
        from models.fusion.physical_encoder import PhysicalDynamicsEncoder
        enc = PhysicalDynamicsEncoder(dim=256).to(device)
        # IMU longer than audio
        out = enc(torch.randn(2, 200, 256, device=device),
                  torch.randn(2, 1, 256, device=device))
        assert out.shape == (2, 200, 256)
        # Audio longer than IMU
        out = enc(torch.randn(2, 1, 256, device=device),
                  torch.randn(2, 200, 256, device=device))
        assert out.shape == (2, 200, 256)


class TestAsymmetricInteraction:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_physical_unchanged(self, device):
        from models.fusion.asymmetric_interaction import AsymmetricInteraction
        asym = AsymmetricInteraction(dim=256, num_blocks=2).to(device)
        v = torch.randn(2, 16, 256, device=device)
        p = torch.randn(2, 200, 256, device=device)
        v_out, p_out = asym(v, p)
        assert torch.allclose(p, p_out), "Physical tokens must be unchanged"
        assert v_out.shape == v.shape

    def test_evidence_gate_range(self, device):
        from models.fusion.asymmetric_interaction import EvidenceGate
        gate = EvidenceGate(dim=256).to(device)
        e = gate(torch.randn(2, 16, 256, device=device))
        assert e.shape == (2, 1)
        assert 0 <= e.min() <= e.max() <= 1


class TestPipelineV3:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def pipeline(self, device):
        import torch.nn as nn
        from models.pipeline_v3 import MultimodalPipelineV3
        from models.modelzoo.common import IdentityStem
        from models.modelzoo.multichannel_tcn import MultiChannelTCN
        from models.fusion.physical_encoder import PhysicalDynamicsEncoder
        from models.fusion.asymmetric_interaction import AsymmetricInteraction, EvidenceGate

        class MockAudio(nn.Module):
            feature_dim = 512
            def forward(self, x, **kw):
                return torch.randn(x.shape[0], 512, device=x.device)

        class MockVideo(nn.Module):
            feature_dim = 768
            def forward(self, x): return x
            def tokenize(self, x=None, **kw):
                inp = x if x is not None else list(kw.values())[0]
                B = inp.shape[0] if isinstance(inp, torch.Tensor) else 2
                dev = inp.device if isinstance(inp, torch.Tensor) else 'cpu'
                return {'tokens': torch.randn(B, 16, 768, device=dev), 'layout': '1d'}

        return MultimodalPipelineV3(
            imu_stems={k: IdentityStem(feature_dim=3) for k in ['accel','gyro','angle']},
            imu_encoder=MultiChannelTCN(output_dim=256),
            audio_stem=MockAudio(), audio_dim=512,
            video_stem=MockVideo(), video_dim=768,
            physical_encoder=PhysicalDynamicsEncoder(dim=256, num_heads=8),
            asymmetric_interaction=AsymmetricInteraction(dim=256, num_blocks=2),
            evidence_gate=EvidenceGate(dim=256),
            mid_fusion_dim=256, num_classes=4,
        ).to(device)

    def test_forward_shapes(self, pipeline, device):
        batch = {
            'imu_accel': torch.randn(2, 512, 3, device=device),
            'imu_gyro': torch.randn(2, 512, 3, device=device),
            'imu_angle': torch.randn(2, 512, 3, device=device),
            'audio': torch.randn(2, 2, 160000, device=device),
            'video': torch.randn(2, 16, 3, 224, 224, device=device),
        }
        logits, aux = pipeline(batch)
        assert logits.shape == (2, 4)
        assert aux['evidence'].shape == (2, 1)
        assert aux['phys_logits'].shape == (2, 4)

    def test_backward(self, pipeline, device):
        batch = {
            'imu_accel': torch.randn(2, 512, 3, device=device),
            'imu_gyro': torch.randn(2, 512, 3, device=device),
            'imu_angle': torch.randn(2, 512, 3, device=device),
            'audio': torch.randn(2, 2, 160000, device=device),
            'video': torch.randn(2, 16, 3, 224, 224, device=device),
        }
        logits, _ = pipeline(batch)
        logits.sum().backward()

    def test_teacher_knowledge_interface(self, pipeline, device):
        batch = {
            'imu_accel': torch.randn(2, 512, 3, device=device),
            'imu_gyro': torch.randn(2, 512, 3, device=device),
            'imu_angle': torch.randn(2, 512, 3, device=device),
            'audio': torch.randn(2, 2, 160000, device=device),
            'video': torch.randn(2, 16, 3, 224, 224, device=device),
        }
        knowledge = pipeline.get_teacher_knowledge(batch)
        for key in ['logits', 'phys_features', 'fused_features', 'evidence_scores', 'phys_logits']:
            assert key in knowledge, f"Missing key: {key}"
        assert knowledge['logits'].shape == (2, 4)
        assert knowledge['evidence_scores'].shape == (2, 1)

    def test_training_mode(self, pipeline, device):
        """In training mode, pipeline should produce gradients."""
        import torch.nn as nn
        pipeline.train()
        batch = {
            'imu_accel': torch.randn(2, 512, 3, device=device),
            'imu_gyro': torch.randn(2, 512, 3, device=device),
            'imu_angle': torch.randn(2, 512, 3, device=device),
            'audio': torch.randn(2, 2, 160000, device=device),
            'video': torch.randn(2, 16, 3, 224, 224, device=device),
        }
        logits, aux = pipeline(batch)
        labels = torch.randint(0, 4, (2,), device=device)
        loss = nn.CrossEntropyLoss()(logits, labels) + 0.3 * nn.CrossEntropyLoss()(aux['phys_logits'], labels)
        loss.backward()
        for name, p in pipeline.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
