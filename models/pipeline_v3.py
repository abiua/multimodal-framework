"""MultimodalPipelineV3 — Physics-First Asymmetric Fusion.

Flow:
    IMU(3ch) -> MultiChannelTCN -> PhysicalDynamicsEncoder -> Physical Tokens
    Audio -> AudioCNN -> Audio Tokens ──────────┘
    Video -> VideoMAE -> Visual Tokens -> AsymmetricInteraction -> EvidenceGate
                                                                       ↓
    Physical Pooling + Gated Visual Pooling -> Fusion -> Classifier
"""
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from .fusion.physical_encoder import PhysicalDynamicsEncoder
from .fusion.asymmetric_interaction import AsymmetricInteraction, EvidenceGate


class MultimodalPipelineV3(nn.Module):
    """Physics-First Asymmetric Fusion Pipeline.

    Three modalities: IMU (3 physical channels), Audio, Video.
    Physical signals (IMU+Audio) form the decision backbone.
    Video provides complementary evidence, modulated by an evidence gate.
    """

    def __init__(
        self,
        imu_stems: Dict[str, nn.Module],
        imu_encoder: nn.Module,
        audio_stem: nn.Module,
        audio_dim: int,
        video_stem: nn.Module,
        video_dim: int,
        physical_encoder: PhysicalDynamicsEncoder,
        asymmetric_interaction: AsymmetricInteraction,
        evidence_gate: EvidenceGate,
        mid_fusion_dim: int,
        num_classes: int,
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        self.imu_stems = nn.ModuleDict(imu_stems)
        self.imu_encoder = imu_encoder
        self.audio_stem = audio_stem
        self.video_stem = video_stem
        self.physical_encoder = physical_encoder
        self.asymmetric_interaction = asymmetric_interaction
        self.evidence_gate = evidence_gate

        D = physical_encoder.dim

        # Dimension projections: backbone output -> unified token dim D
        self.audio_proj = nn.Linear(audio_dim, D) if audio_dim != D else nn.Identity()
        self.video_proj = nn.Linear(video_dim, D) if video_dim != D else nn.Identity()

        # Mid fusion
        self.phys_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, mid_fusion_dim))
        self.vis_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, mid_fusion_dim))
        self.fusion_fc = nn.Sequential(
            nn.Linear(mid_fusion_dim * 2, mid_fusion_dim),
            nn.GELU(), nn.Dropout(dropout_rate),
        )

        # Classifiers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(mid_fusion_dim, num_classes),
        )
        self.phys_classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(mid_fusion_dim, num_classes),
        )

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[Dict]]:
        # 1. IMU: collect 3 channels -> IMU encoder
        imu_channels = {}
        for ch_name in ['accel', 'gyro', 'angle']:
            val = self._resolve_imu_channel(batch, ch_name)
            imu_channels[ch_name] = val
        imu_tokens = self.imu_encoder(imu_channels)  # [B, T_imu, D]

        # 2. Audio: AudioCNN -> [B, D_audio] -> project -> [B, D]
        audio_input = self._resolve_input(batch, 'audio')
        if isinstance(audio_input, torch.Tensor):
            audio_feat = self.audio_stem(audio_input)
        else:
            audio_feat = self.audio_stem(**audio_input)
        audio_tokens = self.audio_proj(audio_feat).unsqueeze(1)  # [B, 1, D]

        # 3. Video: VideoMAE -> tokens [B, T_v, D_video] -> project -> [B, T_v, D]
        video_input = self._resolve_input(batch, 'video')
        if isinstance(video_input, torch.Tensor):
            video_cfg = self.video_stem.tokenize(video_input)
        else:
            video_cfg = self.video_stem.tokenize(**video_input)
        video_tokens = self.video_proj(video_cfg['tokens'])

        # 4. Physical Dynamics Encoder (IMU + Audio)
        physical_tokens = self.physical_encoder(imu_tokens, audio_tokens)

        # 5. Asymmetric Interaction (Video -> Physical)
        visual_out, physical_out = self.asymmetric_interaction(video_tokens, physical_tokens)

        # 6. Evidence Gate
        evidence = self.evidence_gate(visual_out)  # [B, 1]

        # 7. Pooling + Fusion
        phys_pooled = physical_out.mean(dim=1)          # [B, D]
        vis_pooled = evidence * visual_out.mean(dim=1)  # [B, D] — gated by evidence
        phys_feat = self.phys_proj(phys_pooled)
        vis_feat = self.vis_proj(vis_pooled)
        fused = self.fusion_fc(torch.cat([phys_feat, vis_feat], dim=-1))

        # 8. Classify
        logits = self.classifier(fused)
        phys_logits = self.phys_classifier(phys_feat)

        return logits, {
            'evidence': evidence,
            'physical_tokens': physical_out,
            'phys_logits': phys_logits,
        }

    def get_teacher_knowledge(self, batch) -> Dict[str, torch.Tensor]:
        """Distillation interface — returns multi-level teacher knowledge."""
        logits, aux = self.forward(batch)
        return {
            'logits': logits,
            'phys_features': aux['physical_tokens'].mean(dim=1),
            'fused_features': logits,
            'evidence_scores': aux['evidence'],
            'phys_logits': aux['phys_logits'],
        }

    def _resolve_imu_channel(self, batch, ch_name):
        """Resolve IMU channel from various batch formats."""
        full_key = f'imu_{ch_name}'
        if full_key in batch:
            val = batch[full_key]
            if isinstance(val, torch.Tensor):
                return val
            if isinstance(val, dict) and ch_name in val:
                return val[ch_name]
            return val
        for k, v in batch.items():
            if isinstance(v, dict) and ch_name in v:
                return v[ch_name]
            if isinstance(v, dict) and full_key in v:
                return v[full_key]
        raise KeyError(f"IMU channel '{ch_name}' not found. Keys: {list(batch.keys())}")

    @staticmethod
    def _resolve_input(batch, modality):
        """Resolve modality input from batch."""
        if modality in batch:
            return batch[modality]
        for k, v in batch.items():
            if k.startswith(modality) and isinstance(v, torch.Tensor):
                return v
            if isinstance(v, dict) and modality in v:
                return v[modality]
        raise KeyError(f"Modality '{modality}' not found in batch")
