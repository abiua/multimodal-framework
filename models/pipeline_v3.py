"""MultimodalPipelineV3 — Physics-First Asymmetric Fusion.

Flow:
    Physical(IMU/Wave) -> Encoder -> PhysicalDynamicsEncoder -> Physical Tokens
    Audio -> Audio Backbone -> Audio Tokens ──────────┘
    Visual(Video/Image) -> Visual Tokens -> AsymmetricInteraction -> EvidenceGate
                                                                       ↓
    Physical Pooling + Gated Visual Pooling -> Fusion -> Classifier
"""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .fusion.physical_encoder import PhysicalDynamicsEncoder
from .fusion.asymmetric_interaction import AsymmetricInteraction, EvidenceGate


class MultimodalPipelineV3(nn.Module):
    """Physics-First Asymmetric Fusion Pipeline.

    Physical signals (IMU/Wave + Audio) form the decision backbone.
    Visual modality (Video/Image) provides complementary evidence,
    modulated by an evidence gate.

    Args:
        imu_stems: per-channel IMU stems (set to None for single-wave physical source)
        imu_encoder: IMU multi-channel encoder (set to single-wave encoder if imu_stems is None)
        imu_channel_names: IMU channel names to resolve from batch
        audio_stem: audio backbone
        audio_dim: audio backbone output dimension
        visual_stem: visual backbone (video or image)
        visual_dim: visual backbone output dimension
        visual_type: 'video' (uses tokenize) or 'image' (uses forward)
        physical_encoder: PhysicalDynamicsEncoder
        asymmetric_interaction: AsymmetricInteraction
        evidence_gate: EvidenceGate
        mid_fusion_dim: mid-fusion output dimension
        num_classes: number of classes
        dropout_rate: dropout rate
    """

    def __init__(
        self,
        imu_stems: Optional[Dict[str, nn.Module]],
        imu_encoder: nn.Module,
        imu_channel_names: List[str],
        audio_stem: nn.Module,
        audio_dim: int,
        visual_stem: nn.Module,
        visual_dim: int,
        visual_type: str,
        visual_key: str,
        physical_encoder: PhysicalDynamicsEncoder,
        asymmetric_interaction: AsymmetricInteraction,
        evidence_gate: EvidenceGate,
        mid_fusion_dim: int,
        num_classes: int,
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        self.imu_stems = nn.ModuleDict(imu_stems) if imu_stems is not None else None
        self.imu_encoder = imu_encoder
        self.imu_channel_names = imu_channel_names
        self.audio_stem = audio_stem
        self.visual_stem = visual_stem
        self.visual_type = visual_type
        self.visual_key = visual_key
        self.physical_encoder = physical_encoder
        self.asymmetric_interaction = asymmetric_interaction
        self.evidence_gate = evidence_gate

        D = physical_encoder.dim

        # Dimension projections: backbone output -> unified token dim D
        self.audio_proj = nn.Linear(audio_dim, D) if audio_dim != D else nn.Identity()
        self.visual_proj = nn.Linear(visual_dim, D) if visual_dim != D else nn.Identity()

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

    def _encode_physical(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Encode physical modality (multi-channel IMU or single wave)."""
        if self.imu_stems is not None:
            # Multi-channel IMU: resolve each channel
            imu_channels = {}
            for ch_name in self.imu_channel_names:
                val = self._resolve_input(batch, f'imu_{ch_name}')
                if isinstance(val, dict):
                    val = val.get(ch_name, val)
                imu_channels[ch_name] = val
            return self.imu_encoder(imu_channels)  # [B, T, D]
        else:
            # Single physical source (wave): direct encode
            phys_input = self._resolve_input(batch, self.imu_channel_names[0])
            if isinstance(phys_input, dict):
                # Unwrap nested dict from loader transform {modality: tensor}
                name = self.imu_channel_names[0]
                if name in phys_input:
                    phys_input = phys_input[name]
            return self.imu_encoder(phys_input)  # [B, T, D]

    def _encode_visual(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Encode visual modality (video with tokenize, image with forward)."""
        visual_input = self._resolve_input(batch, self.visual_key)
        if self.visual_type == 'video':
            if isinstance(visual_input, torch.Tensor):
                tokens = self.visual_stem.tokenize(visual_input)
            else:
                tokens = self.visual_stem.tokenize(**visual_input)
            return self.visual_proj(tokens['tokens'])  # [B, T_v, D]
        else:
            # Image: forward -> pool -> single token
            if isinstance(visual_input, torch.Tensor):
                feat = self.visual_stem(visual_input)
            else:
                feat = self.visual_stem(**visual_input)
            return self.visual_proj(feat).unsqueeze(1)  # [B, 1, D]

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        # 1. Physical encode
        imu_tokens = self._encode_physical(batch)

        # 2. Audio
        audio_input = self._resolve_input(batch, 'audio')
        if isinstance(audio_input, torch.Tensor):
            audio_feat = self.audio_stem(audio_input)
        else:
            audio_feat = self.audio_stem(**audio_input)
        audio_tokens = self.audio_proj(audio_feat).unsqueeze(1)

        # 3. Visual
        visual_tokens = self._encode_visual(batch)

        # 4. Physical Dynamics Encoder (Physical + Audio)
        physical_tokens = self.physical_encoder(imu_tokens, audio_tokens)

        # 5. Asymmetric Interaction (Visual -> Physical)
        visual_out, physical_out = self.asymmetric_interaction(visual_tokens, physical_tokens)

        # 6. Evidence Gate
        evidence = self.evidence_gate(visual_out)

        # 7. Pooling + Fusion
        phys_pooled = physical_out.mean(dim=1)
        vis_pooled = evidence * visual_out.mean(dim=1)
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
        raise KeyError(f"Modality '{modality}' not found in batch. Keys: {list(batch.keys())}")
