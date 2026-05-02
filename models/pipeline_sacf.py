"""SACFPipeline — Stage-Aware Consensus Fusion v2.

Stage 1: Wave + Audio -> Bidirectional Cross-Attn -> Physical tokens -> phys_pooled -> phys_logits
Stage 2: Image -> FiLM modulation + Gated residual -> f_final -> final_logits
Stage 3: Consensus regularization (computed in training script, NOT in pipeline)
"""
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class SACFPipeline(nn.Module):
    """Stage-Aware Consensus Fusion Pipeline.

    Args:
        wave_encoder: Wave encoder module (outputs [B, T_w, D] or [B, T_w, D_w])
        audio_encoder: AudioTemporalEncoder (outputs [B, T_a, D])
        image_backbone: Image encoder (outputs [B, D_img])
        image_dim: Image backbone output dimension (e.g., 2048)
        temporal_consensus: BidirectionalCrossAttention module
        film_gate: FiLMGate module
        mid_fusion_dim: Mid fusion output dimension (default 256)
        num_classes: Number of classes (default 3)
        modal_dropout: Probability of dropping a modality during training (default 0.3)
        dropout_rate: Dropout rate for classifiers (default 0.35)
    """

    def __init__(
        self,
        wave_encoder: nn.Module,
        audio_encoder: nn.Module,
        image_backbone: nn.Module,
        image_dim: int,
        temporal_consensus: nn.Module,
        film_gate: nn.Module,
        mid_fusion_dim: int = 256,
        num_classes: int = 3,
        modal_dropout: float = 0.3,
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        self.wave_encoder = wave_encoder
        self.audio_encoder = audio_encoder
        self.image_backbone = image_backbone
        self.temporal_consensus = temporal_consensus
        self.film_gate = film_gate

        D = temporal_consensus.dim  # token dimension

        # Dimension projections
        # wave_encoder output dim may or may not match D
        wave_feat_dim = getattr(wave_encoder, 'feature_dim', D)
        self.wave_proj = nn.Linear(wave_feat_dim, D) if wave_feat_dim != D else nn.Identity()
        # audio_encoder has .feature_dim
        audio_feat_dim = getattr(audio_encoder, 'feature_dim', D)
        self.audio_proj = nn.Linear(audio_feat_dim, D) if audio_feat_dim != D else nn.Identity()

        # Pool physical tokens -> project to mid_fusion_dim
        self.phys_pool_proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, mid_fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Final classifier (takes f_final after FiLM + residual)
        self.final_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(mid_fusion_dim, num_classes),
        )

        # Auxiliary classifiers for consensus loss
        self.phys_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(mid_fusion_dim, num_classes),
        )
        self.image_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(mid_fusion_dim, num_classes),
        )

        # Image features -> mid_fusion_dim for image_classifier
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, mid_fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.modal_dropout = modal_dropout

    def _encode_wave(self, wave_input):
        """Encode wave: handle dict unwrap, channel transpose, project."""
        if isinstance(wave_input, dict):
            wave_input = wave_input.get('wave', list(wave_input.values())[0])
        # Handle [B, L, C] -> [B, C, L] if needed (wave format)
        if wave_input.dim() == 3 and wave_input.size(-1) in (6, 3):
            wave_input = wave_input.transpose(1, 2)
        tokens = self.wave_encoder(wave_input)  # [B, T_w, D_w] or [B, T_w, D]
        return self.wave_proj(tokens)

    def _encode_audio(self, audio_input):
        """Encode audio: handle dict unwrap, project."""
        if isinstance(audio_input, dict):
            audio_input = list(audio_input.values())[0]
        tokens = self.audio_encoder(audio_input)  # [B, T_a, D_a]
        return self.audio_proj(tokens)

    def _encode_image(self, image_input):
        """Encode image: handle dict unwrap."""
        if isinstance(image_input, dict):
            image_input = list(image_input.values())[0]
        return self.image_backbone(image_input)  # [B, D_img]

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        # Resolve inputs
        wave_input = self._resolve_input(batch, 'wave')
        audio_input = self._resolve_input(batch, 'audio')
        image_input = self._resolve_input(batch, 'image')

        # Stage 1: Encode + Temporal Consensus
        wave_tokens = self._encode_wave(wave_input)
        audio_tokens = self._encode_audio(audio_input)

        wave_masked = False
        audio_masked = False
        image_masked = False

        if self.training and self.modal_dropout > 0:
            drop = torch.rand(1).item()
            if drop < self.modal_dropout:
                r = torch.rand(1).item()
                if r < 0.33:
                    # Mask wave
                    wave_masked = True
                elif r < 0.66:
                    # Mask audio
                    audio_masked = True
                else:
                    # Mask image (handled later)
                    image_masked = True

        # Temporal consensus handles null-token replacement via BCA masking flags
        physical_tokens = self.temporal_consensus(wave_tokens, audio_tokens, wave_masked, audio_masked)
        phys_pooled = self.phys_pool_proj(physical_tokens.mean(dim=1))
        phys_logits = self.phys_classifier(phys_pooled)

        # Stage 2: Image FiLM + residual
        z_img = self._encode_image(image_input)

        if image_masked:
            f_final = phys_pooled
            image_feat = self.image_proj(z_img)  # still compute for consistent output
            image_logits = self.image_classifier(image_feat)
            film_aux = {
                'film_scale': torch.ones_like(phys_pooled),
                'film_shift': torch.zeros_like(phys_pooled),
                'gate': torch.zeros(phys_pooled.shape[0], 64, device=phys_pooled.device),
            }
        else:
            f_final, film_aux = self.film_gate(z_img, phys_pooled)
            image_feat = self.image_proj(z_img)
            image_logits = self.image_classifier(image_feat)

        final_logits = self.final_classifier(f_final)

        return final_logits, {
            'phys_logits': phys_logits,
            'image_logits': image_logits,
            'wave_masked': wave_masked,
            'audio_masked': audio_masked,
            'image_masked': image_masked,
            'physical_tokens': physical_tokens,
            'phys_pooled': phys_pooled,
            'film_aux': film_aux,
        }

    def get_teacher_knowledge(self, batch) -> Dict[str, torch.Tensor]:
        """Distillation interface -- returns multi-level teacher knowledge."""
        logits, aux = self.forward(batch)
        return {
            'logits': logits,
            'phys_features': aux['phys_pooled'],
            'phys_logits': aux['phys_logits'],
            'image_logits': aux['image_logits'],
            'final_features': logits,
        }

    @staticmethod
    def _resolve_input(batch, modality):
        """Resolve modality input from batch dict."""
        if modality in batch:
            return batch[modality]
        for k, v in batch.items():
            if k.startswith(modality) and isinstance(v, torch.Tensor):
                return v
            if isinstance(v, dict) and modality in v:
                return v[modality]
        raise KeyError(f"Modality '{modality}' not found in batch. Keys: {list(batch.keys())}")
