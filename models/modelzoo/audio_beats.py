"""BEATs backbone — ICML 2023, Microsoft acoustic tokenizer SSL.

Uses SpeechBrain's BEATs implementation. Expects pretrained .pt checkpoint
from Microsoft OneDrive (BEATs_iter3+ AS2M).

Input: raw waveform 16kHz mono → Kaldi fbank → ViT encoder
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone_base import BaseBackbone
from ..registry import register_backbone


@register_backbone('beats', description='BEATs audio encoder (ICML 2023, AudioSet-2M pretrained)', modality='audio')
class BEATsBackbone(BaseBackbone):
    feature_dim = 768

    def __init__(
        self,
        feature_dim: int = 768,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze: bool = False,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        super().__init__()

        from speechbrain.lobes.models.beats import BEATs, BEATsConfig

        if checkpoint_path is None:
            checkpoint_path = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/BEATs/BEATs_iter3_plus_as2m.pt'

        if pretrained:
            import os
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"BEATs checkpoint not found: {checkpoint_path}\n"
                    "Download from: https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf"
                )
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            cfg = BEATsConfig(ckpt['cfg'])
            self.encoder = BEATs(cfg)
            self.encoder.load_state_dict(ckpt['model'], strict=False)
        else:
            cfg = BEATsConfig()
            self.encoder = BEATs(cfg)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x=None, **inputs):
        if x is None:
            if 'waveform' in inputs:
                x = inputs['waveform']
            elif isinstance(inputs, dict):
                for v in inputs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        x = v
                        break

        # Mono: [B, T]
        if x.dim() == 3 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=False)
        elif x.dim() == 3:
            x = x.squeeze(1)

        # BEATs expects [B, T] raw audio at 16kHz, processes internally via Kaldi fbank
        feats, _ = self.encoder.extract_features(x)
        return feats.mean(dim=1)  # [B, 768]
