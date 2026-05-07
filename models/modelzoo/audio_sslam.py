"""SSLAM backbone — ICLR 2025, EAT architecture with mixture SSL pretraining.

HF model: ta012/SSLAM_pretrain (ViT-B, 12 layers, 768-dim)
Input: raw waveform 16kHz mono, up to 10s
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ..backbone_base import BaseBackbone
from ..registry import register_backbone

AS_MEAN = -4.268
AS_STD = 4.569


@register_backbone('sslam', description='SSLAM EAT audio encoder (ICLR 2025, AudioSet-2M pretrained)', modality='audio')
class SSLAMBackbone(BaseBackbone):
    feature_dim = 768

    def __init__(
        self,
        feature_dim: int = 768,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze: bool = False,
        target_length: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.target_length = target_length

        model_dir = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/SSLAM-main'
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from hf_model.configuration_eat import EATConfig
        from hf_model.eat_model import EAT

        config = EATConfig()
        self.encoder = EAT(config)

        if pretrained:
            import os
            ckpt = os.path.join(model_dir, 'hf_model', 'model.safetensors')
            if os.path.exists(ckpt):
                from safetensors.torch import load_file
                state = load_file(ckpt)
                sd = {k.replace('model.', ''): v for k, v in state.items() if k.startswith('model.')}
                missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
                if missing:
                    print(f'[SSLAM] Missing keys: {len(missing)}')
                if unexpected:
                    print(f'[SSLAM] Unexpected keys: {len(unexpected)}')
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, T_frames, 128] using Kaldi fbank (matches SSLAM pretraining)."""
        B = x.shape[0]
        mels = []
        for i in range(B):
            wav = x[i] - x[i].mean()
            mel = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0), htk_compat=True, sample_frequency=16000,
                use_energy=False, window_type='hanning', num_mel_bins=128,
                dither=0.0, frame_shift=10,
            )
            n_frames = mel.shape[0]
            diff = self.target_length - n_frames
            if diff > 0:
                mel = F.pad(mel, (0, 0, 0, diff))
            elif diff < 0:
                mel = mel[:self.target_length]
            mels.append(mel)
        mel = torch.stack(mels, dim=0).to(device=x.device, dtype=x.dtype)
        mel = (mel - AS_MEAN) / (AS_STD * 2)
        return mel

    def forward(self, x=None, **inputs):
        if x is None:
            if 'waveform' in inputs:
                x = inputs['waveform']
            elif isinstance(inputs, dict):
                for v in inputs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        x = v
                        break

        if x.dim() == 3 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=False)
        elif x.dim() == 3:
            x = x.squeeze(1)

        mel = self._to_mel(x)
        mel = mel.unsqueeze(1)  # [B, 1, T, 128]

        x = self.encoder.extract_features(mel)  # [B, N_patches+1, 768]
        return x[:, 0]  # CLS token [B, 768]
