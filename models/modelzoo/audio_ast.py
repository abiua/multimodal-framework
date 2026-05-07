"""AST (Audio Spectrogram Transformer) backbone — HuggingFace-powered.

Registers 'ast' backbone. Uses AudioSet-pretrained ASTModel from HuggingFace
transformers. Expects raw waveform input (16kHz mono) from audio_raw_loader.

Based on: Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from ..backbone_base import BaseBackbone
from ..registry import register_backbone

# AudioSet mel spectrogram normalization constants
AS_MEAN = -4.2677393
AS_STD = 4.5689974


@register_backbone('ast_hf', description='AST Audio Spectrogram Transformer HuggingFace (AudioSet pretrained)', modality='audio')
class ASTBackbone(BaseBackbone):
    """AST backbone for audio classification.

    Input: raw waveform [B, T] or {'waveform': Tensor[B, T]}
    Output: feature vector [B, 768]

    Args:
        model_name: HuggingFace model ID (default MIT/ast-finetuned-audioset-10-10-0.4593)
        freeze: freeze encoder weights
        dropout: dropout rate for the encoder (if supported)
        max_length: max time frames for spectrogram
    """

    feature_dim = 768  # default, overridden in __init__ if hidden_size is specified

    def __init__(
        self,
        feature_dim: int = 768,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze: bool = False,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        max_length: int = 1024,
        num_mel_bins: int = 128,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        f_max: int = 8000,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self._max_length = max_length
        self._num_mel_bins = num_mel_bins

        # GPU-friendly mel spectrogram (configurable for 16k/48k)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=num_mel_bins, f_min=0, f_max=f_max,
            power=2.0, norm=None,
        )
        self.db_transform = T.AmplitudeToDB()

        from transformers import ASTModel

        if pretrained:
            self.encoder = ASTModel.from_pretrained(model_name)
        else:
            from transformers import ASTConfig
            cfg = ASTConfig.from_pretrained(model_name)
            cfg.max_length = max_length
            cfg.num_mel_bins = num_mel_bins
            cfg.hidden_dropout_prob = dropout
            cfg.attention_probs_dropout_prob = dropout
            # Override architecture params from kwargs
            for k in ('num_hidden_layers', 'num_attention_heads',
                       'hidden_size', 'intermediate_size'):
                if k in kwargs:
                    setattr(cfg, k, kwargs[k])
            self.encoder = ASTModel(cfg)

        if 'hidden_size' in kwargs:
            self.feature_dim = kwargs['hidden_size']

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """Convert waveform [B, T] to normalized mel [B, T_frames, n_mels]."""
        mel = self.mel_transform(x)          # [B, n_mels, T_frames]
        mel = self.db_transform(mel + 1e-6)  # amplitude to dB
        mel = (mel - AS_MEAN) / (AS_STD * 2) # AudioSet normalization

        # Pad/truncate to max_length
        Tf = mel.shape[-1]
        if Tf < self._max_length:
            mel = F.pad(mel, (0, self._max_length - Tf))
        elif Tf > self._max_length:
            mel = mel[..., :self._max_length]

        return mel.transpose(1, 2)  # [B, T_frames, n_mels]

    def forward(self, x=None, **inputs):
        if x is None:
            if 'waveform' in inputs:
                x = inputs['waveform']
            elif isinstance(inputs, dict):
                for v in inputs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        x = v
                        break
                if x is None:
                    raise ValueError(f"Cannot find waveform in inputs: {list(inputs.keys())}")

        # Ensure mono: [B, T]
        if x.dim() == 3 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=False)
        elif x.dim() == 3:
            x = x.squeeze(1)

        mel = self._to_mel(x)
        out = self.encoder(mel)
        pooled = out.last_hidden_state.mean(dim=1)  # [B, 768]
        return pooled
