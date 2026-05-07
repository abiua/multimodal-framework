"""BEATs backbone — ICML 2023, Microsoft acoustic tokenizer SSL.

Uses SpeechBrain's BEATs implementation. Supports 16kHz (pretrained) and
48kHz (from-scratch with custom fbank params).

Input: raw waveform mono → Kaldi fbank (configurable sample_rate / n_mels) → ViT encoder
"""

from __future__ import annotations

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi

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
        sample_rate: int = 16000,
        num_mel_bins: int = 128,
        **kwargs,
    ):
        super().__init__()

        from speechbrain.lobes.models.beats import BEATs

        self._sample_rate = sample_rate
        self._num_mel_bins = num_mel_bins

        if pretrained:
            if sample_rate != 16000 or num_mel_bins != 128:
                raise ValueError(
                    f"Pretrained BEATs requires sample_rate=16000, num_mel_bins=128. "
                    f"Got sample_rate={sample_rate}, num_mel_bins={num_mel_bins}. "
                    f"Use pretrained=False for custom audio parameters."
                )
            if checkpoint_path is None:
                checkpoint_path = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/BEATs/BEATs_iter3_plus_AS2M.pt'

            import os
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"BEATs checkpoint not found: {checkpoint_path}\n"
                    "Download from: https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf"
                )
            self.encoder = BEATs(ckp_path=checkpoint_path, freeze=False)
        else:
            self.encoder = BEATs(ckp_path=None, freeze=False)
            # Override preprocess for custom sample_rate / n_mels
            self._patch_preprocess()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _patch_preprocess(self):
        """Replace the instance's preprocess with a version using our sample_rate/n_mels."""
        sample_rate = self._sample_rate
        num_mel_bins = self._num_mel_bins

        def custom_preprocess(self_beats, source, fbank_mean=15.41663, fbank_std=6.55582):
            fbanks = []
            for waveform in source:
                waveform = waveform.unsqueeze(0) * 2**15
                fbank = ta_kaldi.fbank(
                    waveform,
                    num_mel_bins=num_mel_bins,
                    sample_frequency=sample_rate,
                    frame_length=25,
                    frame_shift=10,
                )
                fbanks.append(fbank)
            fbank = torch.stack(fbanks, dim=0)
            return (fbank - fbank_mean) / (2 * fbank_std)

        self.encoder.preprocess = types.MethodType(custom_preprocess, self.encoder)

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

        B = x.shape[0]
        wav_lens = torch.ones(B, device=x.device)
        out = self.encoder.extract_features(x, wav_lens=wav_lens)
        feats = out[0] if isinstance(out, tuple) else out
        return feats.mean(dim=1)  # [B, 768]
