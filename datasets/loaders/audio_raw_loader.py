"""Raw waveform audio loader — passes waveform directly without mel transform."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch

from ..registry import BaseLoader, register_loader

LOGGER = logging.getLogger(__name__)


@register_loader('audio_raw_loader', description='原始波形加载器（不做频谱变换）', modality='audio')
class RawAudioLoader(BaseLoader):
    """Load audio as raw waveform tensor, suitable for AST/BEATs backbones.

    Output:
        {'waveform': Tensor[T]}
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_length: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.max_length = max_length

    def load(self, path: str) -> np.ndarray:
        try:
            import librosa
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception as exc:
            LOGGER.warning("Audio load failed, returning silence: %s; err=%s", path, exc)
            audio = np.zeros(self.max_length, dtype=np.float32)
            return audio

        audio = np.asarray(audio, dtype=np.float32)

        if audio.shape[0] < self.max_length:
            pad = self.max_length - audio.shape[0]
            audio = np.pad(audio, (0, pad), mode='constant')
        else:
            audio = audio[:self.max_length]

        return audio

    def get_transform(self, is_training: bool = True):
        return self._to_tensor

    @staticmethod
    def _to_tensor(audio: np.ndarray) -> Dict[str, torch.Tensor]:
        return {'waveform': torch.tensor(audio, dtype=torch.float32)}
