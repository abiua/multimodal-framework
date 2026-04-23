"""立体声 Mel 频谱图加载器。

放置路径:
    datasets/loaders/audio_loader_stereo.py
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch

from ..registry import BaseLoader, register_loader

LOGGER = logging.getLogger(__name__)


@register_loader('audio_loader_stereo', description='双通道 Mel 频谱图加载器', modality='audio')
class StereoAudioLoader(BaseLoader):
    """将音频文件转换为双通道 Mel 频谱图。

    输出:
        {'mel_spectrogram': Tensor[C=2, H=n_mels, W=time_steps]}
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_length: int = 160000,
        n_mels: int = 224,
        time_steps: int = 224,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.n_mels = n_mels
        self.time_steps = time_steps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.power = power

    def load(self, path: str) -> np.ndarray:
        try:
            import librosa
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, _ = librosa.load(path, sr=self.sample_rate, mono=False)
        except Exception as exc:
            LOGGER.warning("音频文件加载失败，返回静音双通道数据: %s; err=%s", path, exc)
            audio = np.zeros((2, self.max_length), dtype=np.float32)
            return audio

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2 and audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        elif audio.ndim == 2 and audio.shape[0] > 2:
            audio = audio[:2]
        elif audio.ndim != 2:
            raise ValueError(f"不支持的音频形状: {audio.shape}")

        if audio.shape[1] < self.max_length:
            pad = self.max_length - audio.shape[1]
            audio = np.pad(audio, ((0, 0), (0, pad)), mode='constant')
        else:
            audio = audio[:, :self.max_length]

        return audio

    def to_melspectrogram(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        import librosa

        mels = []
        for ch in range(audio.shape[0]):
            mel = librosa.feature.melspectrogram(
                y=audio[ch],
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=self.power,
            )
            mel = mel + 1e-6
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = librosa.util.fix_length(mel, size=self.time_steps, axis=1)
            mel = mel.astype(np.float32)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            mels.append(mel)

        mel_tensor = torch.tensor(np.stack(mels, axis=0), dtype=torch.float32)
        return {'mel_spectrogram': mel_tensor}

    def get_transform(self, is_training: bool = True):
        return self.to_melspectrogram