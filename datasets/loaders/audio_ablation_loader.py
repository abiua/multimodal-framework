"""Ablation audio loader — applies frequency filtering to raw waveform before mel.

Supports four filter modes for controlled frequency-band ablation:
  - full:           no filtering, 0-24kHz (Nyquist for 48kHz)
  - lowpass_8k:     Butterworth lowpass, cutoff 8kHz, zero-phase
  - bandpass_8_24k: Butterworth bandpass 8-23.5kHz, zero-phase
  - resample_16k:   downsample to 16kHz → upsample back to 48kHz

All modes output the same waveform length (48k samples) and the same
mel spectrogram dimensions. Only frequency content differs.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch

from ..registry import BaseLoader, register_loader

LOGGER = logging.getLogger(__name__)


@register_loader('audio_ablation_loader', description='频率消融音频加载器（支持滤波模式）', modality='audio')
class AblationAudioLoader(BaseLoader):
    """Load mono audio at 48kHz, apply frequency filter, extract log-mel spectrogram.

    Output:
        {'mel_spectrogram': Tensor[1, n_mels, time_steps]}
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        max_length: int = 48000,
        n_mels: int = 128,
        time_steps: int = 224,
        n_fft: int = 2048,
        hop_length: int = 480,
        fmax: float = 24000,
        filter_mode: str = 'full',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.n_mels = n_mels
        self.time_steps = time_steps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.filter_mode = filter_mode

        valid_modes = {'full', 'lowpass_8k', 'bandpass_8_24k', 'resample_16k'}
        if filter_mode not in valid_modes:
            raise ValueError(f"filter_mode must be one of {valid_modes}, got '{filter_mode}'")

    # ------------------------------------------------------------------
    # Filter implementations (applied to raw waveform)
    # ------------------------------------------------------------------

    def _apply_lowpass_8k(self, y: np.ndarray) -> np.ndarray:
        """8th-order Butterworth lowpass, cutoff=8000Hz, zero-phase."""
        from scipy.signal import butter, sosfiltfilt
        sos = butter(N=8, Wn=8000, btype='low', fs=self.sample_rate, output='sos')
        return sosfiltfilt(sos, y)

    def _apply_bandpass_8_24k(self, y: np.ndarray) -> np.ndarray:
        """8th-order Butterworth bandpass [8000, 23500]Hz, zero-phase.
        Upper bound set to 23500 to avoid Nyquist boundary (24000).
        """
        from scipy.signal import butter, sosfiltfilt
        sos = butter(N=8, Wn=[8000, 23500], btype='band', fs=self.sample_rate, output='sos')
        return sosfiltfilt(sos, y)

    def _apply_resample_16k(self, y: np.ndarray) -> np.ndarray:
        """Downsample to 16kHz then upsample back to 48kHz.
        Simulates information loss of real 16kHz recording pipeline.
        """
        import librosa
        y_16k = librosa.resample(y, orig_sr=self.sample_rate, target_sr=16000)
        y_48k = librosa.resample(y_16k, orig_sr=16000, target_sr=self.sample_rate)
        if len(y_48k) < self.max_length:
            y_48k = np.pad(y_48k, (0, self.max_length - len(y_48k)))
        else:
            y_48k = y_48k[:self.max_length]
        return y_48k.astype(np.float32)

    # ------------------------------------------------------------------
    # Load + transform
    # ------------------------------------------------------------------

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

        # Pad or truncate to max_length
        if audio.shape[0] < self.max_length:
            pad = self.max_length - audio.shape[0]
            audio = np.pad(audio, (0, pad), mode='constant')
        else:
            audio = audio[:self.max_length]

        # Apply frequency filter
        if self.filter_mode == 'lowpass_8k':
            audio = self._apply_lowpass_8k(audio)
        elif self.filter_mode == 'bandpass_8_24k':
            audio = self._apply_bandpass_8_24k(audio)
        elif self.filter_mode == 'resample_16k':
            audio = self._apply_resample_16k(audio)
        # 'full': no filtering

        return audio

    def to_melspectrogram(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.fmax,
            power=2.0,
        )
        mel = mel + 1e-9  # avoid log(0)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = librosa.util.fix_length(mel, size=self.time_steps, axis=1)
        mel = mel.astype(np.float32)

        # No per-sample normalization — BatchNorm in the model handles it.
        # Per-sample z-score would compensate for missing frequency bands
        # and confound the ablation.

        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]
        return {'mel_spectrogram': mel_tensor}

    def get_transform(self, is_training: bool = True):
        return self.to_melspectrogram
