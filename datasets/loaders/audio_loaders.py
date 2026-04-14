"""音频加载器"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from ..registry import BaseLoader, register_loader


@register_loader('audio_loader', description='标准音频加载器（梅尔频谱图）', modality='audio')
class AudioLoader(BaseLoader):
    """标准音频加载器"""
    
    def __init__(self, sample_rate: int = 16000, max_length: int = 160000, 
                 n_mels: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.n_mels = n_mels
    
    def load(self, path: str) -> np.ndarray:
        try:
            import librosa
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, _ = librosa.load(path, sr=self.sample_rate)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(f"音频文件加载失败，返回静音数据: {path}")
            audio = np.zeros(self.max_length, dtype=np.float32)
        
        # 填充或截断
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]
        
        return audio
    
    def to_melspectrogram(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=self.n_mels)
        mel = librosa.power_to_db(mel, ref=np.max)
        return {'mel_spectrogram': torch.tensor(mel, dtype=torch.float32)}
    
    def get_transform(self, is_training: bool = True):
        return self.to_melspectrogram


@register_loader('audio_loader_raw', description='原始波形音频加载器', modality='audio')
class AudioLoaderRaw(BaseLoader):
    """原始波形音频加载器"""
    
    def __init__(self, sample_rate: int = 16000, max_length: int = 160000, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def load(self, path: str) -> torch.Tensor:
        try:
            import librosa
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, _ = librosa.load(path, sr=self.sample_rate)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(f"音频文件加载失败，返回静音数据: {path}")
            audio = np.zeros(self.max_length, dtype=np.float32)
        
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]
        
        return {'audio': torch.tensor(audio, dtype=torch.float32)}
    
    def get_transform(self, is_training: bool = True):
        return lambda x: x