"""Video数据加载器 — 从MP4提取固定帧数。"""
import torch
import numpy as np
from typing import Dict, Optional
from ..registry import BaseLoader, register_loader


@register_loader('video_loader_frames', description='MP4视频帧提取加载器', modality='video')
class VideoFrameLoader(BaseLoader):
    """从MP4视频中均匀采样固定帧数。

    Args:
        num_frames: 采样帧数 (default 16)
        frame_size: 帧resize尺寸 (default 224)
        mean/std: 归一化参数 (default ImageNet-like for video)
    """

    def __init__(self, num_frames=16, frame_size=224,
                 mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.mean = mean or [0.45, 0.45, 0.45]
        self.std = std or [0.225, 0.225, 0.225]

    def load(self, path: str) -> 'np.ndarray':
        """Read video and sample frames uniformly."""
        import torchvision.io
        video, _, _ = torchvision.io.read_video(path, output_format='TCHW', pts_unit='sec')
        total_frames = video.shape[0]
        if total_frames == 0:
            raise ValueError(f"Video has 0 frames: {path}")
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return video[indices].numpy().astype(np.float32) / 255.0

    def transform(self, data: 'np.ndarray') -> Dict[str, torch.Tensor]:
        """Resize and normalize frames."""
        frames = torch.from_numpy(data).float()  # [T, C, H, W]
        if frames.shape[-1] != self.frame_size or frames.shape[-2] != self.frame_size:
            frames = torch.nn.functional.interpolate(
                frames, size=(self.frame_size, self.frame_size),
                mode='bilinear', align_corners=False)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return {'video': (frames - mean) / std}

    def get_transform(self, is_training=True):
        return self.transform
