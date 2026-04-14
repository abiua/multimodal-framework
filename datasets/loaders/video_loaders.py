"""视频加载器"""

import torch
import numpy as np
from typing import Dict, Any, List
from ..registry import BaseLoader, register_loader


@register_loader('video_loader', description='视频加载器（抽取帧）', modality='video')
class VideoLoader(BaseLoader):
    """视频加载器 - 抽取固定数量的帧"""
    
    def __init__(self, num_frames: int = 16, image_size: int = 224, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.image_size = image_size
    
    def load(self, path: str) -> List[np.ndarray]:
        """加载视频并抽取帧"""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 均匀抽取帧
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
            # 如果帧数不足，复制最后一帧
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            
            return frames
        except ImportError:
            raise ImportError("请安装opencv-python: pip install opencv-python")
    
    def transform_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """将帧转换为张量"""
        try:
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            tensor_frames = [transform(frame) for frame in frames]
            return torch.stack(tensor_frames)  # (num_frames, C, H, W)
        except:
            # 简单转换
            frames = np.array(frames)
            frames = frames.transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            return torch.tensor(frames, dtype=torch.float32) / 255.0
    
    def get_transform(self, is_training: bool = True):
        return self.transform_frames


@register_loader('video_loader_3d', description='3D视频加载器（用于3D CNN）', modality='video')
class VideoLoader3D(BaseLoader):
    """3D视频加载器 - 输出适合3D CNN的格式"""
    
    def __init__(self, num_frames: int = 16, image_size: int = 112, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.image_size = image_size
    
    def load(self, path: str) -> torch.Tensor:
        """加载视频并返回3D张量 (C, T, H, W)"""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    frames.append(frame)
            
            cap.release()
            
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            
            # 转换为 (C, T, H, W) 格式
            video = np.array(frames)  # (T, H, W, C)
            video = video.transpose(3, 0, 1, 2)  # (C, T, H, W)
            video = video.astype(np.float32) / 255.0
            
            return torch.tensor(video)
        except ImportError:
            raise ImportError("请安装opencv-python: pip install opencv-python")
    
    def get_transform(self, is_training: bool = True):
        return lambda x: x