"""图像加载器"""

import torch
import torchvision.transforms as T
from PIL import Image
from typing import Dict, Any
from ..registry import BaseLoader, register_loader


@register_loader('image_loader', description='标准图像加载器', modality='image')
class ImageLoader(BaseLoader):
    """标准图像加载器"""
    
    def __init__(self, image_size: int = 224, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.transform_train = self._build_train_transform()
        self.transform_val = self._build_val_transform()
    
    def _build_train_transform(self):
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _build_val_transform(self):
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')
    
    def get_transform(self, is_training: bool = True):
        return self.transform_train if is_training else self.transform_val


@register_loader('image_loader_simple', description='简单图像加载器（无增强）', modality='image')
class ImageLoaderSimple(BaseLoader):
    """简单图像加载器"""
    
    def __init__(self, image_size: int = 224, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')
    
    def get_transform(self, is_training: bool = True):
        return self.transform