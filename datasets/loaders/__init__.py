"""
DataLoaders - 数据加载器

包含所有可用的数据加载器，通过 @register_loader 装饰器自动注册
"""

from ..registry import LoaderRegistry, register_loader

# 导入加载器以触发注册
from . import image_loaders
from . import text_loaders
from . import audio_loaders
from . import video_loaders
from . import wave_loaders

__all__ = ['LoaderRegistry', 'register_loader']