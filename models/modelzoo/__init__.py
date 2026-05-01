"""
ModelZoo - 模型动物园

包含所有可用的backbone模型，通过 @register_backbone 装饰器自动注册
"""

from ..registry import ModelZoo, register_backbone

# 导入模型以触发注册
from . import image_models
from . import audio_models
from . import text_models
from . import video_models
from . import wave_models
try:
    from . import unireplknet_models
except ImportError:
    pass

__all__ = ['ModelZoo', 'register_backbone']