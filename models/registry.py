"""
ModelZoo - 模型动物园

使用方式：
1. 在modelzoo目录下创建模型文件，如 resnet.py、audiocnn.py
2. 在模型文件中使用 @register_backbone 装饰器注册模型
3. 在配置文件中通过 type 指定模型名称

示例：
    @register_backbone('my_resnet')
    class MyResNet(nn.Module):
        def __init__(self, feature_dim=512, **kwargs):
            ...
        
        def forward(self, x):
            ...
            return features  # shape: (batch, feature_dim)
"""

import os
import importlib
import torch.nn as nn
from typing import Dict, Any, Optional, Type


class ModelZoo:
    """模型动物园 - 管理所有可用的backbone模型"""
    
    _models: Dict[str, Type[nn.Module]] = {}
    _model_info: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, description: str = "", modality: str = "any"):
        """注册模型的装饰器
        
        Args:
            name: 模型名称，用于配置文件中的type字段
            description: 模型描述
            modality: 模态类型 (image/text/audio/any)
        """
        def decorator(model_cls):
            cls._models[name] = model_cls
            cls._model_info[name] = {
                'class': model_cls,
                'description': description,
                'modality': modality,
                'module': model_cls.__module__
            }
            return model_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """获取模型类"""
        if name not in cls._models:
            cls._auto_load_models()
        
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"未找到模型: '{name}'\n"
                f"可用模型: {available}\n"
                f"请确保模型已注册或文件存在于 models/modelzoo/ 目录"
            )
        return cls._models[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """创建模型实例"""
        model_cls = cls.get(name)
        return model_cls(**kwargs)
    
    @classmethod
    def list_models(cls, modality: Optional[str] = None) -> Dict[str, Dict]:
        """列出所有可用模型"""
        cls._auto_load_models()
        
        if modality:
            return {
                name: info for name, info in cls._model_info.items()
                if info['modality'] == modality or info['modality'] == 'any'
            }
        return cls._model_info.copy()
    
    @classmethod
    def _auto_load_models(cls):
        """自动加载modelzoo目录下的所有模型"""
        modelzoo_dir = os.path.dirname(os.path.abspath(__file__))
        
        for filename in os.listdir(modelzoo_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                try:
                    importlib.import_module(f'.modelzoo.{module_name}', package='models')
                except Exception as e:
                    pass  # 静默失败，可能是依赖缺失


# 便捷装饰器
def register_backbone(name: str, description: str = "", modality: str = "any"):
    """注册backbone的便捷装饰器"""
    return ModelZoo.register(name, description, modality)


# 自动加载modelzoo
ModelZoo._auto_load_models()