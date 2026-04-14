"""
DataLoader Registry - 数据加载器注册系统

使用方式：
1. 在datasets/loaders/目录下创建加载器文件
2. 使用 @register_loader 装饰器注册
3. 在配置文件中通过 type 指定加载器名称

示例：
    @register_loader('my_image_loader')
    class MyImageLoader(BaseLoader):
        def load(self, path):
            # 加载逻辑
            return data
"""

import os
import importlib
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List
from pathlib import Path
from PIL import Image


class BaseLoader(ABC):
    """数据加载器基类"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """加载数据"""
        pass
    
    @abstractmethod
    def get_transform(self, is_training: bool = True):
        """获取数据变换"""
        pass


class LoaderRegistry:
    """数据加载器注册表"""
    
    _loaders: Dict[str, Type[BaseLoader]] = {}
    _loader_info: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, description: str = "", modality: str = "any"):
        """注册加载器的装饰器"""
        def decorator(loader_cls):
            cls._loaders[name] = loader_cls
            cls._loader_info[name] = {
                'class': loader_cls,
                'description': description,
                'modality': modality,
                'module': loader_cls.__module__
            }
            return loader_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseLoader]:
        """获取加载器类"""
        if name not in cls._loaders:
            cls._auto_load()
        
        if name not in cls._loaders:
            available = list(cls._loaders.keys())
            raise ValueError(f"未找到加载器: '{name}'，可用加载器: {available}")
        return cls._loaders[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLoader:
        """创建加载器实例"""
        loader_cls = cls.get(name)
        return loader_cls(**kwargs)
    
    @classmethod
    def list_loaders(cls, modality: Optional[str] = None) -> Dict[str, Dict]:
        """列出所有可用加载器"""
        cls._auto_load()
        
        if modality:
            return {
                name: info for name, info in cls._loader_info.items()
                if info['modality'] == modality or info['modality'] == 'any'
            }
        return cls._loader_info.copy()
    
    @classmethod
    def _auto_load(cls):
        """自动加载loaders目录下的所有加载器"""
        loaders_dir = os.path.join(os.path.dirname(__file__), 'loaders')
        if not os.path.exists(loaders_dir):
            return
        
        for filename in os.listdir(loaders_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                try:
                    importlib.import_module(f'.loaders.{module_name}', package='datasets')
                except Exception:
                    pass


# 便捷装饰器
def register_loader(name: str, description: str = "", modality: str = "any"):
    """注册加载器的便捷装饰器"""
    return LoaderRegistry.register(name, description, modality)


# 自动加载
LoaderRegistry._auto_load()