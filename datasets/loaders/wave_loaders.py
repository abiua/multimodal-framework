"""Wave数据加载器 - 处理CSV格式的六轴传感器数据"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from ..registry import BaseLoader, register_loader


@register_loader('wave_loader', description='Wave传感器数据加载器', modality='wave')
class WaveLoader(BaseLoader):
    """Wave传感器数据加载器 - 处理CSV格式的六轴传感器数据"""

    def __init__(self, max_length: int = 512, num_features: int = 6, normalize: bool = True, **kwargs):
        """
        Args:
            max_length: 最大序列长度
            num_features: 特征数量（六轴传感器为6）
            normalize: 是否进行归一化
        """
        super().__init__(**kwargs)
        self.max_length = max_length
        self.num_features = num_features
        self.normalize = normalize

    def load(self, path: str) -> np.ndarray:
        """加载CSV文件"""
        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1)
        except:
            try:
                import pandas as pd
                df = pd.read_csv(path)
                data = df.values
            except:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    data = []
                    for line in lines[1:]:
                        try:
                            row = [float(x) for x in line.strip().split(',')]
                            data.append(row)
                        except:
                            continue
                    data = np.array(data)

        return data

    def transform(self, data: np.ndarray) -> Dict[str, torch.Tensor]:
        """转换数据为模型输入格式"""
        data = np.asarray(data, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            padding = np.zeros((self.max_length - len(data), data.shape[1]))
            data = np.vstack([data, padding])

        if data.shape[1] > self.num_features:
            data = data[:, :self.num_features]
        elif data.shape[1] < self.num_features:
            padding = np.zeros((data.shape[0], self.num_features - data.shape[1]))
            data = np.hstack([data, padding])

        if self.normalize:
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            std[std == 0] = 1
            data = (data - mean) / std
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        attention_mask = np.ones(self.max_length)
        return {
            'wave': torch.tensor(data, dtype=torch.float32)
        }

    def get_transform(self, is_training: bool = True):
        return self.transform


@register_loader('wave_loader_raw', description='原始Wave数据加载器', modality='wave')
class WaveLoaderRaw(BaseLoader):
    """原始Wave数据加载器 - 不进行任何预处理"""

    def __init__(self, max_length: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def load(self, path: str) -> np.ndarray:
        """加载CSV文件"""
        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1)
        except:
            with open(path, 'r') as f:
                lines = f.readlines()
                data = []
                for line in lines[1:]:
                    try:
                        row = [float(x) for x in line.strip().split(',')]
                        data.append(row)
                    except:
                        continue
                data = np.array(data)
        return data

    def transform(self, data: np.ndarray) -> torch.Tensor:
        """简单的张量化"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if len(data) > self.max_length:
            data = data[:self.max_length]

        return {'wave':torch.tensor(data, dtype=torch.float32)}

    def get_transform(self, is_training: bool = True):
        return self.transform