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
        """加载CSV文件 - 兼容IMU CSV格式（中文表头、时间字符串、多列）"""
        import csv
        rows = []
        with open(path, encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                if row:
                    rows.append(row)

        # 提取六轴数据列 (加速度X/Y/Z + 角速度X/Y/Z)
        # 第4到第9列 (0-based index 3-8)，跳过时间、设备名、片上时间
        data = []
        for row in rows[1:]:
            try:
                vals = [float(row[i]) for i in range(3, min(9, len(row)))]
                data.append(vals)
            except (ValueError, IndexError):
                continue

        return np.array(data, dtype=np.float32)

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


@register_loader('imu_channel_loader', description='IMU单通道加载器 (accel/gyro/angle)', modality='imu')
class ImuChannelLoader(BaseLoader):
    """从IMU CSV中提取单个物理通道（3列），独立归一化。

    配置:
        channel: 'accel' | 'gyro' | 'angle'
        max_length: int = 512
    """

    CHANNEL_COLS = {
        'accel': (3, 6),   # 加速度X/Y/Z (cols 3,4,5)
        'gyro':  (6, 9),   # 角速度X/Y/Z (cols 6,7,8)
        'angle': (9, 12),  # 角度X/Y/Z (cols 9,10,11)
    }

    def __init__(self, channel: str = 'accel', max_length: int = 512, **kwargs):
        super().__init__(**kwargs)
        if channel not in self.CHANNEL_COLS:
            raise ValueError(f"Unknown IMU channel: {channel}, expected {list(self.CHANNEL_COLS.keys())}")
        self.channel = channel
        self.max_length = max_length
        self.col_start, self.col_end = self.CHANNEL_COLS[channel]

    def load(self, path: str) -> 'np.ndarray':
        import csv
        rows = []
        with open(path, encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                if row:
                    rows.append(row)

        data = []
        for row in rows[1:]:  # Skip header
            try:
                vals = [float(row[i]) for i in range(self.col_start, min(self.col_end, len(row)))]
                if len(vals) == 3:
                    data.append(vals)
            except (ValueError, IndexError):
                continue

        return np.array(data, dtype=np.float32)

    def transform(self, data: 'np.ndarray') -> Dict[str, torch.Tensor]:
        data = np.asarray(data, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            padding = np.zeros((self.max_length - len(data), data.shape[1]))
            data = np.vstack([data, padding])

        # Per-channel independent normalization
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std == 0] = 1.0
        data = (data - mean) / std
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return {self.channel: torch.tensor(data, dtype=torch.float32)}

    def get_transform(self, is_training: bool = True):
        return self.transform