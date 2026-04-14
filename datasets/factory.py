import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Optional, Any
from pathlib import Path

from .registry import LoaderRegistry, BaseLoader
from utils.distributed import is_dist_avail_and_initialized, get_world_size, get_rank

# 默认加载器映射
DEFAULT_LOADERS = {
    'image': 'image_loader',
    'text': 'text_loader',
    'audio': 'audio_loader',
    'video': 'video_loader'
}


class MultimodalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(
        self,
        data_path: str,
        class_names: List[str],
        modalities: List[str],
        loaders: Dict[str, BaseLoader],
        is_training: bool = True
    ):
        self.data_path = Path(data_path)
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.modalities = modalities
        self.loaders = loaders
        self.is_training = is_training
        
        # 模态文件扩展名映射
        self.modality_extensions = {
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'text': ['.txt'],
            'audio': ['.wav', '.mp3', '.flac', '.ogg'],
            'video': ['.mp4', '.avi', '.mov', '.mkv'],
            'wave': ['.csv']
        }
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载样本列表"""
        samples = []
        
        for class_name in self.class_names:
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            sample_ids = self._get_sample_ids(class_dir)
            
            for sample_id in sample_ids:
                sample = {
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'sample_id': sample_id
                }

                valid = True
                
                for modality in self.modalities:
                    path = self._find_modality_file(class_dir, sample_id, modality)
                    if path:
                        sample[f'{modality}_path'] = str(path)
                    else:
                        valid = False
                        break

                if valid:
                    samples.append(sample)
        
        return samples
    
    def _get_sample_ids(self, class_dir: Path) -> List[str]:
        first_modality = self.modalities[0]
        extensions = self.modality_extensions.get(first_modality, [])
        
        sample_ids = []
        
        # ✅ 优先查子目录
        modality_dir = class_dir / first_modality
        if modality_dir.exists():
            for file in modality_dir.iterdir():
                if file.is_file() and file.suffix.lower() in extensions:
                    sample_ids.append(file.stem)
            return sample_ids
        
        # fallback：类目录
        for file in class_dir.iterdir():
            if file.is_file() and file.suffix.lower() in extensions:
                sample_ids.append(file.stem)

        return sample_ids
    
    def _find_modality_file(self, class_dir: Path, sample_id: str, modality: str) -> Optional[Path]:
        """查找模态文件"""
        extensions = self.modality_extensions.get(modality, [])
        
        # 直接在类目录下查找
        for ext in extensions:
            path = class_dir / f"{sample_id}{ext}"
            if path.exists():
                return path
        
        # 在子目录中查找
        modality_dir = class_dir / modality
        if modality_dir.exists():
            for ext in extensions:
                path = modality_dir / f"{sample_id}{ext}"
                if path.exists():
                    return path
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        item = {
            'class_idx': torch.tensor(sample['class_idx'], dtype=torch.long),
            'class_name': sample['class_name'],
            'sample_id': sample['sample_id']
        }
        
        for modality in self.modalities:
            path_key = f'{modality}_path'
            if path_key not in sample:
                continue
            
            path = sample[path_key]
            loader = self.loaders.get(modality)
            
            if loader is None:
                continue
            
            # 加载数据
            data = loader.load(path)
            
            # 应用变换
            transform = loader.get_transform(self.is_training)
            if transform:
                data = transform(data)
            
            if isinstance(data, dict):
                if len(data) == 1:
                    item[modality] = next(iter(data.values()))
                else:
                    item.update({f'{modality}_{k}': v for k, v in data.items()})
            else:
                item[modality] = data

            # 添加到item
            # if modality == 'text' and isinstance(data, dict):
            #     item.update({f'text_{k}': v for k, v in data.items()})
            # elif modality == 'audio' and isinstance(data, dict):
            #     item.update({f'audio_{k}': v for k, v in data.items()})
            # elif modality == 'wave' and isinstance(data, dict):
            #     item.update({f'wave_{k}': v for k, v in data.items()})
            # else:
            #     item[modality] = data
        # print("DEBUG item keys:", list(item.keys()))
        return item


class DataFactory:
    """数据工厂"""
    
    def __init__(self, config):
        self.config = config
        self.class_names = config.classes.class_names
        self.num_classes = config.classes.num_classes
        
        if len(self.class_names) != self.num_classes:
            raise ValueError(f"类别数量不匹配: num_classes={self.num_classes}, "
                           f"但class_names有{len(self.class_names)}个")
        
        # 创建各模态的加载器
        self.loaders = self._create_loaders()
    
    def _create_loaders(self) -> Dict[str, BaseLoader]:
        """为每个模态创建加载器"""
        loaders = {}
        
        for modality in self.config.data.modalities:
            # 获取该模态的加载器配置
            if modality in self.config.data.loaders:
                loader_cfg = self.config.data.loaders[modality]
                loader_type = loader_cfg.type
                extra_params = loader_cfg.extra_params
            else:
                # 使用默认加载器
                loader_type = DEFAULT_LOADERS.get(modality)
                extra_params = {}
            
            if loader_type is None:
                raise ValueError(f"模态 '{modality}' 没有可用的加载器，请在配置中指定")
            
            # 创建加载器
            loaders[modality] = LoaderRegistry.create(
                loader_type,
                image_size=self.config.data.image_size,
                **extra_params
            )
        
        return loaders
    
    def create_dataset(self, data_path: str, is_training: bool = True) -> MultimodalDataset:
        """创建数据集"""
        return MultimodalDataset(
            data_path=data_path,
            class_names=self.class_names,
            modalities=self.config.data.modalities,
            loaders=self.loaders,
            is_training=is_training
        )
    
    def create_dataloader(self, dataset: MultimodalDataset, is_training: bool = True) -> DataLoader:
        """创建数据加载器"""
        # 检查是否需要分布式采样
        if is_dist_avail_and_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=is_training,
                drop_last=is_training
            )
            shuffle = False  # 使用采样器时不能设置 shuffle
        else:
            sampler = None
            shuffle = is_training
        
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=is_training
        )
    
    def create_train_loader(self) -> DataLoader:
        return self.create_dataloader(
            self.create_dataset(self.config.data.train_path, is_training=True),
            is_training=True
        )
    
    def create_val_loader(self) -> DataLoader:
        return self.create_dataloader(
            self.create_dataset(self.config.data.val_path, is_training=False),
            is_training=False
        )
    
    def create_test_loader(self) -> DataLoader:
        return self.create_dataloader(
            self.create_dataset(self.config.data.test_path, is_training=False),
            is_training=False
        )