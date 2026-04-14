#!/usr/bin/env python
"""
Fish Feeding Intensity 数据集分割和转换脚本

将原始数据集按照 7:2:1 的比例分割成 train/val/test，
并转换成项目可接受的多模态数据格式。
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse


def collect_samples(source_dir: str) -> dict:
    """
    收集所有样本信息
    
    Args:
        source_dir: 原始数据集目录
        
    Returns:
        样本字典 {sample_id: {'class': class_name, 'image': path, 'audio': path, 'wave': path}}
    """
    source_path = Path(source_dir)
    samples = {}
    
    # 获取所有类别
    image_dir = source_path / 'Image'
    classes = [d.name for d in image_dir.iterdir() if d.is_dir()]
    
    print(f"发现类别: {classes}")
    
    for class_name in classes:
        # 获取该类别的所有图像文件
        image_class_dir = image_dir / class_name
        audio_class_dir = source_path / 'Audio' / class_name
        wave_class_dir = source_path / 'Wave' / class_name
        
        image_files = sorted([f for f in image_class_dir.glob('*.jpg')])
        
        for image_file in image_files:
            # 提取样本ID (image_0001 -> 0001)
            sample_id = image_file.stem.replace('image_', '')
            
            # 查找对应的音频和波形文件
            audio_file = audio_class_dir / f'audio_{sample_id}.wav'
            wave_file = wave_class_dir / f'wave_{sample_id}.csv'
            
            if audio_file.exists() and wave_file.exists():
                samples[f"{class_name}_{sample_id}"] = {
                    'class': class_name,
                    'image': str(image_file),
                    'audio': str(audio_file),
                    'wave': str(wave_file)
                }
    
    return samples


def split_dataset(samples: dict, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                  test_ratio: float = 0.1, seed: int = 42) -> dict:
    """
    按比例分割数据集
    
    Args:
        samples: 样本字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        分割后的数据集 {'train': [...], 'val': [...], 'test': [...]}
    """
    random.seed(seed)
    
    # 按类别分组
    class_samples = defaultdict(list)
    for sample_id, sample_info in samples.items():
        class_samples[sample_info['class']].append(sample_id)
    
    split_data = {'train': [], 'val': [], 'test': []}
    
    for class_name, sample_ids in class_samples.items():
        # 打乱顺序
        random.shuffle(sample_ids)
        
        n_samples = len(sample_ids)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # 分割
        split_data['train'].extend(sample_ids[:n_train])
        split_data['val'].extend(sample_ids[n_train:n_train + n_val])
        split_data['test'].extend(sample_ids[n_train + n_val:])
        
        print(f"{class_name}: train={n_train}, val={n_val}, test={n_samples - n_train - n_val}")
    
    return split_data


def organize_dataset(source_dir: str, output_dir: str, samples: dict, 
                     split_data: dict, modalities: list = ['image', 'audio', 'wave']):
    """
    组织数据集到项目格式
    
    Args:
        source_dir: 原始数据集目录
        output_dir: 输出目录
        samples: 样本字典
        split_data: 分割数据
        modalities: 使用的模态列表
    """
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        for sample_id in split_data[split]:
            class_name = samples[sample_id]['class']
            class_dir = output_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    for split in ['train', 'val', 'test']:
        print(f"\n处理 {split} 集...")
        
        for sample_id in split_data[split]:
            sample_info = samples[sample_id]
            class_name = sample_info['class']
            class_dir = output_path / split / class_name
            
            # 获取样本编号
            sample_num = sample_id.split('_', 1)[1]
            
            # 复制图像文件
            if 'image' in modalities:
                src_image = Path(sample_info['image'])
                dst_image = class_dir / f'{sample_num}.jpg'
                if not dst_image.exists():
                    shutil.copy2(src_image, dst_image)
            
            # 复制音频文件到 audio 子目录
            if 'audio' in modalities:
                audio_dir = class_dir / 'audio'
                audio_dir.mkdir(exist_ok=True)
                src_audio = Path(sample_info['audio'])
                dst_audio = audio_dir / f'{sample_num}.wav'
                if not dst_audio.exists():
                    shutil.copy2(src_audio, dst_audio)
            
            # 复制波形文件到 wave 子目录
            if 'wave' in modalities:
                wave_dir = class_dir / 'wave'
                wave_dir.mkdir(exist_ok=True)
                src_wave = Path(sample_info['wave'])
                dst_wave = wave_dir / f'{sample_num}.csv'
                if not dst_wave.exists():
                    shutil.copy2(src_wave, dst_wave)
        
        print(f"  {split} 集处理完成: {len(split_data[split])} 样本")


def print_statistics(output_dir: str):
    """打印数据集统计信息"""
    output_path = Path(output_dir)
    
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_dir = output_path / split
        if not split_dir.exists():
            continue
        
        total = 0
        class_stats = {}
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                # 统计图像文件数量
                n_images = len(list(class_dir.glob('*.jpg')))
                class_stats[class_dir.name] = n_images
                total += n_images
        
        print(f"\n{split}:")
        print(f"  总样本数: {total}")
        for class_name, count in sorted(class_stats.items()):
            print(f"  {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Fish Feeding Intensity 数据集分割')
    parser.add_argument('--source', type=str, 
                       default=os.path.expanduser('~/.cache/huggingface/hub/datasets--ShulongZhang--Multimodal_Fish_Feeding_Intensity/snapshots/43d906534ca9f7012c78f1ef23718df20613de40'),
                       help='原始数据集目录')
    parser.add_argument('--output', type=str, default='data/fish_feeding',
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--modalities', nargs='+', default=['image', 'audio', 'wave'],
                       help='使用的模态')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fish Feeding Intensity 数据集分割")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"分割比例: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"随机种子: {args.seed}")
    print(f"模态: {args.modalities}")
    
    # 收集样本
    print("\n收集样本信息...")
    samples = collect_samples(args.source)
    print(f"总样本数: {len(samples)}")
    
    # 分割数据集
    print("\n分割数据集...")
    split_data = split_dataset(samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
    # 组织数据集
    print("\n组织数据集...")
    organize_dataset(args.source, args.output, samples, split_data, args.modalities)
    
    # 打印统计信息
    print_statistics(args.output)
    
    print("\n" + "=" * 60)
    print("数据集分割完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()