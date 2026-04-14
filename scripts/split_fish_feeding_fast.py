#!/usr/bin/env python
"""
Fish Feeding Intensity 数据集分割和转换脚本 (快速版)

使用符号链接而不是复制文件，加快处理速度。
"""

import os
import random
from pathlib import Path
from collections import defaultdict
import argparse


def collect_samples(source_dir: str) -> dict:
    """收集所有样本信息"""
    source_path = Path(source_dir)
    samples = {}
    
    image_dir = source_path / 'Image'
    classes = [d.name for d in image_dir.iterdir() if d.is_dir()]
    
    print(f"发现类别: {classes}")
    
    for class_name in classes:
        image_class_dir = image_dir / class_name
        audio_class_dir = source_path / 'Audio' / class_name
        wave_class_dir = source_path / 'Wave' / class_name
        
        image_files = sorted([f for f in image_class_dir.glob('*.jpg')])
        
        for image_file in image_files:
            sample_id = image_file.stem.replace('image_', '')
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
    """按比例分割数据集"""
    random.seed(seed)
    
    class_samples = defaultdict(list)
    for sample_id, sample_info in samples.items():
        class_samples[sample_info['class']].append(sample_id)
    
    split_data = {'train': [], 'val': [], 'test': []}
    
    for class_name, sample_ids in class_samples.items():
        random.shuffle(sample_ids)
        
        n_samples = len(sample_ids)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        split_data['train'].extend(sample_ids[:n_train])
        split_data['val'].extend(sample_ids[n_train:n_train + n_val])
        split_data['test'].extend(sample_ids[n_train + n_val:])
        
        print(f"{class_name}: train={n_train}, val={n_val}, test={n_samples - n_train - n_val}")
    
    return split_data


def organize_dataset(output_dir: str, samples: dict, split_data: dict, 
                     modalities: list = ['image', 'audio', 'wave']):
    """组织数据集到项目格式（使用符号链接）"""
    output_path = Path(output_dir)
    
    for split in ['train', 'val', 'test']:
        print(f"\n处理 {split} 集...")
        
        for sample_id in split_data[split]:
            sample_info = samples[sample_id]
            class_name = sample_info['class']
            class_dir = output_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            sample_num = sample_id.split('_', 1)[1]
            
            # 创建图像符号链接
            if 'image' in modalities:
                dst_image = class_dir / f'{sample_num}.jpg'
                if not dst_image.exists():
                    os.symlink(sample_info['image'], dst_image)
            
            # 创建音频符号链接
            if 'audio' in modalities:
                audio_dir = class_dir / 'audio'
                audio_dir.mkdir(exist_ok=True)
                dst_audio = audio_dir / f'{sample_num}.wav'
                if not dst_audio.exists():
                    os.symlink(sample_info['audio'], dst_audio)
            
            # 创建波形符号链接
            if 'wave' in modalities:
                wave_dir = class_dir / 'wave'
                wave_dir.mkdir(exist_ok=True)
                dst_wave = wave_dir / f'{sample_num}.csv'
                if not dst_wave.exists():
                    os.symlink(sample_info['wave'], dst_wave)
        
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
    print("Fish Feeding Intensity 数据集分割 (符号链接版)")
    print("=" * 60)
    
    # 收集样本
    print("\n收集样本信息...")
    samples = collect_samples(args.source)
    print(f"总样本数: {len(samples)}")
    
    # 分割数据集
    print("\n分割数据集...")
    split_data = split_dataset(samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
    # 组织数据集
    print("\n组织数据集...")
    organize_dataset(args.output, samples, split_data, args.modalities)
    
    # 打印统计信息
    print_statistics(args.output)
    
    print("\n" + "=" * 60)
    print("数据集分割完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()