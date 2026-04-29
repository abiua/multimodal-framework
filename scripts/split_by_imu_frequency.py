#!/usr/bin/env python
"""按IMU采样频率划分数据集，使用符号链接节省磁盘空间。"""
import os
import csv
from pathlib import Path
from collections import defaultdict
import argparse

# 频率分组：会话前缀 → 频率标签
FREQ_GROUPS = {
    '0411-19': '10hz', '0412-09': '10hz', '0412-19': '10hz',
    '0414-09': '10hz', '0414-17': '10hz', '0415-09': '10hz', '0415-17': '10hz',
    '0417-09': '5hz',  '0417-17': '5hz',  '0418-09': '5hz',  '0418-18': '5hz',
    '0419-09': '5hz',  '0419-18': '5hz',  '0420-09': '5hz',  '0420-17': '5hz',
    '0416-09': '1hz',
}


def load_labels(labels_path: str) -> dict:
    """加载标签CSV，返回 {segment_id: label}"""
    labels = {}
    with open(labels_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                labels[row[0].strip()] = int(row[1])
    return labels


def get_frequency(segment_id: str) -> str:
    """从 segment_id (e.g. '0411-19/event_000_19-38-10/000') 提取频率标签"""
    session = segment_id.split('/')[0]
    return FREQ_GROUPS.get(session, 'unknown')


def main():
    parser = argparse.ArgumentParser(description='按IMU频率划分数据集')
    parser.add_argument('--source', type=str,
                        default='data/multimodal_data_original',
                        help='原始数据集目录')
    parser.add_argument('--output', type=str,
                        default='data/multimodal_freq_split',
                        help='输出目录')
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    segments_dir = source / 'segments_out'
    labels_path = source / 'labels.csv'

    labels = load_labels(str(labels_path))
    print(f"加载了 {len(labels)} 条标签")

    # 按频率分组统计
    freq_stats = defaultdict(lambda: {'count': 0, 'classes': defaultdict(int)})

    for segment_id, label in labels.items():
        freq = get_frequency(segment_id)
        freq_stats[freq]['count'] += 1
        freq_stats[freq]['classes'][label] += 1

    # 打印统计
    print("\n=== 频率分组统计 ===")
    for freq in ['10hz', '5hz', '1hz']:
        stats = freq_stats[freq]
        print(f"\n{freq.upper()}: {stats['count']} 片段")
        for cls in sorted(stats['classes']):
            print(f"  类别 {cls}: {stats['classes'][cls]}")

    # 创建符号链接
    print("\n=== 创建符号链接 ===")
    segments_abs = segments_dir.resolve()

    for segment_id, label in labels.items():
        freq = get_frequency(segment_id)
        if freq == 'unknown':
            continue

        # 目标: output/freq/label_class/segment_id
        target_dir = output / freq / str(label)
        target_dir.mkdir(parents=True, exist_ok=True)

        # session/event/seg_name
        parts = segment_id.split('/')
        session, event, seg_name = parts[0], parts[1], parts[2]
        src_dir = segments_abs / session / event / seg_name

        if not src_dir.exists():
            continue

        # 为每个模态文件创建符号链接
        # 用唯一名称避免冲突: {session}_{event}_{seg_name}_{mod}.ext
        prefix = f"{session}_{event}_{seg_name}"
        for mod_file in src_dir.iterdir():
            if mod_file.is_file():
                ext = mod_file.suffix
                link_name = target_dir / f"{prefix}{ext}"
                if not link_name.exists():
                    link_name.symlink_to(mod_file.resolve())

    print("\n完成！")
    for freq in ['10hz', '5hz', '1hz']:
        freq_dir = output / freq
        if freq_dir.exists():
            total = sum(1 for _ in freq_dir.rglob('*.mp4'))
            print(f"{freq}: {total} 视频文件")


if __name__ == '__main__':
    main()
