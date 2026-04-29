#!/usr/bin/env python
"""按IMU采样频率划分数据集，使用符号链接节省磁盘空间。

自动检测每个会话的IMU采样率（从rel_time列计算），无需手动配置频率映射。
支持任意采样频率（10Hz/50Hz/100Hz/200Hz等），新数据集加入后直接运行即可。
"""
import os
import csv
from pathlib import Path
from collections import defaultdict
import argparse


def detect_session_frequencies(segments_dir: Path) -> dict:
    """自动检测每个会话的IMU采样频率。

    遍历所有会话目录，读取第一条可用片段的imu.csv，
    从rel_time列计算实际采样率，返回 {session: freq_label} 映射。
    """
    session_freqs = {}

    for session_dir in sorted(segments_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name

        # 找到第一个有效片段
        for event_dir in sorted(session_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            for seg_dir in sorted(event_dir.iterdir()):
                if not seg_dir.is_dir():
                    continue
                imu_file = seg_dir / 'imu.csv'
                if not imu_file.exists():
                    continue

                # 解析rel_time列（最后一列）计算采样率
                rel_times = []
                with open(imu_file, encoding='utf-8-sig', newline='') as f:
                    reader = csv.reader(f, skipinitialspace=True)
                    header = next(reader, [])
                    for row in reader:
                        if row:
                            try:
                                rel_times.append(float(row[-1]))
                            except (ValueError, IndexError):
                                continue

                if len(rel_times) > 1:
                    duration = rel_times[-1] - rel_times[0]
                    rate = (len(rel_times) - 1) / duration if duration > 0 else 0
                    label = f'{round(rate)}hz'
                    session_freqs[session_name] = label
                break  # 只取第一个有效片段
            if session_name in session_freqs:
                break  # 已找到，跳到下一个会话

    return session_freqs


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


def main():
    parser = argparse.ArgumentParser(description='按IMU频率自动划分数据集')
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

    # 自动检测各会话频率
    print("=== 检测IMU采样频率 ===")
    session_freqs = detect_session_frequencies(segments_dir)
    for session, freq in sorted(session_freqs.items()):
        print(f"  {session} → {freq}")

    labels = load_labels(str(labels_path))
    print(f"\n加载了 {len(labels)} 条标签")

    # 按频率分组统计
    freq_order = sorted(set(session_freqs.values()),
                        key=lambda x: int(x.replace('hz', '')))
    freq_stats = defaultdict(lambda: {'count': 0, 'classes': defaultdict(int)})

    for segment_id, label in labels.items():
        session = segment_id.split('/')[0]
        freq = session_freqs.get(session, 'unknown')
        freq_stats[freq]['count'] += 1
        freq_stats[freq]['classes'][label] += 1

    # 打印统计
    print("\n=== 频率分组统计 ===")
    for freq in freq_order:
        stats = freq_stats[freq]
        print(f"\n{freq.upper()}: {stats['count']} 片段")
        for cls in sorted(stats['classes']):
            print(f"  类别 {cls}: {stats['classes'][cls]}")
    unknown = freq_stats.get('unknown', {'count': 0})
    if unknown['count'] > 0:
        print(f"\n未知频率: {unknown['count']} 片段")

    # 创建符号链接
    print("\n=== 创建符号链接 ===")
    segments_abs = segments_dir.resolve()

    for segment_id, label in labels.items():
        session = segment_id.split('/')[0]
        freq = session_freqs.get(session, 'unknown')
        if freq == 'unknown':
            continue

        target_dir = output / freq / str(label)
        target_dir.mkdir(parents=True, exist_ok=True)

        parts = segment_id.split('/')
        session, event, seg_name = parts[0], parts[1], parts[2]
        src_dir = segments_abs / session / event / seg_name

        if not src_dir.exists():
            continue

        prefix = f"{session}_{event}_{seg_name}"
        for mod_file in src_dir.iterdir():
            if mod_file.is_file() and not mod_file.name.startswith('.'):
                ext = mod_file.suffix
                link_name = target_dir / f"{prefix}{ext}"
                if not link_name.exists():
                    link_name.symlink_to(mod_file.resolve())

    # 验证符号链接完整性
    broken = []
    for link in output.rglob('*'):
        if link.is_symlink() and not link.exists():
            broken.append(str(link))
    if broken:
        print(f"\n警告: 发现 {len(broken)} 个损坏的符号链接!")
        for b in broken[:10]:
            print(f"  {b}")
    else:
        print("\n所有符号链接验证通过 (0 个损坏)")

    # 按频率汇总
    print("\n完成！")
    for freq in freq_order:
        freq_dir = output / freq
        if freq_dir.exists():
            total = sum(1 for _ in freq_dir.rglob('*.mp4'))
            print(f"{freq}: {total} 视频文件")


if __name__ == '__main__':
    main()
