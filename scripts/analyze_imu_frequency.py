#!/usr/bin/env python
"""分析不同IMU采样频率对信号特征的影响。"""
import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse


def load_imu_df(csv_path: str) -> pd.DataFrame:
    """加载IMU CSV为DataFrame，兼容中文表头和时间字符串。"""
    rows = []
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if row:
                rows.append(row)

    header = [str(c).strip() for c in rows[0]]
    ncols = len(header)
    data_rows = [row[:ncols] for row in rows[1:]]
    df = pd.DataFrame(data_rows, columns=header)
    df.columns = [str(c).strip() for c in df.columns]

    # 转换六轴数值列
    numeric_cols = ['加速度X(g)', '加速度Y(g)', '加速度Z(g)',
                    '角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def detect_session_frequencies(segments_dir: Path) -> dict:
    """自动检测每个会话的IMU采样频率。"""
    session_freqs = {}
    for session_dir in sorted(segments_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name
        for event_dir in sorted(session_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            for seg_dir in sorted(event_dir.iterdir()):
                if not seg_dir.is_dir():
                    continue
                imu_file = seg_dir / 'imu.csv'
                if not imu_file.exists():
                    continue
                rel_times = []
                with open(imu_file, encoding='utf-8-sig', newline='') as f:
                    reader = csv.reader(f, skipinitialspace=True)
                    next(reader, [])
                    for row in reader:
                        if row:
                            try:
                                rel_times.append(float(row[-1]))
                            except (ValueError, IndexError):
                                continue
                if len(rel_times) > 1:
                    duration = rel_times[-1] - rel_times[0]
                    rate = (len(rel_times) - 1) / duration if duration > 0 else 0
                    session_freqs[session_name] = f'{round(rate)}hz'
                break
            if session_name in session_freqs:
                break
    return session_freqs


def extract_signal_features(df: pd.DataFrame) -> dict:
    """提取六轴信号特征"""
    accel_cols = ['加速度X(g)', '加速度Y(g)', '加速度Z(g)']
    gyro_cols = ['角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']

    features = {}
    for name, cols in [('accel', accel_cols), ('gyro', gyro_cols)]:
        data = df[cols].values
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) == 0:
            continue
        features[f'{name}_mean'] = np.mean(data, axis=0)
        features[f'{name}_std'] = np.std(data, axis=0)
        features[f'{name}_rms'] = np.sqrt(np.mean(data**2, axis=0))
        features[f'{name}_range'] = np.ptp(data, axis=0)
        features[f'{name}_energy'] = np.sum(data**2, axis=0)

    features['n_samples'] = len(df)
    return features


def main():
    parser = argparse.ArgumentParser(description='分析IMU频率影响')
    parser.add_argument('--source', type=str,
                        default='data/multimodal_data_original',
                        help='原始数据集目录')
    parser.add_argument('--output', type=str, default='analysis_output',
                        help='分析输出目录')
    args = parser.parse_args()

    source = Path(args.source)
    segments_dir = source / 'segments_out'
    labels_path = source / 'labels.csv'

    # 检测频率
    print("检测IMU采样频率...")
    session_freqs = detect_session_frequencies(segments_dir)
    freq_order = sorted(set(session_freqs.values()),
                        key=lambda x: int(x.replace('hz', '')))
    print(f"发现频率: {freq_order}")

    # 加载标签
    labels = {}
    with open(labels_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                labels[row[0].strip()] = int(row[1])

    # 收集各频率组的信号特征
    freq_features = defaultdict(list)

    for segment_id, label in labels.items():
        session = segment_id.split('/')[0]
        freq = session_freqs.get(session, 'unknown')
        if freq == 'unknown':
            continue

        parts = segment_id.split('/')
        imu_path = segments_dir / parts[0] / parts[1] / parts[2] / 'imu.csv'
        if not imu_path.exists():
            continue

        try:
            df = load_imu_df(str(imu_path))
            feats = extract_signal_features(df)
            feats['class'] = label
            feats['segment_id'] = segment_id
            freq_features[freq].append(feats)
        except Exception as e:
            print(f"Warning: 处理 {segment_id} 失败: {e}")

    # 打印对比分析
    print("\n" + "=" * 70)
    print("IMU频率影响分析报告")
    print("=" * 70)

    for freq in freq_order:
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        print(f"\n--- {freq.upper()} ({len(feats_list)} 片段) ---")
        print(f"  平均采样点数: {np.mean([f['n_samples'] for f in feats_list]):.1f}")

        for metric in ['accel_std', 'accel_rms', 'accel_range', 'accel_energy',
                       'gyro_std', 'gyro_rms', 'gyro_range', 'gyro_energy']:
            vals = np.array([np.mean(f[metric]) for f in feats_list if metric in f])
            if len(vals):
                print(f"  {metric}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # 分类别对比
    print("\n--- 各类别信号特征 (加速度RMS均值) ---")
    for freq in freq_order:
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        by_class = defaultdict(list)
        for f in feats_list:
            if 'accel_rms' in f:
                by_class[f['class']].append(np.mean(f['accel_rms']))
        print(f"\n{freq.upper()}:")
        for cls in sorted(by_class):
            vals = by_class[cls]
            print(f"  类别 {cls} (n={len(vals)}): mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # 频率分辨率影响
    print("\n--- 频率分辨率影响 ---")
    for freq in freq_order:
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        accel_peaks = []
        for f in feats_list:
            if 'accel_range' in f:
                accel_peaks.append(np.max(f['accel_range']))
        print(f"{freq}: 加速度最大峰峰值 = {np.mean(accel_peaks):.4f}g "
              f"(范围: {np.min(accel_peaks):.4f} - {np.max(accel_peaks):.4f})")

    # 保存文本报告
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'frequency_analysis.txt'
    with open(report_path, 'w') as f:
        f.write("IMU频率影响分析报告\n")
        f.write("=" * 70 + "\n")
        for freq in freq_order:
            feats_list = freq_features[freq]
            if not feats_list:
                continue
            f.write(f"\n{freq.upper()} ({len(feats_list)} 片段)\n")
            f.write(f"  平均采样点数: {np.mean([x['n_samples'] for x in feats_list]):.1f}\n")
            for metric in ['accel_std', 'accel_rms', 'accel_range', 'accel_energy',
                           'gyro_std', 'gyro_rms', 'gyro_range', 'gyro_energy']:
                vals = np.array([np.mean(x[metric]) for x in feats_list if metric in x])
                if len(vals):
                    f.write(f"  {metric}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}\n")
    print(f"\n报告已保存到 {report_path}")

    print("\n分析完成。")


if __name__ == '__main__':
    main()
