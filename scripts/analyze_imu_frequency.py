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

    # 生成对比图表
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    freqs_order = freq_order  # use the auto-detected order from earlier
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c']

    # 图1: 各频率加速度RMS分布对比
    fig, axes = plt.subplots(1, len(freqs_order), figsize=(5*len(freqs_order), 5))
    if len(freqs_order) == 1:
        axes = [axes]
    for ax, freq, color in zip(axes, freqs_order, colors[:len(freqs_order)]):
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        rms_vals = [np.mean(f['accel_rms']) for f in feats_list if 'accel_rms' in f]
        ax.hist(rms_vals, bins=min(20, len(rms_vals)//2+1), color=color, alpha=0.7, edgecolor='black')
        ax.set_title(f'{freq.upper()} (n={len(rms_vals)})')
        ax.set_xlabel('Accel RMS (g)')
        ax.set_ylabel('Frequency')
    fig.suptitle('IMU加速度RMS分布 - 按采样频率', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / 'accel_rms_by_freq.png', dpi=150)
    plt.close(fig)
    print(f"图表已保存到 {output_dir / 'accel_rms_by_freq.png'}")

    # 图2: 各频率下类别间可分离性
    fig, ax = plt.subplots(figsize=(max(12, 3*len(freqs_order)*4), 6))
    x_pos = 0
    tick_labels = []
    bar_width = 0.8
    for freq_idx, freq in enumerate(freqs_order):
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        by_class = defaultdict(list)
        for f in feats_list:
            if 'accel_rms' in f:
                by_class[f['class']].append(np.mean(f['accel_rms']))
        for cls in sorted(by_class):
            vals = by_class[cls]
            color = colors[freq_idx % len(colors)]
            ax.bar(x_pos, np.mean(vals), yerr=np.std(vals) if len(vals) > 1 else 0,
                   color=color, alpha=0.7, capsize=4,
                   label=f'{freq}' if cls == sorted(by_class)[0] else '')
            tick_labels.append(f'{freq}\ncls{cls}')
            x_pos += 1
        if freq_idx < len(freqs_order) - 1:
            x_pos += 0.5  # gap between frequency groups
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Accel RMS (g)')
    ax.set_title('各类别IMU加速度RMS - 按采样频率', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    fig.savefig(output_dir / 'accel_rms_by_class_freq.png', dpi=150)
    plt.close(fig)
    print(f"图表已保存到 {output_dir / 'accel_rms_by_class_freq.png'}")

    # 图3: 采样点数与信号峰峰值关系
    fig, ax = plt.subplots(figsize=(8, 6))
    for freq, color in zip(freqs_order, colors[:len(freqs_order)]):
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        n_samples = [f['n_samples'] for f in feats_list]
        ranges_data = [np.max(f['accel_range']) for f in feats_list if 'accel_range' in f]
        if len(n_samples) > len(ranges_data):
            n_samples = n_samples[:len(ranges_data)]
        elif len(ranges_data) > len(n_samples):
            ranges_data = ranges_data[:len(n_samples)]
        ax.scatter(n_samples, ranges_data, c=color, label=freq, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('采样点数 (2秒窗口)')
    ax.set_ylabel('加速度峰峰值 (g)')
    ax.set_title('采样分辨率 vs 信号峰峰值')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'resolution_vs_range.png', dpi=150)
    plt.close(fig)
    print(f"图表已保存到 {output_dir / 'resolution_vs_range.png'}")

    print("\n分析完成。")


if __name__ == '__main__':
    main()
