# IMU频率划分与影响分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将新采集的多模态测试集按IMU采样频率划分为10Hz/5Hz/1Hz三个子集，分析不同频率对信号特征和模型性能的影响，并修复wave_loader以兼容新的IMU CSV格式。

**Architecture:** 创建频率划分脚本和信号分析脚本两个独立模块。划分脚本按会话级频率标签将segments_out重组为freq_10hz/freq_5hz/freq_1hz目录（符号链接方式节省磁盘空间）。分析脚本对比三种频率下的IMU信号统计特性、各类别分布，并输出可视化报告。

**Tech Stack:** Python, pandas, numpy, matplotlib, pathlib

---

## 频率分组确认

| 频率组 | 会话 | 实际采样率 | 片段数 |
|--------|------|-----------|--------|
| 10Hz | 0411-19, 0412-09, 0412-19, 0414-09, 0414-17, 0415-09, 0415-17 | ~10.0 Hz | 144 |
| 5Hz | 0417-09, 0417-17, 0418-09, 0418-18, 0419-09, 0419-18, 0420-09, 0420-17 | ~5.0 Hz | 173 |
| 1Hz | 0416-09 | ~1.0 Hz | 11 |

分类分布（4类，标签1-4）：1: 65, 2: 54, 3: 60, 4: 149（严重不均衡，类别4占45%）

---

### Task 1: 频率划分脚本 - 将会话按频率分组并创建目录结构

**Files:**
- Create: `scripts/split_by_imu_frequency.py`
- Create: `scripts/__init__.py` (if not exists)

- [ ] **Step 1: 编写频率分组核心逻辑**

```python
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
```

- [ ] **Step 2: 运行频率划分脚本**

Run: `PYTHONPATH=/home/pythoner/abiu/multimodal-framework python scripts/split_by_imu_frequency.py`
Expected: 打印三个频率组的统计信息，创建符号链接目录结构

- [ ] **Step 3: 验证输出目录结构**

Run: `find data/multimodal_freq_split -type f | head -20 && echo "---" && for freq in 10hz 5hz 1hz; do echo "$freq: $(find data/multimodal_freq_split/$freq -name '*.mp4' | wc -l) videos, $(find data/multimodal_freq_split/$freq -name '*.wav' | wc -l) audios, $(find data/multimodal_freq_split/$freq -name '*.csv' | wc -l) imus"; done`
Expected: 三个频率目录下均有对应的模态文件，总数与统计一致

- [ ] **Step 4: 验证符号链接有效**

Run: `python3 -c "
import os
from pathlib import Path
broken = []
for f in Path('data/multimodal_freq_split').rglob('*'):
    if f.is_symlink() and not f.exists():
        broken.append(str(f))
print(f'Broken links: {len(broken)}')
if broken:
    for b in broken[:5]:
        print(f'  {b}')
"` 
Expected: `Broken links: 0`

- [ ] **Step 5: Commit**

```bash
git add scripts/split_by_imu_frequency.py
git commit -m "feat: add IMU frequency-based dataset split script"
```

---

### Task 2: 频率影响分析脚本 - 对比不同频率下的信号特征

**Files:**
- Create: `scripts/analyze_imu_frequency.py`

- [ ] **Step 1: 编写信号对比分析脚本**

```python
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

FREQ_GROUPS = {
    '0411-19': '10hz', '0412-09': '10hz', '0412-19': '10hz',
    '0414-09': '10hz', '0414-17': '10hz', '0415-09': '10hz', '0415-17': '10hz',
    '0417-09': '5hz',  '0417-17': '5hz',  '0418-09': '5hz',  '0418-18': '5hz',
    '0419-09': '5hz',  '0419-18': '5hz',  '0420-09': '5hz',  '0420-17': '5hz',
    '0416-09': '1hz',
}

# IMU CSV中六轴数据列索引 (0-based after CSV parsing)
# 列: 时间,设备名称,片上时间,加速度X,加速度Y,加速度Z,角速度X,角速度Y,角速度Z,...
ACCEL_COLS = slice(3, 6)   # 加速度X,Y,Z
GYRO_COLS = slice(6, 9)    # 角速度X,Y,Z


def load_imu_df(csv_path: str) -> pd.DataFrame:
    """加载IMU CSV为DataFrame"""
    rows = []
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if row:
                rows.append(row)

    header = [str(c).strip() for c in rows[0]]
    data_rows = [row[:31] for row in rows[1:]]
    df = pd.DataFrame(data_rows, columns=header)
    df.columns = [str(c).strip() for c in df.columns]

    # 转换数值列
    numeric_cols = ['加速度X(g)', '加速度Y(g)', '加速度Z(g)',
                    '角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def extract_signal_features(df: pd.DataFrame) -> dict:
    """提取六轴信号特征"""
    accel_cols = ['加速度X(g)', '加速度Y(g)', '加速度Z(g)']
    gyro_cols = ['角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']

    features = {}
    for name, cols in [('accel', accel_cols), ('gyro', gyro_cols)]:
        data = df[cols].values
        data = data[~np.isnan(data).any(axis=1)]  # remove NaN rows
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
        freq = FREQ_GROUPS.get(session, 'unknown')
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
    print("=" * 70)
    print("IMU频率影响分析报告")
    print("=" * 70)

    for freq in ['10hz', '5hz', '1hz']:
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        print(f"\n--- {freq.upper()} ({len(feats_list)} 片段) ---")
        print(f"  平均采样点数: {np.mean([f['n_samples'] for f in feats_list]):.1f}")

        for metric in ['accel_std', 'accel_rms', 'gyro_std', 'gyro_rms']:
            # 堆叠所有片段的该特征（取三轴均值）
            vals = np.array([np.mean(f[metric]) for f in feats_list if metric in f])
            if len(vals):
                print(f"  {metric}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # 分类别对比
    print("\n--- 各类别信号特征 (加速度RMS均值) ---")
    for freq in ['10hz', '5hz', '1hz']:
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

    # 频率分辨率对比：10Hz vs 5Hz 对快速事件捕获能力
    print("\n--- 频率分辨率影响 ---")
    for freq in ['10hz', '5hz', '1hz']:
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        accel_peaks = []
        for f in feats_list:
            if 'accel_range' in f:
                accel_peaks.append(np.max(f['accel_range']))
        print(f"{freq}: 加速度最大峰峰值 = {np.mean(accel_peaks):.4f}g "
              f"(范围: {np.min(accel_peaks):.4f} - {np.max(accel_peaks):.4f})")

    print("\n分析完成。")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行信号分析脚本**

Run: `PYTHONPATH=/home/pythoner/abiu/multimodal-framework python scripts/analyze_imu_frequency.py`
Expected: 打印完整的频率对比分析报告，包含各频率组的信号统计特性

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_imu_frequency.py
git commit -m "feat: add IMU frequency impact analysis script"
```

---

### Task 3: 修复 WaveLoader 以兼容新的 IMU CSV 格式

**Files:**
- Modify: `datasets/loaders/wave_loaders.py:25-46` (WaveLoader.load 方法)

- [ ] **Step 1: 查看当前 load 方法的完整代码**

确认当前代码结构后再修改。

- [ ] **Step 2: 重写 WaveLoader.load 方法**

将 `load` 方法替换为以下实现，正确处理带中文列名和时间字符串的IMU CSV：

```python
def load(self, path: str) -> np.ndarray:
    """加载CSV文件 - 兼容IMU CSV格式"""
    import csv
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if row:
                rows.append(row)

    header = [str(c).strip() for c in rows[0]]
    # 提取六轴数据列 (加速度X/Y/Z + 角速度X/Y/Z)
    # 它们是第4到第9列 (0-based index 3-8)
    data = []
    for row in rows[1:]:
        try:
            vals = [float(row[i]) for i in range(3, min(9, len(row)))]
            data.append(vals)
        except (ValueError, IndexError):
            continue

    return np.array(data, dtype=np.float32)
```

- [ ] **Step 3: 运行单元测试验证**

Run: `python3 -c "
import sys; sys.path.insert(0, '.')
from datasets.loaders.wave_loaders import WaveLoader
import numpy as np

loader = WaveLoader(max_length=512, num_features=6, normalize=True)
data = loader.load('data/multimodal_data_original/segments_out/0411-19/event_000_19-38-10/000/imu.csv')
print(f'Shape: {data.shape}, dtype: {data.dtype}')
assert data.shape[1] == 6, f'Expected 6 features, got {data.shape[1]}'
assert data.shape[0] > 0, 'No rows loaded'

# Test transform
result = loader.transform(data)
print(f'Transform output keys: {list(result.keys())}')
print(f'Wave tensor shape: {result[\"wave\"].shape}')
print('PASS')
"` 
Expected: `Shape: (N, 6), dtype: float32`, transform outputs `wave` tensor

- [ ] **Step 4: 验证不同频率的IMU文件都能正确加载**

Run: `python3 -c "
import sys; sys.path.insert(0, '.')
from datasets.loaders.wave_loaders import WaveLoader
from pathlib import Path

loader = WaveLoader(max_length=512, num_features=6, normalize=True)
base = Path('data/multimodal_data_original/segments_out')

# 测试每种频率的代表性文件
tests = [
    ('10Hz', '0411-19/event_000_19-38-10/000/imu.csv'),
    ('5Hz', '0417-09/event_000_09-05-10/000/imu.csv'),
    ('1Hz', '0416-09/event_000_09-31-04/000/imu.csv'),
]
for label, rel_path in tests:
    path = base / rel_path
    data = loader.load(str(path))
    result = loader.transform(data)
    print(f'{label}: loaded {data.shape[0]} rows → tensor {result[\"wave\"].shape}')
print('ALL PASS')
"` 
Expected: 三个频率均正确加载，10Hz约20行，5Hz约10行，1Hz约3行

- [ ] **Step 5: Commit**

```bash
git add datasets/loaders/wave_loaders.py
git commit -m "fix: update WaveLoader to handle IMU CSV with Chinese headers and time strings"
```

---

### Task 4: 模型在各频率子集上的评估

**Files:**
- Create: `configs/test_imu_freq_10hz.yaml`
- Create: `configs/test_imu_freq_5hz.yaml`
- Create: `configs/test_imu_freq_1hz.yaml`

- [ ] **Step 1: 创建10Hz测试配置**

```yaml
# configs/test_imu_freq_10hz.yaml
data:
  batch_size: 16
  num_workers: 4
  pin_memory: true

  train_path: "data/multimodal_freq_split/10hz"
  val_path: "data/multimodal_freq_split/10hz"
  test_path: "data/multimodal_freq_split/10hz"

  modalities:
    - image
    - audio
    - wave

  loaders:
    image:
      type: image_loader
      extra_params: {}
    audio:
      type: audio_loader_stereo
      extra_params:
        sample_rate: 16000
        max_length: 160000
        n_mels: 224
        time_steps: 224
    wave:
      type: wave_loader
      extra_params:
        max_length: 24
        num_features: 6
        normalize: true

  image_size: 224

classes:
  num_classes: 3
  class_names: ["None", "Strong", "Weak"]

model:
  backbones:
    image:
      type: resnet18
      feature_dim: 512
      pretrained: true
      freeze: false
    audio:
      type: audiocnn
      feature_dim: 512
      pretrained: false
      freeze: false
      extra_params:
        in_channels: 2
    wave:
      type: tcn
      feature_dim: 256
      pretrained: false
      freeze: false

  unified_pipeline:
    token_dim: 256
    position_encodings:
      image: { enabled: true, max_len: 256 }
      audio: { enabled: true, max_len: 256 }
      wave:  { enabled: true, max_len: 24 }
    interaction_blocks:
      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: none
      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: gate
        fusion_kwargs: { gate_hidden_dim: 128, dropout: 0.1 }
      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: cross_attn
        fusion_kwargs: { num_heads: 4, dropout: 0.1 }
      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: token_mix
        fusion_kwargs: { num_heads: 8, dropout: 0.1 }
    mid_fusion_type: attention
    mid_fusion_output_dim: 256
    decision:
      type: identity
      extra_params: {}

  dropout_rate: 0.35

train:
  epochs: 1
  learning_rate: 0.00005
  weight_decay: 0.001
  lr_scheduler: cosine
  warmup_epochs: 0
  optimizer: adamw
  label_smoothing: 0.0
  early_stop: { enabled: false }
  val_interval: 1

eval:
  metrics: [accuracy, precision, recall, f1]
  save_predictions: true
  confusion_matrix: true

system:
  seed: 42
  gpu_ids: [0]
  distributed: false
  fp16: false
  log_interval: 10
  save_interval: 50
  output_dir: output/test_10hz
  tensorboard_enabled: false
  experiment_name: test_10hz
```

- [ ] **Step 2: 创建5Hz和1Hz配置（同上，替换频率路径和output_dir）**

5Hz配置中 `max_length: 12`（约12个采样点/2秒），`train_path: "data/multimodal_freq_split/5hz"`，`output_dir: output/test_5hz`

1Hz配置中 `max_length: 4`（约3个采样点/2秒），`train_path: "data/multimodal_freq_split/1hz"`，`output_dir: output/test_1hz`

- [ ] **Step 3: 运行10Hz评估**

Run: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/pythoner/abiu/multimodal-framework python -m tools.eval --config configs/test_imu_freq_10hz.yaml --checkpoint output/multimodal_v2/best_model.pth`
Expected: 评估完成，输出准确率/precision/recall/F1和混淆矩阵
Note: 模型是3类训练，数据是4类 → 会报类别不匹配错，需要确认现有模型的实际num_classes

- [ ] **Step 4: Commit**

```bash
git add configs/test_imu_freq_10hz.yaml configs/test_imu_freq_5hz.yaml configs/test_imu_freq_1hz.yaml
git commit -m "feat: add per-frequency test configs for IMU frequency evaluation"
```

---

### Task 5: 生成频率影响综合报告

**Files:**
- Modify: `scripts/analyze_imu_frequency.py`（扩展可视化输出）

- [ ] **Step 1: 添加matplotlib可视化图表**

在 `analyze_imu_frequency.py` 的 `main()` 末尾追加：

```python
    # 生成对比图表
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    freqs_order = ['10hz', '5hz', '1hz']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    # 图1: 各频率加速度RMS分布对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, freq, color in zip(axes, freqs_order, colors):
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        rms_vals = [np.mean(f['accel_rms']) for f in feats_list if 'accel_rms' in f]
        ax.hist(rms_vals, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(f'{freq.upper()} (n={len(rms_vals)})')
        ax.set_xlabel('Accel RMS (g)')
        ax.set_ylabel('Frequency')
    fig.suptitle('IMU加速度RMS分布 - 按采样频率', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / 'accel_rms_by_freq.png', dpi=150)
    print(f"图表已保存到 {output_dir / 'accel_rms_by_freq.png'}")

    # 图2: 各频率下类别间可分离性（加速度RMS均值+标准差）
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = 0
    tick_labels = []
    for freq in freqs_order:
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        by_class = defaultdict(list)
        for f in feats_list:
            if 'accel_rms' in f:
                by_class[f['class']].append(np.mean(f['accel_rms']))
        for cls in sorted(by_class):
            vals = by_class[cls]
            ax.bar(x_pos, np.mean(vals), yerr=np.std(vals),
                   color=colors[freqs_order.index(freq)], alpha=0.7,
                   capsize=4, label=f'{freq} class {cls}' if x_pos < 3 else '')
            tick_labels.append(f'{freq}\ncls{cls}')
            x_pos += 1
    ax.set_ylabel('Accel RMS (g)')
    ax.set_title('各类别IMU加速度RMS - 按采样频率', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / 'accel_rms_by_class_freq.png', dpi=150)
    print(f"图表已保存到 {output_dir / 'accel_rms_by_class_freq.png'}")

    # 图3: 采样点数（分辨率）与信号峰峰值关系
    fig, ax = plt.subplots(figsize=(8, 6))
    for freq, color in zip(freqs_order, colors):
        feats_list = freq_features[freq]
        if not feats_list:
            continue
        n_samples = [f['n_samples'] for f in feats_list]
        ranges = [np.max(f['accel_range']) for f in feats_list if 'accel_range' in f]
        ax.scatter(n_samples[:len(ranges)], ranges, c=color, label=freq, alpha=0.6)
    ax.set_xlabel('采样点数 (2秒窗口)')
    ax.set_ylabel('加速度峰峰值 (g)')
    ax.set_title('采样分辨率 vs 信号峰峰值')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'resolution_vs_range.png', dpi=150)
    print(f"图表已保存到 {output_dir / 'resolution_vs_range.png'}")

    print(f"\n所有图表已保存到 {output_dir}/")
```

- [ ] **Step 2: 运行完整分析并生成报告**

Run: `PYTHONPATH=/home/pythoner/abiu/multimodal-framework python scripts/analyze_imu_frequency.py --output analysis_output/imu_freq_report`
Expected: 打印分析报告；`analysis_output/imu_freq_report/` 下生成3张图表

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_imu_frequency.py
git commit -m "feat: add visualization outputs to IMU frequency analysis"
```

---

## 后续步骤（本次plan不实施）

以上5个任务完成后，你将获得：
1. 按频率划分的测试集（符号链接，零额外磁盘开销）
2. IMU信号特征对比分析报告 + 可视化图表
3. 可用的WaveLoader（兼容新IMU CSV格式）
4. 按频率拆分的评测config

下一步（需另开plan）：
- **微调分类头**：现有模型是3分类，需要改为4分类并微调（冻结backbone，只训练classifier head 3→4 权重映射 + fine-tune）
- **低频率数据的插值策略**：1Hz数据只有~3个点/2s，可能需要上采样到与TCN的max_length匹配
- **类别不平衡处理**：类别4占45%，考虑focal loss或重采样
