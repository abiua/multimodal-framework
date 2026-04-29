#!/usr/bin/env python
"""准备微调数据：合并所有频率数据并做80/20分层划分"""
import csv
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

source = Path("data/multimodal_data_original")
segments_dir = source / "segments_out"
labels_path = source / "labels.csv"
output_train = Path("data/finetune_train")
output_test = Path("data/finetune_test")

# 加载标签
labels = {}
with open(labels_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            labels[row[0].strip()] = int(row[1])

# 按类别分组
by_class = defaultdict(list)
for seg_id, label in labels.items():
    by_class[label].append(seg_id)

# 分层80/20划分
train_ids, test_ids = [], []
for cls, segs in sorted(by_class.items()):
    random.shuffle(segs)
    n_train = max(1, int(len(segs) * 0.8))
    train_ids.extend(segs)
    test_ids.extend(segs[n_train:])
    print(f"Class {cls}: {len(segs)} total, {n_train} train, {len(segs) - n_train} test")

print(f"\nTotal: {len(train_ids)} train, {len(test_ids)} test")

# 创建符号链接
segments_abs = segments_dir.resolve()

for split_name, split_ids, output_dir in [
    ("train", train_ids, output_train),
    ("test", test_ids, output_test)
]:
    # 清理旧目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    for seg_id in split_ids:
        parts = seg_id.split('/')
        session, event, seg_name = parts[0], parts[1], parts[2]
        src_dir = segments_abs / session / event / seg_name
        label = labels[seg_id]

        target_dir = output_dir / str(label)
        target_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            continue

        prefix = f"{session}_{event}_{seg_name}"
        for mod_file in src_dir.iterdir():
            if mod_file.is_file() and not mod_file.name.startswith('.'):
                ext = mod_file.suffix
                link_name = target_dir / f"{prefix}{ext}"
                if not link_name.exists():
                    link_name.symlink_to(mod_file.resolve())

    # 统计
    n_videos = sum(1 for _ in output_dir.rglob("*.mp4"))
    print(f"{split_name}: {n_videos} videos created")

print("\nDone!")
