#!/usr/bin/env python
"""Split 5Hz data into stratified train/val/test (70/15/15)."""
import shutil
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

SRC = Path("data/multimodal_freq_split/5hz")
DST = Path("data/v3_5hz")

def main():
    if DST.exists():
        print(f"Removing existing {DST}...")
        shutil.rmtree(DST)

    # Gather all samples per class
    class_samples = {}
    for cls_dir in sorted(SRC.iterdir()):
        if not cls_dir.is_dir():
            continue
        samples = sorted(set(
            f.stem for f in cls_dir.iterdir()
            if f.is_file() and f.suffix in ('.csv', '.jpg', '.wav', '.mp4')
        ))
        class_samples[cls_dir.name] = samples
        print(f"Class {cls_dir.name}: {len(samples)} samples")

    total = sum(len(v) for v in class_samples.values())
    print(f"Total: {total} samples")

    # Stratified split
    all_samples = []
    all_labels = []
    for cls_name, samples in class_samples.items():
        for s in samples:
            all_samples.append((cls_name, s))
            all_labels.append(cls_name)

    train_val, test, tv_labels, _ = train_test_split(
        all_samples, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )
    val_frac = 0.15 / 0.85
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=tv_labels, random_state=43
    )

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for cls_name, sample_id in split_data:
            d = DST / split_name / cls_name
            d.mkdir(parents=True, exist_ok=True)
            for ext in ['.csv', '.jpg', '.wav', '.mp4']:
                src = SRC / cls_name / f"{sample_id}{ext}"
                if src.exists():
                    shutil.copy2(src, d / f"{sample_id}{ext}")

        # Print distribution
        counter = Counter(cls for cls, _ in split_data)
        print(f"\n{split_name}: {len(split_data)} samples")
        for cls_name in sorted(counter):
            print(f"  Class {cls_name}: {counter[cls_name]}")

    print(f"\nDone! Data at {DST.resolve()}")

if __name__ == "__main__":
    main()
