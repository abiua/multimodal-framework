#!/usr/bin/env python
"""Prepare physics-first fusion data: convert raw multimodal data into class-per-subdirectory layout.

Input structure:
  data/multimodal_data_original/
    labels.csv                          # segment_id,label
    segments_out/
      <session-date>/                   # e.g., 0412-09
        <event_id>/                     # e.g., event_000_09-46-07
          <segment_id>/                 # e.g., 000, 001, ...
            audio.wav
            imu.csv
            video.mp4

Output structure:
  data/fish_feeding_v3/
    train/{Weak,Medium,Strong,None}/<sample_id>/    audio.wav, imu.csv, video.mp4
    val/{Weak,Medium,Strong,None}/<sample_id>/      ...
    test/{Weak,Medium,Strong,None}/<sample_id>/     ...
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split


LABEL_NAMES = {1: "Weak", 2: "Medium", 3: "Strong", 4: "None"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare physics-first fusion dataset from raw multimodal data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/multimodal_data_original",
        help="Path to raw multimodal data directory (default: data/multimodal_data_original)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fish_feeding_v3",
        help="Output directory for prepared dataset (default: data/fish_feeding_v3)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> dict:
    """Load segment_id -> label mapping from labels.csv.

    Returns {segment_id: label_name} where label_name is one of
    Weak, Medium, Strong, None.
    """
    labels = {}
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            seg_id = row[0].strip()
            raw_label = int(row[1])
            label_name = LABEL_NAMES.get(raw_label)
            if label_name is None:
                print(f"  [WARNING] Unknown label value {raw_label} for {seg_id}, skipping.")
                continue
            labels[seg_id] = label_name
    return labels


def filter_existing_segments(
    labels: dict, segments_dir: Path
) -> dict:
    """Keep only segments whose source directory actually exists."""
    filtered = {}
    for seg_id, label in labels.items():
        parts = seg_id.split("/")
        if len(parts) != 3:
            print(f"  [WARNING] Unexpected segment_id format: {seg_id}, skipping.")
            continue
        session, event, seg_name = parts
        src_dir = segments_dir / session / event / seg_name
        if src_dir.is_dir():
            filtered[seg_id] = label
        else:
            print(f"  [INFO] Skipping {seg_id}: source directory does not exist.")
    return filtered


def build_splits(
    labels: dict,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple:
    """Perform stratified train/val/test split.

    First splits off test, then splits the remainder into train/val.
    Returns (train_ids, val_ids, test_ids).
    """
    seg_ids = list(labels.keys())
    class_labels = [labels[sid] for sid in seg_ids]

    # Split off test first
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        seg_ids,
        class_labels,
        test_size=test_size,
        stratify=class_labels,
        random_state=seed,
    )

    # Split remainder into train/val
    val_fraction = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=seed + 1,  # different seed to avoid identical split
    )

    return train_ids, val_ids, test_ids


def copy_segment(
    seg_id: str,
    label: str,
    split_name: str,
    segments_dir: Path,
    output_dir: Path,
):
    """Copy audio.wav, imu.csv, video.mp4 to the target directory structure.

    Sample ID is the segment_id with '/' replaced by '_'.
    """
    parts = seg_id.split("/")
    session, event, seg_name = parts
    src_dir = segments_dir / session / event / seg_name
    sample_id = seg_id.replace("/", "_")

    target_dir = output_dir / split_name / label / sample_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy
    modality_files = {
        "audio.wav": src_dir / "audio.wav",
        "imu.csv": src_dir / "imu.csv",
        "video.mp4": src_dir / "video.mp4",
    }

    copied_count = 0
    for fname, src_path in modality_files.items():
        if src_path.is_file():
            dst_path = target_dir / fname
            shutil.copy2(src_path, dst_path)
            copied_count += 1

    if copied_count == 0:
        # Clean up empty directory
        target_dir.rmdir()
        return 0

    return copied_count


def print_distribution(
    labels: dict,
    split_name: str,
    split_ids: list,
):
    """Print label distribution for a given split."""
    counter = Counter(labels[sid] for sid in split_ids)
    total = len(split_ids)
    parts = ", ".join(
        f"{cls_name}: {count} ({count / total * 100:.1f}%)"
        for cls_name in sorted(LABEL_NAMES.values())
        for count in [counter.get(cls_name, 0)]
    )
    print(f"  {split_name}: {total} samples | {parts}")


def main():
    args = parse_args()

    source_dir = Path(args.input)
    segments_dir = source_dir / "segments_out"
    labels_path = source_dir / "labels.csv"
    output_dir = Path(args.output)

    # ── Validate inputs ──────────────────────────────────────────────
    for path, desc in [
        (source_dir, "Input directory"),
        (segments_dir, "Segments directory"),
        (labels_path, "Labels file"),
    ]:
        if not path.exists():
            print(f"ERROR: {desc} does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    if output_dir.exists():
        print(f"ERROR: Output directory already exists: {output_dir}", file=sys.stderr)
        print("Remove it first or choose a different --output path.", file=sys.stderr)
        sys.exit(1)

    # ── Load labels ──────────────────────────────────────────────────
    print("Loading labels...")
    labels_raw = load_labels(labels_path)
    print(f"  Loaded {len(labels_raw)} label entries.")

    # ── Filter by existence ──────────────────────────────────────────
    print("Filtering by source directory existence...")
    labels = filter_existing_segments(labels_raw, segments_dir)
    print(f"  {len(labels)} segments have valid source directories.")

    if len(labels) == 0:
        print("ERROR: No valid segments found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Stratified split ─────────────────────────────────────────────
    print(f"Performing stratified split (val={args.val_size}, test={args.test_size}, seed={args.seed})...")
    train_ids, val_ids, test_ids = build_splits(
        labels, args.val_size, args.test_size, args.seed
    )

    # ── Print distributions ──────────────────────────────────────────
    print("\nLabel distributions:")
    print_distribution(labels, "overall", list(labels.keys()))
    print_distribution(labels, "train", train_ids)
    print_distribution(labels, "val", val_ids)
    print_distribution(labels, "test", test_ids)

    # ── Copy segments ────────────────────────────────────────────────
    print("\nCopying segments...")
    total_copied = 0
    total_segments = 0

    for split_name, split_ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ]:
        for seg_id in split_ids:
            label = labels[seg_id]
            n = copy_segment(seg_id, label, split_name, segments_dir, output_dir)
            if n > 0:
                total_segments += 1
                total_copied += n

    print(f"\nDone! Copied {total_copied} files across {total_segments} segments.")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
