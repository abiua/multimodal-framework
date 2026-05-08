#!/usr/bin/env python3
"""Analyze frequency ablation experiment results.

Reads best metrics from all runs and produces comparison table with mean+-std.

Usage:
    python scripts/analyze_ablation.py
    python scripts/analyze_ablation.py --base-dir output/ablation
    python scripts/analyze_ablation.py --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

GROUPS = ['full_48k', 'lp_8k', 'bp_8_24k', 'resample_16k']
GROUP_LABELS = {
    'full_48k':      'A: Full 48k (0-24kHz)',
    'lp_8k':         'B: LP 8k (0-8kHz)',
    'bp_8_24k':      'C: BP 8-24kHz',
    'resample_16k':  'D: Resample 16k-48k',
}


def extract_metrics(output_dir: Path) -> Optional[dict]:
    """Extract best validation metrics from a training output directory.

    Tries multiple sources in order:
    1. best_model.pth checkpoint metrics
    2. SwanLab summary files
    3. Console log files
    """
    # Source 1: best_model.pth (reads best_val_acc directly from checkpoint)
    best_model = output_dir / 'best_model.pth'
    if best_model.exists():
        try:
            import torch
            ckpt = torch.load(best_model, map_location='cpu', weights_only=True)
            if isinstance(ckpt, dict):
                acc = ckpt.get('best_val_acc')
                if acc is not None:
                    return {
                        'accuracy': float(acc),
                        'f1': None,  # Trainer does not save best F1 in checkpoint
                    }
        except Exception:
            pass

    # Source 2: SwanLab summary
    swanlog_dir = output_dir / 'swanlog'
    if swanlog_dir.exists():
        try:
            for run_dir in sorted(swanlog_dir.iterdir(), reverse=True):
                if run_dir.is_dir():
                    summary_file = run_dir / 'summary.json'
                    if summary_file.exists():
                        with open(summary_file) as f:
                            summary = json.load(f)
                        acc = summary.get('val/accuracy') or summary.get('accuracy') or summary.get('best_accuracy')
                        if acc is not None:
                            return {
                                'accuracy': float(acc),
                                'f1': float(summary.get('val/f1', 0)) if 'val/f1' in summary else None,
                            }
        except Exception:
            pass

    # Source 3: Parse log files for "新的最佳模型! 准确率: XX%"
    for pattern in ['*.log', 'logs/*.log', 'train.log']:
        for log_file in output_dir.glob(pattern):
            try:
                with open(log_file) as f:
                    best_acc = 0.0
                    for line in f:
                        if '新的最佳模型! 准确率:' in line:
                            try:
                                pct_str = line.split(':')[-1].strip().rstrip('%')
                                val = float(pct_str) / 100.0
                                if val > best_acc:
                                    best_acc = val
                            except ValueError:
                                pass
                    if best_acc > 0:
                        return {'accuracy': best_acc, 'f1': None}
            except Exception:
                continue

    return None


def analyze(base_dir: str, json_output: bool = False):
    base = Path(base_dir)
    all_results = {}

    for group in GROUPS:
        group_dir = base / group
        if not group_dir.exists():
            continue

        seed_metrics = []
        for seed_dir in sorted(group_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                continue
            metrics = extract_metrics(seed_dir)
            if metrics:
                seed_metrics.append(metrics)

        if seed_metrics:
            accs = [m['accuracy'] for m in seed_metrics]
            f1_vals = [m['f1'] for m in seed_metrics if m['f1'] is not None]
            all_results[group] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'accuracy_values': [float(a) for a in accs],
                'n_seeds': len(seed_metrics),
            }
            if f1_vals:
                all_results[group]['f1_mean'] = float(np.mean(f1_vals))
                all_results[group]['f1_std'] = float(np.std(f1_vals))
                all_results[group]['f1_values'] = [float(f) for f in f1_vals]

    if not all_results:
        print("No results found. Run training first.")
        sys.exit(1)

    if json_output:
        print(json.dumps(all_results, indent=2))
        return

    # Pretty table
    print(f"\n{'='*75}")
    print("FREQUENCY BAND ABLATION RESULTS")
    print(f"{'='*75}")
    header = f"{'Group':<32s} {'Accuracy':>14s}  {'F1':>14s}  {'Seeds':>6s}"
    print(header)
    print('-' * len(header))

    baseline_acc = None
    for group in GROUPS:
        if group not in all_results:
            continue
        r = all_results[group]
        acc_str = f"{r['accuracy_mean']:.4f} +- {r['accuracy_std']:.4f}"
        f1_str = f"{r.get('f1_mean', 0):.4f} +- {r.get('f1_std', 0):.4f}" if 'f1_mean' in r else 'N/A'
        print(f"{GROUP_LABELS[group]:<32s} {acc_str:>14s}  {f1_str:>14s}  {r['n_seeds']:>6d}")
        if group == 'full_48k':
            baseline_acc = r['accuracy_mean']

    # Delta vs baseline
    if baseline_acc is not None:
        print(f"\n{'='*75}")
        print("DELTA vs Full-48k baseline:")
        print(f"{'='*75}")
        for group in GROUPS:
            if group == 'full_48k' or group not in all_results:
                continue
            r = all_results[group]
            delta = r['accuracy_mean'] - baseline_acc
            pct = (delta / baseline_acc) * 100 if baseline_acc > 0 else 0
            direction = 'v' if delta < 0 else '^'
            print(f"  {GROUP_LABELS[group]:<32s} {direction} {abs(delta):.4f}  ({pct:+.1f}%)")

    # Interpretation guide
    print(f"""
{'='*75}
INTERPRETATION GUIDE
{'='*75}
LP_8k ~= Full_48k  -> high frequencies contribute little, 16kHz sufficient
LP_8k << Full_48k  -> low frequencies alone insufficient for classification
BP_8_24k > chance  -> high frequencies alone carry usable signal (chance=33.3% for 3 classes)
Resample_16k ~= LP_8k    -> resampling loss ~= lowpass, mainly loses highs
Resample_16k << Full_48k -> 16kHz pipeline discards useful information
""")


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation results')
    parser.add_argument('--base-dir', default='output/ablation',
                        help='Base directory containing group subdirectories')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    args = parser.parse_args()
    analyze(args.base_dir, args.json)


if __name__ == '__main__':
    main()
