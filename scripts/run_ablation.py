#!/usr/bin/env python3
"""Run all frequency ablation experiments across multiple seeds (4-GPU DDP).

Usage:
    python scripts/run_ablation.py                      # all 4 groups x 3 seeds = 12 runs
    python scripts/run_ablation.py --group full_48k     # single group x 3 seeds
    python scripts/run_ablation.py --seeds 42,123       # custom seeds
    python scripts/run_ablation.py --dry-run            # print commands without running
    python scripts/run_ablation.py --base-port 29500    # starting master_port
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

GROUPS = {
    'full_48k':      'configs/ablation/full_48k.yaml',
    'lp_8k':         'configs/ablation/lp_8k.yaml',
    'bp_8_24k':      'configs/ablation/bp_8_24k.yaml',
    'resample_16k':  'configs/ablation/resample_16k.yaml',
}

DEFAULT_SEEDS = [42, 123, 456]
PYTHONPATH = '/home/pythoner/abiu/multimodal-framework'
GPU_COUNT = 4


def run_one(group: str, config_path: str, seed: int, base_port: int, dry_run: bool) -> dict:
    """Run one training experiment via torchrun DDP."""
    output_dir = f'output/ablation/{group}/seed_{seed}'
    process_name = f'ablation_{group}_s{seed}'
    master_port = base_port

    # Build torchrun command
    cmd = [
        'torchrun',
        '--nproc_per_node', str(GPU_COUNT),
        '--master_port', str(master_port),
        '-m', 'tools.train',
        '--config', '{TEMP_CONFIG}',  # placeholder, filled below
        '--output-dir', output_dir,
    ]

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {group}  seed={seed}  port={master_port}")
    print(f"  process: {process_name}")
    print(f"  output:  {output_dir}")
    print(f"{'='*70}")

    if dry_run:
        return {'group': group, 'seed': seed, 'output_dir': output_dir, 'dry_run': True}

    # Create temp config with seed override
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['system']['seed'] = seed
    config['system']['output_dir'] = output_dir
    config['system']['experiment_name'] = f"ablation_{group}_s{seed}"

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, dir='/tmp'
    ) as f:
        yaml.dump(config, f, default_flow_style=False)
        temp_config = f.name

    # Replace placeholder with actual temp config path
    cmd[cmd.index('{TEMP_CONFIG}')] = temp_config

    # Build shell command with exec -a for process naming
    shell_cmd = f"exec -a {process_name} " + " ".join(cmd)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    env['PYTHONPATH'] = PYTHONPATH

    t0 = time.time()
    try:
        result = subprocess.run(
            ['bash', '-c', shell_cmd],
            env=env,
            cwd=PYTHONPATH,
        )
        success = result.returncode == 0
    finally:
        try:
            os.unlink(temp_config)
        except OSError:
            pass

    elapsed = time.time() - t0
    status = 'OK' if success else f'FAIL (rc={result.returncode})'
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {group} seed={seed}: {status}  ({elapsed:.0f}s)")

    return {
        'group': group, 'seed': seed, 'output_dir': output_dir,
        'success': success, 'elapsed': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Run frequency ablation experiments (4-GPU DDP)')
    parser.add_argument('--group', type=str, choices=list(GROUPS.keys()),
                        help='Run only one specific group')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated seed list')
    parser.add_argument('--base-port', type=int, default=29500,
                        help='Starting master_port for torchrun (incremented per run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    groups_to_run = [args.group] if args.group else list(GROUPS.keys())

    print(f"Groups:     {groups_to_run}")
    print(f"Seeds:      {seeds}")
    print(f"GPUs:       {GPU_COUNT} (0-{GPU_COUNT-1})")
    print(f"Base port:  {args.base_port}")
    print(f"Total runs: {len(groups_to_run) * len(seeds)}")
    if args.dry_run:
        print("*** DRY RUN — no training will be executed ***")

    results = []
    port_offset = 0
    for group in groups_to_run:
        for seed in seeds:
            port = args.base_port + port_offset
            r = run_one(group, GROUPS[group], seed, port, args.dry_run)
            results.append(r)
            port_offset += 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    succeeded = 0
    failed = 0
    for r in results:
        if r.get('dry_run'):
            status = 'DRY_RUN'
        elif r.get('success'):
            status = 'OK'
            succeeded += 1
        else:
            status = 'FAIL'
            failed += 1
        elapsed = f" ({r.get('elapsed', 0):.0f}s)" if 'elapsed' in r else ''
        print(f"  {r['group']:20s}  seed={r['seed']:3d}  {status}{elapsed}")

    if not args.dry_run:
        print(f"\nSucceeded: {succeeded}, Failed: {failed}")
    print(f"\nOutputs under: output/ablation/")


if __name__ == '__main__':
    main()
