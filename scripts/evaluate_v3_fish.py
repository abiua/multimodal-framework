#!/usr/bin/env python
"""V3 Fish Feeding evaluation + gradient-based modality contribution analysis."""
import argparse, json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models.builder import ModelBuilder
from models.pipeline_v3 import MultimodalPipelineV3
from models.fusion.physical_encoder import PhysicalDynamicsEncoder
from models.fusion.asymmetric_interaction import AsymmetricInteraction, EvidenceGate

# Import WaveEncoder from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_v3_fish import WaveEncoder, build_pipeline_v3

MODALITY_NAMES = {'wave': 'Wave', 'audio': 'Audio', 'image': 'Image'}


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def compute_grad_norm(module):
    total = 0.0
    n = 0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
            n += 1
    return np.sqrt(total) if n > 0 else 0.0


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds, all_labels, evidence_scores = [], [], []
    correct, total = 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        logits, aux = model(batch)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        evidence_scores.extend(aux['evidence'].squeeze(-1).cpu().tolist())
    acc = correct / total
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['None', 'Strong', 'Weak']
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   zero_division=0, output_dict=True)
    return acc, cm, report, all_preds, all_labels, evidence_scores


def run_gradient_analysis(model, loader, device, n_batches=50):
    """Gradient-based modality contribution analysis."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    accum = defaultdict(lambda: defaultdict(list))

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        model.zero_grad()
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        logits, aux = model(batch)
        loss = criterion(logits, labels) + 0.3 * criterion(aux['phys_logits'], labels)
        loss.backward()

        accum['wave']['encoder'].append(compute_grad_norm(model.imu_encoder))
        accum['audio']['stem'].append(compute_grad_norm(model.audio_stem))
        accum['audio']['proj'].append(compute_grad_norm(model.audio_proj))
        accum['image']['stem'].append(compute_grad_norm(model.visual_stem))
        accum['image']['proj'].append(compute_grad_norm(model.visual_proj))
        accum['fusion']['physical_encoder'].append(compute_grad_norm(model.physical_encoder))
        accum['fusion']['asymmetric_interaction'].append(compute_grad_norm(model.asymmetric_interaction))
        accum['fusion']['evidence_gate'].append(compute_grad_norm(model.evidence_gate))
        accum['fusion']['classifier'].append(compute_grad_norm(model.classifier))

        model.zero_grad()

    results = {}
    for group, metrics in accum.items():
        results[group] = {key: {'mean': np.mean(vals), 'std': np.std(vals)}
                         for key, vals in metrics.items()}
    return results


def run_ablation(model, loader, device):
    """Accuracy drop when zeroing each modality."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    def eval_ablated(zero_wave=False, zero_audio=False, zero_image=False):
        correct, total, loss_sum = 0, 0, 0.0
        for batch in loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')

            wave_input = model._resolve_input(batch, 'wave')
            if isinstance(wave_input, dict):
                wave_input = wave_input.get('wave', wave_input)
            if zero_wave:
                wave_input = torch.zeros_like(wave_input)
            wave_tokens = model.imu_encoder(wave_input)

            audio_input = model._resolve_input(batch, 'audio')
            if isinstance(audio_input, torch.Tensor):
                audio_feat = model.audio_stem(audio_input)
            else:
                audio_feat = model.audio_stem(**audio_input)
            if zero_audio:
                audio_feat = torch.zeros_like(audio_feat)
            audio_tokens = model.audio_proj(audio_feat).unsqueeze(1)

            image_input = model._resolve_input(batch, 'image')
            if isinstance(image_input, torch.Tensor):
                image_feat = model.visual_stem(image_input)
            else:
                image_feat = model.visual_stem(**image_input)
            if zero_image:
                image_feat = torch.zeros_like(image_feat)
            image_tokens = model.visual_proj(image_feat).unsqueeze(1)

            pt = model.physical_encoder(wave_tokens, audio_tokens)
            vo, po = model.asymmetric_interaction(image_tokens, pt)
            ev = model.evidence_gate(vo)
            pp = model.phys_proj(po.mean(dim=1))
            vp = model.vis_proj(ev * vo.mean(dim=1))
            fused = model.fusion_fc(torch.cat([pp, vp], dim=-1))
            logits = model.classifier(fused)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        return correct / total, loss_sum / total

    full_acc, full_loss = eval_ablated()
    results = {'full': {'acc': full_acc, 'loss': full_loss}}

    for name, zw, za, zi in [('no_wave', True, False, False),
                               ('no_audio', False, True, False),
                               ('no_image', False, False, True)]:
        acc, loss = eval_ablated(zero_wave=zw, zero_audio=za, zero_image=zi)
        results[name] = {'acc': acc, 'loss': loss, 'acc_drop': full_acc - acc}

    # Physics-only accuracy
    correct, total = 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        _, aux = model(batch)
        correct += (aux['phys_logits'].argmax(1) == labels).sum().item()
        total += len(labels)
    results['physics_only'] = {'acc': correct / total,
                                'acc_drop': full_acc - correct / total}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--grad-batches', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'analysis.log'))

    config = load_config(args.config)
    logger.info(f"Config: {args.config}")

    model = build_pipeline_v3(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded checkpoint epoch={ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', 0):.4f}")

    factory = DataFactory(config)
    test_ds = factory.create_dataset(config.data.test_path, is_training=False)
    bs = min(16, config.data.batch_size)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    logger.info(f"Test set: {len(test_ds)} samples")

    # 1. Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("1. Model Evaluation")
    logger.info("=" * 60)
    acc, cm, report, preds, labels, ev_scores = run_evaluation(model, test_loader, device)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    for cls_key in sorted(report.keys()):
        if cls_key in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        m = report[cls_key]
        logger.info(f"  {cls_key}: prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1-score']:.4f}")
    macro = report['macro avg']
    logger.info(f"  Macro Avg: prec={macro['precision']:.4f} rec={macro['recall']:.4f} f1={macro['f1-score']:.4f}")

    # Evidence analysis
    ev_by_class = defaultdict(list)
    for p, l, e in zip(preds, labels, ev_scores):
        ev_by_class[l].append(e)
    logger.info("\nEvidence Gate scores:")
    for cls_n in sorted(ev_by_class):
        vals = ev_by_class[cls_n]
        logger.info(f"  Class {cls_n}: mean={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # 2. Gradient Analysis
    logger.info("\n" + "=" * 60)
    logger.info("2. Gradient-based Modality Contribution")
    logger.info("=" * 60)
    grad_results = run_gradient_analysis(model, test_loader, device, args.grad_batches)

    mod_grads = {}
    for mod in ['wave', 'audio', 'image']:
        total_g = sum(v['mean'] for v in grad_results[mod].values())
        mod_grads[mod] = total_g

    total_g = sum(mod_grads.values())
    logger.info(f"\n{'Modality':<12} {'Gradient':<16} {'Contribution %':<16}")
    logger.info("-" * 44)
    for mod in ['wave', 'audio', 'image']:
        pct = 100 * mod_grads[mod] / total_g if total_g > 0 else 0
        logger.info(f"{MODALITY_NAMES.get(mod, mod):<12} {mod_grads[mod]:<16.4f} {pct:<16.2f}")

    logger.info("\nFusion component gradients:")
    for key, vals in grad_results['fusion'].items():
        logger.info(f"  {key}: mean={vals['mean']:.4f} +/- {vals['std']:.4f}")

    # 3. Ablation
    logger.info("\n" + "=" * 60)
    logger.info("3. Ablation Study")
    logger.info("=" * 60)
    abl_results = run_ablation(model, test_loader, device)
    for name in ['full', 'no_wave', 'no_audio', 'no_image', 'physics_only']:
        r = abl_results[name]
        logger.info(f"  {name:<15}: acc={r['acc']:.4f}  drop={r.get('acc_drop', 0):.4f}")

    # 4. Summary
    summary = {
        'test_accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'gradient_contribution': {mod: float(v) for mod, v in mod_grads.items()},
        'gradient_pct': {mod: float(100*v/total_g) if total_g > 0 else 0
                        for mod, v in mod_grads.items()},
        'ablation': {k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in abl_results.items()},
        'evidence_by_class': {str(k): {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                             for k, v in ev_by_class.items()},
    }
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {output_dir / 'analysis_summary.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("Modality Contribution Ranking:")
    ranked = sorted(mod_grads.items(), key=lambda x: x[1], reverse=True)
    for rank, (mod, gn) in enumerate(ranked, 1):
        logger.info(f"  {rank}. {MODALITY_NAMES.get(mod, mod)}: {100*gn/total_g:.1f}%")
    logger.info("Done!")


if __name__ == '__main__':
    main()
