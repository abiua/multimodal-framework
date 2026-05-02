#!/usr/bin/env python
"""SACF v2 evaluation + gradient-based modality contribution analysis + ablation."""
import argparse, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory

sys.path.insert(0, str(Path(__file__).parent))
from train_sacf import build_sacf_pipeline

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
    all_preds, all_labels = [], []
    correct, total = 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        logits, _ = model(batch)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = correct / total
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['None', 'Strong', 'Weak']
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   zero_division=0, output_dict=True)
    return acc, cm, report, all_preds, all_labels


def run_gradient_analysis(model, loader, device, n_batches=50):
    """Gradient-based modality contribution analysis for SACF."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    accum = defaultdict(lambda: defaultdict(list))

    alpha, beta, gamma = 0.5, 0.5, 0.3

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        model.zero_grad()
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        logits, aux = model(batch)

        loss_main = criterion(logits, labels)
        loss_phys = criterion(aux['phys_logits'], labels)
        loss_image = criterion(aux['image_logits'], labels)
        p_final = F.log_softmax(logits.detach(), dim=-1)
        loss_cons = 0.5 * (
            F.kl_div(p_final, F.log_softmax(aux['phys_logits'], dim=-1),
                     log_target=True, reduction='batchmean') +
            F.kl_div(p_final, F.log_softmax(aux['image_logits'], dim=-1),
                     log_target=True, reduction='batchmean'))
        loss = loss_main + alpha * loss_phys + beta * loss_image + gamma * loss_cons
        loss.backward()

        accum['wave']['encoder'].append(compute_grad_norm(model.wave_encoder))
        accum['wave']['proj'].append(compute_grad_norm(model.wave_proj))

        accum['audio']['encoder'].append(compute_grad_norm(model.audio_encoder))
        accum['audio']['proj'].append(compute_grad_norm(model.audio_proj))

        accum['image']['backbone'].append(compute_grad_norm(model.image_backbone))
        accum['image']['proj'].append(compute_grad_norm(model.image_proj))

        accum['fusion']['temporal_consensus'].append(compute_grad_norm(model.temporal_consensus))
        accum['fusion']['film_gate'].append(compute_grad_norm(model.film_gate))
        accum['fusion']['final_classifier'].append(compute_grad_norm(model.final_classifier))

        model.zero_grad()

    results = {}
    for group, metrics in accum.items():
        results[group] = {key: {'mean': np.mean(vals), 'std': np.std(vals)}
                         for key, vals in metrics.items()}
    return results


def run_ablation(model, loader, device):
    """Accuracy drop when zeroing each modality in SACF."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    def eval_ablated(zero_wave=False, zero_audio=False, zero_image=False):
        correct, total, loss_sum = 0, 0, 0.0
        for batch in loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')

            wave_input = model._resolve_input(batch, 'wave')
            if isinstance(wave_input, dict):
                wave_input = wave_input.get('wave', list(wave_input.values())[0])
            if zero_wave:
                wave_input = torch.zeros_like(wave_input)
            wave_tokens = model._encode_wave(wave_input)

            audio_input = model._resolve_input(batch, 'audio')
            audio_tokens = model._encode_audio(audio_input)
            if zero_audio:
                audio_tokens = torch.zeros_like(audio_tokens)

            physical_tokens = model.temporal_consensus(wave_tokens, audio_tokens,
                                                       wave_masked=zero_wave,
                                                       audio_masked=zero_audio)
            phys_pooled = model.phys_pool_proj(physical_tokens.mean(dim=1))

            image_input = model._resolve_input(batch, 'image')
            z_img = model._encode_image(image_input)
            if zero_image:
                f_final = phys_pooled
            else:
                f_final, _ = model.film_gate(z_img, phys_pooled)

            logits = model.final_classifier(f_final)
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

    # Wave+Audio only (no Image)
    acc, loss = eval_ablated(zero_image=True)
    results['wave_audio_only'] = {'acc': acc, 'loss': loss, 'acc_drop': full_acc - acc}

    # Per-modal classifier accuracies
    correct, total = 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        _, aux = model(batch)
        correct += (aux['phys_logits'].argmax(1) == labels).sum().item()
        total += len(labels)
    results['physics_only'] = {'acc': correct / total,
                               'acc_drop': full_acc - correct / total}

    correct, total = 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        _, aux = model(batch)
        correct += (aux['image_logits'].argmax(1) == labels).sum().item()
        total += len(labels)
    results['image_only'] = {'acc': correct / total,
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

    model = build_sacf_pipeline(config, device)
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
    acc, cm, report, preds, labels = run_evaluation(model, test_loader, device)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    for cls_key in ['None', 'Strong', 'Weak']:
        m = report[cls_key]
        logger.info(f"  {cls_key}: prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1-score']:.4f}")
    macro = report['macro avg']
    logger.info(f"  Macro Avg: prec={macro['precision']:.4f} rec={macro['recall']:.4f} f1={macro['f1-score']:.4f}")

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
    for name in ['full', 'no_wave', 'no_audio', 'no_image', 'wave_audio_only', 'physics_only', 'image_only']:
        if name in abl_results:
            r = abl_results[name]
            logger.info(f"  {name:<18}: acc={r['acc']:.4f}  drop={r.get('acc_drop', 0):.4f}")

    # 4. Summary JSON
    summary = {
        'test_accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'gradient_contribution': {mod: float(v) for mod, v in mod_grads.items()},
        'gradient_pct': {mod: float(100*v/total_g) if total_g > 0 else 0
                        for mod, v in mod_grads.items()},
        'ablation': {k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in abl_results.items()},
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
