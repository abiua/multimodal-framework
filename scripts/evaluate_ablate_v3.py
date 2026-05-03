#!/usr/bin/env python
"""V3 evaluation + gradient-based modality contribution analysis."""
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
from models.modelzoo.common import IdentityStem
from models.modelzoo.multichannel_tcn import MultiChannelTCN
from models.fusion.physical_encoder import PhysicalDynamicsEncoder
from models.fusion.asymmetric_interaction import AsymmetricInteraction, EvidenceGate

MODALITY_NAMES = {'imu': 'IMU', 'audio': 'Audio', 'video': 'Video'}


def build_pipeline_v3(config, device):
    pipe_cfg = config.model.unified_pipeline
    D = pipe_cfg.token_dim
    imu_stems = {ch: IdentityStem(feature_dim=3).to(device) for ch in ['accel', 'gyro', 'angle']}
    imu_encoder = MultiChannelTCN(channel_dim=64, tcn_channels=[128, 256, 256], output_dim=D).to(device)
    ac = config.model.backbones['audio']
    audio_stem = ModelBuilder.build_backbone(ac.type, ac.feature_dim, ac.pretrained, config.model.dropout_rate, dict(ac.extra_params)).to(device)
    vc = config.model.backbones['video']
    video_stem = ModelBuilder.build_backbone(vc.type, vc.feature_dim, vc.pretrained, config.model.dropout_rate, dict(vc.extra_params)).to(device)
    pc = pipe_cfg.physical_encoder
    physical_encoder = PhysicalDynamicsEncoder(dim=D, num_heads=pc.num_heads, num_cross_attn_layers=pc.num_cross_attn_layers, num_shared_transformer_layers=pc.num_shared_transformer_layers, dropout=pc.dropout).to(device)
    ac2 = pipe_cfg.asymmetric_interaction
    asymmetric_interaction = AsymmetricInteraction(dim=D, num_blocks=ac2.num_blocks, num_heads=ac2.num_heads, dropout=ac2.dropout).to(device)
    ec = pipe_cfg.evidence_gate
    evidence_gate = EvidenceGate(dim=D, hidden_dim=ec.hidden_dim).to(device)
    return MultimodalPipelineV3(
        imu_stems=imu_stems, imu_encoder=imu_encoder,
        imu_channel_names=['accel', 'gyro', 'angle'],
        audio_stem=audio_stem, audio_dim=ac.feature_dim,
        visual_stem=video_stem, visual_dim=vc.feature_dim,
        visual_type='video', visual_key='video',
        physical_encoder=physical_encoder, asymmetric_interaction=asymmetric_interaction,
        evidence_gate=evidence_gate, mid_fusion_dim=pipe_cfg.mid_fusion_output_dim,
        num_classes=config.classes.num_classes, dropout_rate=config.model.dropout_rate
    ).to(device)


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
    report = classification_report(all_labels, all_preds, target_names=['Class1', 'Class2', 'Class3', 'Class4'], zero_division=0, output_dict=True)
    return acc, cm, report, all_preds, all_labels, evidence_scores


def run_gradient_analysis(model, loader, device, n_batches=30):
    """Gradient-based modality contribution analysis for V3."""
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

        accum['imu']['encoder'].append(compute_grad_norm(model.imu_encoder))
        for ch in ['accel', 'gyro', 'angle']:
            accum['imu'][f'stem_{ch}'].append(compute_grad_norm(model.imu_stems[ch]))

        accum['audio']['stem'].append(compute_grad_norm(model.audio_stem))
        accum['audio']['proj'].append(compute_grad_norm(model.audio_proj))

        accum['video']['stem'].append(compute_grad_norm(model.visual_stem))
        accum['video']['proj'].append(compute_grad_norm(model.visual_proj))

        accum['fusion']['physical_encoder'].append(compute_grad_norm(model.physical_encoder))
        accum['fusion']['asymmetric_interaction'].append(compute_grad_norm(model.asymmetric_interaction))
        accum['fusion']['evidence_gate'].append(compute_grad_norm(model.evidence_gate))
        accum['fusion']['classifier'].append(compute_grad_norm(model.classifier))

        model.zero_grad()

    results = {}
    for group, metrics in accum.items():
        results[group] = {key: {'mean': np.mean(vals), 'std': np.std(vals)} for key, vals in metrics.items()}
    return results


def run_ablation(model, loader, device):
    """Accuracy drop when zeroing each modality."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    def eval_ablated(zero_imu=False, zero_audio=False, zero_video=False):
        correct, total, loss_sum = 0, 0, 0.0
        for batch in loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            imu_chs = {}
            for ch in ['accel', 'gyro', 'angle']:
                imu_chs[ch] = model._resolve_imu_channel(batch, ch)
            if zero_imu:
                for k in imu_chs:
                    imu_chs[k] = torch.zeros_like(imu_chs[k])
            imu_tokens = model.imu_encoder(imu_chs)

            ai = model._resolve_input(batch, 'audio')
            af = model.audio_stem(ai) if isinstance(ai, torch.Tensor) else model.audio_stem(**ai)
            if zero_audio:
                af = torch.zeros_like(af)
            audio_tokens = model.audio_proj(af).unsqueeze(1)

            vi = model._resolve_input(batch, 'video')
            vt = model.visual_stem.tokenize(vi) if isinstance(vi, torch.Tensor) else model.visual_stem.tokenize(**vi)
            video_tokens = model.visual_proj(vt['tokens'])
            if zero_video:
                video_tokens = torch.zeros_like(video_tokens)

            pt = model.physical_encoder(imu_tokens, audio_tokens)
            vo, po = model.asymmetric_interaction(video_tokens, pt)
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

    for name, zi, za, zv in [('no_imu', True, False, False),
                               ('no_audio', False, True, False),
                               ('no_video', False, False, True)]:
        acc, loss = eval_ablated(zero_imu=zi, zero_audio=za, zero_video=zv)
        results[name] = {'acc': acc, 'loss': loss, 'acc_drop': full_acc - acc}

    # Physics-only accuracy
    correct, total = 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        _, aux = model(batch)
        correct += (aux['phys_logits'].argmax(1) == labels).sum().item()
        total += len(labels)
    results['physics_only'] = {'acc': correct / total, 'acc_drop': full_acc - correct / total}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--grad-batches', type=int, default=30)
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
    bs = min(8, config.data.batch_size)
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

    # Evidence analysis
    ev_by_class = defaultdict(list)
    for p, l, e in zip(preds, labels, ev_scores):
        ev_by_class[l+1].append(e)
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
    for mod in ['imu', 'audio', 'video']:
        total_g = sum(v['mean'] for v in grad_results[mod].values())
        mod_grads[mod] = total_g

    total_g = sum(mod_grads.values())
    logger.info(f"\n{'Modality':<12} {'Gradient':<16} {'Contribution %':<16}")
    logger.info("-" * 44)
    for mod in ['imu', 'audio', 'video']:
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
    for name in ['full', 'no_imu', 'no_audio', 'no_video', 'physics_only']:
        r = abl_results[name]
        logger.info(f"  {name:<15}: acc={r['acc']:.4f}  drop={r.get('acc_drop', 0):.4f}")

    # 4. Summary
    summary = {
        'test_accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'gradient_contribution': {mod: float(v) for mod, v in mod_grads.items()},
        'gradient_pct': {mod: float(100*v/total_g) if total_g > 0 else 0 for mod, v in mod_grads.items()},
        'ablation': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in abl_results.items()},
        'evidence_by_class': {str(k): {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in ev_by_class.items()},
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
