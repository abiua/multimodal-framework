#!/usr/bin/env python
"""Physics-First Asymmetric Fusion training script."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models.builder import ModelBuilder
from models.pipeline_v3 import MultimodalPipelineV3
from models.modelzoo.common import IdentityStem
from models.modelzoo.multichannel_tcn import MultiChannelTCN
from models.fusion.physical_encoder import PhysicalDynamicsEncoder
from models.fusion.asymmetric_interaction import AsymmetricInteraction, EvidenceGate


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def build_pipeline_v3(config, device):
    pipe_cfg = config.model.unified_pipeline
    D = pipe_cfg.token_dim

    imu_stems = {ch: IdentityStem(feature_dim=3) for ch in ['accel', 'gyro', 'angle']}
    imu_encoder = MultiChannelTCN(channel_dim=64, tcn_channels=[128, 256, 256], output_dim=D)

    audio_cfg = config.model.backbones['audio']
    audio_stem = ModelBuilder.build_backbone(
        audio_cfg.type, audio_cfg.feature_dim, audio_cfg.pretrained,
        config.model.dropout_rate, dict(audio_cfg.extra_params))

    video_cfg = config.model.backbones['video']
    video_stem = ModelBuilder.build_backbone(
        video_cfg.type, video_cfg.feature_dim, video_cfg.pretrained,
        config.model.dropout_rate, dict(video_cfg.extra_params))

    pe_cfg = pipe_cfg.physical_encoder
    physical_encoder = PhysicalDynamicsEncoder(
        dim=D, num_heads=pe_cfg.num_heads,
        num_cross_attn_layers=pe_cfg.num_cross_attn_layers,
        num_shared_transformer_layers=pe_cfg.num_shared_transformer_layers,
        dropout=pe_cfg.dropout)

    ai_cfg = pipe_cfg.asymmetric_interaction
    asymmetric_interaction = AsymmetricInteraction(
        dim=D, num_blocks=ai_cfg.num_blocks, num_heads=ai_cfg.num_heads,
        dropout=ai_cfg.dropout)

    eg_cfg = pipe_cfg.evidence_gate
    evidence_gate = EvidenceGate(dim=D, hidden_dim=eg_cfg.hidden_dim)

    return MultimodalPipelineV3(
        imu_stems=imu_stems, imu_encoder=imu_encoder,
        audio_stem=audio_stem, audio_dim=audio_cfg.feature_dim,
        video_stem=video_stem, video_dim=video_cfg.feature_dim,
        physical_encoder=physical_encoder,
        asymmetric_interaction=asymmetric_interaction,
        evidence_gate=evidence_gate,
        mid_fusion_dim=pipe_cfg.mid_fusion_output_dim,
        num_classes=config.classes.num_classes,
        dropout_rate=config.model.dropout_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'train.log'))

    config = load_config(args.config)
    model = build_pipeline_v3(config, device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Parameters: {sum(p.requires_grad for p in model.parameters()):,} / {n_params:,} trainable"
    )

    data_factory = DataFactory(config)
    train_ds = data_factory.create_dataset(config.data.train_path, is_training=True)
    val_ds = data_factory.create_dataset(config.data.val_path, is_training=False)
    test_ds = data_factory.create_dataset(config.data.test_path, is_training=False)
    bs = min(config.data.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, patience = 0.0, 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            optimizer.zero_grad()
            logits, aux = model(batch)
            fusion_loss = criterion(logits, labels)
            phys_loss = criterion(aux['phys_logits'], labels)
            total_loss = fusion_loss + 0.3 * phys_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += total_loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)
        train_acc = train_correct / train_total
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                labels = batch.pop('class_idx')
                logits, _ = model(batch)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)
        val_acc = val_correct / val_total
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(
                {'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc},
                output_dir / 'best_model.pth'
            )
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

    logger.info(f"Training complete! Best val_acc={best_acc:.4f}")

    ckpt = torch.load(output_dir / 'best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    test_correct, test_total, all_preds, all_labels = 0, 0, [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            logits, _ = model(batch)
            test_correct += (logits.argmax(1) == labels).sum().item()
            test_total += len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    test_acc = test_correct / test_total
    logger.info(f"\nTest Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
    logger.info(f"Pred: {Counter(all_preds)}  True: {Counter(all_labels)}")
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(f"test_acc: {test_acc:.4f}\nbest_val_acc: {best_acc:.4f}\n")


if __name__ == '__main__':
    main()
