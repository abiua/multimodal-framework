#!/usr/bin/env python
"""V3 Physics-First Asymmetric Fusion — Fish Feeding (Image + Wave + Audio)."""
import argparse, os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path

from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models.builder import ModelBuilder
from models.pipeline_v3 import MultimodalPipelineV3
from models.fusion.physical_encoder import PhysicalDynamicsEncoder
from models.fusion.asymmetric_interaction import AsymmetricInteraction, EvidenceGate


class WaveEncoder(nn.Module):
    """6-channel wave sensor encoder -> token sequence [B, T, D]."""

    def __init__(self, in_channels=6, stem_dim=64, tcn_channels=None,
                 kernel_size=7, dropout=0.2, output_dim=256):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [128, 256]

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU(),
        )

        layers = []
        in_ch = stem_dim
        for i, ch in enumerate(tcn_channels):
            layers.append(self._make_tcn_block(in_ch, ch, kernel_size, 2 ** i, dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)

        self.output_proj = nn.Sequential(
            nn.Conv1d(tcn_channels[-1], output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _make_tcn_block(in_ch, out_ch, kernel_size, dilation, dropout):
        class TCNBlock(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
                super().__init__()
                padding = (kernel_size - 1) * dilation
                self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
                self.bn1 = nn.BatchNorm1d(out_ch)
                self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
                self.bn2 = nn.BatchNorm1d(out_ch)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

            def forward(self, x):
                residual = x
                out = self.conv1(x)[:, :, :x.size(2)]
                out = self.bn1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.conv2(out)[:, :, :x.size(2)]
                out = self.bn2(out)
                if self.downsample is not None:
                    residual = self.downsample(residual)
                return self.relu(out + residual)
        return TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)

    def forward(self, x):
        if x.dim() == 3 and x.size(-1) in (6, 3):
            x = x.transpose(1, 2)
        out = self.stem(x)
        out = self.tcn(out)
        out = self.output_proj(out)
        return out.transpose(1, 2)


def build_pipeline_v3(config, device):
    pipe_cfg = config.model.unified_pipeline
    D = pipe_cfg.token_dim

    wave_encoder = WaveEncoder(in_channels=6, output_dim=D).to(device)

    ac = config.model.backbones['audio']
    audio_stem = ModelBuilder.build_backbone(
        ac.type, ac.feature_dim, ac.pretrained,
        config.model.dropout_rate, dict(ac.extra_params)
    ).to(device)

    ic = config.model.backbones['image']
    image_stem = ModelBuilder.build_backbone(
        ic.type, ic.feature_dim, ic.pretrained,
        config.model.dropout_rate, dict(ic.extra_params)
    ).to(device)

    pe_cfg = pipe_cfg.physical_encoder
    physical_encoder = PhysicalDynamicsEncoder(
        dim=D, num_heads=pe_cfg.num_heads,
        num_cross_attn_layers=pe_cfg.num_cross_attn_layers,
        num_shared_transformer_layers=pe_cfg.num_shared_transformer_layers,
        dropout=pe_cfg.dropout
    ).to(device)

    ai_cfg = pipe_cfg.asymmetric_interaction
    asymmetric_interaction = AsymmetricInteraction(
        dim=D, num_blocks=ai_cfg.num_blocks,
        num_heads=ai_cfg.num_heads, dropout=ai_cfg.dropout
    ).to(device)

    eg_cfg = pipe_cfg.evidence_gate
    evidence_gate = EvidenceGate(dim=D, hidden_dim=eg_cfg.hidden_dim).to(device)

    return MultimodalPipelineV3(
        imu_stems=None, imu_encoder=wave_encoder,
        imu_channel_names=['wave'],
        audio_stem=audio_stem, audio_dim=ac.feature_dim,
        visual_stem=image_stem, visual_dim=ic.feature_dim,
        visual_type='image', visual_key='image',
        physical_encoder=physical_encoder,
        asymmetric_interaction=asymmetric_interaction,
        evidence_gate=evidence_gate,
        mid_fusion_dim=pipe_cfg.mid_fusion_output_dim,
        num_classes=config.classes.num_classes,
        dropout_rate=config.model.dropout_rate
    ).to(device)


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, aux = model(batch)
                loss = criterion(logits, labels) + 0.3 * criterion(aux['phys_logits'], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, aux = model(batch)
            loss = criterion(logits, labels) + 0.3 * criterion(aux['phys_logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop('class_idx')
        logits, _ = model(batch)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = local_rank >= 0

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)
    output_dir = Path(args.output)
    if not is_distributed or local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'train.log')) if (not is_distributed or local_rank == 0) else None

    model = build_pipeline_v3(config, device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if not is_distributed or local_rank == 0:
        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    data_factory = DataFactory(config)
    train_ds = data_factory.create_dataset(config.data.train_path, is_training=True)
    val_ds = data_factory.create_dataset(config.data.val_path, is_training=False)

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, sampler=train_sampler,
                                  num_workers=config.data.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, sampler=val_sampler,
                                num_workers=config.data.num_workers, pin_memory=True, drop_last=False)
    else:
        bs = min(config.data.batch_size, len(train_ds))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=config.data.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=config.data.num_workers, pin_memory=True)

    if not is_distributed or local_rank == 0:
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Batch/GPU: {config.data.batch_size}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if config.system.fp16 else None
    best_acc, patience = 0.0, 0

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if is_distributed:
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics = metrics / world_size
            train_loss, train_acc, val_loss, val_acc = metrics[0].item(), metrics[1].item(), metrics[2].item(), metrics[3].item()

        if not is_distributed or local_rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                sd = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save({'model_state_dict': sd, 'epoch': epoch, 'val_acc': best_acc}, output_dir / 'best_model.pth')
                logger.info(f"  Saved best (val_acc={best_acc:.4f})")
            else:
                patience += 1
                if patience >= args.patience:
                    logger.info(f"Early stop at epoch {epoch+1}")
                    break

    if not is_distributed or local_rank == 0:
        logger.info(f"Training done. Best val_acc={best_acc:.4f}")
    if is_distributed:
        torch.cuda.synchronize()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
