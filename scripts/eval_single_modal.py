"""Quick single-modality evaluation using simple classifiers — DDP compatible."""
import argparse, os, torch, torch.nn as nn
import torch.distributed as dist
import numpy as np
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils.config import load_config
from datasets import DataFactory
from models.builder import ModelBuilder

def to_device(obj, device):
    if isinstance(obj, torch.Tensor): return obj.to(device)
    if isinstance(obj, dict): return {k: to_device(v, device) for k, v in obj.items()}
    return obj

def train_single(config, modality, backbone_cfg, feature_dim, num_classes, device, local_rank, world_size):
    is_dist = world_size > 1

    backbone = ModelBuilder.build_backbone(
        backbone_cfg.type, feature_dim,
        backbone_cfg.get('pretrained', False), 0.0, dict(backbone_cfg.get('extra_params', {}))
    ).to(device)
    head = nn.Linear(feature_dim, num_classes).to(device)
    model = nn.Sequential(backbone, head)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    factory = DataFactory(config)
    train_ds = factory.create_dataset(config.data.train_path, is_training=True)
    test_ds = factory.create_dataset(config.data.test_path, is_training=False)

    if is_dist:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=64, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=64, sampler=test_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    best_acc = 0.0
    for epoch in range(30):
        if is_dist:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            inp = batch.get(modality)
            if isinstance(inp, dict):
                inp = inp.get(modality, list(inp.values())[0]) if inp else None
            if inp is None: continue
            feat = model[0](inp) if not is_dist else model.module[0](inp)
            logits = (model[1](feat) if not is_dist else model.module[1](feat))
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = to_device(batch, device)
                labels = batch.pop('class_idx')
                inp = batch.get(modality)
                if isinstance(inp, dict):
                    inp = inp.get(modality, list(inp.values())[0]) if inp else None
                if inp is None: continue
                feat = model[0](inp) if not is_dist else model.module[0](inp)
                logits = (model[1](feat) if not is_dist else model.module[1](feat))
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        acc = correct / total if total > 0 else 0
        if is_dist:
            t = torch.tensor([acc], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            acc = t.item() / world_size
        if acc > best_acc:
            best_acc = acc
        if epoch >= 10 and acc < best_acc * 0.98:
            break

    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--modality', type=str, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_dist = local_rank >= 0
    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)
    NC = config.classes.num_classes
    modality = args.modality

    if modality not in config.model.backbones:
        raise ValueError(f"Modality '{modality}' not in backbones config")

    cfg = config.model.backbones[modality]
    fd = cfg.feature_dim if hasattr(cfg, 'feature_dim') else cfg.get('feature_dim', 512)

    if not is_dist or local_rank == 0:
        print(f"Training {modality} (backbone={cfg.type}, dim={fd}, 4-GPU DDP)...")

    acc = train_single(config, modality, cfg, fd, NC, device, local_rank, world_size)

    if not is_dist or local_rank == 0:
        print(f"  {modality}: Best test acc = {acc:.4f}")

    if is_dist:
        torch.cuda.synchronize()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
