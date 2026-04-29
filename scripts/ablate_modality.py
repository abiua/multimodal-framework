#!/usr/bin/env python
"""单模态消融实验：分别仅用image/audio/wave训练，判断各模态独立分类能力"""
import argparse, torch, torch.nn as nn
from pathlib import Path
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models import ModelBuilder


def to_device(obj, device):
    if isinstance(obj, torch.Tensor): return obj.to(device)
    elif isinstance(obj, dict): return {k: to_device(v, device) for k, v in obj.items()}
    return obj


class SingleModalityModel(nn.Module):
    """单模态分类器：stem → pool → classifier"""
    def __init__(self, stem, feature_dim, num_classes, dropout=0.3):
        super().__init__()
        self.stem = stem
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, batch):
        # batch contains only the target modality
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                feat = self.stem(batch[key])
                return self.classifier(feat)
            elif isinstance(batch[key], dict):
                feat = self.stem(**batch[key])
                return self.classifier(feat)
        raise ValueError("No tensor found in batch")


class SingleModalityDataset(Dataset):
    """包装MultimodalDataset，只返回指定模态"""
    def __init__(self, full_dataset, modality):
        self.ds = full_dataset
        self.modality = modality

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        label = item['class_idx']
        # Return only this modality's data
        mod_key = f'{self.modality}_path'
        if mod_key in item:
            path = item[mod_key]
            loader = self.ds.loaders[self.modality]
            data = loader.load(path)
            transform = loader.get_transform(self.ds.is_training)
            if transform:
                data = transform(data)
            return {self.modality: data, 'class_idx': label}
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--modality', type=str, required=True, choices=['image','audio','wave'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'ablate.log'))

    config = load_config(args.config)
    config.classes.num_classes = 4
    config.data.modalities = [args.modality]

    # Build full model, extract only the relevant stem
    full_model = ModelBuilder.build_model(config)
    stem = full_model.stems[args.modality]

    # Determine feature dim
    if args.modality == 'image': feature_dim = 512
    elif args.modality == 'audio': feature_dim = 512
    else: feature_dim = 256

    model = SingleModalityModel(stem, feature_dim, 4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"单模态 [{args.modality}] 模型: {n_params:,} 参数")

    # Data
    data_factory = DataFactory(config)
    train_full = data_factory.create_dataset(config.data.train_path, is_training=True)
    val_full = data_factory.create_dataset(config.data.val_path, is_training=False)
    test_full = data_factory.create_dataset(config.data.test_path, is_training=False)

    train_ds = SingleModalityDataset(train_full, args.modality)
    val_ds = SingleModalityDataset(val_full, args.modality)
    test_ds = SingleModalityDataset(test_full, args.modality)

    bs = min(config.data.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    logger.info(f"训练: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, patience = 0.0, 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        scheduler.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                labels = batch.pop('class_idx')
                logits = model(batch)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        val_acc = correct / total

        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss/total:.4f} Acc:{train_acc:.4f} | Val: {val_loss/total:.4f} Acc:{val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc; patience = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

    logger.info(f"训练完成! 最佳val_acc={best_acc:.4f}")

    # Test
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            logits = model(batch)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    logger.info(f"\nTest Accuracy: {test_acc:.4f} ({sum(1 for p,l in zip(all_preds, all_labels) if p==l)}/{len(all_labels)})")
    logger.info(f"预测分布: {Counter(all_preds)}")
    logger.info(f"真实分布: {Counter(all_labels)}")

    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(f"modality: {args.modality}\n")
        f.write(f"test_acc: {test_acc:.4f}\n")
        f.write(f"best_val_acc: {best_acc:.4f}\n")
        f.write(f"pred_dist: {dict(Counter(all_preds))}\n")
        f.write(f"true_dist: {dict(Counter(all_labels))}\n")

    logger.info(f"结果已保存到 {output_dir / 'summary.txt'}")


if __name__ == '__main__':
    main()
