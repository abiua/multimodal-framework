#!/usr/bin/env python
"""按频率独立训练：加载预训练权重 → 改4分类头 → 训练 → 评测"""
import argparse, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models import ModelBuilder


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='预训练权重路径')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--freeze-backbone', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'train.log'))

    # 1. 构建4分类模型
    config = load_config(args.config)
    config.classes.num_classes = 4
    model = ModelBuilder.build_model(config).to(device)

    # 2. 加载预训练权重（跳过3分类head）
    ckpt = torch.load(args.checkpoint, map_location=device)
    filtered = {k: v for k, v in ckpt['model_state_dict'].items() if not k.startswith('classifier')}
    model.load_state_dict(filtered, strict=False)
    logger.info(f"预训练权重加载完成，分类头随机初始化 (4类)")

    # 3. 冻结策略
    trainable_params = []
    for name, param in model.named_parameters():
        if args.freeze_backbone:
            if 'classifier' in name or 'decision' in name:
                param.requires_grad = True
                trainable_params.append(name)
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    if args.freeze_backbone:
        logger.info(f"仅训练: {trainable_params}")

    # 4. 数据加载
    data_factory = DataFactory(config)
    train_ds = data_factory.create_dataset(config.data.train_path, is_training=True)
    val_ds = data_factory.create_dataset(config.data.val_path, is_training=False)
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False)
    logger.info(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

    # 小数据集用更小batch
    if len(train_ds) < 50:
        logger.info("小数据集(<50样本): 使用batch_size=4")
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # 5. 训练配置
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01 if len(train_ds) < 100 else 0.001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, patience_counter = 0.0, 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        train_acc = train_correct / train_total
        scheduler.step()

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                labels = batch.pop('class_idx')
                logits = model(batch)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc},
                       output_dir / 'best_model.pth')
            logger.info(f"  ✓ 保存最佳模型 (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"训练完成! 最佳val_acc={best_acc:.4f}")

    # 6. 最终测试集评测
    logger.info("\n=== 测试集评测 ===")
    best_ckpt = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    test_ds = data_factory.create_dataset(config.data.test_path, is_training=False)
    test_loader = DataLoader(test_ds, batch_size=4 if len(test_ds) < 50 else config.data.batch_size, shuffle=False)
    logger.info(f"测试集: {len(test_ds)} 样本")

    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            logits = model(batch)
            test_correct += (logits.argmax(1) == labels).sum().item()
            test_total += len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = test_correct / test_total
    logger.info(f"Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")

    # Per-class metrics
    from collections import Counter
    logger.info(f"\n预测分布: {Counter(all_preds)}")
    logger.info(f"真实分布: {Counter(all_labels)}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\n混淆矩阵:\n{cm}")
    logger.info(f"\n分类报告:\n{classification_report(all_labels, all_preds, zero_division=0)}")

    logger.info(f"\n模型保存至: {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
