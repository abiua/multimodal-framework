#!/usr/bin/env python
"""模态平衡训练：模态Dropout + 辅助分类头，防止强模态压制弱模态"""
import argparse, torch, torch.nn as nn
from pathlib import Path
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models import ModelBuilder


def to_device(obj, device):
    if isinstance(obj, torch.Tensor): return obj.to(device)
    elif isinstance(obj, dict): return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--modality-dropout', type=float, default=0.2)
    parser.add_argument('--aux-lambda', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'train.log'))

    # 1. 构建模型（4分类，带辅助头）
    config = load_config(args.config)
    config.classes.num_classes = 4
    # 注入模态平衡参数
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
    config.model.unified_pipeline.modality_dropout = args.modality_dropout
    config.model.unified_pipeline.aux_classifiers = True
    OmegaConf.set_struct(config, True)

    model = ModelBuilder.build_model(config).to(device)

    # 2. 加载预训练权重（跳过分类头）
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    filtered = {k: v for k, v in ckpt['model_state_dict'].items()
                if not k.startswith('classifier') and 'aux_classifiers' not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(f"预训练权重加载: {len(missing)} missing, {len(unexpected)} unexpected (expected: classifier & aux heads)")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数: {n_trainable:,} / {n_params:,} trainable")
    logger.info(f"模态Dropout: {args.modality_dropout}, Aux Lambda: {args.aux_lambda}")

    # 3. 数据加载
    data_factory = DataFactory(config)
    train_ds = data_factory.create_dataset(config.data.train_path, is_training=True)
    val_ds = data_factory.create_dataset(config.data.val_path, is_training=False)
    test_ds = data_factory.create_dataset(config.data.test_path, is_training=False)

    bs = min(config.data.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    logger.info(f"训练: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")

    # 4. 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, patience = 0.0, 0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            labels = batch.pop('class_idx')
            optimizer.zero_grad()

            logits, aux_logits = model(batch)

            # 融合loss
            fusion_loss = criterion(logits, labels)

            # 辅助loss
            aux_losses = []
            if aux_logits is not None:
                for m, aux_logit in aux_logits.items():
                    aux_losses.append(criterion(aux_logit, labels))

            total_loss = fusion_loss
            if aux_losses:
                aux_mean = sum(aux_losses) / len(aux_losses)
                total_loss = fusion_loss + args.aux_lambda * aux_mean
            else:
                aux_mean = torch.tensor(0.0)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        train_acc = train_correct / train_total
        scheduler.step()

        # --- Val ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                labels = batch.pop('class_idx')
                logits, _ = model(batch)  # val时不drop
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc; patience = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc},
                       output_dir / 'best_model.pth')
            logger.info(f"  ✓ 保存最佳模型 (val_acc={val_acc:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

    logger.info(f"训练完成! 最佳val_acc={best_acc:.4f}")

    # 5. Test
    best_ckpt = torch.load(output_dir / 'best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
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
    logger.info(f"预测分布: {Counter(all_preds)}")
    logger.info(f"真实分布: {Counter(all_labels)}")

    # 保存汇总
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(f"modality_dropout: {args.modality_dropout}\n")
        f.write(f"aux_lambda: {args.aux_lambda}\n")
        f.write(f"test_acc: {test_acc:.4f}\n")
        f.write(f"best_val_acc: {best_acc:.4f}\n")
        f.write(f"pred_dist: {dict(Counter(all_preds))}\n")
        f.write(f"true_dist: {dict(Counter(all_labels))}\n")

    logger.info(f"结果已保存到 {output_dir / 'summary.txt'}")


if __name__ == '__main__':
    main()
