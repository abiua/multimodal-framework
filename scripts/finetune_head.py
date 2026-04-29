#!/usr/bin/env python
"""微调分类头：加载3分类checkpoint → 替换为4分类head → 仅训练分类头"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models import ModelBuilder
from evaluators import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default='output/finetune_head')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=str(output_dir / 'finetune.log'))

    # 1. 构建模型（4分类）
    config = load_config(args.config)
    config.classes.num_classes = 4  # 确保4分类
    model = ModelBuilder.build_model(config)
    model = model.to(device)
    logger.info(f"模型构建完成，分类头: {model.classifier}")

    # 2. 加载预训练权重（跳过分类头参数，因为3→4类size不匹配）
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_state = ckpt['model_state_dict']
    # 过滤掉classifier相关参数，让4分类head保持随机初始化
    filtered_state = {k: v for k, v in model_state.items() if not k.startswith('classifier')}
    # 也过滤decision中可能与class相关的参数（identity decision没有可训练参数，安全）
    model.load_state_dict(filtered_state, strict=False)
    skipped = [k for k in model_state if k.startswith('classifier')]
    logger.info(f"Checkpoint加载完成，跳过的分类头参数: {skipped}")

    # 3. 冻结所有层，仅训练分类头和decision
    for name, param in model.named_parameters():
        if 'classifier' in name or 'decision' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 4. 数据加载
    data_factory = DataFactory(config)
    train_dataset = data_factory.create_dataset(config.data.train_path, is_training=True)
    val_dataset = data_factory.create_dataset(config.data.val_path, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False)
    logger.info(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    # 递归移动tensor到device
    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_device(v) for v in obj]
        return obj

    # 5. 训练
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            batch = to_device(batch)
            labels = batch.pop('class_idx')
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        train_acc = train_correct / train_total
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss/train_total:.4f}, Acc: {train_acc:.4f}")

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch)
                labels = batch.pop('class_idx')
                logits = model(batch)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss/val_total:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc},
                       output_dir / 'best_finetune.pth')
            logger.info(f"  保存最佳模型 (val_acc={val_acc:.4f})")

    logger.info(f"微调完成! 最佳val_acc={best_acc:.4f}")
    logger.info(f"模型保存到 {output_dir / 'best_finetune.pth'}")


if __name__ == '__main__':
    main()
