#!/usr/bin/env python
"""分析不同频率下各模态的梯度贡献"""
import argparse, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
import numpy as np
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


def get_modality_gradients(model, batch, device):
    """前向+反向传播，获取各模态stem的梯度范数"""
    model.train()  # 需要梯度
    model.zero_grad()

    batch = to_device(batch, device)
    labels = batch.pop('class_idx')

    # 前向
    logits = model(batch)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    # 收集各模态stem的梯度范数
    grad_norms = {}
    for mod_name in ['image', 'audio', 'wave']:
        stem = model.stems[mod_name]
        total_grad_norm = 0.0
        total_param_norm = 0.0
        n_params = 0
        for name, param in stem.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item() ** 2
                total_param_norm += param.data.norm(2).item() ** 2
                n_params += 1
        grad_norms[mod_name] = {
            'grad_norm': np.sqrt(total_grad_norm),
            'param_norm': np.sqrt(total_param_norm),
            'n_params': n_params
        }

    # 也收集interaction blocks中各模态token的梯度
    # 获取tokenizer投影层的梯度
    for mod_name in ['image', 'audio', 'wave']:
        proj = model.tokenizer.projection.projections[mod_name]
        proj_grad = 0.0
        n = 0
        for name, param in proj.named_parameters():
            if param.grad is not None:
                proj_grad += param.grad.norm(2).item() ** 2
                n += 1
        if mod_name in grad_norms:
            grad_norms[mod_name]['proj_grad_norm'] = np.sqrt(proj_grad)

    # 收集classifier梯度（作为参考baseline）
    cls_grad = 0.0
    for name, param in model.classifier.named_parameters():
        if param.grad is not None:
            cls_grad += param.grad.norm(2).item() ** 2
    grad_norms['classifier'] = {'grad_norm': np.sqrt(cls_grad)}

    model.zero_grad()
    return grad_norms, loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-batches', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    config.classes.num_classes = 4

    model = ModelBuilder.build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"模型已加载: {args.checkpoint}")

    data_factory = DataFactory(config)
    test_ds = data_factory.create_dataset(config.data.test_path, is_training=False)
    loader = DataLoader(test_ds, batch_size=min(8, config.data.batch_size), shuffle=True)
    print(f"测试集: {len(test_ds)} 样本")

    # 累积梯度统计
    accum = defaultdict(lambda: {'grad_norm': [], 'param_norm': [], 'proj_grad_norm': []})
    losses = []

    print(f"\nRunning {args.n_batches} batches...")
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        grad_norms, loss = get_modality_gradients(model, batch, device)
        losses.append(loss)
        for mod, metrics in grad_norms.items():
            for k, v in metrics.items():
                if k in accum[mod]:
                    accum[mod][k].append(v)
                else:
                    accum[mod][k] = [v]

    # 汇总结果
    print(f"\n{'='*70}")
    print(f"各模态梯度贡献分析")
    print(f"频率: {args.config.split('_')[-1].replace('.yaml','')}")
    print(f"样本批次数: {len(losses)}, 平均Loss: {np.mean(losses):.4f}")
    print(f"{'='*70}")

    print(f"\n{'模态':<12} {'Stem梯度范数':<16} {'投影层梯度':<16} {'参数范数':<16} {'梯度/参数比':<14}")
    print("-" * 74)

    total_grad = 0
    modality_grads = {}
    for mod in ['image', 'audio', 'wave']:
        if mod not in accum:
            continue
        g = accum[mod]
        avg_grad = np.mean(g['grad_norm'])
        avg_param = np.mean(g['param_norm']) if g['param_norm'] else 1.0
        avg_proj = np.mean(g['proj_grad_norm']) if 'proj_grad_norm' in g and g['proj_grad_norm'] else 0
        ratio = avg_grad / avg_param if avg_param > 0 else 0
        modality_grads[mod] = avg_grad
        total_grad += avg_grad
        print(f"{mod:<12} {avg_grad:<16.4f} {avg_proj:<16.4f} {avg_param:<16.4f} {ratio:<14.6f}")

    # 各模态占比
    print(f"\n{'模态':<12} {'梯度贡献占比':<16}")
    print("-" * 28)
    for mod in ['image', 'audio', 'wave']:
        if mod in modality_grads and total_grad > 0:
            pct = 100 * modality_grads[mod] / total_grad
            bar = '█' * int(pct / 2)
            print(f"{mod:<12} {pct:5.1f}% {bar}")

    # Classifier梯度（reference）
    if 'classifier' in accum:
        cls_grad = np.mean(accum['classifier']['grad_norm'])
        print(f"\n分类头梯度范数 (参考): {cls_grad:.4f}")

    # 梯度随时间的变化（前5个batch的详细值）
    print(f"\n--- 前5个batch详细梯度 ---")
    print(f"{'Batch':<8} {'Image':<14} {'Audio':<14} {'Wave':<14} {'Loss':<10}")
    print("-" * 60)
    for i in range(min(5, len(losses))):
        img = accum['image']['grad_norm'][i] if i < len(accum['image']['grad_norm']) else 0
        aud = accum['audio']['grad_norm'][i] if i < len(accum['audio']['grad_norm']) else 0
        wav = accum['wave']['grad_norm'][i] if i < len(accum['wave']['grad_norm']) else 0
        print(f"{i:<8} {img:<14.4f} {aud:<14.4f} {wav:<14.4f} {losses[i]:<10.4f}")


if __name__ == '__main__':
    main()
