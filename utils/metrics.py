import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = 'weighted',
    num_classes: Optional[int] = None
) -> Dict[str, any]:
    """计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率（用于ROC/PR曲线）
        class_names: 类别名称列表
        average: 平均方式 ('weighted', 'macro', 'micro', None)
        num_classes: 类别数量
    
    Returns:
        包含所有指标的字典
    """
    
    metrics = {}
    
    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # 分类报告
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
        metrics['classification_report'] = report
        
        # 每个类别的指标
        metrics['per_class'] = {}
        for cls_name in class_names:
            if cls_name in report:
                metrics['per_class'][cls_name] = {
                    'precision': report[cls_name]['precision'],
                    'recall': report[cls_name]['recall'],
                    'f1': report[cls_name]['f1-score'],
                    'support': report[cls_name]['support']
                }
    
    # 如果提供了预测概率，计算ROC和PR曲线
    if y_probs is not None:
        if not np.isfinite(y_probs).all():
            metrics['warning'] = 'y_probs contains NaN or Inf, skip ROC/PR metrics'
            return metrics
        if num_classes is None:
            num_classes = len(np.unique(y_true))
        
        roc_metrics = calculate_roc_metrics(y_true, y_probs, num_classes, class_names)
        metrics.update(roc_metrics)
        
        pr_metrics = calculate_pr_metrics(y_true, y_probs, num_classes, class_names)
        metrics.update(pr_metrics)
    
    return metrics


def calculate_roc_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """计算ROC相关指标
    
    Args:
        y_true: 真实标签
        y_probs: 预测概率 (n_samples, n_classes)
        num_classes: 类别数量
        class_names: 类别名称列表
    
    Returns:
        ROC相关指标字典
    """
    metrics = {}
    
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if num_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    # 计算每类的ROC曲线和AUC
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        if np.sum(y_true_bin[:, i]) > 0:  # 确保该类有样本
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 1])
            roc_auc[i] = 0.0
    
    # 计算micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 计算macro-average AUC
    all_auc = [roc_auc[i] for i in range(num_classes)]
    roc_auc["macro"] = np.mean(all_auc)
    
    metrics['roc'] = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }
    
    return metrics


def calculate_pr_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """计算Precision-Recall相关指标
    
    Args:
        y_true: 真实标签
        y_probs: 预测概率 (n_samples, n_classes)
        num_classes: 类别数量
        class_names: 类别名称列表
    
    Returns:
        PR相关指标字典
    """
    metrics = {}
    
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if num_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    # 计算每类的PR曲线和AP
    precision = {}
    recall = {}
    ap = {}
    
    for i in range(num_classes):
        if np.sum(y_true_bin[:, i]) > 0:  # 确保该类有样本
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            ap[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        else:
            precision[i] = np.array([1, 0])
            recall[i] = np.array([0, 1])
            ap[i] = 0.0
    
    # 计算macro-average AP
    all_ap = [ap[i] for i in range(num_classes)]
    ap["macro"] = np.mean(all_ap)
    
    metrics['pr'] = {
        'precision': precision,
        'recall': recall,
        'ap': ap
    }
    
    return metrics


def get_predictions_with_probs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """获取模型预测结果和概率
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        (true_labels, predicted_labels, predicted_probs)
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = model(batch)
            
            # 获取概率和预测
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(batch['class_idx'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(metrics: Dict[str, float], logger=None):
    """打印指标"""
    
    output_func = logger.info if logger else print
    
    output_func("=" * 50)
    output_func("Evaluation Metrics:")
    output_func("=" * 50)
    
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            output_func(f"\nConfusion Matrix:")
            output_func(str(value))
        elif key == 'classification_report':
            if isinstance(value, dict):
                output_func(f"\nClassification Report:")
                import json
                output_func(json.dumps(value, indent=2))
            else:
                output_func(f"\nClassification Report:")
                output_func(str(value))
        elif key in ['roc', 'pr']:
            continue  # 跳过曲线数据
        elif key == 'per_class':
            output_func(f"\nPer-Class Metrics:")
            for cls_name, cls_metrics in value.items():
                output_func(f"  {cls_name}: P={cls_metrics['precision']:.4f}, "
                          f"R={cls_metrics['recall']:.4f}, F1={cls_metrics['f1']:.4f}")
        else:
            output_func(f"{key}: {value:.4f}")
    
    output_func("=" * 50)
