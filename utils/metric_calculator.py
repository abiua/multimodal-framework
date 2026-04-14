"""指标计算器"""
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize


class MetricCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
        """计算基础指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        }
    
    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def calculate_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """计算每类指标"""
        if class_names is None:
            class_names = [f'class_{i}' for i in range(len(np.unique(y_true)))]
        
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
        
        return {
            cls_name: {
                'precision': report[cls_name]['precision'],
                'recall': report[cls_name]['recall'],
                'f1': report[cls_name]['f1-score'],
                'support': report[cls_name]['support']
            }
            for cls_name in class_names if cls_name in report
        }
    
    @staticmethod
    def calculate_roc_metrics(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """计算ROC相关指标"""
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        if num_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        fpr, tpr, roc_auc = {}, {}, {}
        
        for i in range(num_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                fpr[i] = np.array([0, 1])
                tpr[i] = np.array([0, 1])
                roc_auc[i] = 0.0
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_auc = [roc_auc[i] for i in range(num_classes)]
        roc_auc["macro"] = np.mean(all_auc)
        
        return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    @staticmethod
    def calculate_pr_metrics(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """计算PR相关指标"""
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        if num_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        precision, recall, ap = {}, {}, {}
        
        for i in range(num_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                ap[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
            else:
                precision[i] = np.array([1, 0])
                recall[i] = np.array([0, 1])
                ap[i] = 0.0
        
        all_ap = [ap[i] for i in range(num_classes)]
        ap["macro"] = np.mean(all_ap)
        
        return {'precision': precision, 'recall': recall, 'ap': ap}
    
    @staticmethod
    @torch.no_grad()
    def get_predictions_with_probs(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> tuple:
        """获取模型预测结果和概率"""
        model.eval()
        
        all_labels, all_preds, all_probs = [], [], []
        
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(batch['class_idx'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
