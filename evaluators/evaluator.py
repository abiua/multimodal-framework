import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import setup_logger
from utils.metrics import calculate_metrics, print_metrics


class Evaluator:
    """评估器"""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        test_loader,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.device = device
        
        # 移动模型到设备
        self.model.to(device)
        
        # 设置日志
        self.logger = setup_logger(
            log_file=os.path.join(config.system.output_dir, 'eval.log')
        )
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """评估模型"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in self.test_loader:
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch['class_idx'].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # 计算指标
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            class_names=self.config.classes.class_names
        )
        
        # 保存预测结果
        if self.config.eval.save_predictions:
            self._save_predictions(all_preds, all_labels, all_probs)
        
        # 绘制混淆矩阵
        if self.config.eval.confusion_matrix:
            self._plot_confusion_matrix(metrics['confusion_matrix'])
        
        return metrics
    
    def _save_predictions(
        self,
        preds: List[int],
        labels: List[int],
        probs: List[np.ndarray]
    ):
        """保存预测结果"""
        predictions = {
            'predictions': preds,
            'labels': labels,
            'probabilities': probs
        }
        
        save_path = os.path.join(self.config.system.output_dir, 'predictions.npz')
        np.savez(save_path, **predictions)
        self.logger.info(f'预测结果已保存: {save_path}')
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.classes.class_names,
            yticklabels=self.config.classes.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        save_path = os.path.join(self.config.system.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f'混淆矩阵已保存: {save_path}')
    
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f'模型已加载: {checkpoint_path}')
    
    def run(self) -> Dict[str, Any]:
        """运行评估"""
        self.logger.info('开始评估...')
        
        metrics = self.evaluate()
        
        # 打印指标
        print_metrics(metrics, self.logger)
        
        self.logger.info('评估完成!')
        
        return metrics