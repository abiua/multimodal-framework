import os
import numpy as np
import torch
from typing import Dict, Optional, List, Any
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard日志记录器
    
    用于记录训练过程中的各种指标和可视化数据，支持：
    - 标量指标（loss、accuracy等）
    - 混淆矩阵可视化
    - ROC曲线和PR曲线
    - 特征分布可视化（t-SNE）
    - 模型图
    - 学习率曲线
    - 训练效率指标
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "default",
        enabled: bool = True
    ):
        """
        Args:
            log_dir: TensorBoard日志目录
            experiment_name: 实验名称，用于区分不同实验
            enabled: 是否启用TensorBoard记录
        """
        self.enabled = enabled
        self.writer = None
        
        if self.enabled:
            # 创建日志目录
            log_path = os.path.join(log_dir, "tensorboard", experiment_name)
            os.makedirs(log_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_path)
            self.log_dir = log_path
    
    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        """记录标量值"""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """记录多个标量值"""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
    
    def add_confusion_matrix(
        self,
        tag: str,
        cm: np.ndarray,
        global_step: int,
        class_names: Optional[List[str]] = None
    ):
        """记录混淆矩阵为图像
        
        Args:
            tag: 标签名称
            cm: 混淆矩阵
            global_step: 全局步数
            class_names: 类别名称列表
        """
        if not self.enabled or self.writer is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制混淆矩阵
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        if class_names:
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix')
        
        # 在格子中显示数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        self.writer.add_figure(tag, fig, global_step)
        plt.close(fig)
    
    def add_roc_curve(
        self,
        tag: str,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        global_step: int
    ):
        """记录ROC曲线
        
        Args:
            tag: 标签名称
            fpr: False Positive Rate
            tpr: True Positive Rate
            auc: AUC值
            global_step: 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        fig.tight_layout()
        
        self.writer.add_figure(tag, fig, global_step)
        plt.close(fig)
    
    def add_pr_curve(
        self,
        tag: str,
        precision: np.ndarray,
        recall: np.ndarray,
        ap: float,
        global_step: int
    ):
        """记录Precision-Recall曲线
        
        Args:
            tag: 标签名称
            precision: 精确率数组
            recall: 召回率数组
            ap: Average Precision
            global_step: 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        fig.tight_layout()
        
        self.writer.add_figure(tag, fig, global_step)
        plt.close(fig)
    
    def add_per_class_metrics(
        self,
        tag_prefix: str,
        metrics_per_class: Dict[str, Dict[str, float]],
        global_step: int
    ):
        """记录每个类别的指标
        
        Args:
            tag_prefix: 标签前缀
            metrics_per_class: 每个类别的指标字典
                格式: {class_name: {metric_name: value}}
            global_step: 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 准备数据
        class_names = list(metrics_per_class.keys())
        metric_names = list(metrics_per_class[class_names[0]].keys())
        
        # 为每个指标创建柱状图
        for metric_name in metric_names:
            values = [metrics_per_class[cls][metric_name] for cls in class_names]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(class_names))
            ax.bar(x, values)
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} per Class')
            ax.set_ylim([0, 1.05])
            
            # 在柱子上方显示数值
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            fig.tight_layout()
            self.writer.add_figure(f'{tag_prefix}/{metric_name}', fig, global_step)
            plt.close(fig)
    
    def add_scalar_metrics_bar(
        self,
        tag: str,
        metrics: Dict[str, float],
        global_step: int
    ):
        """记录指标柱状图
        
        Args:
            tag: 标签名称
            metrics: 指标字典
            global_step: 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(metrics))
        ax.bar(x, list(metrics.values()))
        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax.set_title('Evaluation Metrics')
        
        # 在柱子上方显示数值
        for i, (k, v) in enumerate(metrics.items()):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        fig.tight_layout()
        self.writer.add_figure(tag, fig, global_step)
        plt.close(fig)
    
    def add_feature_distribution(
        self,
        tag: str,
        features: np.ndarray,
        labels: np.ndarray,
        global_step: int,
        method: str = 'tsne',
        class_names: Optional[List[str]] = None,
        max_samples: int = 1000
    ):
        """记录特征分布可视化（t-SNE或UMAP降维）
        
        Args:
            tag: 标签名称
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签数组
            global_step: 全局步数
            method: 降维方法 ('tsne' 或 'umap')
            class_names: 类别名称列表
            max_samples: 最大采样数（避免计算过慢）
        """
        if not self.enabled or self.writer is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 采样（如果样本数过多）
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        # 降维
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        embedding = reducer.fit_transform(features)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            label_name = class_names[label] if class_names and label < len(class_names) else f'Class {label}'
            ax.scatter(embedding[mask, 0], embedding[mask, 1], label=label_name, alpha=0.6, s=20)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'Feature Distribution ({method.upper()})')
        ax.legend(loc='best', fontsize='small')
        fig.tight_layout()
        
        self.writer.add_figure(tag, fig, global_step)
        plt.close(fig)
    
    def add_model_graph(self, model: torch.Tensor, input_example: Dict[str, torch.Tensor]):
        """记录模型图
        
        Args:
            model: 模型
            input_example: 输入样例
        """
        if not self.enabled or self.writer is None:
            return
        
        try:
            self.writer.add_graph(model, input_example)
        except Exception as e:
            print(f"Warning: Could not add model graph to TensorBoard: {e}")
    
    def add_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """记录超参数和对应的指标
        
        Args:
            hparams: 超参数字典
            metrics: 对应的指标
        """
        if not self.enabled or self.writer is None:
            return
        
        # 将所有值转换为TensorBoard支持的类型
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v
            elif isinstance(v, (list, tuple)):
                clean_hparams[k] = str(v)
            else:
                clean_hparams[k] = str(v)
        
        self.writer.add_hparams(clean_hparams, metrics)
    
    def add_learning_rate(
        self,
        lr: float,
        global_step: int,
        param_group_idx: int = 0
    ):
        """记录学习率
        
        Args:
            lr: 学习率值
            global_step: 全局步数
            param_group_idx: 参数组索引
        """
        if self.enabled and self.writer:
            self.writer.add_scalar(f'learning_rate/group_{param_group_idx}', lr, global_step)
    
    def add_training_efficiency(
        self,
        epoch_time: float,
        throughput: float,
        gpu_memory_mb: Optional[float],
        global_step: int
    ):
        """记录训练效率指标
        
        Args:
            epoch_time: epoch训练时间（秒）
            throughput: 吞吐量（samples/sec）
            gpu_memory_mb: GPU内存使用（MB）
            global_step: 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        self.writer.add_scalar('efficiency/epoch_time_seconds', epoch_time, global_step)
        self.writer.add_scalar('efficiency/throughput_samples_per_sec', throughput, global_step)
        
        if gpu_memory_mb is not None:
            self.writer.add_scalar('efficiency/gpu_memory_mb', gpu_memory_mb, global_step)
    
    def add_gradient_norm(
        self,
        model: torch.nn.Module,
        global_step: int
    ):
        """记录梯度范数
        
        Args:
            model: 模型
            global_step: 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 记录每层的梯度范数
                self.writer.add_scalar(f'gradient_norm/{name}', param_norm.item(), global_step)
        
        if param_count > 0:
            total_norm = total_norm ** 0.5
            self.writer.add_scalar('gradient_norm/total', total_norm, global_step)
    
    def add_weight_distribution(
        self,
        model: torch.nn.Module,
        global_step: int
    ):
        """记录权重分布直方图
        
        Args:
            model: 模型
            global_step: 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        for name, param in model.named_parameters():
            if param.data.numel() > 0:
                self.writer.add_histogram(f'weights/{name}', param.data, global_step)
    
    def close(self):
        """关闭TensorBoard writer"""
        if self.writer:
            self.writer.close()
    
    def flush(self):
        """刷新TensorBoard writer"""
        if self.writer:
            self.writer.flush()
