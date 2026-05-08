import os
import sys
import traceback
import numpy as np
import torch
from typing import Dict, Optional, List, Any


def _get_api_key() -> Optional[str]:
    """从项目文件中读取 SwanLab API key"""
    key_files = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "swanlab_api_key.txt"),
        os.path.join(os.getcwd(), "swanlab_api_key.txt"),
    ]
    for f in key_files:
        if os.path.exists(f):
            with open(f, "r") as fh:
                key = fh.read().strip()
                if key:
                    return key
    return None


class SwanLabLogger:
    """SwanLab 日志记录器，接口对齐 TensorBoardLogger"""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "default",
        project: str = "MM",
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = enabled
        self.run = None

        if self.enabled:
            import swanlab

            api_key = _get_api_key()
            if api_key:
                os.environ["SWANLAB_API_KEY"] = api_key

            try:
                log_path = os.path.join(log_dir, "swanlab")
                os.makedirs(log_path, exist_ok=True)

                self.run = swanlab.init(
                    project=project,
                    experiment_name=experiment_name,
                    logdir=log_path,
                    config=config or {},
                )
                self.log_dir = log_path
            except Exception:
                self.enabled = False
                print(f"[SwanLab] Init failed:\n{traceback.format_exc()}", file=sys.stderr)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        if self.enabled and self.run is not None:
            import swanlab
            swanlab.log({tag: scalar_value}, step=global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        if self.enabled and self.run is not None:
            import swanlab
            data = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            swanlab.log(data, step=global_step)

    def _log_figure(self, tag: str, fig, global_step: int):
        if not self.enabled or self.run is None:
            import matplotlib.pyplot as plt
            plt.close(fig)
            return
        import swanlab
        try:
            swanlab.log({tag: swanlab.Image(fig)}, step=global_step)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def add_confusion_matrix(
        self, tag: str, cm: np.ndarray, global_step: int,
        class_names: Optional[List[str]] = None,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        if class_names:
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()
        self._log_figure(tag, fig, global_step)

    def add_roc_curve(
        self, tag: str, fpr: np.ndarray, tpr: np.ndarray,
        auc: float, global_step: int,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        self._log_figure(tag, fig, global_step)

    def add_pr_curve(
        self, tag: str, precision: np.ndarray, recall: np.ndarray,
        ap: float, global_step: int,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {ap:.4f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()
        self._log_figure(tag, fig, global_step)

    def add_per_class_metrics(
        self, tag_prefix: str,
        metrics_per_class: Dict[str, Dict[str, float]],
        global_step: int,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class_names = list(metrics_per_class.keys())
        metric_names = list(metrics_per_class[class_names[0]].keys())

        for metric_name in metric_names:
            values = [metrics_per_class[cls][metric_name] for cls in class_names]
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(class_names))
            ax.bar(x, values)
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} per Class")
            ax.set_ylim([0, 1.05])
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
            fig.tight_layout()
            self._log_figure(f"{tag_prefix}/{metric_name}", fig, global_step)

    def add_feature_distribution(
        self, tag: str, features: np.ndarray, labels: np.ndarray,
        global_step: int, method: str = "tsne",
        class_names: Optional[List[str]] = None, max_samples: int = 1000,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")

        embedding = reducer.fit_transform(features)

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            label_name = (
                class_names[label] if class_names and label < len(class_names)
                else f"Class {label}"
            )
            ax.scatter(embedding[mask, 0], embedding[mask, 1], label=label_name, alpha=0.6, s=20)

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(f"Feature Distribution ({method.upper()})")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        self._log_figure(tag, fig, global_step)

    def add_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        if self.enabled and self.run is not None:
            import swanlab
            swanlab.log({f"hparam/{k}": metrics.get(k, 0) for k in metrics}, step=0)

    def add_learning_rate(self, lr: float, global_step: int, param_group_idx: int = 0):
        self.add_scalar(f"learning_rate/group_{param_group_idx}", lr, global_step)

    def add_training_efficiency(
        self, epoch_time: float, throughput: float,
        gpu_memory_mb: Optional[float], global_step: int,
    ):
        self.add_scalar("efficiency/epoch_time_seconds", epoch_time, global_step)
        self.add_scalar("efficiency/throughput_samples_per_sec", throughput, global_step)
        if gpu_memory_mb is not None:
            self.add_scalar("efficiency/gpu_memory_mb", gpu_memory_mb, global_step)

    def add_gradient_norm(self, model: torch.nn.Module, global_step: int):
        if not self.enabled or self.run is None:
            return
        import swanlab

        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                swanlab.log({f"gradient_norm/{name}": param_norm.item()}, step=global_step)

        if total_norm > 0:
            swanlab.log({"gradient_norm/total": total_norm ** 0.5}, step=global_step)

    def add_weight_distribution(self, model: torch.nn.Module, global_step: int):
        # SwanLab 不支持直方图，跳过
        pass

    def flush(self):
        pass

    def close(self):
        if self.enabled and self.run is not None:
            import swanlab
            swanlab.finish()
            self.run = None

    def add_model_graph(self, model, input_example):
        pass
