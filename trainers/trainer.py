import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Optional, Any, Tuple, List
import numpy as np

# 统一的分布式工具导入
from utils.distributed import (
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    save_on_master,
    reduce_dict,
    barrier
)

# 统一的日志导入
from utils.logger import setup_logger

# 统一的指标计算导入
from utils.metrics import calculate_metrics, get_predictions_with_probs


class Trainer:
    """训练器（支持分布式训练）"""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        print("Train dataset size:", len(self.train_loader.dataset))
        print("Train loader size:", len(self.train_loader))
        
        # 检查是否启用分布式训练
        self.distributed = config.system.distributed and is_dist_avail_and_initialized()
        
        # 设置设备
        if device is None:
            if self.distributed:
                self.device = torch.device(f'cuda:{get_rank() % torch.cuda.device_count()}')
            elif torch.cuda.is_available():
                self.device = torch.device(f'cuda:{config.system.gpu_ids[0]}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # 移动模型到设备
        self.model = model.to(self.device)
        
        # 设置分布式数据并行
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                output_device=self.device.index if self.device.type == 'cuda' else None,
                find_unused_parameters=True
            )
        
        # 设置优化器
        self.optimizer = self._build_optimizer()
        
        # 设置学习率调度器
        self.lr_scheduler = self._build_lr_scheduler()
        
        # 设置损失函数
        self.criterion = self._build_criterion()
        
        # 混合精度训练
        self.scaler = GradScaler("cuda") if config.system.fp16 else None
        
        # 设置日志（只在主进程记录）
        if is_main_process():
            self.logger = setup_logger(
                log_file=os.path.join(config.system.output_dir, 'train.log')
            )
        else:
            self.logger = None
        
        # 设置SwanLab日志（只在主进程）
        self.swanlab_logger = None
        if is_main_process():
            sw_enabled = getattr(config.system, 'swanlab_enabled', True)
            if sw_enabled:
                try:
                    from utils.swanlab_logger import SwanLabLogger
                    experiment_name = getattr(config.system, 'experiment_name', 'default')
                    swanlab_project = getattr(config.system, 'swanlab_project', 'MM')
                    self.swanlab_logger = SwanLabLogger(
                        log_dir=config.system.output_dir,
                        experiment_name=experiment_name,
                        project=swanlab_project,
                        enabled=True,
                    )
                    if self.swanlab_logger.run is not None:
                        self._log(f"SwanLab日志已启用，项目: {swanlab_project}, 实验: {experiment_name}")
                    else:
                        self._log("警告: SwanLab初始化失败")
                except Exception as e:
                    import traceback
                    self._log(f"警告: 无法初始化SwanLab: {e}\n{traceback.format_exc()}")
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.global_step = 0
        
        # Early Stop 状态
        self.early_stop_enabled = config.train.early_stop.enabled if hasattr(config.train, 'early_stop') else False
        self.early_stop_patience = config.train.early_stop.patience if hasattr(config.train, 'early_stop') else 10
        self.early_stop_min_delta = config.train.early_stop.min_delta if hasattr(config.train, 'early_stop') else 0.001
        self.early_stop_monitor = config.train.early_stop.monitor if hasattr(config.train, 'early_stop') else 'accuracy'
        self.early_stop_mode = config.train.early_stop.mode if hasattr(config.train, 'early_stop') else 'max'
        self.early_stop_counter = 0
        self.best_monitored_value = None
        
        # 训练效率统计
        self.epoch_start_time = None
        self.training_samples_processed = 0
        
        # 类别名称（用于TensorBoard可视化）
        self.class_names = getattr(config.classes, 'class_names', None)
        self.num_classes = getattr(config.classes, 'num_classes', 10)
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        # 获取模型参数（处理 DDP 包装）
        model_params = self.model.module.parameters() if self.distributed else self.model.parameters()
        
        if self.config.train.optimizer == 'adam':
            return optim.Adam(
                model_params,
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay
            )
        elif self.config.train.optimizer == 'sgd':
            return optim.SGD(
                model_params,
                lr=self.config.train.learning_rate,
                momentum=self.config.train.momentum,
                weight_decay=self.config.train.weight_decay
            )
        elif self.config.train.optimizer == 'adamw':
            return optim.AdamW(
                model_params,
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.train.optimizer}")
        
    def _handle_non_finite_grads(self, batch, batch_idx) -> bool:
        """检查非有限梯度；DDP 下任意 rank 出现异常，所有 rank 一起跳过该 batch。"""
        model_to_check = self.model.module if self.distributed else self.model

        bad_names = []
        for name, param in model_to_check.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                bad_names.append(name)

        local_bad = 1 if bad_names else 0
        bad_flag = torch.tensor([local_bad], device=self.device, dtype=torch.int)

        if self.distributed:
            dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)

        if bad_flag.item() > 0:
            sample_ids = batch.get('sample_id', None)
            class_names = batch.get('class_name', None)

            print(
                f"[rank{get_rank()}] Skip non-finite grad batch: "
                f"epoch={self.current_epoch}, batch_idx={batch_idx}, "
                f"local_bad_layers={bad_names[:20]}, "
                f"sample_ids={sample_ids}, class_names={class_names}",
                flush=True
            )

            self.optimizer.zero_grad(set_to_none=True)
            return True

        return False    
    
    def _build_lr_scheduler(self):
        """构建学习率调度器"""
        if self.config.train.lr_scheduler == 'cosine':
            warmup_epochs = int(getattr(self.config.train, 'warmup_epochs', 0))
            total_epochs = int(self.config.train.epochs)
            base_lr = float(self.config.train.learning_rate)

            cosine = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_epochs - warmup_epochs, 1),
                eta_min=base_lr * 0.01
            )

            if warmup_epochs > 0:
                warmup = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                return optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs]
                )

            return cosine

        elif self.config.train.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.train.step_size,
                gamma=self.config.train.gamma
            )

        elif self.config.train.lr_scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.train.gamma,
                patience=10
            )

        else:
            return None
    
    def _build_criterion(self) -> nn.Module:
        """构建损失函数"""
        if self.config.train.label_smoothing > 0:
            return nn.CrossEntropyLoss(
                label_smoothing=self.config.train.label_smoothing,
                weight=self._get_class_weights()
            )
        else:
            return nn.CrossEntropyLoss(weight=self._get_class_weights())
    
    def _get_class_weights(self) -> Optional[torch.Tensor]:
        """获取类别权重"""
        if self.config.classes.class_weights:
            return torch.tensor(
                self.config.classes.class_weights,
                dtype=torch.float32,
                device=self.device
            )
        return None
    
    def _log(self, message: str):
        """记录日志（只在主进程）"""
        if self.logger is not None:
            self.logger.info(message)
    
    def _set_epoch_for_sampler(self, epoch: int):
        """为分布式采样器设置 epoch"""
        if self.distributed:
            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader, 'sampler') and isinstance(self.val_loader.sampler, DistributedSampler):
                self.val_loader.sampler.set_epoch(epoch)
    
    def _check_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """检查是否应该 early stop"""
        if not self.early_stop_enabled or not val_metrics:
            return False
        
        # 获取监控的指标值
        if self.early_stop_monitor == 'accuracy':
            current_value = val_metrics.get('accuracy', 0.0)
        elif self.early_stop_monitor == 'val_loss':
            current_value = val_metrics.get('val_loss', float('inf'))
        else:
            return False
        
        # 初始化最佳值
        if self.best_monitored_value is None:
            self.best_monitored_value = current_value
            return False
        
        # 检查是否有改善
        if self.early_stop_mode == 'max':
            improved = current_value > self.best_monitored_value + self.early_stop_min_delta
        else:  # mode == 'min'
            improved = current_value < self.best_monitored_value - self.early_stop_min_delta
        
        if improved:
            self.best_monitored_value = current_value
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.early_stop_patience:
                self._log(f'Early stopping triggered! No improvement for {self.early_stop_patience} epochs.')
                return True
            return False
    
    def _get_gpu_memory(self) -> Optional[float]:
        """获取GPU内存使用（MB）"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        return None
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def _move_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        return obj
    
    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.epoch_start_time = time.time()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()

            # 移动数据到设备
            batch = self._move_to_device(batch)

            batch_size = batch['class_idx'].size(0)

            # 前向传播
            if self.config.system.fp16:
                with autocast("cuda"):
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch['class_idx'])
            else:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['class_idx'])

            # 数值稳定性检查：先检查输出和loss
            if not torch.isfinite(outputs).all():
                raise ValueError(
                    f"Non-finite outputs detected at epoch={self.current_epoch}, "
                    f"batch_idx={batch_idx}"
                )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss detected at epoch={self.current_epoch}, "
                    f"batch_idx={batch_idx}, loss={loss.item()}"
                )

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)

            if self.config.system.fp16 and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                if self._handle_non_finite_grads(batch, batch_idx):
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self._handle_non_finite_grads(batch, batch_idx):
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

            # 再检查参数是否被更新成 NaN/Inf
            model_to_check = self.model.module if self.distributed else self.model
            has_non_finite_param = False
            for name, param in model_to_check.named_parameters():
                if param is not None and not torch.isfinite(param).all():
                    has_non_finite_param = True
                    raise ValueError(
                        f"Non-finite parameter detected after optimizer step: "
                        f"{name}, epoch={self.current_epoch}, batch_idx={batch_idx}"
                    )

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch['class_idx'].size(0)
            correct += predicted.eq(batch['class_idx']).sum().item()

            self.global_step += 1
            self.training_samples_processed += batch_size

            # SwanLab: 记录batch级别指标
            if self.swanlab_logger and self.global_step % self.config.system.log_interval == 0:
                self.swanlab_logger.add_scalar('train/batch_loss', loss.item(), self.global_step)
                self.swanlab_logger.add_scalar('train/batch_acc', 100.0 * correct / total, self.global_step)
                self.swanlab_logger.add_scalar('train/learning_rate', self._get_current_lr(), self.global_step)
                self.swanlab_logger.add_scalar('train/grad_norm', float(grad_norm), self.global_step)

            # 日志（只在主进程）
            if is_main_process() and batch_idx % self.config.system.log_interval == 0:
                self._log(
                    f'Epoch: {self.current_epoch} | '
                    f'Batch: {batch_idx}/{len(self.train_loader)} | '
                    f'Loss: {loss.item():.4f} | '
                    f'GradNorm: {float(grad_norm):.4f} | '
                    f'Acc: {100.0 * correct / total:.2f}%'
                )

        # 计算epoch指标
        epoch_time = time.time() - self.epoch_start_time
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        throughput = total / epoch_time if epoch_time > 0 else 0

        # SwanLab: 记录epoch级别指标
        if self.swanlab_logger:
            self.swanlab_logger.add_scalar('train/epoch_loss', epoch_loss, self.current_epoch)
            self.swanlab_logger.add_scalar('train/epoch_acc', epoch_acc, self.current_epoch)
            self.swanlab_logger.add_scalar('train/epoch_time', epoch_time, self.current_epoch)
            self.swanlab_logger.add_scalar('train/throughput', throughput, self.current_epoch)

            gpu_memory = self._get_gpu_memory()
            if gpu_memory is not None:
                self.swanlab_logger.add_scalar('system/gpu_memory_mb', gpu_memory, self.current_epoch)

            # 记录权重分布（每10个epoch记录一次）
            if self.current_epoch % 10 == 0:
                model = self.model.module if self.distributed else self.model
                self.swanlab_logger.add_weight_distribution(model, self.current_epoch)

            self.swanlab_logger.flush()

        # 分布式训练时汇总指标
        if self.distributed:
            metrics = {
                'train_loss': epoch_loss,
                'train_acc': epoch_acc
            }
            reduced_metrics = reduce_dict(metrics)
            return reduced_metrics

        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        all_features = []  # 用于特征可视化
        
        for batch in self.val_loader:
            # 移动数据到设备
            batch = self._move_to_device(batch)
            
            # 前向传播
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch['class_idx'])
            
            # 统计
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += batch['class_idx'].size(0)
            correct += predicted.eq(batch['class_idx']).sum().item()
            
            # 收集预测结果用于详细指标计算
            all_labels.extend(batch['class_idx'].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 收集特征用于可视化（每20个epoch记录一次）
            if self.swanlab_logger and self.current_epoch % 20 == 0:
                try:
                    model = self.model.module if self.distributed else self.model
                    features = model.extract_features(batch)
                    all_features.extend(features.cpu().numpy())
                except Exception:
                    pass  # 如果提取特征失败，跳过
        
        # 计算本地指标
        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        
        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # TensorBoard: 记录验证指标
        if self.swanlab_logger:
            self.swanlab_logger.add_scalar('val/loss', val_loss, self.current_epoch)
            self.swanlab_logger.add_scalar('val/accuracy', val_acc * 100, self.current_epoch)
            
            # 计算详细指标
            detailed_metrics = calculate_metrics(
                y_true=all_labels,
                y_pred=all_preds,
                y_probs=all_probs,
                class_names=self.class_names,
                average='macro',
                num_classes=self.num_classes
            )
            
            if detailed_metrics:
                # 记录整体指标
                self.swanlab_logger.add_scalars(
                    'val/metrics',
                    {
                        'accuracy': detailed_metrics.get('accuracy', 0),
                        'precision': detailed_metrics.get('precision', 0),
                        'recall': detailed_metrics.get('recall', 0),
                        'f1': detailed_metrics.get('f1', 0)
                    },
                    self.current_epoch
                )
                
                # 记录混淆矩阵
                if 'confusion_matrix' in detailed_metrics:
                    self.swanlab_logger.add_confusion_matrix(
                        'val/confusion_matrix',
                        detailed_metrics['confusion_matrix'],
                        self.current_epoch,
                        self.class_names
                    )
                
                # 记录每类指标
                if 'per_class' in detailed_metrics:
                    self.swanlab_logger.add_per_class_metrics(
                        'val',
                        detailed_metrics['per_class'],
                        self.current_epoch
                    )
                
                # 记录ROC曲线（每5个epoch记录一次）
                if 'roc' in detailed_metrics and self.current_epoch % 5 == 0:
                    roc_data = detailed_metrics['roc']
                    # 记录macro-average ROC
                    self.swanlab_logger.add_roc_curve(
                        'val/roc_macro',
                        roc_data['fpr']['micro'],  # 使用micro作为代表
                        roc_data['tpr']['micro'],
                        roc_data['auc'].get('macro', roc_data['auc'].get('micro', 0)),
                        self.current_epoch
                    )
                
                # 记录PR曲线（每5个epoch记录一次）
                if 'pr' in detailed_metrics and self.current_epoch % 5 == 0:
                    pr_data = detailed_metrics['pr']
                    # 记录第一个有样本的类别的PR曲线作为代表
                    for i in range(self.num_classes):
                        if i in pr_data['precision']:
                            self.swanlab_logger.add_pr_curve(
                                f'val/pr_class_{i}',
                                pr_data['precision'][i],
                                pr_data['recall'][i],
                                pr_data['ap'].get(i, 0),
                                self.current_epoch
                            )
                            break
                
                # 记录特征分布（每20个epoch记录一次）
                if all_features and self.current_epoch % 20 == 0:
                    try:
                        self.swanlab_logger.add_feature_distribution(
                            'val/feature_distribution',
                            np.array(all_features),
                            all_labels,
                            self.current_epoch,
                            method='tsne',
                            class_names=self.class_names,
                            max_samples=500
                        )
                    except Exception as e:
                        self._log(f"Warning: 无法记录特征分布: {e}")
            
            self.swanlab_logger.flush()
        
        # 分布式训练时汇总指标
        if self.distributed:
            metrics = {
                'val_loss': val_loss,
                'accuracy': val_acc
            }
            reduced_metrics = reduce_dict(metrics)
            return reduced_metrics
        
        return {
            'val_loss': val_loss,
            'accuracy': val_acc
        }
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """保存检查点（只在主进程）"""
        if not is_main_process():
            return
        
        # 获取模型状态字典（处理 DDP 包装）
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'global_step': self.global_step,
            'early_stop_counter': self.early_stop_counter,
            'best_monitored_value': self.best_monitored_value
        }
        
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        save_path = os.path.join(self.config.system.output_dir, filename)
        torch.save(checkpoint, save_path)
        self._log(f'检查点已保存: {save_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        map_location = self.device if not self.distributed else f'cuda:{get_rank() % torch.cuda.device_count()}'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # 加载模型状态字典（处理 DDP 包装）
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.global_step = checkpoint['global_step']
        
        if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # 恢复 Early Stop 状态
        if 'early_stop_counter' in checkpoint:
            self.early_stop_counter = checkpoint['early_stop_counter']
        if 'best_monitored_value' in checkpoint:
            self.best_monitored_value = checkpoint['best_monitored_value']
        
        self._log(f'检查点已加载: {checkpoint_path}')
    
    def train(self):
        """完整训练流程"""
        self._log('开始训练...')
        self._log(f'训练配置: {self.config}')
        
        if self.distributed:
            self._log(f'分布式训练: 世界大小={get_world_size()}, 当前排名={get_rank()}')
        
        # TensorBoard: 记录超参数
        if self.swanlab_logger:
            hparams = {
                'batch_size': self.config.data.batch_size,
                'learning_rate': self.config.train.learning_rate,
                'epochs': self.config.train.epochs,
                'optimizer': self.config.train.optimizer,
                'lr_scheduler': self.config.train.lr_scheduler,
                'weight_decay': self.config.train.weight_decay,
                'label_smoothing': self.config.train.label_smoothing,
                'backbone': getattr(self.config.model, 'backbone', 'unknown'),
                'fusion_type': getattr(self.config.model, 'fusion_type', 'none'),
                'num_classes': self.num_classes,
                'fp16': self.config.system.fp16
            }
            # 先保存超参数，训练结束后再记录最终指标
            self._hparams = hparams
        
        for epoch in range(self.config.train.epochs):
            self.current_epoch = epoch
            
            # 设置采样器的 epoch
            self._set_epoch_for_sampler(epoch)
            
            # 训练一个epoch
            train_metrics = self.train_one_epoch()
            self._log(
                f'Epoch {epoch} - '
                f'Train Loss: {train_metrics["train_loss"]:.4f}, '
                f'Train Acc: {train_metrics["train_acc"]:.2f}%'
            )
            
            # 验证
            val_metrics = {}
            val_interval = getattr(self.config.train, 'val_interval', 1)
            if self.val_loader and (epoch + 1) % val_interval == 0:
                val_metrics = self.validate()
                if val_metrics:
                    self._log(
                        f'Epoch {epoch} - '
                        f'Val Loss: {val_metrics["val_loss"]:.4f}, '
                        f'Val Acc: {val_metrics["accuracy"] * 100:.2f}%'
                    )
                    
                    # 保存最佳模型
                    if val_metrics['accuracy'] > self.best_val_acc:
                        self.best_val_acc = val_metrics['accuracy']
                        self.save_checkpoint('best_model.pth')
                        self._log(f'新的最佳模型! 准确率: {self.best_val_acc * 100:.2f}%')
                    
                    # 检查 Early Stop
                    if self._check_early_stop(val_metrics):
                        self._log('Early stopping triggered. Training stopped.')
                        break
            
            # 保存检查点
            if (epoch + 1) % self.config.system.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # 更新学习率
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics and 'accuracy' in val_metrics:
                        self.lr_scheduler.step(val_metrics['accuracy'])
                else:
                    self.lr_scheduler.step()
            
            # 同步所有进程
            if self.distributed:
                barrier()
        
        # SwanLab: 记录最终超参数和指标
        if self.swanlab_logger and hasattr(self, '_hparams'):
            final_metrics = {
                'hparam/best_val_acc': self.best_val_acc,
                'hparam/final_epoch': self.current_epoch
            }
            self.swanlab_logger.add_hyperparameters(self._hparams, final_metrics)
            self.swanlab_logger.close()
        
        self._log('训练完成!')
