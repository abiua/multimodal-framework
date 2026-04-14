"""训练循环管理器"""
import time
import torch
import torch.distributed as dist
from typing import Dict, Optional, Any, List


def _is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _reduce_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not dist.is_available() or not dist.is_initialized():
        return input_dict
    with torch.no_grad():
        values = list(input_dict.values())
        dist.all_reduce(torch.tensor(values).to(next(iter(input_dict.values())).device))
        return {k: v / _get_world_size() for k, v in zip(input_dict.keys(), values)}


class TrainingLoop:
    """训练循环管理器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[torch.device] = None,
        fp16: bool = False,
        lr_scheduler=None,
        logger=None,
        tb_logger=None,
        class_names: Optional[List[str]] = None,
        num_classes: int = 10
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fp16 = fp16
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.tb_logger = tb_logger
        self.class_names = class_names
        self.num_classes = num_classes
        
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
        # 分布式相关
        self.distributed = dist.is_available() and dist.is_initialized()
        
        # 指标统计
        self.epoch_start_time = None
        self.training_samples_processed = 0
    
    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.epoch_start_time = time.time()
        
        total_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_size = batch['class_idx'].size(0)
            
            # 前向传播
            if self.fp16 and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch['class_idx'])
            else:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['class_idx'])
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.fp16 and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(batch['class_idx']).sum().item()
            
            self.global_step += 1
            self.training_samples_processed += batch_size
            
            # TensorBoard记录
            if self.tb_logger and self.global_step % 10 == 0:
                self.tb_logger.add_scalar('train/batch_loss', loss.item(), self.global_step)
                self.tb_logger.add_scalar('train/batch_acc', 100. * correct / total, self.global_step)
                self.tb_logger.add_scalar('train/learning_rate', self._get_current_lr(), self.global_step)
            
            # 日志
            if _is_main_process() and self.logger and batch_idx % 10 == 0:
                self.logger.info(
                    f'Epoch: {self.current_epoch} | '
                    f'Batch: {batch_idx}/{len(self.train_loader)} | '
                    f'Loss: {loss.item():.4f} | '
                    f'Acc: {100. * correct / total:.2f}%'
                )
        
        # 计算epoch指标
        epoch_time = time.time() - self.epoch_start_time
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        throughput = total / epoch_time if epoch_time > 0 else 0
        
        # TensorBoard记录epoch指标
        if self.tb_logger and _is_main_process():
            self.tb_logger.add_scalar('train/epoch_loss', epoch_loss, self.current_epoch)
            self.tb_logger.add_scalar('train/epoch_acc', epoch_acc, self.current_epoch)
            self.tb_logger.add_scalar('train/epoch_time', epoch_time, self.current_epoch)
            self.tb_logger.add_scalar('train/throughput', throughput, self.current_epoch)
            self.tb_logger.flush()
        
        # 分布式汇总
        if self.distributed:
            return _reduce_dict({'train_loss': epoch_loss, 'train_acc': epoch_acc})
        
        return {'train_loss': epoch_loss, 'train_acc': epoch_acc}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds, all_probs = [], [], []
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch['class_idx'])
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += batch['class_idx'].size(0)
            correct += predicted.eq(batch['class_idx']).sum().item()
            
            all_labels.extend(batch['class_idx'].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        
        # 分布式汇总
        if self.distributed:
            return _reduce_dict({'val_loss': val_loss, 'accuracy': val_acc})
        
        return {'val_loss': val_loss, 'accuracy': val_acc}
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_epoch_for_sampler(self, epoch: int):
        """为分布式采样器设置epoch"""
        if self.distributed:
            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, dist.DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader, 'sampler') and isinstance(self.val_loader.sampler, dist.DistributedSampler):
                self.val_loader.sampler.set_epoch(epoch)
