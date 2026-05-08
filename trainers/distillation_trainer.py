import torch
import torch.nn as nn
from trainers.trainer import Trainer
from models.distillation.contrastive_loss import ContrastiveDistillationLoss


class DistillationTrainer(Trainer):
    """蒸馏训练器，继承自基础训练器"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config,
        train_loader,
        val_loader=None,
        device=None
    ):
        """
        初始化蒸馏训练器
        
        Args:
            teacher_model: 预训练的教师模型
            student_model: 学生模型
            config: 配置对象
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
        """
        # 先初始化父类，但暂时不调用 Trainer.__init__ 的完整逻辑
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{config.system.gpu_ids[0]}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # 冻结教师模型
        self.teacher_model = teacher_model.to(self.device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # 学生模型
        self.student_model = student_model.to(self.device)
        
        # 初始化蒸馏损失函数
        distill_config = getattr(config, 'distillation', {})
        self.distillation_loss = ContrastiveDistillationLoss(
            temperature=distill_config.get('temperature', 0.07),
            alpha=distill_config.get('alpha', 0.5),
            contrastive_weight=distill_config.get('contrastive_weight', 0.3)
        )
        
        # 调用父类初始化（部分参数会覆盖）
        super().__init__(self.student_model, config, train_loader, val_loader, device)
        
        # 重新设置模型引用
        self.model = self.student_model
        
        # 重置优化器以确保只优化学生模型参数
        self.optimizer = self._build_optimizer()
    
    def train_one_epoch(self) -> dict:
        """
        训练一个epoch，使用知识蒸馏
        
        Returns:
            包含训练指标的字典
        """
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0.0
        total_hard_loss = 0.0
        total_kl_loss = 0.0
        total_contrastive_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            hard_labels = batch['class_idx']
            batch_size = hard_labels.size(0)
            
            # 教师模型前向传播（无梯度）
            with torch.no_grad():
                teacher_logits = self.teacher_model(batch)
            
            # 学生模型前向传播
            if self.config.system.fp16:
                with torch.cuda.amp.autocast():
                    student_logits = self.student_model(batch)
                    # 计算蒸馏损失
                    distill_loss, loss_dict = self.distillation_loss(
                        student_logits, teacher_logits, hard_labels
                    )
            else:
                student_logits = self.student_model(batch)
                # 计算蒸馏损失
                distill_loss, loss_dict = self.distillation_loss(
                    student_logits, teacher_logits, hard_labels
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.config.system.fp16 and self.scaler is not None:
                self.scaler.scale(distill_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                distill_loss.backward()
                self.optimizer.step()
            
            # 统计损失
            total_loss += distill_loss.item()
            total_hard_loss += loss_dict['hard_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            total_contrastive_loss += loss_dict['contrastive_loss'].item()
            
            # 计算准确率
            _, predicted = student_logits.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
            
            self.global_step += 1
            
            # SwanLab: 记录batch级别指标
            if self.swanlab_logger and self.global_step % self.config.system.log_interval == 0:
                self.swanlab_logger.add_scalar('train/batch_loss', distill_loss.item(), self.global_step)
                self.swanlab_logger.add_scalar('train/batch_acc', 100. * correct / total, self.global_step)
                self.swanlab_logger.add_scalar('train/batch_hard_loss', loss_dict['hard_loss'].item(), self.global_step)
                self.swanlab_logger.add_scalar('train/batch_kl_loss', loss_dict['kl_loss'].item(), self.global_step)
                self.swanlab_logger.add_scalar('train/batch_contrastive_loss', loss_dict['contrastive_loss'].item(), self.global_step)
                self.swanlab_logger.add_scalar('train/learning_rate', self._get_current_lr(), self.global_step)
            
            # 日志（只在主进程）
            if batch_idx % self.config.system.log_interval == 0:
                self._log(
                    f'Epoch: {self.current_epoch} | '
                    f'Batch: {batch_idx}/{len(self.train_loader)} | '
                    f'Loss: {distill_loss.item():.4f} | '
                    f'Acc: {100. * correct / total:.2f}%'
                )
        
        # 计算epoch指标
        epoch_loss = total_loss / len(self.train_loader)
        epoch_hard_loss = total_hard_loss / len(self.train_loader)
        epoch_kl_loss = total_kl_loss / len(self.train_loader)
        epoch_contrastive_loss = total_contrastive_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # SwanLab: 记录epoch级别指标
        if self.swanlab_logger:
            self.swanlab_logger.add_scalar('train/epoch_loss', epoch_loss, self.current_epoch)
            self.swanlab_logger.add_scalar('train/epoch_hard_loss', epoch_hard_loss, self.current_epoch)
            self.swanlab_logger.add_scalar('train/epoch_kl_loss', epoch_kl_loss, self.current_epoch)
            self.swanlab_logger.add_scalar('train/epoch_contrastive_loss', epoch_contrastive_loss, self.current_epoch)
            self.swanlab_logger.add_scalar('train/epoch_acc', epoch_acc, self.current_epoch)
            self.swanlab_logger.flush()
        
        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'hard_loss': epoch_hard_loss,
            'kl_loss': epoch_kl_loss,
            'contrastive_loss': epoch_contrastive_loss
        }