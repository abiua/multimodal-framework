import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveDistillationLoss(nn.Module):
    """对比蒸馏损失函数，结合硬标签损失、KL 散度蒸馏损失和 InfoNCE 对比损失"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.5,
        contrastive_weight: float = 0.3,
    ):
        """
        初始化对比蒸馏损失
        
        Args:
            temperature: 温度参数τ，控制分布平滑度 (默认 0.07)
            alpha: 硬标签损失和蒸馏损失的权重平衡 (默认 0.5)
            contrastive_weight: 对比损失权重 (默认 0.3)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.contrastive_weight = contrastive_weight
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        计算对比蒸馏损失
        
        Args:
            student_logits: 学生模型的 logits，形状 [batch_size, num_classes]
            teacher_logits: 教师模型的 logits，形状 [batch_size, num_classes]
            hard_labels: 真实标签，形状 [batch_size]
        
        Returns:
            total_loss: 总损失
            losses: 包含各分项损失的字典
        """
        batch_size = student_logits.size(0)
        device = student_logits.device
        
        # 1. 硬标签损失
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        
        # 2. KL 散度蒸馏损失
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        
        # 3. InfoNCE 对比损失
        # 归一化学生和教师 logits
        student_norm = F.normalize(student_logits, p=2, dim=1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=1)
        
        # 计算余弦相似度矩阵 [batch_size, batch_size]
        # student_norm: [batch, num_classes], teacher_norm: [batch, num_classes]
        similarity_matrix = student_norm @ teacher_norm.T / self.temperature
        
        # 创建标签：对角线元素为正样本对
        labels = torch.arange(batch_size, device=device)
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        # 总损失
        distillation_loss = (1 - self.alpha) * hard_loss + self.alpha * kl_loss * (self.temperature ** 2)
        total_loss = distillation_loss + self.contrastive_weight * contrastive_loss
        
        losses = {
            'hard_loss': hard_loss,
            'kl_loss': kl_loss,
            'contrastive_loss': contrastive_loss,
            'distillation_loss': distillation_loss,
        }
        
        return total_loss, losses
