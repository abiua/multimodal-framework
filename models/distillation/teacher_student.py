import torch
import torch.nn as nn
from models.builder import ModelBuilder


class TeacherStudentWrapper(nn.Module):
    """教师-学生模型包装器，用于知识蒸馏"""
    
    def __init__(self, teacher_config, student_config):
        """
        初始化教师-学生模型包装器
        
        Args:
            teacher_config: 教师配置 (三模态)
            student_config: 学生配置 (双模态)
        """
        super().__init__()
        self.teacher_config = teacher_config
        self.student_config = student_config
        
        # 构建教师模型
        self.teacher = ModelBuilder.build_model(teacher_config)
        self.teacher.eval()
        
        # 冻结教师模型参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 构建学生模型
        self.student = ModelBuilder.build_model(student_config)
    
    def forward(self, batch, mode='student'):
        """
        前向传播
        
        Args:
            batch: 输入批次数据
            mode: 模式，'student' 或 'teacher' 或 'both'
            
        Returns:
            根据模式返回对应的 logits 输出
        """
        if mode == 'teacher':
            with torch.no_grad():
                teacher_logits = self.teacher(batch)
            return teacher_logits
        elif mode == 'student':
            student_logits = self.student(batch)
            return student_logits
        elif mode == 'both':
            with torch.no_grad():
                teacher_logits = self.teacher(batch)
            student_logits = self.student(batch)
            return teacher_logits, student_logits
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def get_teacher(self):
        """获取教师模型"""
        return self.teacher
    
    def get_student(self):
        """获取学生模型"""
        return self.student