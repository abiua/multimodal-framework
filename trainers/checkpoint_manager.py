"""检查点管理器"""
import os
import torch
import torch.distributed as dist
from typing import Dict, Any


def _is_main_process() -> bool:
    """检查是否为主进程"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _get_rank() -> int:
    """获取进程排名"""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_acc: float,
        global_step: int,
        lr_scheduler=None,
        **kwargs
    ):
        """保存检查点（只在主进程）"""
        if not _is_main_process():
            return
        
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'global_step': global_step,
            **kwargs
        }
        
        if lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        
        save_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lr_scheduler=None
    ) -> Dict[str, Any]:
        """加载检查点"""
        map_location = device if not dist.is_initialized() else f'cuda:{_get_rank() % torch.cuda.device_count()}'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        model_state.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        results = {
            'epoch': checkpoint['epoch'],
            'best_val_acc': checkpoint['best_val_acc'],
            'global_step': checkpoint['global_step']
        }
        
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        return results
