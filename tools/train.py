import argparse
import torch
import torch.multiprocessing as mp
import random
import numpy as np
from pathlib import Path
import os

from utils.config import load_config
from utils.logger import setup_logger
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    is_main_process
)
from datasets import DataFactory
from models import ModelBuilder
from trainers import Trainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态分类模型训练')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--data-path', type=str, help='数据路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--num-classes', type=int, help='类别数量')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true',
                       help='启用分布式训练')
    parser.add_argument('--world-size', type=int, default=-1,
                       help='分布式训练的总进程数')
    parser.add_argument('--rank', type=int, default=-1,
                       help='当前进程的排名')
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='本地进程排名（由 torchrun 自动设置）')
    parser.add_argument('--dist-url', type=str, default='env://',
                       help='分布式训练的初始化URL')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                       help='分布式训练的后端')
    
    return parser.parse_args()


def main_worker(rank, world_size, args, config):
    """分布式训练的工作进程"""
    # 设置分布式环境
    if args.distributed:
        setup_distributed(rank, world_size, args.dist_backend, args.dist_url)
    
    # 设置随机种子
    set_seed(config.system.seed + rank)
    
    # 创建输出目录
    Path(config.system.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    if args.distributed:
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{config.system.gpu_ids[0]}')
    else:
        device = torch.device('cpu')
    
    # 设置日志
    logger = setup_logger(
        log_file=str(Path(config.system.output_dir) / 'train.log') if is_main_process() else None
    )
    
    if is_main_process():
        logger.info(f'配置: {config}')
        logger.info(f'设备: {device}')
        if args.distributed:
            logger.info(f'分布式训练: 世界大小={world_size}, 当前排名={rank}')
    
    # 创建数据工厂
    data_factory = DataFactory(config)
    
    # 创建数据加载器
    train_loader = data_factory.create_train_loader()
    val_loader = data_factory.create_val_loader() if config.data.val_path else None
    
    if is_main_process():
        logger.info(f'训练样本数: {len(train_loader.dataset)}')
        if val_loader:
            logger.info(f'验证样本数: {len(val_loader.dataset)}')
    
    # 创建模型
    model = ModelBuilder.build_model(config)
    
    if is_main_process():
        logger.info(f'模型结构:\n{model}')
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'总参数数: {total_params:,}')
        logger.info(f'可训练参数数: {trainable_params:,}')
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # 恢复训练
    if config.system.resume:
        trainer.load_checkpoint(config.system.resume)
    
    # 开始训练
    trainer.train()
    
    # 清理分布式环境
    if args.distributed:
        cleanup_distributed()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置（已移除 backbone 相关覆盖）
    if args.data_path:
        config.data.train_path = args.data_path
        if 'val' in args.data_path:   # 简单处理，避免硬编码 replace
            config.data.val_path = args.data_path.replace('train', 'val')
        else:
            config.data.val_path = args.data_path.replace('train', 'val')  # 按需调整
    if args.output_dir:
        config.system.output_dir = args.output_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.learning_rate = args.lr
    if args.num_classes:
        config.classes.num_classes = args.num_classes
    if args.resume:
        config.system.resume = args.resume
    
    # 设置GPU
    if args.gpu:
        config.system.gpu_ids = [int(x) for x in args.gpu.split(',') if x.strip()]
    
    # 检查是否通过 torchrun/torch.distributed.launch 启动
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
    
    # 设置分布式训练
    if args.distributed:
        config.system.distributed = True
        
        if args.world_size == -1:
            if 'WORLD_SIZE' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
            else:
                world_size = torch.cuda.device_count()
        else:
            world_size = args.world_size
        
        if args.rank == -1:
            if 'RANK' in os.environ:
                rank = int(os.environ['RANK'])
            else:
                rank = 0
        else:
            rank = args.rank
        
        if not torch.cuda.is_available():
            raise RuntimeError('分布式训练需要 CUDA 支持')
        
        if world_size > torch.cuda.device_count():
            raise RuntimeError(f'请求的进程数 ({world_size}) 超过可用 GPU 数量 ({torch.cuda.device_count()})')
        
        if 'RANK' in os.environ:
            main_worker(rank, world_size, args, config)
        else:
            mp.spawn(
                main_worker,
                args=(world_size, args, config),
                nprocs=world_size,
                join=True
            )
    else:
        # 单卡训练
        config.system.distributed = False
        main_worker(0, 1, args, config)


if __name__ == '__main__':
    main()