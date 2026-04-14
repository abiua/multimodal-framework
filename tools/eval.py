import argparse
import torch
from pathlib import Path

from utils.config import load_config
from utils.logger import setup_logger
from datasets import DataFactory
from models import ModelBuilder
from evaluators import Evaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态分类模型评估')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data-path', type=str, help='测试数据路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.data_path:
        config.data.test_path = args.data_path
    if args.output_dir:
        config.system.output_dir = args.output_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    # 设置GPU
    if args.gpu:
        config.system.gpu_ids = [int(x) for x in args.gpu.split(',')]
    
    # 设置设备
    device = torch.device(f'cuda:{config.system.gpu_ids[0]}' 
                         if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    Path(config.system.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        log_file=str(Path(config.system.output_dir) / 'eval.log')
    )
    
    logger.info(f'配置: {config}')
    logger.info(f'设备: {device}')
    
    # 创建数据工厂
    data_factory = DataFactory(config)
    
    # 创建测试数据加载器
    test_loader = data_factory.create_test_loader()
    
    logger.info(f'测试样本数: {len(test_loader.dataset)}')
    
    # 创建模型
    model = ModelBuilder.build_model(config)
    model = model.to(device)
    
    logger.info(f'模型结构:\n{model}')
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        config=config,
        test_loader=test_loader,
        device=device
    )
    
    # 加载模型
    evaluator.load_model(args.checkpoint)
    
    # 运行评估
    metrics = evaluator.run()
    
    logger.info(f'评估结果: {metrics}')


if __name__ == '__main__':
    main()