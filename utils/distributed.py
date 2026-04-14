import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Tuple


def is_dist_avail_and_initialized() -> bool:
    """检查分布式训练是否可用且已初始化"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """获取分布式训练的总进程数"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """获取当前进程的排名"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """检查当前进程是否为主进程"""
    return get_rank() == 0


def get_local_rank() -> int:
    """获取本地排名（节点内排名）"""
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = 'nccl',
    init_method: str = 'env://'
) -> None:
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程排名
        world_size: 总进程数
        backend: 通信后端 ('nccl' for GPU, 'gloo' for CPU)
        init_method: 初始化方法
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_distributed() -> None:
    """清理分布式训练环境"""
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def distributed_wrapper(
    fn,
    world_size: int,
    backend: str = 'nccl'
):
    """
    分布式训练包装函数
    
    Args:
        fn: 要执行的函数，接收 rank 和 world_size 作为参数
        world_size: 总进程数
        backend: 通信后端
    """
    mp.spawn(
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    对所有进程的张量求平均值
    
    Args:
        tensor: 要平均的张量
        
    Returns:
        平均后的张量
    """
    if not is_dist_avail_and_initialized():
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    # 克隆张量以避免修改原始数据
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    reduced_tensor /= world_size
    
    return reduced_tensor


def reduce_dict(input_dict: dict) -> dict:
    """
    对字典中的所有值进行规约求和
    
    Args:
        input_dict: 输入字典
        
    Returns:
        规约后的字典
    """
    if not is_dist_avail_and_initialized():
        return input_dict
    
    world_size = get_world_size()
    if world_size == 1:
        return input_dict
    
    # 获取当前设备
    device = get_device_for_rank(get_rank())
    
    # 将字典值转换为张量
    keys = sorted(input_dict.keys())
    values = [input_dict[key] for key in keys]
    
    # 转换为张量并移动到 GPU
    tensor_values = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensor_values.append(value.clone().detach().to(device))
        else:
            tensor_values.append(torch.tensor(value, dtype=torch.float32, device=device))
    
    # 将所有张量拼接在一起
    tensor = torch.stack(tensor_values)
    
    # 规约
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 转换回字典
    reduced_dict = {}
    for i, key in enumerate(keys):
        reduced_dict[key] = tensor[i].item() / world_size
    
    return reduced_dict


def save_on_master(state: dict, filepath: str) -> None:
    """
    只在主进程保存文件
    
    Args:
        state: 要保存的状态字典
        filepath: 文件路径
    """
    if is_main_process():
        torch.save(state, filepath)


def barrier() -> None:
    """同步所有进程"""
    if is_dist_avail_and_initialized():
        dist.barrier()


def get_device_for_rank(rank: int) -> torch.device:
    """
    根据排名获取设备
    
    Args:
        rank: 进程排名
        
    Returns:
        设备对象
    """
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        return torch.device(f'cuda:{device_id}')
    return torch.device('cpu')
