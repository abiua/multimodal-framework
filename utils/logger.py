import os
import sys
import logging
from typing import Optional


def setup_logger(
    name: str = "multimodal",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """设置日志记录器"""
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(fmt)
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger