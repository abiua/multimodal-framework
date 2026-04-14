import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from omegaconf import OmegaConf, MISSING, DictConfig


@dataclass
class ModalityLoaderConfig:
    """单个模态的数据加载器配置"""
    type: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """数据配置"""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_path: str = MISSING
    val_path: str = MISSING
    test_path: str = MISSING
    modalities: List[str] = field(default_factory=lambda: ["image"])
    
    loaders: Dict[str, ModalityLoaderConfig] = field(default_factory=dict)
    image_size: int = 224
    augmentations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassConfig:
    """类别配置"""
    num_classes: int = MISSING
    class_names: List[str] = field(default_factory=list)
    class_weights: Optional[List[float]] = None


@dataclass
class BackboneConfig:
    """单个模态的backbone配置"""
    type: str = "resnet18"
    pretrained: bool = True
    feature_dim: int = 512
    freeze: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """模型配置（优化后：只保留 backbones，单/多模态写法完全统一）"""
    backbones: Dict[str, BackboneConfig] = field(default_factory=dict)
    
    # 融合配置（仅多模态时生效）
    fusion_type: str = "concat"
    fusion_hidden_dim: int = 512
    
    # 分类头配置（所有模态共用）
    dropout_rate: float = 0.1
    classifier_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [256])


@dataclass
class EarlyStopConfig:
    """Early Stop 配置"""
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "accuracy"
    mode: str = "max"


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    step_size: int = 30
    gamma: float = 0.1
    optimizer: str = "adam"
    momentum: float = 0.9
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    val_interval: int = 1


@dataclass
class EvalConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    save_predictions: bool = True
    confusion_matrix: bool = True


@dataclass
class SystemConfig:
    """系统配置"""
    seed: int = 42
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    distributed: bool = False
    fp16: bool = False
    log_interval: int = 10
    save_interval: int = 10
    output_dir: str = "output"
    resume: str = ""
    dist_backend: str = "nccl"
    dist_url: str = "env://"
    tensorboard_enabled: bool = True
    experiment_name: str = "default"


@dataclass
class Config:
    """主配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    classes: ClassConfig = field(default_factory=ClassConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def load_config(config_path: Optional[str] = None, **kwargs) -> DictConfig:
    """加载配置的便捷函数（保持你原来的调用方式）"""
    # 1. 构建结构化 schema（带默认值和类型）
    schema = OmegaConf.structured(Config)
    
    if config_path and os.path.exists(config_path):
        # 2. 加载 YAML 并与 schema 合并（自动填充默认值 + 类型校验）
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(schema, user_cfg)
    else:
        cfg = schema
    
    # 3. 命令行 kwargs 覆盖（如果有）
    if kwargs:
        override = OmegaConf.create(kwargs)
        cfg = OmegaConf.merge(cfg, override)
    
    # 4. 可选：运行时校验（推荐加上）
    validate_config(cfg)
    
    return cfg


def validate_config(cfg: DictConfig):
    """简单配置校验"""
    if not cfg.data.modalities:
        raise ValueError("data.modalities 不能为空")
    
    if not cfg.model.backbones:
        raise ValueError("model.backbones 不能为空")
    
    # 检查每个 modality 是否都有对应的 backbone 配置
    for modality in cfg.data.modalities:
        if modality not in cfg.model.backbones:
            raise ValueError(f"模态 '{modality}' 在 model.backbones 中未定义，请检查 YAML 配置")
    
    # 其他你关心的校验可以继续添加