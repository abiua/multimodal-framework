from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from omegaconf import OmegaConf, MISSING, DictConfig


DEFAULT_MODALITY_LOADERS = {
    "image": "image_loader",
    "text": "text_loader",
    "audio": "audio_loader",
    "video": "video_loader",
}

ALLOWED_FUSION_TYPES = {"concat", "add", "attention"}
ALLOWED_STAGE_FUSION_MODES = {"identity", "mean", "sum"}
ALLOWED_MID_FUSION_TYPES_V2 = {"concat", "add", "attention", "none"}


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
    """单个模态的 backbone 配置"""
    type: str = "resnet18"
    pretrained: bool = True
    feature_dim: int = 512
    freeze: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """模型配置"""

    # 统一的 backbone 配置入口（单/多模态一致）
    backbones: Dict[str, BackboneConfig] = field(default_factory=dict)

    # 最终 feature-level fusion（多模态时可选）
    fusion_type: str = "concat"
    fusion_hidden_dim: int = 512
    mid_fusion_enabled: bool = True

    # 分类头配置
    dropout_rate: float = 0.1
    classifier_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [256])

    # staged / mid-fusion 路径
    use_staged_forward: bool = False
    fusion_stages: List[int] = field(default_factory=list)  # 0-based
    stage_fusion_common_dim: int = 128
    stage_fusion_mode: str = "identity"

    # 新统一流水线（为 None 时走旧路径）
    unified_pipeline: Optional[UnifiedPipelineConfig] = None


@dataclass
class InteractionBlockConfig:
    """单个 InteractionBlock 配置"""
    transform_type: str = "transformer"
    transform_kwargs: Dict[str, Any] = field(default_factory=dict)
    fusion_type: str = "none"
    fusion_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionConfig:
    """Decision 模块配置"""
    type: str = "identity"
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PEResolutionConfig:
    """Position Encoding 按模态配置"""
    enabled: bool = True
    max_len: int = 256


@dataclass
class UnifiedPipelineConfig:
    """统一多模态流水线配置（新）"""
    token_dim: int = 256
    interaction_blocks: List[InteractionBlockConfig] = field(default_factory=list)
    position_encodings: Dict[str, PEResolutionConfig] = field(default_factory=dict)
    mid_fusion_type: str = "attention"
    mid_fusion_output_dim: int = 256
    mid_fusion_enabled: bool = True
    decision: DecisionConfig = field(default_factory=DecisionConfig)


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


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数，当前值: {value}")


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数，当前值: {value}")


def _require_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} 必须在 [0, 1] 范围内，当前值: {value}")


def validate_config(cfg: DictConfig) -> None:
    """运行时配置校验"""

    # ---------------------------
    # data
    # ---------------------------
    if not cfg.data.modalities:
        raise ValueError("data.modalities 不能为空")

    if len(set(cfg.data.modalities)) != len(cfg.data.modalities):
        raise ValueError(f"data.modalities 不能有重复项，当前值: {list(cfg.data.modalities)}")

    _require_positive_int("data.batch_size", int(cfg.data.batch_size))
    _require_non_negative_int("data.num_workers", int(cfg.data.num_workers))
    _require_positive_int("data.image_size", int(cfg.data.image_size))

    for split_name in ("train_path", "val_path", "test_path"):
        split_value = getattr(cfg.data, split_name)
        if split_value in (None, ""):
            raise ValueError(f"data.{split_name} 不能为空")

    # ---------------------------
    # classes
    # ---------------------------
    _require_positive_int("classes.num_classes", int(cfg.classes.num_classes))

    if cfg.classes.class_names:
        if len(cfg.classes.class_names) != int(cfg.classes.num_classes):
            raise ValueError(
                f"class_names 数量({len(cfg.classes.class_names)}) "
                f"与 num_classes({cfg.classes.num_classes}) 不一致"
            )

    if cfg.classes.class_weights is not None:
        if len(cfg.classes.class_weights) != int(cfg.classes.num_classes):
            raise ValueError(
                f"class_weights 数量({len(cfg.classes.class_weights)}) "
                f"与 num_classes({cfg.classes.num_classes}) 不一致"
            )

    # ---------------------------
    # model.backbones
    # ---------------------------
    if not cfg.model.backbones:
        raise ValueError("model.backbones 不能为空")

    for modality in cfg.data.modalities:
        if modality not in cfg.model.backbones:
            raise ValueError(f"模态 '{modality}' 在 model.backbones 中未定义")

        backbone_cfg = cfg.model.backbones[modality]

        if not getattr(backbone_cfg, "type", ""):
            raise ValueError(f"模态 '{modality}' 的 backbone.type 不能为空")

        _require_positive_int(
            f"model.backbones.{modality}.feature_dim",
            int(backbone_cfg.feature_dim),
        )

        if modality not in DEFAULT_MODALITY_LOADERS and modality not in cfg.data.loaders:
            raise ValueError(
                f"模态 '{modality}' 没有默认 loader，请在 data.loaders.{modality} 中显式指定"
            )

        if modality in cfg.data.loaders:
            loader_cfg = cfg.data.loaders[modality]
            if not loader_cfg.type:
                raise ValueError(f"data.loaders.{modality}.type 不能为空")

    # ---------------------------
    # model.fusion / head
    # ---------------------------
    if cfg.model.fusion_type not in ALLOWED_FUSION_TYPES:
        raise ValueError(
            f"model.fusion_type 必须是 {sorted(ALLOWED_FUSION_TYPES)} 之一，"
            f"当前值: {cfg.model.fusion_type}"
        )

    _require_positive_int("model.fusion_hidden_dim", int(cfg.model.fusion_hidden_dim))
    _require_probability("model.dropout_rate", float(cfg.model.dropout_rate))

    if cfg.model.classifier_hidden_dims is not None:
        for i, hidden_dim in enumerate(cfg.model.classifier_hidden_dims):
            _require_positive_int(f"model.classifier_hidden_dims[{i}]", int(hidden_dim))

    # ---------------------------
    # staged / mid-fusion
    # ---------------------------
    if cfg.model.use_staged_forward:
        if cfg.model.stage_fusion_mode not in ALLOWED_STAGE_FUSION_MODES:
            raise ValueError(
                f"model.stage_fusion_mode 必须是 {sorted(ALLOWED_STAGE_FUSION_MODES)} 之一，"
                f"当前值: {cfg.model.stage_fusion_mode}"
            )

        _require_positive_int(
            "model.stage_fusion_common_dim",
            int(cfg.model.stage_fusion_common_dim),
        )

        fusion_stages = list(cfg.model.fusion_stages or [])
        for i, stage_idx in enumerate(fusion_stages):
            if not isinstance(stage_idx, int):
                raise ValueError(f"model.fusion_stages[{i}] 必须是整数，当前值: {stage_idx}")
            if stage_idx < 0:
                raise ValueError(f"model.fusion_stages[{i}] 不能为负数，当前值: {stage_idx}")

        if len(set(fusion_stages)) != len(fusion_stages):
            raise ValueError(f"model.fusion_stages 不能有重复项，当前值: {fusion_stages}")
    else:
        if cfg.model.fusion_stages:
            raise ValueError(
                "当 model.use_staged_forward=False 时，model.fusion_stages 应为空"
            )

    # ---------------------------
    # unified pipeline v2
    # ---------------------------
    if cfg.model.unified_pipeline is not None:
        pipe = cfg.model.unified_pipeline
        _require_positive_int("unified_pipeline.token_dim", int(pipe.token_dim))
        _require_positive_int("unified_pipeline.mid_fusion_output_dim", int(pipe.mid_fusion_output_dim))

        if pipe.mid_fusion_type not in ALLOWED_MID_FUSION_TYPES_V2:
            raise ValueError(
                f"unified_pipeline.mid_fusion_type 必须是 {sorted(ALLOWED_MID_FUSION_TYPES_V2)} 之一"
            )

        from models.fusion.registry import FusionRegistry as _FR
        for i, blk in enumerate(pipe.interaction_blocks):
            if blk.fusion_type not in _FR.list_all():
                raise ValueError(
                    f"interaction_blocks[{i}].fusion_type '{blk.fusion_type}' 无效，"
                    f"可用: {_FR.list_all()}"
                )

    # ---------------------------
    # train
    # ---------------------------
    _require_positive_int("train.epochs", int(cfg.train.epochs))
    if float(cfg.train.learning_rate) <= 0:
        raise ValueError(f"train.learning_rate 必须大于 0，当前值: {cfg.train.learning_rate}")
    if float(cfg.train.weight_decay) < 0:
        raise ValueError(f"train.weight_decay 不能小于 0，当前值: {cfg.train.weight_decay}")
    _require_non_negative_int("train.warmup_epochs", int(cfg.train.warmup_epochs))
    _require_positive_int("train.val_interval", int(cfg.train.val_interval))

    _require_probability("train.label_smoothing", float(cfg.train.label_smoothing))
    if float(cfg.train.mixup_alpha) < 0:
        raise ValueError(f"train.mixup_alpha 不能小于 0，当前值: {cfg.train.mixup_alpha}")
    if float(cfg.train.cutmix_alpha) < 0:
        raise ValueError(f"train.cutmix_alpha 不能小于 0，当前值: {cfg.train.cutmix_alpha}")

    if cfg.train.early_stop.enabled:
        _require_positive_int("train.early_stop.patience", int(cfg.train.early_stop.patience))
        if float(cfg.train.early_stop.min_delta) < 0:
            raise ValueError(
                f"train.early_stop.min_delta 不能小于 0，当前值: {cfg.train.early_stop.min_delta}"
            )
        if cfg.train.early_stop.mode not in {"min", "max"}:
            raise ValueError(
                f"train.early_stop.mode 必须是 ['min', 'max'] 之一，当前值: {cfg.train.early_stop.mode}"
            )

    # ---------------------------
    # system
    # ---------------------------
    _require_positive_int("system.log_interval", int(cfg.system.log_interval))
    _require_positive_int("system.save_interval", int(cfg.system.save_interval))

    if not cfg.system.output_dir:
        raise ValueError("system.output_dir 不能为空")
    if not cfg.system.experiment_name:
        raise ValueError("system.experiment_name 不能为空")


def load_config(config_path: Optional[str] = None, **kwargs) -> DictConfig:
    """
    加载配置：
    1. 先构建结构化 schema
    2. 再 merge 用户 YAML
    3. 最后 merge 运行时 kwargs override
    """

    schema = OmegaConf.structured(Config)

    if config_path is None:
        cfg = schema
    else:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(schema, user_cfg)

    if kwargs:
        override = OmegaConf.create(kwargs)
        cfg = OmegaConf.merge(cfg, override)

    validate_config(cfg)
    return cfg