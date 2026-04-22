import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from .registry import ModelZoo
from .heads import ClassifierHead, MultimodalFusion

class StageableBackbone(nn.Module):
    num_stages = 4

    def init_state(self, **inputs):
        """把原始输入转成后续逐 stage 传播的 state"""
        raise NotImplementedError

    def forward_stage(self, state, stage_idx: int):
        """只执行一个 stage，返回新的 state"""
        raise NotImplementedError

    def forward_head(self, state):
        """stage 全跑完后，做 pool/proj，输出最终特征向量"""
        raise NotImplementedError

    def forward(self, **inputs):
        state = self.init_state(**inputs)
        for stage_idx in range(self.num_stages):
            state = self.forward_stage(state, stage_idx)
        return self.forward_head(state)
    
class StageFusionAdapter(nn.Module):
    """
    对每个模态当前 stage 的 state 先做全局汇聚 -> 投到公共空间 -> 融合 ->
    再投回各模态当前通道数，以 residual 方式回注。
    """
    def __init__(self, stage_dims, common_dim=128, mode="mean", dropout=0.0):
        super().__init__()
        if not stage_dims:
            raise ValueError("stage_dims 不能为空")

        self.stage_dims = dict(stage_dims)
        self.common_dim = common_dim
        self.mode = mode

        self.to_common = nn.ModuleDict({
            modality: nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, common_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for modality, dim in self.stage_dims.items()
        })

        self.back_to_modality = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(common_dim, dim),
                nn.Tanh(),
            )
            for modality, dim in self.stage_dims.items()
        })

        self.gates = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(common_dim, dim),
                nn.Sigmoid(),
            )
            for modality, dim in self.stage_dims.items()
        })

    def _summarize_state(self, state):
        """
        返回:
            summary: [B, C]
            layout:  当前 state 的布局标记
        支持:
            - Tensor [B, C]
            - Tensor [B, C, T]
            - Tensor [B, C, H, W]
            - Dict state, 其中 state["x"] 为 [B, L, C]（给后续 stageable text 预留）
        """
        if isinstance(state, dict):
            if "x" not in state:
                raise TypeError("dict state 必须包含键 'x'")
            x = state["x"]
            if x.dim() != 3:
                raise TypeError(f"暂不支持的 dict state 形状: {tuple(x.shape)}")
            return x.mean(dim=1), "blc"

        if not isinstance(state, torch.Tensor):
            raise TypeError(f"不支持的 state 类型: {type(state)}")

        if state.dim() == 2:
            return state, "bc"
        if state.dim() == 3:
            # 当前 wave/tcn 是 [B, C, T]
            return state.mean(dim=-1), "bct"
        if state.dim() == 4:
            return state.mean(dim=(-2, -1)), "bchw"

        raise TypeError(f"不支持的 state 形状: {tuple(state.shape)}")

    def _inject(self, state, delta, gate, layout):
        mod = delta * gate

        if layout == "bc":
            return state + mod

        if layout == "bct":
            return state + mod.unsqueeze(-1)

        if layout == "bchw":
            return state + mod.unsqueeze(-1).unsqueeze(-1)

        if layout == "blc":
            out = dict(state)
            out["x"] = state["x"] + mod.unsqueeze(1)
            return out

        raise ValueError(f"未知 layout: {layout}")

    def forward(self, states):
        summaries = {}
        layouts = {}

        for modality, state in states.items():
            summary, layout = self._summarize_state(state)
            summaries[modality] = summary
            layouts[modality] = layout

        projected = []
        for modality, summary in summaries.items():
            projected.append(self.to_common[modality](summary))

        stacked = torch.stack(projected, dim=0)

        if self.mode == "sum":
            fused = stacked.sum(dim=0)
        else:
            fused = stacked.mean(dim=0)

        out = {}
        for modality, state in states.items():
            delta = self.back_to_modality[modality](fused)
            gate = self.gates[modality](fused)
            out[modality] = self._inject(state, delta, gate, layouts[modality])

        return out

class MultimodalClassifier(nn.Module):
    """多模态分类模型(重构版)"""
    
    def __init__(self, config, backbones, classifier_head, fusion=None,
                 use_staged_forward=False, fusion_stages=(1, 2)):
        super().__init__()
        self.config = config
        self.backbones = backbones
        self.classifier_head = classifier_head
        self.fusion = fusion

        self.use_staged_forward = use_staged_forward
        self.fusion_stages = set(fusion_stages) if fusion_stages is not None else set()

        self.num_stages = min(
            getattr(backbone, "num_stages", 4) for backbone in self.backbones.values()
        ) if len(self.backbones) > 0 else 4

        self.stage_fusions = nn.ModuleDict()
        if self.use_staged_forward and self.fusion_stages and len(self.backbones) > 1:
            common_dim = getattr(getattr(config, "model", None), "stage_fusion_common_dim", 128)
            mode = getattr(getattr(config, "model", None), "stage_fusion_mode", "mean")
            dropout = getattr(getattr(config, "model", None), "dropout_rate", 0.0)

            for stage_idx in sorted(self.fusion_stages):
                stage_dims = self._collect_stage_dims(stage_idx)
                self.stage_fusions[str(stage_idx)] = StageFusionAdapter(
                    stage_dims=stage_dims,
                    common_dim=common_dim,
                    mode=mode,
                    dropout=dropout,
                )

    def _collect_stage_dims(self, stage_idx: int):
        stage_dims = {}
        for modality, backbone in self.backbones.items():
            dims = getattr(backbone, "stage_dims", None)
            if dims is None:
                raise ValueError(
                    f"backbone '{modality}' 未定义 stage_dims，无法启用中期 stage fusion"
                )
            if stage_idx >= len(dims):
                raise ValueError(
                    f"backbone '{modality}' 的 stage_dims 长度不足，无法访问 stage {stage_idx}"
                )
            stage_dims[modality] = dims[stage_idx]
        return stage_dims
    
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """前向传播"""
        if self.use_staged_forward:
            features = self.extract_features_staged(batch)
        else:
            features = self.extract_features(batch)
        return self.classifier_head(features)
    
    def extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """提取特征统一入口"""
        features = []

        # print("DEBUG registered backbones:", list(self.backbones.keys()))
        # if isinstance(batch, dict):
        #     print("DEBUG batch keys:", list(batch.keys()))
        # else:
        #     print("DEBUG batch type:", type(batch))
        
        # 遍历已注册的backbone
        for modality, backbone in self.backbones.items():
            if modality in batch:
                # 假设 batch[modality] 是一个字典，包含该模态所需的所有参数
                modality_inputs = batch[modality]
                # if hasattr(modality_inputs, "shape"):
                #     print(f"DEBUG {modality} input shape: {modality_inputs.shape}")
                # else:
                #     print(f"DEBUG {modality} input type: {type(modality_inputs)}")
                if isinstance(modality_inputs, torch.Tensor):
                    # 如果是张量，假设backbone默认输入参数名为x
                    feat = backbone(x=modality_inputs)
                else:
                    # 动态解包传参，彻底消灭 if-elif
                    feat = backbone(**modality_inputs)
                features.append(feat)

        if not features:
            raise ValueError("输入批次中没有找到与已注册 backbone 匹配的模态数据")

        if len(features) == 1:
            return features[0]
            
        if self.fusion:
            return self.fusion(features)
            
        return torch.cat(features, dim=1)
    
    def extract_features_staged(self, batch):
        states = {}

        # init
        for modality, backbone in self.backbones.items():
            if modality not in batch:
                continue
            inputs = batch[modality]
            if isinstance(inputs, torch.Tensor):
                states[modality] = backbone.init_state(x=inputs)
            else:
                states[modality] = backbone.init_state(**inputs)

        if not states:
            raise ValueError("输入批次中没有找到可用模态数据")

        # stage loop
        for stage_idx in range(self.num_stages):
            for modality, backbone in self.backbones.items():
                if modality not in states:
                    continue
                states[modality] = backbone.forward_stage(states[modality], stage_idx)

            if stage_idx in self.fusion_stages:
                states = self.stage_fusions[str(stage_idx)](states)

        # head
        feats = []
        for modality, backbone in self.backbones.items():
            if modality not in states:
                continue
            feats.append(backbone.forward_head(states[modality]))

        if len(feats) == 1:
            return feats[0]
        if self.fusion:
            return self.fusion(feats)
        return torch.cat(feats, dim=1)


class ModelBuilder:
    """模型构建器(重构版)"""
    
    @staticmethod
    def build_backbone(
        backbone_type: str,
        feature_dim: int,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """构建backbone - 直接从ModelZoo获取"""
        return ModelZoo.create(
            backbone_type,
            feature_dim=feature_dim,
            pretrained=pretrained,
            dropout=dropout_rate,
            **(extra_params or {})
        )
    
    @staticmethod
    def build_classifier_head(
        in_features: int,
        num_classes: int,
        dropout_rate: float = 0.1,
        hidden_dims: Optional[List[int]] = None
    ) -> nn.Module:
        """构建分类头"""
        return ClassifierHead(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            hidden_dims=hidden_dims
        )
    
    @staticmethod
    def build_fusion(
        feature_dims: List[int],
        output_dim: int,
        fusion_type: str = 'concat',
        dropout_rate: float = 0.1
    ) -> nn.Module:
        """构建融合模块"""
        return MultimodalFusion(
            feature_dims=feature_dims,
            output_dim=output_dim,
            fusion_type=fusion_type,
            dropout_rate=dropout_rate
        )
    
    @classmethod
    def build_model(cls, config) -> MultimodalClassifier:
        if not hasattr(config.model, 'backbones') or not config.model.backbones:
            raise ValueError("config.model.backbones 字段不能为空")
        """统一构建逻辑（单/多模态完全一致）"""
        backbones = nn.ModuleDict()
        feature_dims = []

        for modality in config.data.modalities:
            # 直接从 backbones[modality] 获取配置（必须存在）
            if modality not in config.model.backbones:
                raise ValueError(f"模态 {modality} 在 config.model.backbones 中未定义")

            cfg = config.model.backbones[modality]

            backbone_type = getattr(cfg, 'type', 'resnet18')

            backbone = cls.build_backbone(
                backbone_type=backbone_type,
                feature_dim=cfg.feature_dim,
                pretrained=getattr(cfg, 'pretrained', True),
                dropout_rate=config.model.dropout_rate,
                extra_params=getattr(cfg, 'extra_params', {})
            )

            # 冻结参数
            if getattr(cfg, 'freeze', False):
                for param in backbone.parameters():
                    param.requires_grad = False

            backbones[modality] = backbone
            feature_dims.append(getattr(backbone, 'feature_dim', cfg.feature_dim))

        # 决定是否需要 fusion
        fusion = None
        output_dim = feature_dims[0]

        if len(feature_dims) > 1:
            fusion = cls.build_fusion(
                feature_dims=feature_dims,
                output_dim=config.model.fusion_hidden_dim,
                fusion_type=config.model.fusion_type,
                dropout_rate=config.model.dropout_rate
            )
            output_dim = config.model.fusion_hidden_dim
        # 单模态时 fusion 保持 None，output_dim 就是单个 backbone 的维度

        classifier_head = cls.build_classifier_head(
            in_features=output_dim,
            num_classes=config.classes.num_classes,
            dropout_rate=config.model.dropout_rate,
            hidden_dims=config.model.classifier_hidden_dims
        )

        return MultimodalClassifier(
            config=config,
            backbones=backbones,
            classifier_head=classifier_head,
            fusion=fusion,
            use_staged_forward=getattr(config.model, "use_staged_forward", False),
            fusion_stages=getattr(config.model, "fusion_stages", ()),
        )