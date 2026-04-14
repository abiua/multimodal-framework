import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ClassifierHead(nn.Module):
    """分类头"""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout_rate: float = 0.1,
        hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()
        
        if hidden_dims:
            layers = []
            prev_dim = in_features
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.classifier(x)


class MultimodalFusion(nn.Module):
    """多模态融合模块"""
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int,
        fusion_type: str = 'concat',
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.feature_dims = feature_dims
        
        if fusion_type == 'concat':
            total_dim = sum(feature_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        elif fusion_type == 'add':
            # 确保所有维度相同
            assert len(set(feature_dims)) == 1, "加法融合要求所有特征维度相同"
            self.fusion = nn.Sequential(
                nn.Linear(feature_dims[0], output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        elif fusion_type == 'attention':
            # 注意力融合
            self.query = nn.Linear(output_dim, output_dim)
            self.key = nn.Linear(output_dim, output_dim)
            self.value = nn.Linear(output_dim, output_dim)
            
            # 将各模态特征投影到相同维度
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in feature_dims
            ])
            
            self.output_projection = nn.Linear(output_dim, output_dim)
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        if self.fusion_type == 'concat':
            fused = torch.cat(features, dim=1)
            return self.fusion(fused)
        
        elif self.fusion_type == 'add':
            # 投影到相同维度并相加
            projected = [proj(feat) for proj, feat in zip(self.projections, features)]
            fused = torch.stack(projected, dim=0).sum(dim=0)
            return self.fusion(fused)
        
        elif self.fusion_type == 'attention':
            # 投影各模态特征
            projected = [proj(feat) for proj, feat in zip(self.projections, features)]
            
            # 堆叠特征 [num_modalities, batch_size, output_dim]
            stacked = torch.stack(projected, dim=0)
            
            # 计算注意力
            query = self.query(stacked.mean(dim=0, keepdim=True))  # [1, batch, output_dim]
            key = self.key(stacked)  # [num_modalities, batch, output_dim]
            value = self.value(stacked)  # [num_modalities, batch, output_dim]
            
            # 注意力分数
            attention_scores = torch.bmm(
                query.transpose(0, 1),
                key.transpose(0, 1).transpose(1, 2)
            )  # [batch, 1, num_modalities]
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # 加权融合
            fused = torch.bmm(
                attention_weights,
                value.transpose(0, 1)
            ).squeeze(1)  # [batch, output_dim]
            
            return self.output_projection(fused)