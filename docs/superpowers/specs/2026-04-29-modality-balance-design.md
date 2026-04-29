# Modality Balance: 防止强模态压制弱模态

> 基于单模态消融实验结论设计。核心发现：Audio在5Hz单模态达87.95%，但multimodal仅84.34%（image拖累）；10Hz下融合有效(93.33%)。

## 方案

### P0: 模态Dropout + 辅助分类头

**模态Dropout:**
- 训练时以概率p随机屏蔽一个模态（设p=0.2 per modality）
- 屏蔽方式：将该模态token置零或跳过该模态的forward
- 效果：强制audio/wave独立学习，防止image梯度主导

**辅助分类头:**
- 每个模态的pooled token后接独立分类头
- 单模态loss + 融合loss联合训练
- Loss = fusion_loss + λ * (img_loss + aud_loss + wav_loss) / 3
- λ=0.3 初始，可调
- 效果：确保每个modal encoder学到判别特征

### P1: 样本级模态门控（后续）

在mid-fusion前加learnable gate:
- gate = softmax(MLP([pooled_img, pooled_aud, pooled_wav]))
- fused = sum(gate_i * pooled_i)
- 温度退火防止门控退化

### P2: Wave手工特征（后续）

- 加速度模长、角速度模长
- 信号jerk（三阶差分）
- 频带能量（FFT后分3频带）
- 作为TCN的额外输入通道
