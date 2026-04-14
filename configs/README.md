# 配置文件使用说明

本框架支持多种模态的配置，包括图像、文本、音频和视频。以下是如何配置不同模态的示例：

## 1. 图像模态配置

```yaml
data:
  modalities:
    - image
  
  loaders:
    image:
      type: image_loader
      extra_params:
        # 可选参数
        resize: [224, 224]  # [height, width]
        normalize: true

model:
  backbones:
    image:
      type: resnet18  # 可选: resnet18, resnet50, efficientnet_b0, mobilenet_v2
      pretrained: true
      feature_dim: 512
      freeze: false
```

## 2. 文本模态配置

```yaml
data:
  modalities:
    - text
  
  loaders:
    text:
      type: text_loader
      extra_params:
        max_length: 128
        vocab_size: 30000

model:
  backbones:
    text:
      type: textlstm  # 可选: textlstm, textgru, textcnn
      feature_dim: 512
      freeze: false
      extra_params:
        vocab_size: 30000
        embed_dim: 256
        num_layers: 2
```

## 3. 音频模态配置

```yaml
data:
  modalities:
    - audio
  
  loaders:
    audio:
      type: audio_loader
      extra_params:
        sample_rate: 16000
        n_mels: 128

model:
  backbones:
    audio:
      type: audiocnn  # 可选: audiocnn, audiocnn_deep
      feature_dim: 512
      freeze: false
      extra_params:
        n_mels: 128
```

## 4. 视频模态配置

```yaml
data:
  modalities:
    - video
  
  loaders:
    video:
      type: video_loader
      extra_params:
        num_frames: 16
        frame_size: [224, 224]

model:
  backbones:
    video:
      type: videocnn  # 可选: videocnn, videocnn3d
      feature_dim: 512
      freeze: false
```

## 5. 多模态配置示例

```yaml
data:
  modalities:
    - image
    - text
  
  loaders:
    image:
      type: image_loader
      extra_params: {}
    
    text:
      type: text_loader
      extra_params:
        max_length: 128
        vocab_size: 30000

model:
  backbones:
    image:
      type: resnet18
      pretrained: true
      feature_dim: 512
      freeze: false
    
    text:
      type: textlstm
      feature_dim: 512
      freeze: false
      extra_params:
        vocab_size: 30000
        embed_dim: 256
        num_layers: 2
  
  fusion_type: concat  # 可选: concat, add, attention
  fusion_hidden_dim: 512
```

## 6. 模型配置参数说明

- `type`: 模态类型，如resnet18、textlstm等
- `pretrained`: 是否使用预训练模型
- `feature_dim`: 特征维度
- `freeze`: 是否冻结参数
- `extra_params`: 额外参数，根据不同模型类型有所不同

## 7. 融合配置参数说明

- `fusion_type`: 融合方式，可选concat、add、attention
- `fusion_hidden_dim`: 融合后的隐藏层维度

## 8. 其他配置参数

- `batch_size`: 批处理大小
- `num_workers`: 数据加载器的工作线程数
- `pin_memory`: 是否将数据加载到GPU内存中
- `train_path`: 训练数据路径
- `val_path`: 验证数据路径
- `test_path`: 测试数据路径