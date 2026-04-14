# 多模态分类训练框架

一个灵活、可扩展的多模态分类训练框架，支持图像、文本、音频、视频等多种模态的分类任务。

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [配置详解](#配置详解)
- [使用手册](#使用手册)
- [扩展开发](#扩展开发)
- [示例项目](#示例项目)

---

## 项目简介

本框架旨在提供一个统一的多模态分类训练平台，用户可以通过简单的配置文件定义数据、模型和训练参数，无需编写大量代码即可完成多模态分类任务的训练和评估。

框架采用模块化设计，支持：
- **多模态融合**：图像、文本、音频、视频等模态的自由组合
- **灵活配置**：通过 YAML 配置文件控制所有参数
- **可扩展性**：支持自定义模型和数据加载器
- **分布式训练**：支持单机多卡分布式训练
- **混合精度训练**：支持 FP16 混合精度加速

---

## 核心特性

- ✅ **多模态支持**：图像、文本、音频、视频
- ✅ **独立配置**：每个模态可独立配置 backbone 和数据加载器
- ✅ **融合策略**：拼接 (concat)、加法 (add)、注意力 (attention)
- ✅ **混合精度**：支持 FP16 混合精度训练
- ✅ **参数冻结**：支持冻结指定模态的 backbone 参数
- ✅ **Early Stop**：支持早停机制，防止过拟合
- ✅ **学习率调度**：支持 cosine、step、plateau 等调度策略
- ✅ **分布式训练**：支持单机多卡 DDP 训练
- ✅ **可扩展**：ModelZoo 和 Loader 注册系统

---

## 项目结构

```
multimodal-framework/
├── configs/                    # 配置文件目录
│   ├── default.yaml           # 默认配置
│   └── fish_feeding.yaml      # 鱼类喂食强度示例配置
├── data/                      # 数据目录
├── datasets/                  # 数据集模块
│   ├── __init__.py
│   ├── registry.py            # 数据加载器注册系统
│   ├── factory.py             # 数据工厂
│   └── loaders/               # 各模态数据加载器
│       ├── image_loaders.py   # 图像加载器
│       ├── text_loaders.py    # 文本加载器
│       ├── audio_loaders.py   # 音频加载器
│       └── video_loaders.py   # 视频加载器
├── models/                    # 模型模块
│   ├── __init__.py
│   ├── registry.py            # ModelZoo 注册系统
│   ├── builder.py             # 模型构建器
│   ├── heads/                 # 分类头和融合模块
│   │   └── classifier.py
│   └── modelzoo/              # 所有 backbone 模型
│       ├── image_models.py    # 图像模型
│       ├── text_models.py     # 文本模型
│       └── audio_models.py    # 音频模型
├── trainers/                  # 训练器模块
│   └── trainer.py
├── evaluators/                # 评估器模块
│   └── evaluator.py
├── utils/                     # 工具模块
│   ├── config.py              # 配置系统
│   ├── logger.py              # 日志
│   └── metrics.py             # 评估指标
├── tools/                     # 工具脚本
│   ├── train.py               # 训练脚本
│   └── eval.py                # 评估脚本
├── output/                    # 输出目录
├── scripts/                   # 辅助脚本
├── requirements.txt           # 依赖列表
└── README.md                  # 本文档
```

---

## 环境要求

- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA >= 11.0 (可选，用于 GPU 训练)

### 依赖包

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
pyyaml>=5.4.0
Pillow>=8.0.0
tqdm>=4.60.0
librosa>=0.8.0  # 可选，用于音频处理
```

---

## 安装指南

### 1. 克隆项目

```bash
git clone <repository-url>
cd multimodal-framework
```

### 2. 创建虚拟环境（推荐）

```bash
conda create -n multimodal python=3.8
conda activate multimodal
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 PyTorch（根据 CUDA 版本）

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 快速开始

### 1. 准备数据

按照以下结构组织数据：

```
data/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   ├── image1.txt          # 文本（可选）
│   │   └── audio/image1.wav    # 音频（可选）
│   ├── class_1/
│   └── ...
├── val/
└── test/
```

### 2. 配置类别

编辑配置文件 `configs/default.yaml`：

```yaml
classes:
  num_classes: 5
  class_names:
    - cat
    - dog
    - bird
    - fish
    - rabbit
```

### 3. 配置模态

```yaml
data:
  modalities:
    - image
    - text
```

### 4. 开始训练

```bash
python -m tools.train --config configs/default.yaml
```

### 5. 评估模型

```bash
python -m tools.eval --config configs/default.yaml --checkpoint output/best_model.pth
```

---

## 配置详解

### 数据配置 (data)

```yaml
data:
  batch_size: 32              # 批次大小
  num_workers: 4              # 数据加载线程数
  pin_memory: true            # 是否使用锁页内存
  
  train_path: "data/train"    # 训练数据路径
  val_path: "data/val"        # 验证数据路径
  test_path: "data/test"      # 测试数据路径
  
  modalities:                 # 使用的模态列表
    - image
    - text
  
  loaders:                    # 每个模态的数据加载器配置
    image:
      type: image_loader      # 加载器类型
      extra_params: {}        # 额外参数
    
    text:
      type: text_loader
      extra_params:
        max_length: 128       # 最大序列长度
        vocab_size: 30000     # 词汇表大小
  
  image_size: 224             # 图像尺寸
```

### 类别配置 (classes)

```yaml
classes:
  num_classes: 10             # 类别数量
  class_names:                # 类别名称列表
    - class_0
    - class_1
    # ...
  class_weights:              # 类别权重（可选）
    - 1.0
    - 1.0
```

### 模型配置 (model)

```yaml
model:
  backbone: resnet18          # 单模态时的默认 backbone
  pretrained: true            # 是否使用预训练权重
  feature_dim: 512            # 特征维度
  
  backbones:                  # 多模态时每个模态的 backbone 配置
    image:
      type: resnet18
      pretrained: true
      feature_dim: 512
      freeze: false           # 是否冻结参数
      extra_params: {}
    
    text:
      type: textlstm
      pretrained: false
      feature_dim: 512
      freeze: false
      extra_params:
        vocab_size: 30000
        embed_dim: 256
        num_layers: 2
  
  fusion_type: concat         # 融合方式: concat, add, attention
  fusion_hidden_dim: 512      # 融合隐藏层维度
  
  dropout_rate: 0.1           # Dropout 比率
  classifier_hidden_dims:     # 分类头隐藏层维度（可选）
    - 256
    - 128
  
  multimodal: true            # 是否多模态
```

### 训练配置 (train)

```yaml
train:
  epochs: 100                 # 训练轮数
  learning_rate: 0.001        # 学习率
  weight_decay: 0.0001        # 权重衰减
  
  lr_scheduler: cosine        # 学习率调度器: cosine, step, plateau
  warmup_epochs: 5            # 预热轮数
  step_size: 30               # StepLR 的步长
  gamma: 0.1                  # 学习率衰减因子
  
  optimizer: adam              # 优化器: adam, sgd, adamw
  momentum: 0.9               # SGD 动量
  
  label_smoothing: 0.1        # 标签平滑
  mixup_alpha: 0.0            # Mixup alpha
  cutmix_alpha: 0.0           # CutMix alpha
  
  # Early Stop 配置
  early_stop:
    enabled: true             # 是否启用 early-stop
    patience: 10              # 等待轮数
    min_delta: 0.001          # 最小改善阈值
    monitor: accuracy         # 监控指标: accuracy, val_loss
    mode: max                 # 模式: max, min
```

### 系统配置 (system)

```yaml
system:
  seed: 42                    # 随机种子
  gpu_ids:                    # GPU ID 列表
    - 0
  distributed: false          # 是否启用分布式训练
  fp16: false                 # 是否启用混合精度训练
  
  log_interval: 10            # 日志打印间隔（批次）
  save_interval: 10           # 检查点保存间隔（轮数）
  
  output_dir: output          # 输出目录
  resume: ""                  # 恢复训练的检查点路径
  
  dist_backend: nccl          # 分布式后端
  dist_url: env://            # 分布式 URL
```

---

## 使用手册

### 训练命令

```bash
# 基本训练
python -m tools.train --config configs/default.yaml

# 指定 GPU
python -m tools.train --config configs/default.yaml --gpu 0

# 恢复训练
python -m tools.train --config configs/default.yaml --resume output/checkpoint_epoch_50.pth

# 分布式训练（单机多卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config configs/default.yaml

# 打开tensorboard
tensorboard --port=6006 --logdir=output/fish_feeding_image_only_4gpu/tensorboard
```

### 评估命令

```bash
# 评估最佳模型
python -m tools.eval --config configs/default.yaml --checkpoint output/best_model.pth

# 评估指定检查点
python -m tools.eval --config configs/default.yaml --checkpoint output/checkpoint_epoch_50.pth
```

### 输出文件

训练完成后，`output/` 目录下会生成以下文件：

```
output/
├── best_model.pth           # 最佳模型检查点
├── checkpoint_epoch_*.pth   # 定期保存的检查点
├── train.log                # 训练日志
└── config.yaml              # 使用的配置文件
```

### 检查点内容

每个检查点包含以下内容：

```python
checkpoint = {
    'epoch': 当前轮数,
    'model_state_dict': 模型参数,
    'optimizer_state_dict': 优化器状态,
    'best_val_acc': 最佳验证准确率,
    'global_step': 全局步数,
    'lr_scheduler_state_dict': 学习率调度器状态,
    'early_stop_counter': 早停计数器,
    'best_monitored_value': 最佳监控值
}
```

---

## 扩展开发

### 添加自定义 Backbone

1. 在 `models/modelzoo/` 目录下创建新文件：

```python
# models/modelzoo/my_models.py
import torch.nn as nn
from models import register_backbone

@register_backbone('my_backbone', description='我的自定义模型', modality='image')
class MyBackbone(nn.Module):
    def __init__(self, feature_dim=512, pretrained=False, freeze=False, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        # 定义你的模型结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

2. 在 `models/modelzoo/__init__.py` 中导入：

```python
from .my_models import MyBackbone
```

3. 在配置文件中使用：

```yaml
model:
  backbones:
    image:
      type: my_backbone
      feature_dim: 512
```

### 添加自定义数据加载器

1. 在 `datasets/loaders/` 目录下创建新文件：

```python
# datasets/loaders/my_loaders.py
import torch
from datasets import register_loader, BaseLoader
from torchvision import transforms

@register_loader('my_loader', description='我的自定义加载器', modality='image')
class MyLoader(BaseLoader):
    def __init__(self, image_size=224, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
    
    def load(self, path):
        # 实现你的数据加载逻辑
        from PIL import Image
        image = Image.open(path).convert('RGB')
        return image
    
    def get_transform(self, is_training=True):
        if is_training:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
```

2. 在 `datasets/loaders/__init__.py` 中导入：

```python
from .my_loaders import MyLoader
```

3. 在配置文件中使用：

```yaml
data:
  loaders:
    image:
      type: my_loader
      extra_params:
        image_size: 224
```

---

## 示例项目

### 鱼类喂食强度分类

这是一个三模态（图像、音频、波形）分类的示例：

```bash
# 使用鱼类喂食配置训练
python -m tools.train --config configs/fish_feeding.yaml
```

配置特点：
- 使用图像、音频、六轴传感器波形三种模态
- 图像使用 ResNet18 backbone
- 音频和波形使用 AudioCNN backbone
- 使用注意力融合策略
- 启用混合精度训练

---

## 可用组件

### ModelZoo (Backbone)

| 模态 | 模型名称 | 特征维度 | 说明 |
|------|----------|----------|------|
| 图像 | resnet18 | 512 | ResNet-18 |
| 图像 | resnet50 | 2048 | ResNet-50 |
| 图像 | resnet101 | 2048 | ResNet-101 |
| 图像 | efficientnet_b0 | 1280 | EfficientNet-B0 |
| 图像 | mobilenet_v2 | 1280 | MobileNet-V2 |
| 文本 | textlstm | 512 | LSTM 文本模型 |
| 文本 | textgru | 512 | GRU 文本模型 |
| 文本 | textcnn | 512 | CNN 文本模型 |
| 音频 | audiocnn | 512 | CNN 音频模型 |
| 音频 | audiocnn_deep | 512 | 深层 CNN 音频模型 |
| 音频 | audioresnet | 512 | ResNet 音频模型 |

### Loaders (数据加载器)

| 模态 | 加载器名称 | 说明 |
|------|------------|------|
| 图像 | image_loader | 标准图像加载器（含数据增强） |
| 图像 | image_loader_simple | 简单加载器（无增强） |
| 文本 | text_loader | 标准文本加载器 |
| 文本 | text_loader_char | 字符级文本加载器 |
| 音频 | audio_loader | 梅尔频谱图加载器 |
| 音频 | audio_loader_raw | 原始波形加载器 |
| 视频 | video_loader | 视频帧抽取加载器 |
| 视频 | video_loader_3d | 3D CNN 视频加载器 |
| 波形 | wave_loader | 六轴传感器波形加载器 |

### 融合方式

| 类型 | 说明 |
|------|------|
| concat | 特征拼接，将所有模态的特征向量拼接后输入分类器 |
| add | 特征相加，要求所有模态特征维度相同 |
| attention | 注意力融合，使用注意力机制自适应融合多模态特征 |

### 学习率调度器

| 类型 | 说明 |
|------|------|
| cosine | 余弦退火调度 |
| step | 等间隔衰减 |
| plateau | 基于验证指标的自适应衰减 |

---

## 常见问题

### Q: 如何使用单模态训练？

设置 `model.multimodal: false`，并在 `data.modalities` 中只保留一个模态。

### Q: 如何冻结部分模态的参数？

在对应模态的 backbone 配置中设置 `freeze: true`。

### Q: 训练中断后如何继续？

使用 `--resume` 参数指定检查点路径：

```bash
python -m tools.train --config configs/default.yaml --resume output/checkpoint_epoch_50.pth
```

### Q: 如何调整 Early Stop 参数？

在配置文件的 `train.early_stop` 部分调整：

```yaml
train:
  early_stop:
    enabled: true
    patience: 20      # 增加耐心值
    min_delta: 0.0001 # 减小最小改善阈值
    monitor: val_loss # 监控验证损失
    mode: min         # 损失越小越好
```

---

## 许可证

本项目采用 MIT 许可证。
