跨模态对比蒸馏实现 Todolist
1. ✅ 创建目录结构
逻辑思路：
- 在 models/ 下创建 distillation/ 子目录
- 创建 __init__.py 导出核心类
- 目录结构：
    models/distillation/
  ├── __init__.py
  ├── contrastive_loss.py
  └── teacher_student.py
  
---
2. ✅ 实现对比蒸馏损失函数 (contrastive_loss.py)
逻辑思路：
- 创建 ContrastiveDistillationLoss 类，继承 nn.Module
- __init__ 参数：
  - temperature: 温度参数τ (默认 0.07)，控制分布平滑度
  - alpha: 硬标签损失和蒸馏损失的权重平衡 (默认 0.5)
  - contrastive_weight: 对比损失权重 (默认 0.3)
- forward 方法输入：student_logits, teacher_logits, hard_labels
- 计算三部分损失：
  1. 硬标签损失: F.cross_entropy(student_logits, hard_labels)
  2. KL 散度蒸馏损失: 
     - 对教师 logits 做温度缩放：softmax(teacher_logits / T)
     - 计算 KL 散度：KL(student_soft || teacher_soft)
  3. InfoNCE 对比损失:
     - 归一化学生和教师 logits
     - 计算余弦相似度矩阵
     - 使用交叉熵损失让学生靠近教师
- 返回：总损失 + 各分项损失字典
---
3. ✅ 实现教师 - 学生模型包装器 (teacher_student.py)
逻辑思路：
- 创建 TeacherStudentWrapper 类，继承 nn.Module
- __init__ 参数：
  - teacher_config: 教师配置 (三模态)
  - student_config: 学生配置 (双模态)
- 构建教师模型：
  - 使用 ModelBuilder.build_model(teacher_config)
  - 设置 teacher.eval() 并冻结参数 (requires_grad=False)
- 构建学生模型：
  - 使用 ModelBuilder.build_model(student_config)
  - 保持可训练状态
- forward 方法：
  - 支持分别调用教师和学生模型
  - 返回对应的 logits 输出
- 辅助方法：
  - get_teacher(): 获取教师模型
  - get_student(): 获取学生模型
---
4. ✅ 实现蒸馏训练器 (distillation_trainer.py)
逻辑思路：
- 创建 DistillationTrainer 类，继承 Trainer
- __init__ 新增参数：
  - teacher_model: 预训练的教师模型
  - config.distillation: 蒸馏超参数配置
- 冻结教师模型：
    for param in teacher_model.parameters():
      param.requires_grad = False
  teacher_model.eval()
  - 重写 train_one_epoch() 方法：
  1. 学生正常训练模式
  2. 教师固定为评估模式
  3. 对每个 batch：
     - 教师前向传播 (无梯度)
     - 学生前向传播
     - 计算蒸馏损失
     - 反向传播更新学生参数
  4. 记录损失字典到 TensorBoard
- 保留原有的 validate() 方法 (只评估学生模型)
---
5. ✅ 创建蒸馏配置文件 (cross_modal.yaml)
逻辑思路：
- 文件路径：configs/distillation/cross_modal.yaml
- 配置结构：
    # 教师配置 (三模态)
  teacher:
    data:
      modalities: [image, audio, wave]
      batch_size: 32
    model:
      multimodal: true
      fusion_type: attention
      backbones:
        image: {type: resnet50, pretrained: true, feature_dim: 2048}
        audio: {type: audioresnet, pretrained: true, feature_dim: 512}
        wave: {type: wavecnn, pretrained: true, feature_dim: 512}
  
  # 学生配置 (双模态)
  student:
    data:
      modalities: [audio, wave]
      batch_size: 32
    model:
      multimodal: true
      fusion_type: concat
      backbones:
        audio: {type: audiocnn, pretrained: false, feature_dim: 512}
        wave: {type: wavecnn_small, pretrained: false, feature_dim: 256}
  
  # 蒸馏超参数
  distillation:
    temperature: 4.0      # 温度参数
    alpha: 0.7            # 蒸馏损失权重
    contrastive_weight: 0.3  # 对比损失权重
  
  # 训练配置
  train:
    epochs: 100
    learning_rate: 0.001
    optimizer: adamw
  
---
6. ✅ 创建蒸馏训练脚本 (train_distill.py)
逻辑思路：
- 文件路径：tools/train_distill.py
- 主函数流程：
  1. 解析命令行参数 (config, teacher_checkpoint 等)
  2. 加载配置文件
  3. 创建数据加载器 (使用学生配置)
  4. 构建教师 - 学生包装器
  5. 加载教师预训练权重：
          teacher_checkpoint = torch.load(teacher_path)
     wrapper.teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
       6. 创建蒸馏训练器
  7. 调用 trainer.train()
  8. 保存最佳学生模型
---
7. ✅ 测试蒸馏训练流程
逻辑思路：
- 运行训练脚本：
    python -m tools.train_distill --config configs/distillation/cross_modal.yaml
  - 检查点：
  1. 教师模型是否正确加载并冻结
  2. 学生模型是否正常训练
  3. 蒸馏损失是否下降
  4. TensorBoard 日志是否正常记录
  5. 显存使用是否合理
---
8. ✅ 评估学生模型性能
逻辑思路：
- 运行评估脚本：
    python -m tools.eval --config configs/distillation/cross_modal.yaml \
    --checkpoint output/distill_student/best_model.pth
  - 对比指标：
  - 教师模型准确率 (三模态)
  - 学生模型准确率 (双模态，蒸馏后)
  - 基线学生模型准确率 (无蒸馏)
- 预期结果：
  - 蒸馏后的学生 > 无蒸馏的学生
  - 接近或达到教师性能
---
关键技术点总结
模块
温度缩放
KL 散度
InfoNCE
总损失
祝你实现顺利！有任何问题随时问我。