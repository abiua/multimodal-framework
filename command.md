# 训练

## 单卡训练
CUDA_VISIBLE_DEVICES=0  PYTHONPATH=/home/pythoner/abiu/multimodal-framework torchrun --nproc_per_node=1   --master_port=29501   -m tools.train   --config configs/fish_feeding_unireplknet.yaml

## 多卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3  PYTHONPATH=/home/pythoner/abiu/multimodal-framework torchrun --nproc_per_node=4   --master_port=29501   -m tools.train   --config configs/fish_feeding_unireplknet.yaml

# 验证


