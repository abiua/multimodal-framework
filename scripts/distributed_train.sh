#!/bin/bash

# ============================================================================
# 分布式训练启动脚本
# ============================================================================

# 使用方法:
# 1. 单机多卡训练:
#    ./scripts/distributed_train.sh --config configs/default.yaml --data-path data/train
#
# 2. 指定 GPU 数量:
#    ./scripts/distributed_train.sh --config configs/default.yaml --data-path data/train --nproc 4
#
# 3. 使用 torchrun (推荐):
#    torchrun --nproc_per_node=4 -m tools.train --config configs/default.yaml --distributed

# 默认参数
CONFIG="configs/default.yaml"
DATA_PATH=""
OUTPUT_DIR=""
BATCH_SIZE=""
EPOCHS=""
LR=""
BACKBONE=""
NUM_CLASSES=""
RESUME=""
GPUS="0"
NPROC=1
USE_TORCHRUN=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --num-classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --torchrun)
            USE_TORCHRUN=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查 CUDA 是否可用
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "错误: CUDA 不可用"
    exit 1
fi

# 获取 GPU 数量
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "可用 GPU 数量: $NUM_GPUS"

# 构建命令参数
CMD_ARGS=""

if [ -n "$DATA_PATH" ]; then
    CMD_ARGS="$CMD_ARGS --data-path $DATA_PATH"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD_ARGS="$CMD_ARGS --output-dir $OUTPUT_DIR"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD_ARGS="$CMD_ARGS --batch-size $BATCH_SIZE"
fi

if [ -n "$EPOCHS" ]; then
    CMD_ARGS="$CMD_ARGS --epochs $EPOCHS"
fi

if [ -n "$LR" ]; then
    CMD_ARGS="$CMD_ARGS --lr $LR"
fi

if [ -n "$BACKBONE" ]; then
    CMD_ARGS="$CMD_ARGS --backbone $BACKBONE"
fi

if [ -n "$NUM_CLASSES" ]; then
    CMD_ARGS="$CMD_ARGS --num-classes $NUM_CLASSES"
fi

if [ -n "$RESUME" ]; then
    CMD_ARGS="$CMD_ARGS --resume $RESUME"
fi

# 启动分布式训练
if [ "$USE_TORCHRUN" = true ]; then
    echo "使用 torchrun 启动分布式训练..."
    torchrun \
        --nproc_per_node=$NPROC \
        --master_port=29500 \
        -m tools.train \
        --config $CONFIG \
        --distributed \
        $CMD_ARGS
else
    echo "使用 mp.spawn 启动分布式训练..."
    python -m tools.train \
        --config $CONFIG \
        --distributed \
        --world-size $NPROC \
        --gpu $GPUS \
        $CMD_ARGS
fi