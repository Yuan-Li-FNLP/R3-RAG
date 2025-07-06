#!/bin/bash

set -e  # 启用错误立即退出

# 自动识别显卡数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)

# 设置 CUDA_VISIBLE_DEVICES 和 tensor-parallel-size
if [ $gpu_count -gt 0 ]; then
    # 生成从0到gpu_count-1的显卡ID，以逗号分隔
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    tensor_parallel_size=$gpu_count  # 假设 tensor-parallel-size = GPU 数量
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "tensor-parallel-size: $tensor_parallel_size"
else
    echo "没有检测到显卡"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES FORCE_TORCHRUN=1 llamafactory-cli train examples/llama3_full_sft_llama.yaml
