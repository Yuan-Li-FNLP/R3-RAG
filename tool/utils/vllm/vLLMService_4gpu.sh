#!/bin/bash

# 自动识别显卡数量
# gpu_count=$(nvidia-smi --list-gpus | wc -l)
gpu_count=4
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

# 自动识别 host
host=$(hostname -I | awk '{print $1}')
echo "host: $host"

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/Qwen/Qwen2.5-14B-Instruct  \
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/Qwen/Qwen2.5-14B-Instruct  \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size $tensor_parallel_size \
    --port 8002 \
    --device cuda \
    --dtype float16 \
    --host $host