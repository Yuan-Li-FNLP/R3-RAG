#!/bin/bash

# 自动识别显卡数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
# 设置 CUDA_VISIBLE_DEVICES 和 tensor-parallel-size
if [ $gpu_count -gt 0 ]; then
    # 生成从0到gpu_count-1的显卡ID，以逗号分隔
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "没有检测到显卡"
    exit 1
fi

# 自动识别 host
host=$(hostname -I | awk '{print $1}')
echo "host: $host"

# CUDA_VISIBLE_DEVICES=0,1 uvicorn retriever_service:app --host $host --port 8001
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uvicorn retriever_service:app --host $host --port 8001