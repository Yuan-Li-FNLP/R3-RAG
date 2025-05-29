#!/bin/bash
if command -v nvidia-smi &>/dev/null; then
    gpu_count=$(nvidia-smi -L | wc -l)
    echo "GPU List:"
    nvidia-smi -L
    echo ""
    echo "INFO: There are ${gpu_count} GPUs."
else
    echo "No nvidia-smi Commond"
    # exit 1  
fi

# 获取IP并保存到变量
IP=$(hostname -I | awk '{print $1}')
if [ -z "$IP" ]; then
    echo "Error: Failed to get IP"
    exit 2
else
    echo "INFO: Get IP:${IP}"
fi
