#!/bin/bash

retriever_port=8001
vllm_port=8002

# 获取主机IP
host=$(hostname -I | awk '{print $1}')
echo "host: $host"

# 创建主session
tmux new-session -d -s service

# 创建retriever窗口
tmux new-window -t service:1 -n retriever
tmux send-keys -t service:retriever "
while true; do
    echo '启动retriever服务...'
    CUDA_VISIBLE_DEVICES=0,1 uvicorn retriever_service:app --host $host --port $retriever_port
    echo 'retriever服务退出，3秒后重启...'
    sleep 3
done
" Enter

# 创建vllm窗口
tmux new-window -t service:2 -n vllm
tmux send-keys -t service:vllm "
while true; do
    echo '启动vllm服务...'
    CUDA_VISIBLE_DEVICES=2 vllm serve /remote-home1/yli/Model/Generator/Qwen2.5/14B/instruct \\
        --gpu-memory-utilization 0.9 \\
        --tensor-parallel-size 1 \\
        --port $vllm_port \\
        --device cuda \\
        --dtype float16 \\
        --host $host
    echo 'vllm服务退出，3秒后重启...'
    sleep 3
done
" Enter

# 返回主窗口
tmux select-window -t service:0

# 在主窗口显示监控信息
tmux send-keys -t service:0 "
echo '服务管理面板 - 按 Ctrl+C 退出所有服务'
echo '窗口列表：'
echo '  0: 主控制台 (当前)'
echo '  1: retriever服务 (端口: $retriever_port)'
echo '  2: vllm服务 (端口: $vllm_port)'
echo '主机IP: $host'
echo ''
echo '切换窗口: Ctrl+b 然后按数字键'
echo '查看日志: 切换到对应窗口'
echo ''
trap 'tmux kill-session -t service; exit' INT
while true; do
    echo -n '.'
    sleep 5
done
" Enter

# 附加到session
tmux attach-session -t service