#!/bin/bash
input_file="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/datasets/hotpotqa/train.jsonl"
output_file="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/format_openrlhf/hotpotqa/train.jsonl"

# 提取 output_file 所在目录
record_dir=$(dirname "$output_file")
# 检查 record_dir 是否存在，不存在则创建
if [ ! -d "$record_dir" ]; then
    echo "目录不存在，正在创建：$record_dir"
    mkdir -p "$record_dir"  # 使用 -p 选项创建多级目录
else
    echo "目录已存在：$record_dir"
fi

python3 format.py "${input_file}" "${output_file}"