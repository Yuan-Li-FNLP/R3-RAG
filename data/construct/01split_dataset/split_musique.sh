#!/bin/bash

# 设置参数
CHUNK_SIZE=20  # 每个文件的行数
INPUT_FILE="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/datasets/musique/train.jsonl"  # 输入文件路径
OUTPUT_DIR="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/split/musique/train"   # 输出目录路径

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file $INPUT_FILE does not exist"
    exit 1
fi

# 运行Python脚本
python split.py \
    --chunk_size "$CHUNK_SIZE" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR"

# 检查Python脚本的执行状态
if [ $? -eq 0 ]; then
    echo "Split completed successfully"
else
    echo "Split failed"
    exit 1
fi