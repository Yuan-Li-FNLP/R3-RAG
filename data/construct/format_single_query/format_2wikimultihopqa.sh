#!/bin/bash

# 输入文件夹路径
INPUT_FILE="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/merge/2wikimultihopqa/train/merge.jsonl"
# 输出文件夹路径，必须是json文件
OUTPUT_FILE="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/format/2wikimultihopqa/train/format.json"

# 格式化为标准的llama factory json
python3 format.py "${INPUT_FILE}" "${OUTPUT_FILE}"