#!/bin/bash

# 输入文件夹路径
INPUT_FILE="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/merge/hotpotqa/train_3w/merge.jsonl"
# 输出文件夹路径，必须是json文件
OUTPUT_FILE="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/format/hotpotqa/train_3w/format.json"

# 格式化为标准的llama factory json
python3 format.py "${INPUT_FILE}" "${OUTPUT_FILE}"