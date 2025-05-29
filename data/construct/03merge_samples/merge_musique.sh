#!/bin/bash

# 输入文件夹路径
INPUT_DIR="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/sample/musique/train"
# 输出文件夹路径
OUTPUT_DIR="/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/merge/musique/train"

# 合并文件
# 合并后的输出文件路径
OUTPUT_FILE_v1="${OUTPUT_DIR}/merge_tempv1.jsonl"
# 确保输出文件夹存在
mkdir -p $OUTPUT_DIR
# 合并 JSONL 文件
cat ${INPUT_DIR}/*.jsonl > $OUTPUT_FILE_v1
echo "合并完成，输出文件为: ${OUTPUT_FILE_v1}"

# 格式化为标准的jsonl
OUTPUT_FILE_v2="${OUTPUT_DIR}/merge.jsonl"
python3 format.py "$OUTPUT_FILE_v1" "$OUTPUT_FILE_v2"
echo "格式化完成，输出文件为: ${OUTPUT_FILE_v2}"

# statistic：统计数据量和可用数据
OUTPUT_FILE_v3="${OUTPUT_DIR}/log.txt"
python3 statistic.py "${OUTPUT_FILE_v2}" "${OUTPUT_FILE_v3}"
echo "数据统计完成，输出文件为: ${OUTPUT_FILE_v3}"
