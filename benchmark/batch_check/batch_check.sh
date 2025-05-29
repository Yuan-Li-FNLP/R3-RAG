#!/bin/bash

# 定义参数，用户可以自行修改这些值
INPUT_FILE="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/benchmark/NaiveRAG/metrics/llama-3.1-8b-instruct_2wikimultihopqa_17_bm25/records.jsonl"             # 输入JSON文件路径：要求是一个list
OUTPUT_FILE="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/benchmark/NaiveGeneration/metrics/llama-3.1-8b-instruct_2wikimultihopqa"                          # 输出JSON文件路径：要求是一个list
QUESTION_KEY="question"                            # 问题的键名：要求是一个str
GOLDEN_ANSWERS_KEY="golden_answers"                # 标准答案的键名:要求是一个list，元素个数可以为1
ANSWER_KEY="answer"                                # 需要检查的答案的键名：要求是一个str
CHECK_MODEL="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/Qwen/Qwen2.5-72B-Instruct"        # 模型的本地存储路径：用于检查模型，请使用qwen 72B

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 启动Python脚本并传递参数
python batch_check.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --question_key "$QUESTION_KEY" \
  --golden_answers_key "$GOLDEN_ANSWERS_KEY" \
  --answer_key "$ANSWER_KEY" \
  --check_model "$CHECK_MODEL"
