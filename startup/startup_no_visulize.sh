#!/bin/bash

# 自动识别 host
host=$(hostname -I | awk '{print $1}')
echo "host: $host"

CUDA_VISIBLE_DEVICES=0 python RRAG.py \
    --model_path /remote-home1/yli/Workspace/R3RAG/train/Models/SFT/llama \
    --stop_token_ids 128009 \
    --num_passages_one_retrieval 3 \
    --num_passages_one_split_retrieval 5 \
    --max_num_passages 80 \
    --num_search_one_attempt 10 \
    --api_try_counter 3 \
    --retriver_url "http://10.176.58.103:8001/search" \
    --openai_model_name "/remote-home1/yli/Model/Generator/Qwen2.5/14B/instruct" \
    --api_url "http://10.176.58.103:8002" \
    --openai_suffix "/v1" \
    --api_key "EMPTY" 
