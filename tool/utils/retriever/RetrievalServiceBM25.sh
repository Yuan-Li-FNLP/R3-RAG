#!/bin/bash

# 自动识别 host
host=$(hostname -I | awk '{print $1}')
echo "host: $host"

uvicorn retriever_service_bm25:app --host $host --port 8001
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uvicorn retriever_service:app --host $host --port 8001