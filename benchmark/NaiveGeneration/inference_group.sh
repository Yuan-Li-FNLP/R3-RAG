#!/bin/bash

# 定义模型路径
MODEL_PATHS=(
    "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/meta-llama/Llama-3.1-8B-Instruct"
    "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/Qwen/Qwen2.5-7B-Instruct"
)

# 定义模型名称
MODEL_NAMES=(
    "llama-3.1-8b-instruct"
    "qwen2.5-7b-instruct"
)

# 定义模型的tensor_parallel_size
MODEL_TP_SIZES=(
    8  # Llama model
    4  # Qwen model
)

# 定义数据集路径
DATASET_PATHS=(
    "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/2wikimultihopqa/dev.jsonl"
    "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/hotpotqa/dev.jsonl"
    "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/musique/dev.jsonl"
)

# 定义数据集名称
DATASET_NAMES=(
    "2wikimultihopqa"
    "hotpotqa"
    "musique"
)

# 循环遍历每个模型
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    model_path=${MODEL_PATHS[$i]}
    model_name=${MODEL_NAMES[$i]}
    tp_size=${MODEL_TP_SIZES[$i]}

    # 设置CUDA_VISIBLE_DEVICES
    if [ $tp_size -eq 4 ]; then
        cuda_devices="0,1,2,3"
    else
        cuda_devices="0,1,2,3,4,5,6,7"
    fi

    # 循环遍历每个数据集
    for ((j=0; j<${#DATASET_PATHS[@]}; j++)); do
        dataset_path=${DATASET_PATHS[$j]}
        dataset_name=${DATASET_NAMES[$j]}
        
        # 组合模型名和数据集名
        run_name="${model_name}_${dataset_name}"
        log_dir="metrics/${run_name}"
        
        echo "==============================================="
        echo "Running with model: ${model_name}"
        echo "Dataset: ${dataset_name}"
        echo "Tensor Parallel Size: ${tp_size}"
        echo "CUDA Devices: ${cuda_devices}"
        echo "Log directory: ${log_dir}"
        echo "==============================================="
        
        # 创建日志目录
        mkdir -p ${log_dir}
        
        # 运行命令
        CUDA_VISIBLE_DEVICES=${cuda_devices} python src/inference.py \
            --model_path=${model_path} \
            --log_dir=${log_dir} \
            --dev_dataset_path=${dataset_path} \
            --tensor_parallel_size=${tp_size} \
            > ${log_dir}/run.log 2>&1
        
        # 检查上一个命令是否成功执行
        if [ $? -ne 0 ]; then
            echo "Error: Failed to run model ${model_name} on dataset ${dataset_name}"
            echo "Check logs at ${log_dir}/run.log"
            exit 1
        else
            echo "Successfully completed model ${model_name} on dataset ${dataset_name}"
        fi
    done
done

echo "All runs completed successfully!"