#!/bin/bash

# 模型路径设置
split_path="check_mode_path"
split_host=""
split_port=""

# 四个模型的地址
llama_sft_model_path="llama_sft_model_path"
llama_rl_model_path="llama_rl_model_path"
qwen_sft_model_path="qwen_sft_model_path"
qwen_rl_model_path="qwen_rl_model_path"

# 各模型对应的stop_token_id和tp值
llama_stop_token_id=128009
qwen_stop_token_id=151645
llama_tp=8
qwen_tp=4

# 检索器服务的端口号（统一为一个）及其IP地址
retrieve_port="8001"
retrieve_host_e5="retrieve_host_e5"
retrieve_host_bge="retrieve_host_bge"
retrieve_host_bm25="retrieve_host_bm25"

# 通用参数
num_search_one_attempt=5

# 创建一个数组来保存所有模型名称
model_names=()

# 执行函数
run_experiment() {
    local model_type=$1  # llama 或 qwen
    local model_variant=$2  # sft 或 rl
    local retriever_type=$3  # e5 或 bge 或 bm25
    local num_docs=$4  # 文档数量(1-10)
    
    # 设置模型路径和参数
    if [ "$model_type" == "llama" ] && [ "$model_variant" == "sft" ]; then
        model_path=$llama_sft_model_path
        stop_token=$llama_stop_token_id
        tp=$llama_tp
    elif [ "$model_type" == "llama" ] && [ "$model_variant" == "rl" ]; then
        model_path=$llama_rl_model_path
        stop_token=$llama_stop_token_id
        tp=$llama_tp
    elif [ "$model_type" == "qwen" ] && [ "$model_variant" == "sft" ]; then
        model_path=$qwen_sft_model_path
        stop_token=$qwen_stop_token_id
        tp=$qwen_tp
    elif [ "$model_type" == "qwen" ] && [ "$model_variant" == "rl" ]; then
        model_path=$qwen_rl_model_path
        stop_token=$qwen_stop_token_id
        tp=$qwen_tp
    fi
    
    # 设置检索主机
    if [ "$retriever_type" == "e5" ]; then
        retrieve_host=$retrieve_host_e5
    elif [ "$retriever_type" == "bge" ]; then
        retrieve_host=$retrieve_host_bge
    else
        retrieve_host=$retrieve_host_bm25
    fi
    
    # 设置模型名称
    model_name="${model_variant}_${model_type}_base_${retriever_type}_docs_${num_docs}_valish"

    # 将模型名称添加到数组
    model_names+=("$model_name")

    # 创建日志目录
    log_dir="metrics/${model_name}"
    mkdir -p ${log_dir}
    
    echo "Running experiment: $model_name with ${retriever_type}, num_docs=${num_docs} on $retrieve_host"
    
    # 根据模型类型选择GPU配置
    if [ "$model_type" == "llama" ]; then
        # llama需要8张卡
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/inference_new.py \
            --model_path=${model_path} \
            --log_dir=${log_dir} \
            --retrieve_url=http://${retrieve_host}:${retrieve_port}/search \
            --num_search_one_attempt=${num_search_one_attempt} \
            --stop_token_id $stop_token \
            --split_url=http://${split_host}:${split_port}/split_query \
            --num_of_docs $num_docs \
            --tp $tp
    else
        # qwen只需要4张卡
        CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference_new.py \
            --model_path=${model_path} \
            --log_dir=${log_dir} \
            --retrieve_url=http://${retrieve_host}:${retrieve_port}/search \
            --num_search_one_attempt=${num_search_one_attempt} \
            --stop_token_id $stop_token \
            --split_url=http://${split_host}:${split_port}/split_query \
            --num_of_docs $num_docs \
            --tp $tp
    fi
        
    echo "Finished experiment: $model_name"
    echo "-------------------------------------"
}

# 计算指标函数
calculate_metrics() {
    local model_name=$1
    local log_dir="metrics/${model_name}"
    
    echo "Calculating metrics for $model_name"
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/cal_metric.py \
        --model_path=${split_path} \
        --log_dir=${log_dir} \
        --num_search_one_attempt=${num_search_one_attempt} \
        > ${log_dir}/run.log 2>&1
        
    echo "Metrics calculation completed for $model_name"
    echo "-------------------------------------"
}

# 主执行流程
echo "Starting all experiments..."

# 记录所有模型名称到文件
model_names_file="all_model_names.txt"
> $model_names_file  # 清空文件内容

# 先执行所有实验
echo "Running all experiments..."
for model_type in "llama" "qwen"; do
    for model_variant in "sft" "rl"; do
        for retriever_type in "e5" "bge" "bm25"; do
            for num_docs in {1..10}; do
                run_experiment "$model_type" "$model_variant" "$retriever_type" "$num_docs"
                echo "${model_variant}_${model_type}_base_${retriever_type}_docs_${num_docs}_valish" >> $model_names_file
            done
        done
    done
done

echo "All experiments completed!"

echo "Starting metrics calculation for all models..."
for model_name in "${model_names[@]}"; do
    calculate_metrics "$model_name"
done

echo "All metrics calculations completed!"
echo "All model names have been saved to $model_names_file"