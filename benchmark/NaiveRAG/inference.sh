# reminder:
# 1. remember to change the model path and dataset path
# 2. remember to change the model name!!! please use model name and dataset name
# 3. use 8 gpus.(vllm tensor_parallel_size=8)
# 4. QWEN-7b need set 4gpus
retrieve_host="10.244.69.142"
retrieve_port="8001"

# model_path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/meta-llama/Llama-3.1-8B-Instruct'
dev_dataset_path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/2wikimultihopqa/dev.jsonl'
model_name='llama_instruct_2wiki'

log_dir=metrics/${model_name}
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/inference.py \
    --model_path=${model_path} \
    --log_dir=${log_dir} \
    --dev_dataset_path=${dev_dataset_path} \
    --retrieve_url=http://${retrieve_host}:${retrieve_port}/search \
    --tensor_parallel_size 8 \
    --num_of_docs 17 \
    > ${log_dir}/run.log 2>&1