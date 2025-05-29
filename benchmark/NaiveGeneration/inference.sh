# reminder:
# 1. remember to change the model path and dataset path
# 2. remember to change the model name!!! please use model name and dataset name
# 3. use 8 gpus.(vllm tensor_parallel_size=8)
model_path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/meta-llama/Llama-3.1-8B-Instruct'
# '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/Qwen/Qwen2.5-7B-Instruct'
dev_dataset_path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/2wikimultihopqa/dev.jsonl'
# '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/hotpotqa/dev.jsonl'
# '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/musique/dev.jsonl'
model_name='llama_instruct_2wiki'
tp_size=8

log_dir=metrics/${model_name}
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/inference.py \
    --model_path=${model_path} \
    --log_dir=${log_dir} \
    --dev_dataset_path=${dev_dataset_path} \
    --tensor_parallel_size=${tp_size} \
    > ${log_dir}/run.log 2>&1