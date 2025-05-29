split_path="check_model_name"
retrieve_host="retrieve_host"
retrieve_port="8001"
split_host=""
split_port=""

llama_stop_token_id=128009
num_docs=3
tp=8
model_path="R3-RAG_model"
model_name="the_name_of_the_inference_case"
num_search_one_attempt=5
log_dir=metrics/${model_name}
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/inference_new.py \
    --model_path=${model_path} \
    --log_dir=${log_dir} \
    --retrieve_url=http://${retrieve_host}:${retrieve_port}/search \
    --num_search_one_attempt=${num_search_one_attempt} \
    --stop_token_id $llama_stop_token_id \
    --num_of_docs $num_docs \
    --tp $tp \
    --split_url=http://${split_host}:${split_port}/split_query 