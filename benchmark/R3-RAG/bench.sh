split_path="check_model_name"

model_name="dir_name"
num_search_one_attempt=5
log_dir=metrics/${model_name}
mkdir -p ${log_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/cal_metric.py \
    --model_path=${split_path} \
    --log_dir=${log_dir} \
    --num_search_one_attempt=${num_search_one_attempt} \
    > ${log_dir}/run.log 2>&1