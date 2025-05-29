#!/bin/bash

model_names=(
    "dir_name"
)

# Paths configuration
split_path="check_model_name"
num_search_one_attempt=5

# Process each model in sequence
for model_name in "${model_names[@]}"; do
    echo "========================================"
    echo "Processing model: ${model_name}"
    echo "========================================"
    
    # Create log directory
    log_dir="metrics/${model_name}"
    mkdir -p "${log_dir}"
    
    # Run the command
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/cal_metric.py \
        --model_path=${split_path} \
        --log_dir=${log_dir} \
        --num_search_one_attempt=${num_search_one_attempt} \
        > ${log_dir}/run.log 2>&1
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed processing for ${model_name}"
    else
        echo "Error processing ${model_name}. Check log at ${log_dir}/run.log"
        # Uncomment the line below if you want to stop on error
        # exit 1
    fi
    
    echo "Finished processing ${model_name}"
    echo ""
done

echo "All models have been processed."