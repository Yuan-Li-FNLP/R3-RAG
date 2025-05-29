python main.py \
    --model_api_key "deepseek_api_key" \
    --model_api_url "https://api.deepseek.com/beta" \
    --model_api_name "deepseek-chat" \
    --retriever_host "10.176.52.122" \
    --retriever_port "8002" \
    --input_dir  "/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/split/2wikimultihopqa/train" \
    --output_dir "/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/sample/2wikimultihopqa/train" \
    --worker_num_max 64 \

    # --debug_mode