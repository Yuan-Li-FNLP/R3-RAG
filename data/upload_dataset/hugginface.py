from huggingface_hub import HfApi, upload_file

api = HfApi(token="your_token")
repo_id = "Yuan-Li-FNLP/R3-RAG-ColdStartTrainingData"

# 上传 2wikimultihopqa
upload_file(
    path_or_fileobj="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/data/construct/format_single_query/2wikimultihopqa/train/format.json",
    path_in_repo="2wikimultihopqa.json",
    repo_id=repo_id,
    repo_type="dataset"
)

# 上传 hotpotqa
upload_file(
    path_or_fileobj="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/data/construct/format_single_query/hotpotqa/train_3w/format.json",
    path_in_repo="hotpotqa.json",
    repo_id=repo_id,
    repo_type="dataset"
)

# 上传 musique
upload_file(
    path_or_fileobj="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/data/construct/format_single_query/musique/train/format.json",
    path_in_repo="musique.json",
    repo_id=repo_id,
    repo_type="dataset"
)

