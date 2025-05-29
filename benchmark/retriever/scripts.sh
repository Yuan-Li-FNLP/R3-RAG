retriever_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/intfloat/e5-base-v2"
retriever_bge_path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/BAAI/bge-large-en-v1.5'
index_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/index/e5_Flat.index"
index_path_bm25="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/index_bm25/bm25"
index_path_bge='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/index_bge_large/bge_Flat.index'
corpus_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/processed_corpus.jsonl"
retrieve_host="10.244.151.169"
retrieve_port="8001"

wiki_index_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/enwiki/2018/wiki18_100w_e5.index"
wiki_corpus_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/enwiki/2018/wiki18_100w.jsonl"

script=$1
if [ "$script" == "retriever" ]; then
    CUDA_VISIBLE_DEVICES=0,1 python src/retrive_server.py \
        --host=${retrieve_host} --port=${retrieve_port} \
        --model_path=${retriever_path} \
        --index_path=${index_path} \
        --corpus_path=${corpus_path}
elif [ "$script" == "retrieverbm25" ]; then
    python src/retrive_server_bm25.py \
        --host=${retrieve_host} --port=${retrieve_port} \
        --index_path=${index_path_bm25} \
        --corpus_path=${corpus_path}
elif [ "$script" == "retrieverbge" ]; then
    CUDA_VISIBLE_DEVICES=0,1 python src/retrive_server_bge.py \
        --host=${retrieve_host} --port=${retrieve_port} \
        --model_path=${retriever_bge_path} \
        --index_path=${index_path_bge} \
        --corpus_path=${corpus_path}
elif [ "$script" == "retrieverwiki" ]; then
    CUDA_VISIBLE_DEVICES=0,1 python src/retrive_server.py \
        --host=${retrieve_host} --port=${retrieve_port} \
        --model_path=${retriever_path} \
        --index_path=${wiki_index_path} \
        --corpus_path=${wiki_corpus_path}
else
    echo "Invalid script name. Please choose from 'retriever', 'split', or 'eval'."
fi
