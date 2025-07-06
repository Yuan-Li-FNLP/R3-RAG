CUDA_VISIBLE_DEVICES=0,1 python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/processed_corpus.jsonl \
    --save_dir /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/index_bm25 \
    --bm25_backend bm25s
