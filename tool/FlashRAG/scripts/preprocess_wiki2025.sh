CUDA_VISIBLE_DEVICES=0,1 python preprocess_wiki.py --dump_path /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/enwiki/enwiki-20250220-pages-articles.xml.bz2  \
                        --save_path /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/enwiki/20250220/test_sample.jsonl \
                        --chunk_by sentence \
                        --chunk_size 512 \
                        --num_workers 32