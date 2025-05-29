from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flashrag.retriever import DenseRetriever
from flashrag.config import Config
import uvicorn
from typing import List, Any

# 定义请求体模型
class QueryRequest(BaseModel):
    query: str
app = FastAPI(title="Dense Retriever Service", version="1.0")
# 使用全局变量存储retriever实例
dense_retriever: DenseRetriever = None

# 初始化检索器
@app.on_event("startup")
def load_retriever():
    global dense_retriever
    if dense_retriever is None:
        print("Loading Retriever...")
        retrieval_config = {
                'retrieval_method': 'e5',
                'retrieval_model_path': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/intfloat/e5-base-v2',
                'retrieval_query_max_length': 256,
                'retrieval_use_fp16': True,
                'retrieval_topk': 5,
                'retrieval_batch_size': 32,
                'index_path': "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/index/e5_Flat.index",
                'corpus_path': "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/processed_corpus.jsonl",
                'save_retrieval_cache': False,
                'use_retrieval_cache': False,
                'retrieval_cache_path': None,
                'use_reranker': False,
                'faiss_gpu': True,
                'use_sentence_transformer': False,
                'retrieval_pooling_method': 'mean',
                "instruction": None,
            }
        dense_retriever = DenseRetriever(retrieval_config)
        print("Retriever loaded successfully.")

@app.post("/search")
def search(query_request: QueryRequest):
    query = query_request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        retrieval_results = dense_retriever.search(query)
        return retrieval_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
