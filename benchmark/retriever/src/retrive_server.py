from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
from flashrag.retriever import DenseRetriever
import uvicorn
from typing import List, Any

class QueryRequest(BaseModel):
    querys: list
app = FastAPI(title="Dense Retriever Service", version="1.0")

def get_args():
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--corpus_path', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print("Loading Retriever...")
    retrieval_config = {
            'retrieval_method': 'e5',
            'retrieval_model_path': args.model_path,
            'retrieval_query_max_length': 256,
            'retrieval_use_fp16': True,
            'retrieval_topk': 17,
            'retrieval_batch_size': 32,
            'index_path': args.index_path,
            'corpus_path': args.corpus_path,
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
    async def search(query_request: QueryRequest):
        querys = query_request.querys
        if not querys:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        try:
            retrieval_results = dense_retriever.batch_search(querys)
            return retrieval_results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    uvicorn.run(app,host=args.host, port=args.port)