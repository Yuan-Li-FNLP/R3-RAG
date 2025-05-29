from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
from flashrag.retriever import BM25Retriever
import uvicorn
from typing import List, Dict, Any
import numpy as np
import json

class QueryRequest(BaseModel):
    querys: list

app = FastAPI(title="BM25 Retriever Service", version="1.0")

def get_args():
    parser = argparse.ArgumentParser(description='BM25 Retriever Service')
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--corpus_path', type=str, required=True)
    args = parser.parse_args()
    return args

def convert_numpy_to_lists(array):
    """将NumPy数组转换为嵌套列表结构"""
    if isinstance(array, np.ndarray):
        # 先转为Python列表
        result = array.tolist()
        return result
    return array

@app.post("/search")
async def search(query_request: QueryRequest):
    querys = query_request.querys
    if not querys:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        # 调用检索器获取结果
        raw_results = dense_retriever.batch_search(querys)
        print(f"Raw results type: {type(raw_results)}")
        
        # 将NumPy数组转为Python列表
        results_list = convert_numpy_to_lists(raw_results)
        
        # # 进一步处理确保结果可序列化
        # # 如果结果包含文档对象，转换为简单字典
        # final_results = []
        # for batch in results_list:
        #     batch_results = []
        #     for item in batch:
        #         if isinstance(item, dict):
        #             # 仅保留可序列化的键值对
        #             simple_item = {}
        #             for k, v in item.items():
        #                 if isinstance(k, str):  # 确保键是字符串
        #                     if isinstance(v, (str, int, float, bool, type(None))):
        #                         simple_item[k] = v
        #                     else:
        #                         simple_item[k] = str(v)  # 将复杂值转为字符串
        #             batch_results.append(simple_item)
        #         else:
        #             # 非字典类型转换为简单字典
        #             batch_results.append({"content": str(item)})
        #     final_results.append(batch_results)
        
        # return final_results
        
        return results_list
            
    except Exception as e:
        import traceback
        error_detail = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    args = get_args()
    print("Loading Retriever...")
    retrieval_config = {
        'retrieval_method': 'bm25',
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
        "bm25_backend": 'bm25s',
    }
    dense_retriever = BM25Retriever(retrieval_config)
    print("Retriever loaded successfully.")
    
    uvicorn.run(app, host=args.host, port=args.port)