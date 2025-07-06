import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.pipeline import SelfAskPipeline, IRCOTPipeline
from flashrag.prompt import PromptTemplate

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str)
# parser.add_argument("--retriever_path", type=str)
# args = parser.parse_args()
model_path = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/Qwen/Qwen2.5-7B-Instruct"
retriever_path = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Model/intfloat/e5-base-v2"

config_dict = {
    "data_dir": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/tool/FlashRAG/examples/quick_start/dataset",
    "model2path": {"e5": retriever_path, "Qwen2.5-7B-Instruct": model_path},
    "retrieval_topk": 1,
    'faiss_gpu': False,
    "retrieval_method": "e5",
    'index_path': "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/index/e5_Flat.index",
    'corpus_path': "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/processed_corpus.jsonl",
    # "index_path": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/tool/FlashRAG/examples/quick_start/indexes/e5_Flat.index",
    # "corpus_path": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/R3RAG/tool/FlashRAG/examples/quick_start/indexes/general_knowledge.jsonl",
    "generator_model": "Qwen2.5-7B-Instruct",
    "metrics": ["em", "f1", "acc"],
    "save_intermediate_data": True,
}

config = Config(config_dict=config_dict)

all_split = get_dataset(config)
test_data = all_split["test"]
prompt_templete = PromptTemplate(
    config,
    system_prompt="Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
    user_prompt="Question: {question}\nAnswer:",
)


# pipeline = SequentialPipeline(config, prompt_template=prompt_templete)
# pipeline = SelfAskPipeline(config, prompt_template=prompt_templete)
pipeline = IRCOTPipeline(config, max_iter=10, prompt_template=prompt_templete)


output_dataset = pipeline.run(test_data, do_eval=True)
print("---generation output---")
print(output_dataset.pred)
