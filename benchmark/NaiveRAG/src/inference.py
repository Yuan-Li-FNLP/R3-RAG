from vllm import LLM, SamplingParams
import json,os,random
import pdb
from datasets import load_dataset
import tqdm
import requests
import time
import argparse
DEBUG = False
POST_BATCH_SIZE = 2048
SSL_RETRY = 8

def get_args():
    parser = argparse.ArgumentParser(description="naive_generation_benchmark")
    parser.add_argument('--model_path', type=str, default='llama-7b-hf', help="Path to the model.")
    parser.add_argument('--log_dir', type=str, default='~/logs', help="Path to the model.")
    parser.add_argument('--dev_dataset_path', type=str, default='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/R3RAG/original_datasets/hotpotqa/dev.jsonl', help="Path to the model.")
    parser.add_argument('--retrieve_url', type=str, required=True, help="Retrieve url.")
    parser.add_argument('--num_of_docs', type=int, default=17, help="eos token id to stop.") 
    parser.add_argument('--tensor_parallel_size', type=int, default=8, help="eos token id to stop.") 

    args = parser.parse_args()
    return args
def mystrip(one_str):
    one_str = one_str.strip()
    one_str = one_str.strip("\\n")
    one_str = one_str.strip("#")
    return one_str
def extract_substring2(text, start_str, stop_strs):
    start_index = text.find(start_str)
    if start_index == -1:
        return None
    start = start_index + len(start_str)
    
    end = len(text)
    
    for stop_str in stop_strs:
        temp_index = text.find(stop_str, start)
        if temp_index != -1 and temp_index < end:
            end = temp_index
    if start < end:
        return mystrip(text[start:end])
    else:
        return None
def split_response(response):
    mydict = {
        "original":response
    }
    str_analysis = "The problem analysis:"
    str_query = "The retrieval query:"
    str_answer = "The final answer:"
    stop_strs = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####"]
    stop_strs_query = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep", "?"]
    stop_strs_answer = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep"]
    
    start_index = response.find(str_analysis)
    if start_index==-1:    
        mydict['analysis']=None
        return mydict
    else:
        mydict["analysis"]=extract_substring2(response, str_analysis, stop_strs)
    start_index_query = response.find(str_query, start_index+len(str_analysis))
    start_index_answer = response.find(str_answer, start_index+len(str_analysis))
    if start_index_query==-1 and start_index_answer==-1:
        mydict['analysis']=None
        return mydict
    elif start_index_query!=-1 and start_index_answer!=-1:
        if start_index_query<start_index_answer:
            mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
        else:
            mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    elif start_index_query!=-1:
        mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
    elif start_index_answer!=-1:
        mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    else:
        raise ValueError
    return mydict
def GetRetrieval(retrieve_url, querys):
    res = []
    for i in tqdm.tqdm(range(0, len(querys), POST_BATCH_SIZE)):
        subset = querys[i:i+POST_BATCH_SIZE]
        for _ in range(SSL_RETRY):
            response = requests.post(retrieve_url, json={"querys": subset}, headers={"Content-Type": "application/json"})
            if response.status_code == 200 and response.json():
                res.extend(response.json())
                break
        else:
            print(f"Fail info: {response.text}")
            raise ValueError(f"Failed to retrieve query:{i} ~ {i + POST_BATCH_SIZE}!!!!!!!!!!")
    return res
def solve(args):
    ckpt , records = solve_init(args)
    solve_main(args, ckpt, records, temperature=0)
    remain_idxs = [i for i , record in enumerate(records) if 'answer' not in record]
    print(f"Remain records: {len(remain_idxs)}")
    if len(remain_idxs) == 0:
        pass
    else:
        solve_directly(args, ckpt, records, temperature=0)
        
    with open(os.path.join(args.log_dir , "records.jsonl"), "w", encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
def solve_init(args):
    if DEBUG:
        ckpt = LLM(
            model=args.model_path, 
            tensor_parallel_size=1
        )
    else:
        ckpt = LLM(
            model=args.model_path, 
            tensor_parallel_size=args.tensor_parallel_size
        )
    print("ckpt is ready.")
    # 读取 JSON 数据
    dataset  = load_dataset('json', data_files=args.dev_dataset_path)['train']
    # 如果处于调试模式，只取少量数据
    if DEBUG:
        # 获取数据集大小
        dataset_size = len(dataset)
        # 随机采样8个索引（或小于数据集大小的数量）
        sample_size = min(8, dataset_size)
        sampled_indices = random.sample(range(dataset_size), sample_size)
        # 只保留采样的数据
        dataset = dataset.select(sampled_indices)
    records = []
    query_list = []
    for i, data in enumerate(dataset):
        record = {}
        record['question'] = data['question']
        record['golden_answers'] = data['golden_answers']
        record['state'] = "undo"
        record['resample_times'] = 0
        records.append(record)
        query_list.append(data['question'])
    # batch retrieve
    doc_list = GetRetrieval(args.retrieve_url, query_list)
    for doc_one, record in zip(doc_list,records):
        # print(args.num_of_docs)
        # print(len(doc_one))
        record['doc'] = "\n".join([doc_one_one['contents'] for doc_one_one in doc_one[:args.num_of_docs]])
    return ckpt , records
def generate_naive_rag_prompt(question, doc):
    system_message = f"""Answer the question based on the given document. Only give me the answer and do not output any other words.
The following are given documents.
{doc}
"""
    user_message = f"""The question: {question}"""
    message_list = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return message_list
def generate_naive_rag_cot_prompt(question, doc):
    system_message = """You are a helpful assistant that answers questions based on document retrieval with step-by-step reasoning.

For any question, please structure your response in this format:
The problem analysis: [Provide detailed step-by-step reasoning]
The final answer: [Provide the concise final answer]

Example:
User: The question: What was the company's revenue in 2023?
Assistant:
The problem analysis: I need to find information about the company's revenue in 2023. Looking at the provided document, I can see in the third paragraph that "the company's total revenue for fiscal year 2023 reached $128 million." This clearly states the exact revenue figure I'm looking for.
The final answer: The company's revenue in 2023 was $128 million.

Please carefully analyze the provided documents, ensure your answer is fully based on the document content, and use step-by-step reasoning to reach accurate conclusions."""

    # Add documents to system message
    system_message += f"""

The following are the provided documents:
{doc}
"""

    # User message with the question
    user_message = f"""The question: {question}"""
    message_list = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return message_list
def solve_main(args, ckpt, records, temperature=0):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=512)
    messages = [generate_naive_rag_cot_prompt(record['question'],record['doc']) for record in records]
    outputs = ckpt.chat(messages, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    vals = [split_response(output) for output in outputs]
        
    for i, val in enumerate(vals):
        records[i]['output'] = val['original']
        if 'answer' in val:
            records[i]['answer'] = val['answer']
            records[i]['state'] = "done"
        else:
            records[i]['state'] = "wrong"
def solve_directly(args, ckpt, records, temperature=0):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=512)
    
    # 提取需要处理的data id（这是可以自定义的，默认是没有提取出答案的都算）
    remain_idxs = [i for i , record in enumerate(records) if 'answer' not in record]
    messages = [generate_naive_rag_prompt(records[remain_idx]['question'], records[remain_idx]['doc']) for remain_idx in remain_idxs]
    
    outputs = ckpt.chat(messages, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
        
    # for i, val in enumerate(vals):
    for output , remain_idx in zip(outputs , remain_idxs):  
        records[remain_idx]['answer'] = output
        records[remain_idx]['state'] = "done"
        records[remain_idx]['resample_times'] = records[remain_idx].get('resample', 0) + 1

if __name__ == "__main__":
    start = time.time()
    print(f"Start at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    
    args = get_args()
    solve(args)

    # Record the end time
    end = time.time()
    print(f"End at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")

    # Calculate the elapsed time
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.2f} seconds")