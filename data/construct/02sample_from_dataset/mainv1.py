import pdb

import argparse
parser = argparse.ArgumentParser(description='Process files with parameters')
# 添加必须的参数
parser.add_argument('--model_api_key', type=str, required=True)
parser.add_argument('--model_api_url', type=str, required=True)
parser.add_argument('--model_api_name', type=str, required=True)
parser.add_argument('--retriever_host', type=str, required=True)
parser.add_argument('--retriever_port', type=str, required=True)
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--worker_num_max', type=int, required=True)
parser.add_argument('--debug_mode', action='store_true', help='A boolean flag')
args = parser.parse_args()

__DEBUG__=args.debug_mode
API_try_counter=3
if __DEBUG__:
    pdb.set_trace()

# 加载模型api
from openai import OpenAI
client = OpenAI(
    api_key=args.model_api_key,
    base_url=args.model_api_url,
)
model=args.model_api_name

# 加载corpus
import requests
def GetRetrieval(query):
    url = "http://"+f"{args.retriever_host}"+":"+f"{args.retriever_port}"+"/search"
    payload = {
        "query": query
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        retrieval_results = response.json()
        return retrieval_results
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# 文件处理
import json
import re
def write_data_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as f:  # 'a' 是追加模式
        # json.dump(data, f, ensure_ascii=False, indent=4)
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')  # 每个字典单独占一行
def extract_substring(text, str1, str2=""):
    # 查找 str1 的最后一次出现的位置
    index1 = text.find(str1)
    # 如果 str1 找不到，直接返回空字符串
    if index1 == -1:
        return ""
    # 如果 str2 为空，返回从 str1 最后一个字符之后到文本末尾的所有内容
    if not str2:
        return text[index1 + len(str1):]
    # 查找 str2 第一次出现的位置
    index2 = text.find(str2)
    # 如果 str2 找不到，返回空字符串
    if index2 == -1:
        return ""
    # 确保提取的区间合法
    start = index1 + len(str1)  # 从 str1 的最后一个字符之后开始
    end = index2  # 到 str2 的开始位置之前
    if start < end:  # 如果区间合法，返回截取的内容
        return text[start:end]
    else:
        return ""  # 如果区间无效，返回空字符串 
def extract_substring2(text, start_str, stop_strs):
    # 查找 start_str 的最后一次出现的位置
    start_index = text.find(start_str)
    if start_index == -1:
        return ""
    start = start_index + len(start_str)
    
    # 初始化一个很大的结束索引
    end = len(text)
    
    # 查找 stop_strs 中每个字符串在 start 之后的位置，取最小的那个
    for stop_str in stop_strs:
        temp_index = text.find(stop_str, start)
        if temp_index != -1 and temp_index < end:
            end = temp_index
    # 提取子字符串
    if start < end:
        return text[start:end].strip()
    else:
        return ""
def clean_string(str):
    # str = str.strip(" ")
    str = str.strip("\n")
    str = str.strip("#")
    return str
def truncate_after_first_occurrence(str1, str2):
    # 查找 str2 在 str1 中的首次出现位置
    index = str1.find(str2)
    # 如果未找到，返回原始的 str1
    if index == -1:
        return str1
    # 计算 str2 结束后的下一个字符的位置
    end_index = index + len(str2) + 1
    # 如果 end_index 超过 str1 的长度，返回整个 str1
    if end_index > len(str1):
        return str1
    # 截取到 end_index
    return str1[:end_index]
def split_response(response):
    mydict = {
        "original":response
    }
    str_analysis = "The problem analysis:"
    str_query = "The retrieval query:"
    str_answer = "The final answer:"
    # 定义停止标识字符串列表
    # stop_strs = [str_analysis, str_query, str_answer, "####", "The retrieval documents:", "###"]
    stop_strs = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####"]

    if (str_query in response) and (str_answer in response):
        mydict["analysis"] = clean_string(extract_substring2(response, str_analysis, stop_strs))
        mydict["query"]    = clean_string(truncate_after_first_occurrence(extract_substring2(response, str_query,    stop_strs), "?"))
        mydict["answer"]   = clean_string(extract_substring2(response, str_answer,   stop_strs))
    elif str_analysis not in response:
        mydict["analysis"]=None
        mydict["query"]=None
        mydict["answer"]=None
    elif (str_query not in response) and (str_answer not in response):
        mydict["analysis"]=None
        mydict["query"]=None
        mydict["answer"]=None
    elif str_query in response:
        mydict["analysis"] = clean_string(extract_substring(response, str_analysis, str_query))
        mydict["query"]    = clean_string(truncate_after_first_occurrence(extract_substring2(response, str_query,    stop_strs), "?"))
    elif str_answer in response:
        mydict["analysis"] = clean_string(extract_substring(response, str_analysis, str_answer))
        mydict["answer"] = clean_string(extract_substring2(response, str_answer, stop_strs))
    return mydict
def check_correctness(str1, str2, standard_answers, client, model):
    standard_answers_formatted = "\n".join([f"- {ans}" for ans in standard_answers])
    prompt = f"""
Question: {str1}
Given Answer: {str2}

Standard Answer List:
{standard_answers_formatted}

Please perform a correctness analysis to determine whether the given answer satisfies the requirements of any one of the standard answers listed above.

Output format:
Correctness analysis: [Your analysis]
Final answer: True or False
"""
    prompt = f"""
Question: {str1}
Given Answer: {str2}

Standard Answer List:
{standard_answers_formatted}

Please perform a correctness analysis to determine whether the given answer satisfies the requirements of any one of the standard answers listed above. In your analysis, consider the relevance, accuracy, and completeness of the answer.

Output format:
Correctness analysis: [Your analysis]
Final answer: True or False

**Guidelines**:
- **True**: If the given answer is relevant, accurate, and complete based on the standard answers.
- **False**: If the answer is irrelevant, inaccurate, incomplete, or does not provide the required information, even if it doesn't directly contradict the standard answers.

**Example**:

---
Question: What nationality is the director of film Wedding Night In Paradise (1950 Film)?
Given Answer: The film "Wedding Night In Paradise" from 1950 does not exist or is not documented, so the nationality of its director cannot be determined.

Standard Answer List:
- Hungarian

Correctness analysis: The given answer does not address the nationality of the director and instead claims the film is undocumented. Assuming the film exists and the standard answer is "Hungarian," the answer is irrelevant and incomplete as it fails to provide the required nationality information.

Final answer: False
---
"""
    prompt = f"""
Question: {str1}
Given Answer: {str2}

Standard Answer List:
{standard_answers_formatted}

Please perform a correctness analysis based on the following criteria:

1. **Existence of an Answer**: Determine whether the question has a definitive answer among the standard answers provided.
2. **Comparison with Standard Answers**: If an answer exists, evaluate whether the given answer satisfies any one of the standard answers.
3. **Handling Uncertainty**: If the question has a definitive answer but the given answer indicates uncertainty or inability to determine, consider the final answer as False.

Output format:
Correctness analysis: [Your detailed analysis]
Final answer: True or False
"""
    for counter in range(API_try_counter):
        try:
            completion = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=256,
                temperature=0.8
            )
            mydict = split_answer(completion.choices[0].text)
            final_answer = mydict.get('Final answer', 'False').strip().lower()
            if final_answer == 'true':
                return True, completion.choices[0].text
            else:
                return False, completion.choices[0].text
        except Exception as e:
            print(f"An error occurred: {e}")
    return False, ""
def split_answer(response_text):
    result = {}
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith("Correctness analysis:"):
            result['Correctness analysis'] = line.replace("Correctness analysis:", "").strip()
        elif line.startswith("Final answer:"):
            result['Final answer'] = line.replace("Final answer:", "").strip()
    return result
def GetStepbystepRetrieval(mydict, retrieved_ids, documents):
    doc = []
    attempts_retrieval = 3
    for one_attempt_retrieval in range(attempts_retrieval):
        retrieval_results = GetRetrieval(mydict.get("query"))
        if retrieval_results is not None:
            break
    if retrieval_results is None:
        mydict["doc"]=""
        return
    for result in retrieval_results:
        if result['id'] not in retrieved_ids:
            retrieved_ids.append(result['id'])
            documents.append(result['contents'].split('\n')[-1])
            doc.append(result['contents'])
        if len(documents) >= num_passages_one_retrieval:
            break
    document = '\n'.join(doc)
    mydict["doc"]=document
def GetStepbystepRetrievalv2(mydict, retrieved_ids, documents, client, model):
    doc = []
    attempts_retrieval = 3
    for one_attempt_retrieval in range(attempts_retrieval):
        retrieval_results = GetRetrieval(mydict.get("query"))
        if retrieval_results is not None:
            break
    if retrieval_results is None:
        mydict["split_queries"] = [""]
        mydict["doc"]=""
        return
    for result in retrieval_results:
        if result['id'] not in retrieved_ids:
            retrieved_ids.append(result['id'])
            documents.append(result['contents'].split('\n')[-1])
            doc.append(result['contents'])
        if len(documents) >= num_passages_one_retrieval:
            break
    split_queries = split_query(mydict.get("query"), client, model)
    mydict["split_queries"] = split_queries
    for one_query in split_queries:
        retrieval_results = GetRetrieval(one_query)
        for result in retrieval_results:
            if result['id'] not in retrieved_ids:
                retrieved_ids.append(result['id'])
                documents.append(result['contents'].split('\n')[-1])
                doc.append(result['contents'])
            if len(documents) >= num_passages_one_retrieval:
                break
    document = '\n'.join(doc)
    mydict["doc"]=document
def split_query(query, client, model):
    prompt = f"""
You are an intelligent query decomposer. Based on the user's query, determine whether it needs to be broken down into smaller sub-questions. If it does, provide a list of sub-questions; otherwise, return a list containing only the original query.

Example 1:
Input: "Who is the director of 'Danger: Diabolik' and what is the release year of the film?"
Output: ["Who is the director of 'Danger: Diabolik'?", "What is the release year of the film 'Danger: Diabolik'?"]

Example 2:
Input: "What is the capital of France?"
Output: ["What is the capital of France?"]

Example 3:
Input: "Explain the theory of relativity."
Output: ["Explain the theory of relativity."]

Example 4:
Input: "List all the planets in the solar system and describe their main characteristics."
Output: ["List all the planets in the solar system.", "Describe the main characteristics of each planet."]

Now, given the following input query, provide the corresponding output list:

Input: "{query}"
Output:
"""
    for counter in range(API_try_counter):
        try:
            # Call the OpenAI API to generate the response
            response = client.completions.create(model=model, prompt=prompt, max_tokens=256, temperature=0.9)
            # Extract the generated text
            generated_text = response.choices[0].text.strip()
            # Use regular expressions to extract the list items
            # Assumes the output format is ["Question 1", "Question 2", ...]
            matches = re.findall(r'"(.*?)"', generated_text)
            # If no matches are found, return the original query in a list
            if not matches:
                return [query]
            return matches
        except Exception as e:
            print(f"An error occurred: {e}")
            # In case of an error, return the original query
    return [query]

# 生成搜素链
from prompt_template import *
num_passages_one_retrieval = 3
num_attempts_one_question = 7
num_search_one_attempt = 7
num_attempts_one_generation=2 # 一次生成格式太差，重新生成的次数

# 加载数据集
import os
input_dir = args.input_dir # 需要修改
output_dir = args.output_dir
log_path = os.path.join(output_dir, "log.txt")

# 读取文件列表
input_file_name_list=["train0.jsonl", "train1.jsonl"]
extension = ".jsonl"
input_file_name_list = []
# 遍历文件夹
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(extension.lower()):
            # 获取完整路径
            # full_path = os.path.join(root, file)
            input_file_name_list.append(file)
# 设置进程数
import multiprocessing
num_processes = min(len(input_file_name_list), multiprocessing.cpu_count(), args.worker_num_max)
print(f"Processer Number: {num_processes}")

# 单进程开始
def worker(input_file_name):
    input_file_path = os.path.join(input_dir, input_file_name)
    filename, extension = os.path.splitext(input_file_name)

    from datasets import load_dataset
    dataset = load_dataset('json', data_files=input_file_path, split='train')

    output_file_path = os.path.join(output_dir, input_file_name)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        pass

    processed_question_counter=0
    question_counter=0
    for id, data_one in enumerate(dataset):
        if __DEBUG__:
            if id>0:
                break
        question_counter+=1
        data_init = {
            "id_source": data_one["id"],
            "question": data_one["question"],
            "golden_answers": data_one["golden_answers"],
            "search_chain_fail":[],
            "search_chain_success":[],
            "fail_reason": [],
            "success_reason":[],
        }
        for search_attempts_counter in range(num_attempts_one_question): # 生成数据的允许失败最大次数
            if __DEBUG__:
                print(f"The {search_attempts_counter+1} attempts to generate for Problem {data_init['id_source']}.")
            search_chain = []
            retrieved_ids = []
            documents = []
            for search_counter in range(num_search_one_attempt): # 检索链的长度
                if __DEBUG__:
                    print(f"The {search_counter+1} Step to generate for Problem {data_init['id_source']}.")
                if search_counter==0:
                    prompt=prompt_question_initv2(data_init['question'])
                else:
                    prompt=prompt_question_newv2(search_chain, data_init['question'])
                    # prompt=prompt_question_endv3(search_chain, data_init['question'])
                for generation_counter in range(num_attempts_one_generation):
                    if search_attempts_counter==0:
                        completion = client.completions.create(model=model, prompt=prompt, max_tokens=256, temperature=0.0)
                    else:
                        completion = client.completions.create(model=model, prompt=prompt, max_tokens=256, temperature=0.9)
                    mydict=split_response(completion.choices[0].text)  
                    if (mydict.get("query") is not None) or (mydict.get("answer") is not None): # 判断单步生成的格式是否有问题
                        break
                if (mydict.get("query") is None) and (mydict.get("answer") is None): # 一个步骤生成的格式有问题，这个检索链丢弃
                    print("Format Error: Throw the chain.")
                    if __DEBUG__:
                        print(mydict)
                    break            
                elif mydict.get("answer") is not None:
                    if __DEBUG__:
                        print(mydict)
                    search_chain.append(mydict)
                    break # 模型认为自己回答完毕
                elif mydict.get("query") is not None:
                    if search_attempts_counter==0:
                        GetStepbystepRetrieval(mydict, retrieved_ids, documents)
                    else:
                        GetStepbystepRetrievalv2(mydict, retrieved_ids, documents, client, model)
                    if __DEBUG__:
                        print(mydict)
                    search_chain.append(mydict)
                else:
                    if __DEBUG__:
                        print(mydict)
                    raise ValueError("mydict is not the valid format.")
                    # ERROR
                if search_counter==num_search_one_attempt-1: # 最后一次但是还是没有生成答案
                    if __DEBUG__:
                        print(mydict)
                        print(f"The {search_counter+1} Step of the {search_attempts_counter+1} attempts to generate for Problem {data_init['id_source']}: Last step, but no answer.")
                    break
                pass
            
            # 如果存在提取最后一步
            last_step=None
            if len(search_chain)==0:
                continue
            else:
                last_step = search_chain[-1]
            if last_step.get("answer") is None:# 最后一步还是没有生成答案
                continue
            else:
                correctness, reason = check_correctness(data_init['question'], last_step.get("answer"), data_init["golden_answers"], client, model)
                if correctness:
                    print(f"The {search_attempts_counter+1} attempts to generate for Problem {data_init['id_source']}: Success.")
                    if __DEBUG__:
                        print("Reason:\n")
                        print(repr(reason))
                    data_init["success_reason"].append(reason)
                    data_init["search_chain_success"].append(search_chain)
                    data_init["id"]=processed_question_counter # 成功id从0开始
                    processed_question_counter+=1 # 成功counter从1开始
                    break
                else:
                    print(f"The {search_attempts_counter+1} attempts to generate for Problem {data_init['id_source']}: Fail.")
                    if __DEBUG__:
                        print(f"answer: {last_step.get('answer')} vs Golden_answer: {data_init['golden_answers']}")
                        print("Reason:\n")
                        print(repr(reason))
                    data_init["fail_reason"].append(reason)
                    data_init["search_chain_fail"].append(search_chain)
                    continue
        write_data_to_jsonl(output_file_path, data_init)
        if __DEBUG__:
            print(f"File: {input_file_name}, processed_question_counter: {processed_question_counter}")
            print(f"File: {input_file_name}, question_counter: {question_counter}")
        pass
    with multiprocessing.Lock():  # 使用锁确保日志写入的原子性
        mode = 'a' if os.path.exists(log_path) else 'w'
        # mode = 'a'
        with open(log_path, mode, encoding='utf-8') as f:
            f.write(f"{filename} {processed_question_counter} {question_counter}\n")
    print(f"File: {input_file_name}, processed_question_counter: {processed_question_counter}")
    print(f"File: {input_file_name}, question_counter: {question_counter}")

    if not __DEBUG__:
        try:
            os.remove(input_file_path)
            print(f"文件 '{input_file_path}' 已成功删除。")
        except FileNotFoundError:
            print(f"错误: 文件 '{input_file_path}' 未找到。")
        except PermissionError:
            print(f"错误: 没有权限删除文件 '{input_file_path}'。")
        except Exception as e:
            print(f"删除文件时发生错误: {e}")
    # return processed_question_counter, question_counter


# 创建进程池
with multiprocessing.Pool(num_processes) as pool:
    # results = pool.map(worker, input_file_name_list)
    pool.map(worker, input_file_name_list)

# 汇总结果
# total_processed_question_counter = sum(r[0] for r in results)
# total_question_counter = sum(r[1] for r in results)
# print(f"Total processed questions: {total_processed_question_counter}")
# print(f"Total questions: {total_question_counter}")
