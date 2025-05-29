import argparse, re, requests, random, pdb, tqdm
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from openai import OpenAI

def mystrip(one_str):
    one_str = one_str.strip()
    # one_str = one_str.strip("\\n")
    # 移除开头的所有"\\n"
    one_str = re.sub(r'^(\\n)+', '', one_str)
    # 移除结尾的所有"\\n"
    one_str = re.sub(r'(\\n)+$', '', one_str)  
    one_str = one_str.strip("#")  
    return one_str

def extract_substring2(text, start_str, stop_strs):
    # 查找 start_str 的最后一次出现的位置
    start_index = text.find(start_str)
    if start_index == -1:
        return None
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
        if start_str=="The retrieval query:" and end < len(text):
            if text[end]=='?':
                return mystrip(text[start:(end+1)]) # 会自动截断到字符串的末尾，而不会引发错误
            else:
                return mystrip(text[start:(end)]) # 会自动截断到字符串的末尾，而不会引发错误
        else:
            return mystrip(text[start:(end)]) # 会自动截断到字符串的末尾，而不会引发错误
    else:
        return None

def split_by_question_mark(text):
    if not isinstance(text, str):
        return None
    if not text:
        return ['']
    result = re.findall(r'[^?]*\?', text)
    return result if result else [text]

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

def GetRetrieval(query, retriver_url):
    payload = {
        "query": query
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(retriver_url, json=payload, headers=headers)
    if response.status_code == 200:
        retrieval_results = response.json()
        return retrieval_results
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def GetStepbystepRetrieval(mydict, retrieved_ids, documents, config):
    doc = []
    attempts_retrieval = 3
    for one_attempt_retrieval in range(attempts_retrieval):
        retrieval_results = GetRetrieval(mydict.get("query"), config['retriver_url'])
        if retrieval_results is not None:
            break
    if retrieval_results is None:
        mydict["doc"]=""
        return
    for i,result in enumerate(retrieval_results):
        if len(retrieved_ids) >= config['max_num_passages']:
            break
        if (i) >= config['num_passages_one_retrieval']:
            break
        if result['id'] not in retrieved_ids:
            retrieved_ids.append(result['id'])
            documents.append(result['contents'].split('\n')[-1])
            doc.append(result['contents'])
    document = '\n'.join(doc)
    mydict["doc"]=document

def GetStepbystepRetrievalv2(mydict, retrieved_ids, documents, config):
    doc = []
    attempts_retrieval = 3
    for one_attempt_retrieval in range(attempts_retrieval):
        retrieval_results = GetRetrieval(mydict.get("query"), config['retriver_url'])
        if retrieval_results is not None:
            break
    if retrieval_results is None:
        mydict["split_queries"] = [""]
        mydict["doc"]=""
        return
    for i, result in enumerate(retrieval_results):
        if len(retrieved_ids) >= config['max_num_passages']:
            break
        if (i) >= config['num_passages_one_retrieval']:
            break
        if result['id'] not in retrieved_ids:
            retrieved_ids.append(result['id'])
            documents.append(result['contents'].split('\n')[-1])
            doc.append(result['contents'])

    if mydict.get("split_queries"):
        pass
    else:
        split_queries = split_query(mydict.get("query"), config)
        mydict["split_queries"] = split_queries
    flag=False

    all_valid_docs = []
    temp_retrieved_ids = retrieved_ids.copy()
    for one_query in split_queries:
        retrieval_results = GetRetrieval(one_query, config['retriver_url'])
        if not retrieval_results:
            continue
        for result in retrieval_results:
            if result['id'] not in temp_retrieved_ids:
                flag=True
                all_valid_docs.append({
                    'id': result['id'],
                    'doc': result['contents']
                })
                temp_retrieved_ids.append(result['id'])
    if flag==False:
        mydict["doc"]=""
    else:
        if len(all_valid_docs) > config['num_passages_one_split_retrieval']:
            selected_docs = random.sample(all_valid_docs, config['num_passages_one_split_retrieval'])
        else:
            selected_docs = all_valid_docs  
        for doc_item in selected_docs:
            retrieved_ids.append(doc_item['id'])
            documents.append(doc_item['doc'].split('\n')[-1])
            doc.append(doc_item['doc'])  
        document = '\n'.join(doc)
        mydict["doc"] = document

def split_query(query, config):
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
    for counter in range(config['api_try_counter']):
        try:
            conversation = [{"role": "system","content": "You are a helpful assistant"},{"role": "user","content": prompt}]
            chat_response = config['client'].chat.completions.create(
                model=config['model_name'],
                messages=conversation,
                max_tokens=512,
                temperature=0.0
            )
            generated_text=chat_response.choices[0].message.content.strip()
            matches = re.findall(r'"(.*?)"', generated_text)
            if not matches:
                return [query]
            return matches
        except Exception as e:
            print(f"An error occurred: {e}")
    return [query]

class ChatSession:
    def __init__(self, llm, sampling_params, config):
        # 初始化LLM
        self.llm = llm
        # 设置采样参数，包括停止token id
        self.sampling_params = sampling_params
        # 配置参数
        self.config = config
        # 初始化对话历史
        self.input = ""
        self.output = ""
        self.search_chain = []
        self.retrieved_ids = []
        self.documents = []

    def get_answer(self, question, search_chain_length, split_flag=True):
        self.input = f"The question: {question}"
        self.output = ""
        length = 0
        while True:
            if length>=search_chain_length:
                break
            else:
                self.input = self.input + self.output
                conversation = [{"role": "system","content": "You are a helpful assistant"},{"role": "user","content": self.input}]
                outputs = self.llm.chat(
                    messages=conversation,
                    sampling_params=self.sampling_params
                )
                # stream_output = self.output + outputs# 流式显示
                generated_text = outputs[0].outputs[0].text
                mydict=split_response(generated_text)
                if mydict.get('answer'):
                    answer=mydict.get('answer')
                    self.search_chain.append(mydict)
                    length+=1
                    self.output = self.output + generated_text
                    break
                elif mydict.get('query'):
                    if split_flag:
                        GetStepbystepRetrievalv2(mydict, self.retrieved_ids, self.documents, self.config)
                    else:
                        GetStepbystepRetrieval(mydict, self.retrieved_ids, self.documents, self.config)
                    self.search_chain.append(mydict)
                    length+=1
                    self.output = self.output + generated_text +"\n"+ f"The retrieval documents: {mydict['doc']}" +"\n"
        pdb.set_trace()
        self.history_down = []
        self.search_chain = []
        self.retrieved_ids = []
        self.documents = []
        return self.output

def main():
    parser = argparse.ArgumentParser(description="使用vLLM部署Llama模型进行多轮对话")
    parser.add_argument("--model_path", type=str, required=True, help="Llama模型的路径")
    parser.add_argument("--stop_token_ids", type=int, nargs="+", default=[], help="停止生成的token id列表")
    
    # 检索相关参数
    parser.add_argument("--num_passages_one_retrieval", type=int, default=3, help="单次检索的文档数量")
    parser.add_argument("--num_passages_one_split_retrieval", type=int, default=5, help="分割查询检索的文档数量")
    parser.add_argument("--max_num_passages", type=int, default=80, help="最大文档数量")
    parser.add_argument("--num_search_one_attempt", type=int, default=10, help="单次尝试的搜索数量")
    parser.add_argument("--api_try_counter", type=int, default=3, help="API重试次数")
    parser.add_argument("--retriver_url", type=str, default="http://10.176.58.103:8001/search", help="检索器URL")
    
    # OpenAI API相关参数
    parser.add_argument("--openai_model_name", type=str, default="/remote-home1/yli/Model/Generator/Qwen2.5/14B/instruct", help="OpenAI模型名称")
    parser.add_argument("--api_url", type=str, default="http://10.176.58.103:8002", help="API URL")
    parser.add_argument("--openai_suffix", type=str, default="/v1", help="OpenAI API后缀")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API密钥")
    
    
    args = parser.parse_args()

    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_url + args.openai_suffix,
    )
    
    # 配置字典
    config = {
        'num_passages_one_retrieval': args.num_passages_one_retrieval,
        'num_passages_one_split_retrieval': args.num_passages_one_split_retrieval,
        'max_num_passages': args.max_num_passages,
        'num_search_one_attempt': args.num_search_one_attempt,
        'api_try_counter': args.api_try_counter,
        'retriver_url': args.retriver_url,
        'model_name': args.openai_model_name,
        'api_url': args.api_url,
        'openai_suffix': args.openai_suffix,
        'client': client,
    }

    # 初始化LLM
    llm = LLM(model=args.model_path, trust_remote_code=True)
    # 设置采样参数，包括停止token id
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=512,
        stop_token_ids=args.stop_token_ids if args.stop_token_ids else []
    )
    # 创建聊天会话
    session = ChatSession(llm, sampling_params, config)
    question = "What's the place of birth of the former member of The Sunnyboys?"
    answer = session.get_answer(question, search_chain_length=10, split_flag=False)
    pdb.set_trace()

if __name__ == "__main__":
    main()