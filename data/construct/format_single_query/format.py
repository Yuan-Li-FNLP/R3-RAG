import sys, json, os, pdb
from datasets import load_dataset

def mystrip(one_str):
    one_str = one_str.strip()
    one_str = one_str.strip("\\n")
    return one_str

def GenerateTrainingData(data_one):
    trainingdatalist = []
    instruction_str = ""
    output_str = ""
    flag = True
    for i, data_one_step in enumerate(data_one["search_chain_success"][0]):
        if i==0:# 结尾不加\n，每个字符串strip()
            instruction_str = f"The question: {data_one['question']}"
            if data_one_step.get('answer'): # 一部分答案由于输出格式错误，会含有answer和query，所以必须先answer
                analysis_str = mystrip(data_one_step['analysis'])
                answer_str = mystrip(data_one_step['answer'])
                output_str = f"Step {i+1}:\nThe problem analysis: {analysis_str}\nThe final answer: {answer_str}"
            elif data_one_step.get('query'):
                analysis_str = mystrip(data_one_step['analysis'])
                query_str = mystrip(data_one_step['query'])
                output_str = f"Step {i+1}:\nThe problem analysis: {analysis_str}\nThe retrieval query: {query_str}"
            else:
                if data_one_step.get('query')=="":
                    print(f"The data {data_one['id_source']} format wrong! The query string is empty!")   
                else:
                    print(f"The data {data_one['id_source']} format wrong!")                  
                # pdb.set_trace()
                flag=False
        else:
            doc_str = mystrip(data_one['search_chain_success'][0][i-1]['doc'])
            instruction_str = mystrip(instruction_str)
            output_str = mystrip(output_str)
            instruction_str = instruction_str+"\n"+output_str+"\n"+f"The retrieval documents: {doc_str}"
            if data_one_step.get('answer'):
                analysis_str = mystrip(data_one_step['analysis'])
                answer_str = mystrip(data_one_step['answer'])
                output_str = f"Step {i+1}:\nThe problem analysis: {analysis_str}\nThe final answer: {answer_str}"
            elif data_one_step.get('query'):
                analysis_str = mystrip(data_one_step['analysis'])
                query_str = mystrip(data_one_step['query'])
                output_str = f"Step {i+1}:\nThe problem analysis: {analysis_str}\nThe retrieval query: {query_str}"
            else:
                if data_one_step.get('query')=="":
                    print(f"The data {data_one['id_source']} format wrong! The query string is empty!")   
                else:
                    print(f"The data {data_one['id_source']} format wrong!")                 
                # pdb.set_trace()
                flag=False
        trainingdata = {
            "instruction": instruction_str,
            "input": "",
            "output": output_str,
        }
        trainingdatalist.append(trainingdata)
    return flag, trainingdatalist

def format(input_file, output_file):
    counter=0
    dataset = load_dataset('json', data_files=input_file)
    formatted_data = []
    for i, data_one in enumerate(dataset["train"]):
        if len(data_one["search_chain_success"])==0:
            continue
        flag, mylist = GenerateTrainingData(data_one)
        if flag:
            counter+=len(mylist)
            formatted_data.extend(mylist)
        else:
            print(f"Skip the question {data_one['id_source']} for the wrong format!")
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)
    
    # 获取文件路径
    log_dir = os.path.dirname(output_file)
    log_file = "log.txt"
    log_path = os.path.join(log_dir, log_file)
    with open(log_path, 'w', encoding='utf-8') as file:
        file.write(f"Total  data rows: {counter}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python for 2 params")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    _, input_file_extension = os.path.splitext(input_file)
    _, output_file_extension = os.path.splitext(output_file)
    if input_file_extension.lower() != '.jsonl' or output_file_extension.lower() != '.json':
        print("File format wrong!")
        sys.exit(1)
    format(input_file, output_file)
