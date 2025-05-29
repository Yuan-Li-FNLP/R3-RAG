# import json
# import pdb
# # 直接加载整个文件
# def load_jsonl(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#     return data

# 使用示例
# file_path = '/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/merge/2wikimultihopqa/train/merge.jsonl'
# from datasets import load_dataset
# dataset = load_dataset('json', data_files=file_path)
# data = load_jsonl(file_path)
# pdb.set_trace()   

import sys
from datasets import load_dataset

def statistic(input_file, output_file):
    dataset = load_dataset('json', data_files=input_file)
    total=len(dataset["train"])
    success=0
    for data_one in dataset["train"]:
        if len(data_one["search_chain_success"]) > 0:
            success+=1
    print(f"Total  data rows: {total}")
    print(f"Useful data rows: {success}")
    print(f"Utilization rate: {success/total}")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"Total  data rows: {total}\n")
        file.write(f"Useful data rows: {success}\n")
        file.write(f"Utilization rate: {success/total}\n")

if __name__ == "__main__":
    # 获取命令行参数：输入文件路径
    if len(sys.argv) != 3:
        print("Usage: python statistic.py for 2 params")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 打印数据
    statistic(input_file, output_file)
