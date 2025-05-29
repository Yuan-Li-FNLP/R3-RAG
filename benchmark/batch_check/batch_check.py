import argparse
import json
import torch
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Union
import os, pdb, copy, re, string, ast
from collections import Counter
from datasets import load_dataset

DEBUG=False

def parse_args():
    parser = argparse.ArgumentParser(description='Check answers using LLM')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--question_key', type=str, required=True, help='Key for questions in the input data')
    parser.add_argument('--golden_answers_key', type=str, required=True, help='Key for golden answers in the input data')
    parser.add_argument('--answer_key', type=str, required=True, help='Key for answers to check in the input data')
    parser.add_argument('--check_model', type=str, required=True, help='Path to the local model')
    return parser.parse_args()
def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from input JSON file."""
    # with open(input_file, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    dataset = load_dataset('json', data_files=input_file)
    # data = dataset['train']
    data = list(dataset['train'])
    if DEBUG:
        data = data[:5]  # 取前5个样本用于调试
        
    if not isinstance(data, list):
        raise ValueError("Input data should be a list")
    
    if not data:
        raise ValueError("Input data is empty")
    
    # Check if the first item has the required keys
    first_item = data[0]
    if not isinstance(first_item, dict):
        raise ValueError("Each item in the data should be a dictionary")
    
    return data
def solve_init():
    args = parse_args()
    data = load_data(args.input)
    llm = LLM(model=args.check_model, tensor_parallel_size=4)
    return args, data, llm

def process(text):
    # lower case
    text = text.lower()
    # remove_articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # unify white space
    text = " ".join(text.split())
    # remove punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return text
def check_prompt(question, output, standard_answers):
    standard_answers_formatted = "\n".join([f"- {ans}\n" for ans in standard_answers])
    return f"""
Please perform a correctness analysis based on the following criteria:
1. Existence of an Answer: Determine whether the question has a definitive answer among the standard answers provided.
2. Comparison with Standard Answers: If an answer exists, evaluate whether the given answer satisfies any one of the standard answers.
3. Handling Uncertainty: If the question has a definitive answer but the given answer indicates uncertainty or inability to determine, consider the final answer as False.

Output format:
Correctness analysis: [Your detailed analysis]
Final answer: True or False

Guidelines:
True: If the given answer is relevant, accurate, and complete based on the standard answers.
False: If the answer is irrelevant, inaccurate, incomplete, or does not provide the required information, even if it doesn't directly contradict the standard answers.

Example:
Model Input:
Question: What nationality is the director of film Wedding Night In Paradise (1950 Film)?
Given Answer: The film "Wedding Night In Paradise" from 1950 does not exist or is not documented, so the nationality of its director cannot be determined.
Standard Answer List:
- Hungarian

Model Output:
Correctness analysis: The given answer does not address the nationality of the director and instead claims the film is undocumented. Assuming the film exists and the standard answer is "Hungarian," the answer is irrelevant and incomplete as it fails to provide the required nationality information.
Final answer: False

Now, given the following input, provide the corresponding output:
Model Input:
Question: {question}
Given Answer: {output}
Standard Answer List:
{standard_answers_formatted}
Model Output:
"""
def F1(golds: list[str], response: str) -> tuple[float, float, float]:
    true_counters = [Counter(gold.split()) for gold in golds]
    pred_counter = Counter(response.split())
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for gold in true_counters:
        common = sum((gold & pred_counter).values())
        if common == 0:
            continue
        p = 1.0 * common / sum(pred_counter.values())
        r = 1.0 * common / sum(gold.values())
        f1_ = (2 * p * r) / (p + r)
        precision = max(p, precision)
        recall = max(r, recall)
        f1 = max(f1, f1_)
    return f1, recall, precision
def split_answer(response_text):
    result = {}
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith("Correctness analysis:"):
            result['Correctness analysis'] = line.replace("Correctness analysis:", "").strip()
        elif line.startswith("Final answer:"):
            result['Final answer'] = line.replace("Final answer:", "").strip()
    return result
def solve_core(args, records, llm, temperature=0):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=512)
    judge_queue = []
    # 跳过无答案，朴素em为1的record
    for i, record in enumerate(records):
        gts = [process(ans) for ans in record['golden_answers']]
        if 'answer' in record and not record.get('skip_bench', False) and record.get("answer", ""):
            answer = process(record['answer'])
            record['em'] = float(answer in gts)
            record['f1'] = F1(gts , answer)[0]
            if answer in gts:
                record['em_processed'] = True
                record['f1_processed'] = 1.0
                record['skip_bench'] = True
                record['correctness'] = True
                record['correctness_reason'] = "The answer is in the standard answers."
            else:
                judge_queue.append((
                    i , 
                    [
                        {"role": "system","content": "You are a helpful assistant"},
                        {"role": "user","content": check_prompt(record['question'], record['answer'], record['golden_answers'])}
                    ]
                ))
        else:
            record['em_processed'] = False
            record['f1_processed'] = 0.0
            record['skip_bench'] = True
            record['correctness'] = False
            record['correctness_reason'] = "No answer"
            record['em'] = False
            record['f1'] = 0.0
    
    outputs = llm.chat([val[1] for val in judge_queue], sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    for output , (i , _) in zip(outputs , judge_queue):
        judge = split_answer(output)
        if 'correctness_list' not in records[i]:
            records[i]['correctness_list'] = []
            records[i]['correctness_list'].append(judge.get('Final answer', 'False').strip().lower() == 'true')
        else:
            records[i]['correctness_list'].append(judge.get('Final answer', 'False').strip().lower() == 'true')
        if 'correctness_reason' not in records[i]:
            records[i]['correctness_reason'] = []
            records[i]['correctness_reason'].append(judge.get('Correctness analysis', ''))
        else:
            records[i]['correctness_reason'].append(judge.get('Correctness analysis', ''))

def em_prompt(question, answer):
    return f"""
Your task is to extract a list of short candidate answers from a detailed sentence answer for a given question.

### Instructions:
- Provide multiple candidate answers whenever possible.
- All candidate answers must have exactly the same meaning and precisely answer the original question.
- For dates or numbers, include multiple common formats.
- For people's names or places, include common variations, abbreviations, or full names.
- For yes/no questions, include ["Yes"] or ["No"] and other synonymous expressions.

### Important formatting guidelines (must strictly follow):
- Your output must be a Python-style list of strings, for example: ["Answer1", "Answer2", "Answer3"]
- Ensure your output can be directly parsed by Python's `ast.literal_eval()` function.
- Do NOT mix quotation marks improperly. Maintain consistent and proper quotation usage to ensure your list is valid JSON format.

### Examples:

Example 1 (Date Question):
Question: When was the People's Republic of China founded?
Answer: The founding ceremony of the People's Republic of China was officially held on October 1st, 1949.
Candidate Answers: ["October 1st, 1949", "1949-10-01", "1949.10.01", "Oct 1, 1949"]

Example 2 (Person Name):
Question: Who wrote the novel "1984"?
Answer: "1984" is a novel authored by the British writer George Orwell, whose real name was Eric Arthur Blair.
Candidate Answers: ["George Orwell", "Eric Arthur Blair"]

Example 3 (Place Name):
Question: What is the capital of the United States?
Answer: The capital city of the United States is Washington, D.C., commonly known as Washington DC or simply Washington.
Candidate Answers: ["Washington, D.C.", "Washington DC", "Washington"]

Example 4 (Yes/No Question - positive):
Question: Did Albert Einstein develop the theory of relativity?
Answer: Yes, Albert Einstein is credited with developing the theory of relativity.
Candidate Answers: ["Yes"]

Example 5 (Yes/No Question - negative):
Question: Is Mercury the farthest planet from the Sun?
Answer: No, Mercury is actually the closest planet to the Sun, not the farthest.
Candidate Answers: ["No"]

### Now, based on the following question and answer, carefully provide the candidate answers as instructed (strictly following the JSON and quotation guidelines above):

Question: {question}
Answer: {answer}
Candidate Answers:
"""
def normalize_answer(s):
    """
    Normalize text by removing articles, punctuation, and extra whitespace, and converting to lowercase.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def whitespace_fix(text):
        return ' '.join(text.split())
    def lower(text):
        return text.lower()
    return whitespace_fix(remove_articles(remove_punctuation(lower(s))))
def extract_candidate_answers_strict(llm_output):
    # 检查输入合法性
    if not isinstance(llm_output, str) or not llm_output.strip():
        print("Warning: LLM output is empty or invalid.")
        return False
    # 正则匹配第一个出现的中括号内容
    match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
    if not match:
        print("Warning: No list found in LLM output.")
        return False
    candidate_str = match.group()
    # 尝试用 ast.literal_eval 安全解析
    try:
        candidate_list = ast.literal_eval(candidate_str)
    except (ValueError, SyntaxError):
        print("Warning: Unable to parse candidate answers using ast.literal_eval.")
        return False
    # 确保解析结果是一个列表
    if not isinstance(candidate_list, list):
        print("Warning: Parsed content is not a list.")
        return False
    # 对列表中的每个元素严格检查
    valid_candidate_list = []
    for item in candidate_list:
        if not isinstance(item, (str, int, float)):
            print(f"Warning: Invalid item type detected: {item} ({type(item)}).")
            return False  # 发现任何异常类型，直接返回False
        item_str = str(item).strip()
        valid_candidate_list.append(item_str)
    return valid_candidate_list
def compute_em_f1(gold_list: list[str], pred_list: list[str]) -> dict:
    """
    Compute EM and F1 using your provided F1 function.
    
    Args:
        gold_list (list[str]): List of ground-truth answers.
        pred_list (list[str]): List of predicted candidate answers.
        
    Returns:
        dict: {"em": EM score, "f1": F1 score}
    """
    # normalize answers first
    normalized_golds = [normalize_answer(gold) for gold in gold_list]
    normalized_preds = [normalize_answer(pred) for pred in pred_list]

    # Compute EM: exact match if any prediction matches any gold after normalization
    em = float(any(pred == gold for pred in normalized_preds for gold in normalized_golds))

    # Compute F1: take the maximum F1 score across all predictions
    max_f1 = 0.0
    for pred in normalized_preds:
        f1, _, _ = F1(normalized_golds, pred)
        max_f1 = max(max_f1, f1)
    return em, max_f1
def get_em(args, records, llm):
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    judge_queue = []
    # 跳过无答案，朴素em为1的record
    for i , record in enumerate(records):
        if 'answer' in record and not record.get('skip_bench', False):
            judge_queue.append((
                i , 
                [
                    {"role": "system","content": "You are a helpful assistant"},
                    {"role": "user","content": em_prompt(record['question'], record['answer'])}
                ]
            ))
    outputs = llm.chat([val[1] for val in judge_queue], sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    
    judge_queue_resample = []    
    for output , (i , _) in zip(outputs , judge_queue):
        judge = extract_candidate_answers_strict(output)
        if judge is False:
            print(f"Warning: Failed to extract candidate answers for record {i}.")
            judge_queue_resample.append((
                i , 
                [
                    {"role": "system","content": "You are a helpful assistant"},
                    {"role": "user","content": em_prompt(records[i]['question'], records[i]['answer'])}
                ]
            ))
            pass
        else:
            records[i]['answer_processed_list'] = judge
            records[i]['answer_processed_list'].append(process(records[i]['answer']))
    
    if judge_queue_resample:
        outputs = llm.chat([val[1] for val in judge_queue_resample], sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        for output , (i , _) in zip(outputs , judge_queue_resample):
            judge = extract_candidate_answers_strict(output)
            if judge is False:
                print(f"Warning: Finally failed to extract candidate answers for record {i}.")
                records[i]['answer_processed_list'] = [records[i]['answer']]
            else:
                records[i]['answer_processed_list'] = judge
                records[i]['answer_processed_list'].append(process(records[i]['answer']))

    # 根据answer_processed_list和golden_answers计算em_processed和f1_processed
    for record in records:
        if 'answer_processed_list' in record:
            record['em_processed'] , record['f1_processed'] = compute_em_f1(record['golden_answers'] , record['answer_processed_list'])
        elif 'em' in record:
            record['em_processed'] = record['em']
            record['f1_processed'] = record['f1']
        else:
            record['em_processed'] = False
            record['f1_processed'] = 0

def statistics(args, records, llm):
    # 计算records里面的['correctness']：通过['correctness_list']属性中True和False的个数，谁多就是谁，并提取对应reason作为['correctness_reason']
    for record in records:
        if 'answer' in record and not record.get('skip_bench', False):
            record['correctness'] = len([val for val in record['correctness_list'] if val]) >= len([val for val in record['correctness_list'] if not val])
            record['correctness_reason'] = ""

    # 初始化统计量
    total_em = 0.0
    total_em_processed = 0.0
    total_f1 = 0.0
    total_f1_processed = 0.0
    total_judge = 0.0
    count = 0

    # 遍历所有记录，计算总的 EM、F1 和 judge
    for record in records:
        total_em += record['em']
        total_em_processed += record['em_processed']
        total_f1 += record['f1']
        total_f1_processed += record['f1_processed']
        total_judge += record['correctness']
        count += 1
    # 计算平均值
    if count > 0:
        avg_em = total_em / count
        avg_em_processed = total_em_processed / count
        avg_f1 = total_f1 / count
        avg_f1_processed = total_f1_processed / count
        avg_judge = total_judge / count
    else:
        avg_em = avg_em_processed = avg_f1 = avg_f1_processed = avg_judge = 0.0
    # 保存过程数据到输出文件
    directory = os.path.dirname(args.input)
    full_filename = os.path.basename(args.input)
    filename, extension = os.path.splitext(full_filename)
    if extension.startswith('.'):
        extension = extension[1:]
    output_path1 = os.path.join(directory, filename+"_processed.json")
    output_path2 = os.path.join(directory, filename+"_results.txt")

    processed_results = {
        "total_em": total_em,
        "total_em_processed": total_em_processed,
        "total_f1": total_f1,
        "total_f1_processed": total_f1_processed,
        "total_judge": total_judge,
        "count": count,
        "avg_em": avg_em,
        "avg_em_processed": avg_em_processed,
        "avg_f1": avg_f1,
        "avg_f1_processed": avg_f1_processed,
        "avg_judge": avg_judge
    }
    
    with open(output_path1, "w", encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)

    # 保存统计结果到输出文件
    with open(output_path2, "w", encoding='utf-8') as f:
        f.write(f"Total EM: {total_em:.4f}\n")
        f.write(f"Total EM Processed: {total_em_processed:.4f}\n")
        f.write(f"Total F1: {total_f1:.4f}\n")
        f.write(f"Total F1 Processed: {total_f1_processed:.4f}\n")
        f.write(f"Total Judge: {total_judge:.4f}\n")
        f.write(f"Number of records: {count}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Average EM: {avg_em:.4f}\n")
        f.write(f"Average EM Processed: {avg_em_processed:.4f}\n")
        f.write(f"Average F1: {avg_f1:.4f}\n")
        f.write(f"Average F1 Processed: {avg_f1_processed:.4f}\n")
        f.write(f"Average Judge: {avg_judge:.4f}\n")

def main():
    args, data, llm = solve_init()
    solve_core(args, data, llm, temperature=0)
    get_em(args, data, llm)
    statistics(args, data, llm)
    # with open(os.path.join(args.log_dir, "records_processed.jsonl"), "w", encoding='utf-8') as f:
    #     for record in records:
    #         json.dump(record, f, ensure_ascii=False)
    #         f.write('\n')
    return

if __name__ == "__main__":
    main()