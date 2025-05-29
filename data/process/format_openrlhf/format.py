import sys, pdb, json
from datasets import load_dataset

def statistic(input_file, output_file):
    dataset = load_dataset('json', data_files=input_file)
    total=len(dataset["train"])
    data_store = []
    counter=0
    for i, data_one in enumerate(dataset["train"]):
        if i<30000:
            continue
        elif i==30256:
            break 
        counter+=1
        data_one_store = {
            "id": counter-1,
            "question": data_one["question"],
            "golden_answers": data_one["golden_answers"],
            "context_messages": f"The question: {data_one['question']}"
        }
        data_store.append(data_one_store)
    # 以 JSONL 格式写入
    with open(output_file, "w", encoding="utf-8") as f:
        for data_one in data_store:
            json.dump(data_one, f, ensure_ascii=False)
            f.write("\n")  # 每个 JSON 需要换行
        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python statistic.py for 2 params")
        sys.exit(1)

    input = sys.argv[1]
    output = sys.argv[2]

    statistic(input, output)
