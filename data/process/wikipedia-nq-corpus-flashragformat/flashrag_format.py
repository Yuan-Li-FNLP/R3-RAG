from datasets import load_dataset
import json

# 加载原始数据集
dataset = load_dataset('json', data_files='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus/corpus.jsonl.gz', split='train')

# 输出文件路径
output_file = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/yli/workspace/Datasets/Tevatron/wikipedia-nq-corpus-flashragformat/processed_corpus.jsonl'

# 打开输出文件（以写模式）
with open(output_file, 'w') as f_out:
    # 遍历数据集
    for example in dataset:
        # 构建新格式的数据
        new_doc = {
            'id': int(example['docid'])-1,  # 将 'docid' 改为 'id'
            'contents': f"{example['title']}\n{example['text']}"+ r"\n"  # 拼接 title 和 text
        }
        # 将处理后的数据转换为 JSON 格式并写入文件
        f_out.write(json.dumps(new_doc) + '\n')

print(f"处理后的数据已保存到 {output_file}")
