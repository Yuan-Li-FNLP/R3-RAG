import json
import os
from pathlib import Path
import argparse

def split_jsonl(chunk_size: int, input_file: str, output_dir: str):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    chunk = []
    chunk_id = 0
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 解析JSON行 
            try:
                json_obj = json.loads(line.strip())
                chunk.append(json_obj)
                
                # 当chunk达到指定大小时写入文件
                if len(chunk) == chunk_size:
                    output_file = os.path.join(output_dir, f'{chunk_id}.jsonl')
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        for item in chunk:
                            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    chunk = []
                    chunk_id += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue
    
    # 处理最后剩余的chunk
    if chunk:
        output_file = os.path.join(output_dir, f'{chunk_id}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for item in chunk:
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Split JSONL file into chunks')
    parser.add_argument('--chunk_size', type=int, help='Number of lines per chunk')
    parser.add_argument('--input_file', type=str, help='Input JSONL file path')
    parser.add_argument('--output_dir', type=str, help='Output directory path')
    
    args = parser.parse_args()
    
    if args.chunk_size <= 0:
        parser.error("Chunk size must be positive")
        
    if not os.path.exists(args.input_file):
        parser.error(f"Input file {args.input_file} does not exist")
        
    split_jsonl(args.chunk_size, args.input_file, args.output_dir)

if __name__ == "__main__":
    main()