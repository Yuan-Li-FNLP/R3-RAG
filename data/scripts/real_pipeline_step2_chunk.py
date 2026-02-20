"""
真实的知识库创建流程 - 阶段2: 分块（切成小段）

作用：
- 把长文档切成小块（避免单个文档太长）
- 保持块之间有重叠（避免信息丢失）
- 转换成FlashRAG所需的JSONL格式

使用方法：
    python real_pipeline_step2_chunk.py --input extracted_texts.json --output corpus.jsonl --chunk_size 500
"""

import json
import re
import argparse

def smart_chunk(text, chunk_size=500, overlap=50):
    """
    智能分块函数

    Args:
        text: 要切分的文本
        chunk_size: 每块的目标大小（字符数）
        overlap: 块之间的重叠字符数

    Returns:
        List of text chunks
    """
    # 按段落分割（双换行符）
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果单个段落就超过chunk_size，强制切分
        if len(para) > chunk_size:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 强制切分超长段落
            for i in range(0, len(para), chunk_size - overlap):
                sub_chunk = para[i:i + chunk_size]
                chunks.append(sub_chunk.strip())

            continue

        # 如果加上这段不超过限制
        if len(current_chunk) + len(para) + 2 < chunk_size:  # +2 for \n\n
            current_chunk += para + "\n\n"
        else:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())

            # 开始新块（包含重叠）
            if overlap > 0 and current_chunk:
                # 从上一块取最后overlap个字符
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para + "\n\n"
            else:
                current_chunk = para + "\n\n"

    # 保存最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def main():
    parser = argparse.ArgumentParser(description='阶段2: 分块 - 把长文档切成小块')
    parser.add_argument('--input', required=True, help='输入JSON文件（step1的输出）')
    parser.add_argument('--output', default='corpus.jsonl', help='输出JSONL文件')
    parser.add_argument('--chunk_size', type=int, default=500, help='每块大小（字符数）')
    parser.add_argument('--overlap', type=int, default=50, help='块之间重叠（字符数）')

    args = parser.parse_args()

    print("="*60)
    print("阶段2: 分块 - 切成小段")
    print("="*60)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"块大小: {args.chunk_size} 字符")
    print(f"重叠: {args.overlap} 字符")
    print()

    # 读取结构化数据
    with open(args.input, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"读取了 {len(documents)} 个文档")
    print()

    # 分块处理
    all_chunks = []
    chunk_id = 0

    for doc_idx, doc in enumerate(documents, 1):
        source_file = doc.get('source_file', f'unknown_{doc_idx}')
        text = doc.get('full_text', '')
        text_length = len(text)

        print(f"[{doc_idx}/{len(documents)}] {source_file}")
        print(f"  原始长度: {text_length:,} 字符")

        if text_length == 0:
            print(f"  ⚠️  跳过（无内容）")
            continue

        # 分块
        chunks = smart_chunk(text, chunk_size=args.chunk_size, overlap=args.overlap)

        print(f"  生成块数: {len(chunks)}")
        print(f"  平均长度: {sum(len(c) for c in chunks) // len(chunks):,} 字符")

        # 添加到总列表
        for chunk_text in chunks:
            all_chunks.append({
                "id": str(chunk_id),
                "contents": chunk_text,
                "_metadata": {  # 元数据（不会包含在最终JSONL中）
                    "source_file": source_file,
                    "chunk_index": len(all_chunks)
                }
            })
            chunk_id += 1

        print()

    # 保存为JSONL格式（FlashRAG要求）
    with open(args.output, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            # 只保留id和contents（FlashRAG格式）
            flashrag_doc = {
                "id": chunk["id"],
                "contents": chunk["contents"]
            }
            f.write(json.dumps(flashrag_doc, ensure_ascii=False) + '\n')

    print("="*60)
    print(f"✅ 分块完成！")
    print(f"   总块数: {len(all_chunks)}")
    print(f"   保存到: {args.output}")
    print("="*60)

    # 显示统计
    chunk_lengths = [len(c["contents"]) for c in all_chunks]
    print(f"\n📊 块大小统计:")
    print(f"   最小: {min(chunk_lengths):,} 字符")
    print(f"   最大: {max(chunk_lengths):,} 字符")
    print(f"   平均: {sum(chunk_lengths) // len(chunk_lengths):,} 字符")

    # 显示示例
    print(f"\n示例块（前3个）:")
    for i in range(min(3, len(all_chunks))):
        chunk = all_chunks[i]
        preview = chunk["contents"][:150].replace('\n', ' ')
        print(f"  [{chunk['id']}] {preview}...")

if __name__ == "__main__":
    main()
