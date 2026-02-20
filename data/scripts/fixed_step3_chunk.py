"""
阶段3: 分块 - 使用1500字符（不是500！）
"""

import json
import re
import argparse

def smart_chunk(text, chunk_size=1500, overlap=150):
    """
    智能分块

    Args:
        text: 要分块的文本
        chunk_size: 块大小（字符数）【默认1500，推荐值】
        overlap: 重叠大小（字符数）【默认150，约10%】

    Returns:
        List of text chunks

    说明:
        1500字符 ≈ 240-300个英文单词
                 ≈ 210-270个德语单词
                 ≈ 2-3个完整段落
    """
    # 按段落分割（两个或更多换行符）
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果单个段落超过chunk_size，强制切分
        if len(para) > chunk_size:
            # 先保存当前块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 强制切分超长段落
            for i in range(0, len(para), chunk_size - overlap):
                sub_chunk = para[i:i + chunk_size]
                if sub_chunk.strip():
                    chunks.append(sub_chunk.strip())

            continue

        # 检查加上这段是否超过限制
        potential_length = len(current_chunk) + len(para) + 2  # +2 for \n\n

        if potential_length < chunk_size:
            # 不超过限制，加入当前块
            current_chunk += para + "\n\n"
        else:
            # 超过限制，保存当前块，开始新块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # 开始新块（包含重叠）
            if overlap > 0 and len(current_chunk) > overlap:
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
    parser = argparse.ArgumentParser(description='阶段3: 分块')
    parser.add_argument('--input', required=True, help='输入JSON文件（阶段2输出）')
    parser.add_argument('--output', default='corpus.jsonl', help='输出JSONL文件')
    parser.add_argument('--chunk_size', type=int, default=1500,
                       help='块大小（字符数），默认1500')
    parser.add_argument('--overlap', type=int, default=150,
                       help='重叠大小（字符数），默认150')

    args = parser.parse_args()

    print("="*60)
    print("阶段3: 分块")
    print("="*60)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"块大小: {args.chunk_size} 字符")
    print(f"重叠: {args.overlap} 字符")
    print()

    # 读取清洗后的数据
    with open(args.input, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"读取了 {len(documents)} 个文档\n")

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        source_file = doc['source_file']
        text = doc['cleaned_text']
        text_length = len(text)

        print(f"分块: {source_file}")
        print(f"  原始长度: {text_length:,} 字符")

        if text_length == 0:
            print(f"  ⚠️  跳过（无内容）\n")
            continue

        # 分块
        chunks = smart_chunk(text, chunk_size=args.chunk_size, overlap=args.overlap)

        print(f"  生成块数: {len(chunks)}")

        # 统计块大小
        chunk_sizes = [len(c) for c in chunks]
        avg_size = sum(chunk_sizes) // len(chunks) if chunks else 0
        min_size = min(chunk_sizes) if chunks else 0
        max_size = max(chunk_sizes) if chunks else 0

        print(f"  平均块大小: {avg_size:,} 字符")
        print(f"  最小块: {min_size:,} 字符")
        print(f"  最大块: {max_size:,} 字符")

        # 显示示例块
        if chunks:
            print(f"  示例块: {chunks[0][:100]}...")

        # 添加到总列表
        for chunk_text in chunks:
            all_chunks.append({
                "id": str(chunk_id),
                "contents": chunk_text,
                "_source": source_file  # 元数据（不会保存到JSONL）
            })
            chunk_id += 1

        print()

    # 保存为JSONL（FlashRAG格式）
    with open(args.output, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            # FlashRAG只需要id和contents
            flashrag_doc = {
                "id": chunk["id"],
                "contents": chunk["contents"]
            }
            f.write(json.dumps(flashrag_doc, ensure_ascii=False) + '\n')

    print("="*60)
    print(f"✅ 阶段3完成！")
    print(f"   总块数: {len(all_chunks)}")
    print(f"   保存到: {args.output}")
    print("="*60)

    # 统计
    chunk_sizes = [len(c["contents"]) for c in all_chunks]
    print(f"\n📊 块大小统计:")
    print(f"   最小: {min(chunk_sizes):,} 字符")
    print(f"   最大: {max(chunk_sizes):,} 字符")
    print(f"   平均: {sum(chunk_sizes) // len(chunk_sizes):,} 字符")
    print(f"   总字符数: {sum(chunk_sizes):,}")

    # 显示示例块
    print(f"\n示例块（前3个）:")
    for i in range(min(3, len(all_chunks))):
        chunk = all_chunks[i]
        preview = chunk["contents"][:150].replace('\n', ' ')
        print(f"  [{chunk['id']}] ({len(chunk['contents'])}字符) {preview}...")

if __name__ == "__main__":
    main()
