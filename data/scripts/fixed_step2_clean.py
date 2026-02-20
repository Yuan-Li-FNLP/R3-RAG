"""
阶段2: 数据清洗（独立步骤）
"""

import json
import re
import unicodedata
import argparse

def clean_text(text):
    """
    清洗文本，去除PDF噪音，标准化格式
    """
    if not text:
        return ""

    # 1. 统一Unicode（处理德语umlauts: ä, ö, ü, ß）
    text = unicodedata.normalize('NFC', text)

    # 2. 去除页码
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Seite \d+ von \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*/\s*\d+', '', text)  # 1/20格式

    # 3. 去除页眉页脚常见词
    text = re.sub(r'\bDRAFT\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCONFIDENTIAL\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bTechnische Universität München\b', '', text)
    text = re.sub(r'\bTUM\b', '', text)

    # 4. 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 5. 修复PDF提取的常见错误
    # 修复连字符断词（德语常见）
    # 例如: "Infor-\nmation" → "Information"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # 6. 压缩多余空白
    # 多个空格 → 一个空格
    text = re.sub(r' +', ' ', text)

    # 多个换行 → 最多两个换行（保留段落结构）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. 去除每行首尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 8. 标准化课程代码格式
    # "IN 2064" → "IN2064"
    text = re.sub(r'\b([A-Z]{2,3})\s+(\d{4})\b', r'\1\2', text)

    # 9. 去除零宽字符
    text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)

    return text.strip()

def main():
    parser = argparse.ArgumentParser(description='阶段2: 数据清洗')
    parser.add_argument('--input', required=True, help='输入JSON文件（阶段1输出）')
    parser.add_argument('--output', default='cleaned_documents.json', help='输出JSON文件')

    args = parser.parse_args()

    print("="*60)
    print("阶段2: 数据清洗")
    print("="*60)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print()

    # 读取阶段1的输出
    with open(args.input, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"读取了 {len(documents)} 个文档\n")

    cleaned_docs = []
    total_chars_removed = 0

    for doc in documents:
        source_file = doc['source_file']
        raw_text = doc['raw_text']

        print(f"清洗: {source_file}")
        print(f"  原始长度: {len(raw_text):,} 字符")

        # 清洗
        cleaned_text = clean_text(raw_text)
        chars_removed = len(raw_text) - len(cleaned_text)
        total_chars_removed += chars_removed

        print(f"  清洗后长度: {len(cleaned_text):,} 字符")
        print(f"  去除了: {chars_removed:,} 字符 ({chars_removed/len(raw_text)*100:.1f}%)")

        cleaned_docs.append({
            "source_file": source_file,
            "cleaned_text": cleaned_text,
            "num_chars": len(cleaned_text)
        })

        # 显示清洗效果示例
        if chars_removed > 100:
            print(f"  示例（清洗前）: {raw_text[:150]}")
            print(f"  示例（清洗后）: {cleaned_text[:150]}")

        print()

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)

    print("="*60)
    print(f"✅ 阶段2完成！")
    print(f"   文档数: {len(cleaned_docs)}")
    print(f"   总共去除: {total_chars_removed:,} 字符")
    print(f"   保存到: {args.output}")
    print("="*60)

    # 统计
    total_chars_after = sum(d['num_chars'] for d in cleaned_docs)
    print(f"\n📊 统计:")
    print(f"   清洗后总字符数: {total_chars_after:,}")

if __name__ == "__main__":
    main()
