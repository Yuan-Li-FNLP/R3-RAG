"""
阶段1: 结构化 - 提取文本（只提取，不清洗）
"""

import fitz  # PyMuPDF
import pandas as pd
import json
import argparse
from pathlib import Path

def extract_from_pdf(pdf_path):
    """从PDF提取原始文本（不清洗）"""
    print(f"  提取PDF: {pdf_path.name}")

    try:
        doc = fitz.open(pdf_path)
        raw_text = ""

        for page_num, page in enumerate(doc, 1):
            raw_text += page.get_text()

            if page_num % 10 == 0:
                print(f"    进度: {page_num}/{len(doc)} 页")

        doc.close()

        return {
            "source_file": pdf_path.name,
            "raw_text": raw_text,
            "num_pages": len(doc),
            "num_chars": len(raw_text)
        }
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='阶段1: 提取文本')
    parser.add_argument('--input', required=True, help='输入文件夹')
    parser.add_argument('--output', default='all_documents.json', help='输出JSON文件')

    args = parser.parse_args()
    input_dir = Path(args.input)

    print("="*60)
    print("阶段1: 结构化 - 提取原始文本")
    print("="*60)
    print(f"输入: {input_dir}")
    print(f"输出: {args.output}")
    print()

    all_docs = []

    # 处理PDF
    pdf_files = list(input_dir.glob("*.pdf"))
    if pdf_files:
        print(f"找到 {len(pdf_files)} 个PDF文件\n")
        for pdf_file in pdf_files:
            doc_data = extract_from_pdf(pdf_file)
            if doc_data:
                all_docs.append(doc_data)
                print(f"  ✅ 提取了 {doc_data['num_chars']:,} 字符\n")

    # 保存
    if all_docs:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        print("="*60)
        print(f"✅ 阶段1完成!")
        print(f"   文档数: {len(all_docs)}")
        total_chars = sum(d['num_chars'] for d in all_docs)
        print(f"   总字符数: {total_chars:,}")
        print(f"   保存到: {args.output}")
        print("="*60)

        # 显示示例
        print("\n示例文档:")
        sample = all_docs[0]
        print(f"  文件: {sample['source_file']}")
        print(f"  页数: {sample['num_pages']}")
        print(f"  字符数: {sample['num_chars']:,}")
        print(f"  预览: {sample['raw_text'][:200]}...")
    else:
        print("❌ 没有提取到任何数据")

if __name__ == "__main__":
    main()
