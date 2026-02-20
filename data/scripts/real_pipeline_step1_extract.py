"""
真实的知识库创建流程 - 阶段1: 结构化（提取文本）

作用：
- 从PDF文件中提取文本
- 从CSV文件中提取数据
- 统一保存为JSON格式

使用方法：
    python real_pipeline_step1_extract.py --input /path/to/your/files --output extracted_texts.json
"""

import fitz  # PyMuPDF
import pandas as pd
import json
import argparse
from pathlib import Path

def extract_from_pdf(pdf_path):
    """从PDF提取文本"""
    print(f"  处理PDF: {pdf_path.name}")

    try:
        doc = fitz.open(pdf_path)
        full_text = ""

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            full_text += text

            # 显示进度
            if page_num % 10 == 0:
                print(f"    已处理 {page_num}/{len(doc)} 页")

        doc.close()

        return {
            "source_file": pdf_path.name,
            "file_type": "pdf",
            "full_text": full_text,
            "length": len(full_text),
            "pages": len(doc)
        }
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return None

def extract_from_csv(csv_path):
    """从CSV提取数据"""
    print(f"  处理CSV: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path)
        print(f"    找到 {len(df)} 行数据")

        # 假设CSV有这些列（根据实际情况调整）
        # course_code, title, ects, lecturer, semester, prerequisites

        extracted_rows = []

        for idx, row in df.iterrows():
            # 构建文本（根据你的CSV列名调整）
            text_parts = []

            for col_name, value in row.items():
                if pd.notna(value):  # 跳过空值
                    text_parts.append(f"{col_name}: {value}")

            text = "\n".join(text_parts)

            extracted_rows.append({
                "source_file": csv_path.name,
                "file_type": "csv",
                "full_text": text,
                "length": len(text),
                "row_index": idx
            })

        return extracted_rows
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return []

def extract_from_txt(txt_path):
    """从TXT提取文本"""
    print(f"  处理TXT: {txt_path.name}")

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        return {
            "source_file": txt_path.name,
            "file_type": "txt",
            "full_text": full_text,
            "length": len(full_text)
        }
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='阶段1: 结构化 - 从各种文件提取文本')
    parser.add_argument('--input', required=True, help='输入文件夹路径')
    parser.add_argument('--output', default='extracted_texts.json', help='输出JSON文件')

    args = parser.parse_args()

    input_dir = Path(args.input)

    if not input_dir.exists():
        print(f"❌ 文件夹不存在: {input_dir}")
        return

    print("="*60)
    print("阶段1: 结构化 - 提取文本")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {args.output}")
    print()

    extracted_data = []

    # 处理PDF文件
    pdf_files = list(input_dir.glob("*.pdf"))
    if pdf_files:
        print(f"📄 找到 {len(pdf_files)} 个PDF文件")
        for pdf_file in pdf_files:
            result = extract_from_pdf(pdf_file)
            if result:
                extracted_data.append(result)
                print(f"    ✅ 提取了 {result['length']:,} 个字符")

    # 处理CSV文件
    csv_files = list(input_dir.glob("*.csv"))
    if csv_files:
        print(f"\n📊 找到 {len(csv_files)} 个CSV文件")
        for csv_file in csv_files:
            results = extract_from_csv(csv_file)
            if results:
                extracted_data.extend(results)
                print(f"    ✅ 提取了 {len(results)} 行数据")

    # 处理TXT文件
    txt_files = list(input_dir.glob("*.txt"))
    if txt_files:
        print(f"\n📝 找到 {len(txt_files)} 个TXT文件")
        for txt_file in txt_files:
            result = extract_from_txt(txt_file)
            if result:
                extracted_data.append(result)
                print(f"    ✅ 提取了 {result['length']:,} 个字符")

    # 保存结果
    if extracted_data:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)

        print()
        print("="*60)
        print(f"✅ 结构化完成！")
        print(f"   提取了 {len(extracted_data)} 个文档")
        total_chars = sum(doc['length'] for doc in extracted_data)
        print(f"   总字符数: {total_chars:,}")
        print(f"   保存到: {args.output}")
        print("="*60)

        # 显示示例
        print("\n示例文档:")
        sample = extracted_data[0]
        print(json.dumps({
            "source_file": sample["source_file"],
            "file_type": sample["file_type"],
            "length": sample["length"],
            "preview": sample["full_text"][:200] + "..."
        }, ensure_ascii=False, indent=2))
    else:
        print("❌ 没有提取到任何数据！")

if __name__ == "__main__":
    main()
