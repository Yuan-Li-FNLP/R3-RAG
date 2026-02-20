#!/bin/bash
# 一键处理混合格式、多语言TUM数据
# 支持: PDF + HTML, 德语 + 英语

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================
# 参数检查
# ============================================
if [ $# -eq 0 ]; then
    echo "使用方法: bash process_mixed_data.sh <数据目录>"
    echo ""
    echo "示例:"
    echo "  bash process_mixed_data.sh /home/user/tum_data_mixed"
    echo ""
    echo "数据目录结构应该是:"
    echo "  <数据目录>/"
    echo "    ├── folder1/  (包含PDF和/或HTML文件)"
    echo "    ├── folder2/"
    echo "    └── ..."
    exit 1
fi

DATA_DIR="$1"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误: 目录不存在: $DATA_DIR"
    exit 1
fi

cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        混合格式、多语言TUM知识库处理器                            ║"
echo "║        支持: PDF + HTML × 德语 + 英语                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 数据目录: $DATA_DIR"
echo ""

# ============================================
# 检查依赖
# ============================================
echo "检查依赖..."
python3 -c "import fitz, bs4, unicodedata" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖库"
    echo "请安装: pip3 install PyMuPDF beautifulsoup4"
    exit 1
fi
echo "✅ 依赖检查通过"
echo ""

# ============================================
# 扫描数据目录
# ============================================
echo "扫描数据目录..."
folders=($(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d))

if [ ${#folders[@]} -eq 0 ]; then
    echo "⚠️  警告: 没有找到子文件夹"
    echo "是否直接处理根目录的文件？[y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        folders=("$DATA_DIR")
    else
        exit 1
    fi
fi

echo "找到 ${#folders[@]} 个文件夹:"
for folder in "${folders[@]}"; do
    pdf_count=$(find "$folder" -maxdepth 1 -name "*.pdf" | wc -l)
    html_count=$(find "$folder" -maxdepth 1 -name "*.html" -o -name "*.htm" | wc -l)
    echo "  📁 $(basename "$folder"): ${pdf_count} PDF + ${html_count} HTML"
done
echo ""

# ============================================
# 阶段1: 提取文本
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段1/3: 提取文本（PDF + HTML）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for folder in "${folders[@]}"; do
    folder_name=$(basename "$folder")
    output_file="extracted_${folder_name}.json"

    echo "处理: $folder_name"

    # 检查是否有PDF或HTML文件
    pdf_count=$(find "$folder" -maxdepth 1 -name "*.pdf" | wc -l)
    html_count=$(find "$folder" -maxdepth 1 -name "*.html" -o -name "*.htm" | wc -l)

    if [ $pdf_count -eq 0 ] && [ $html_count -eq 0 ]; then
        echo "  ⚠️  跳过（没有PDF或HTML文件）"
        continue
    fi

    # 提取
    python3 extract_multiformat.py \
        --input "$folder" \
        --output "$output_file" \
        --formats pdf,html

    echo ""
done

echo "✅ 阶段1完成"
echo ""

# ============================================
# 阶段2: 清洗文本（多语言）
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段2/3: 清洗文本（德语 + 英语）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for extracted_file in extracted_*.json; do
    if [ -f "$extracted_file" ]; then
        cleaned_file="cleaned_${extracted_file#extracted_}"

        echo "清洗: $extracted_file"
        python3 clean_multilingual.py \
            --input "$extracted_file" \
            --output "$cleaned_file"

        echo ""
    fi
done

echo "✅ 阶段2完成"
echo ""

# ============================================
# 阶段3: 合并所有清洗后的文件
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段3/3: 合并和切分"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "合并所有清洗后的文件..."

# 合并JSON文件
python3 - <<'PYTHON'
import json
import glob

all_docs = []
for file in glob.glob("cleaned_*.json"):
    print(f"  读取: {file}")
    with open(file, 'r', encoding='utf-8') as f:
        docs = json.load(f)
        all_docs.extend(docs)

print(f"\n总文档数: {len(all_docs)}")

# 统计
stats = {'de': 0, 'en': 0, 'unknown': 0, 'pdf': 0, 'html': 0}
for doc in all_docs:
    lang = doc.get('language', 'unknown')
    fmt = doc.get('format', 'unknown')
    stats[lang] = stats.get(lang, 0) + 1
    stats[fmt] = stats.get(fmt, 0) + 1

print(f"  🇩🇪 德语: {stats['de']}")
print(f"  🇬🇧 英语: {stats['en']}")
print(f"  📄 PDF: {stats['pdf']}")
print(f"  🌐 HTML: {stats['html']}")

# 保存合并结果
with open('all_cleaned_merged.json', 'w', encoding='utf-8') as f:
    json.dump(all_docs, f, ensure_ascii=False, indent=2)

print(f"\n💾 保存到: all_cleaned_merged.json")
PYTHON

echo ""
echo "✅ 阶段3完成"
echo ""

# ============================================
# 后续步骤提示
# ============================================
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   ✅ 数据处理完成！                                                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 结果文件:"
echo "   - all_cleaned_merged.json (所有清洗后的文档)"
echo ""
echo "🔄 下一步操作:"
echo ""
echo "1️⃣  切分文档:"
echo "   python3 smart_chunker_for_tum.py \\"
echo "       --input all_cleaned_merged.json \\"
echo "       --output final_corpus.jsonl"
echo ""
echo "2️⃣  构建多语言索引:"
echo "   bash build_index_with_model_choice.sh"
echo ""
echo "3️⃣  测试跨语言检索:"
echo "   python3 test_cross_lingual.py --model intfloat/multilingual-e5-base"
echo ""
echo "💡 提示: 详细说明请查看 多格式多语言处理指南.md"
echo ""
