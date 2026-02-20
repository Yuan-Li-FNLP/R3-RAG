#!/bin/bash
# 一键处理TUM数据的所有类型文档
# 使用方法: bash quick_process_tum_data.sh /path/to/tum_data

set -e

echo "╔════════════════════════════════════════════════════╗"
echo "║   TUM数据智能处理 - 针对7种文档类型               ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "❌ 错误：请提供数据目录路径"
    echo ""
    echo "使用方法:"
    echo "  bash $0 /path/to/tum_data"
    echo ""
    echo "你的数据目录应该包含这些子文件夹:"
    echo "  1_course_intros/    - 200门课程介绍"
    echo "  2_module_plans/     - 8个模块规划"
    echo "  3_lectures/         - 30门课程讲义"
    echo "  4_exercises/        - 30门练习题"
    echo "  5_scripts/          - 30门Script"
    echo "  6_references/       - 30门参考文献"
    echo "  7_project_docs/     - 2个项目文档"
    echo ""
    exit 1
fi

DATA_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/../tum_kb_work"

mkdir -p "${WORK_DIR}"

# 检查数据目录
if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ 数据目录不存在: ${DATA_DIR}"
    exit 1
fi

echo "📁 数据目录: ${DATA_DIR}"
echo "📁 工作目录: ${WORK_DIR}"
echo ""

# 检查子文件夹
required_folders=(
    "1_course_intros"
    "2_module_plans"
    "3_lectures"
    "4_exercises"
    "5_scripts"
    "6_references"
    "7_project_docs"
)

echo "检查文件夹结构..."
missing_folders=()
for folder in "${required_folders[@]}"; do
    if [ ! -d "${DATA_DIR}/${folder}" ]; then
        missing_folders+=("${folder}")
        echo "  ⚠️  未找到: ${folder}"
    else
        file_count=$(find "${DATA_DIR}/${folder}" -type f | wc -l)
        echo "  ✅ ${folder}: ${file_count} 个文件"
    fi
done

if [ ${#missing_folders[@]} -gt 0 ]; then
    echo ""
    echo "❌ 缺少必需的文件夹。请创建："
    for folder in "${missing_folders[@]}"; do
        echo "  mkdir -p ${DATA_DIR}/${folder}"
    done
    exit 1
fi

echo ""
read -p "继续处理? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

cd "${WORK_DIR}"

# ============================================
# 阶段1: 提取所有文档的文本
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段1/4: 提取文本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for folder in "${required_folders[@]}"; do
    echo "提取: ${folder}..."
    python3 "${SCRIPT_DIR}/fixed_step1_extract.py" \
        --input "${DATA_DIR}/${folder}" \
        --output "extracted_${folder}.json"
done

echo "✅ 阶段1完成"

# ============================================
# 阶段2: 清洗所有文本
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段2/4: 数据清洗"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for folder in "${required_folders[@]}"; do
    if [ -f "extracted_${folder}.json" ]; then
        echo "清洗: ${folder}..."
        python3 "${SCRIPT_DIR}/fixed_step2_clean.py" \
            --input "extracted_${folder}.json" \
            --output "cleaned_${folder}.json"
    fi
done

echo "✅ 阶段2完成"

# ============================================
# 阶段3: 智能切分（根据文档类型）
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段3/4: 智能切分"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 创建处理脚本
cat > process_all.py <<'PYTHON'
import json
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from smart_chunker_for_tum import TUMDocumentChunker

chunker = TUMDocumentChunker()

# 处理映射：文件夹 -> 文档类型
folder_to_type = {
    '1_course_intros': 'course_intro',
    '2_module_plans': 'module_plan',
    '3_lectures': 'lecture',
    '4_exercises': 'exercise',
    '5_scripts': 'script',
    '6_references': 'references',
    '7_project_docs': 'project'
}

for folder, doc_type in folder_to_type.items():
    json_file = f'cleaned_{folder}.json'
    print(f"处理 {folder} ({doc_type})...")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        for doc in documents:
            # 提取文档ID
            filename = doc['source_file']
            doc_id = filename.replace('.pdf', '').replace('.txt', '')

            chunker.process_document(
                text=doc['cleaned_text'],
                doc_type=doc_type,
                doc_id=doc_id
            )

        print(f"  ✅ 完成\n")
    except FileNotFoundError:
        print(f"  ⚠️  文件不存在，跳过\n")

# 导出
chunker.export_to_jsonl('final_corpus.jsonl')
chunker.print_statistics()
PYTHON

python3 process_all.py

echo "✅ 阶段3完成"

# ============================================
# 阶段4: 构建索引
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段4/4: 构建索引"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

INDEX_DIR="${SCRIPT_DIR}/../../indexes/tum_complete_kb"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 检测到GPU"
    USE_FP16="--use_fp16"
    BATCH_SIZE=128
else
    echo "💻 使用CPU模式"
    USE_FP16=""
    BATCH_SIZE=64
fi

python3 -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-base \
    --corpus_path final_corpus.jsonl \
    --save_dir "${INDEX_DIR}" \
    ${USE_FP16} \
    --batch_size ${BATCH_SIZE} \
    --pooling_method mean \
    --faiss_type Flat

echo "✅ 阶段4完成"

# ============================================
# 完成
# ============================================
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   ✅ TUM知识库创建完成！                          ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "📊 生成的文件:"
echo "   - final_corpus.jsonl (语料库)"
echo "   - ${INDEX_DIR}/ (索引)"
echo ""
echo "📈 文档类型统计:"
echo "   1. 课程介绍: ~200块（不切分）"
echo "   2. 模块规划: ~8-16块（按模块切分）"
echo "   3. 课程讲义: ~600-900块（1500字符切分）"
echo "   4. 练习题: ~150-300块（按题切分）"
echo "   5. Script: ~60-150块（1500字符切分）"
echo "   6. 参考文献: ~30块（不切分）"
echo "   7. 项目文档: ~5-10块（按章节切分）"
echo ""
echo "🚀 下一步:"
echo "   测试检索:"
echo "   python3 ${SCRIPT_DIR}/real_pipeline_step3_test.py \\"
echo "           --index_path ${INDEX_DIR}"
echo ""
