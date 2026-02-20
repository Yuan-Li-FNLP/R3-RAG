#!/bin/bash
# 完整流程 - 正确版本（数据清洗独立，块大小1500字符）
#
# 使用方法:
#   bash fixed_run_all.sh /path/to/your/files

set -e  # 遇到错误立即退出

echo "╔════════════════════════════════════════════════════╗"
echo "║   TUM课程知识库 - 完整流程（修正版）              ║"
echo "║   5个独立阶段：提取→清洗→分块→索引→测试          ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "❌ 错误：请提供输入文件夹路径"
    echo ""
    echo "使用方法:"
    echo "  bash $0 /path/to/your/files"
    echo ""
    echo "示例:"
    echo "  bash $0 /home/user/my_course_data"
    echo ""
    exit 1
fi

INPUT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 输出目录
WORK_DIR="${SCRIPT_DIR}/../kb_build"
mkdir -p "${WORK_DIR}"

STEP1_OUTPUT="${WORK_DIR}/step1_extracted.json"
STEP2_OUTPUT="${WORK_DIR}/step2_cleaned.json"
STEP3_OUTPUT="${WORK_DIR}/step3_corpus.jsonl"
INDEX_DIR="${SCRIPT_DIR}/../../indexes/tum_kb_1500"

# 检查输入目录
if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ 输入目录不存在: ${INPUT_DIR}"
    exit 1
fi

# 统计文件
file_count=$(find "${INPUT_DIR}" -type f \( -name "*.pdf" -o -name "*.csv" -o -name "*.txt" \) | wc -l)

if [ "$file_count" -eq 0 ]; then
    echo "❌ 输入目录中没有找到PDF、CSV或TXT文件"
    exit 1
fi

echo "📁 输入目录: ${INPUT_DIR}"
echo "   找到 $file_count 个文件"
echo ""
echo "📁 工作目录: ${WORK_DIR}"
echo "📁 索引目录: ${INDEX_DIR}"
echo ""

read -p "继续吗? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ============================================================
# 阶段1: 结构化 - 提取原始文本
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段1/5: 结构化 - 提取原始文本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "${SCRIPT_DIR}/fixed_step1_extract.py" \
    --input "${INPUT_DIR}" \
    --output "${STEP1_OUTPUT}"

if [ $? -ne 0 ]; then
    echo "❌ 阶段1失败"
    exit 1
fi

echo ""
read -p "阶段1完成，按Enter继续..." dummy

# ============================================================
# 阶段2: 数据清洗（独立步骤）
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段2/5: 数据清洗"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "${SCRIPT_DIR}/fixed_step2_clean.py" \
    --input "${STEP1_OUTPUT}" \
    --output "${STEP2_OUTPUT}"

if [ $? -ne 0 ]; then
    echo "❌ 阶段2失败"
    exit 1
fi

echo ""
read -p "阶段2完成，按Enter继续..." dummy

# ============================================================
# 阶段3: 分块（1500字符）
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段3/5: 分块（1500字符）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "${SCRIPT_DIR}/fixed_step3_chunk.py" \
    --input "${STEP2_OUTPUT}" \
    --output "${STEP3_OUTPUT}" \
    --chunk_size 1500 \
    --overlap 150

if [ $? -ne 0 ]; then
    echo "❌ 阶段3失败"
    exit 1
fi

echo ""
read -p "阶段3完成，按Enter继续..." dummy

# ============================================================
# 阶段4: 构建索引
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段4/5: 构建索引"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "这一步会下载AI模型（~1GB），可能需要几分钟..."
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 检测到GPU"
    USE_FP16="--use_fp16"
    BATCH_SIZE=128
else
    echo "💻 使用CPU模式"
    USE_FP16=""
    BATCH_SIZE=32
fi

python3 -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-base \
    --corpus_path "${STEP3_OUTPUT}" \
    --save_dir "${INDEX_DIR}" \
    ${USE_FP16} \
    --max_length 512 \
    --batch_size ${BATCH_SIZE} \
    --pooling_method mean \
    --faiss_type Flat

if [ $? -ne 0 ]; then
    echo "❌ 阶段4失败"
    exit 1
fi

echo ""
read -p "阶段4完成，按Enter继续..." dummy

# ============================================================
# 阶段5: 测试
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段5/5: 测试检索"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "${SCRIPT_DIR}/real_pipeline_step3_test.py" \
    --index_path "${INDEX_DIR}" \
    --model_path intfloat/multilingual-e5-base

# ============================================================
# 完成
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   ✅ 知识库创建完成！                             ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "📊 生成的文件:"
echo "   1. 原始文本: ${STEP1_OUTPUT}"
echo "   2. 清洗文本: ${STEP2_OUTPUT}"
echo "   3. 分块语料库: ${STEP3_OUTPUT}"
echo "   4. 检索索引: ${INDEX_DIR}/"
echo ""
echo "📈 参数:"
echo "   - 块大小: 1500字符"
echo "   - 重叠: 150字符"
echo "   - 模型: multilingual-e5-base"
echo ""
echo "🚀 下一步:"
echo "   1. 启动检索服务:"
echo "      cd ../../benchmark/retriever"
echo "      python src/retrive_server.py --index_path ${INDEX_DIR}"
echo ""
echo "   2. 测试查询:"
echo "      curl -X POST http://localhost:5000/search \\"
echo "           -H 'Content-Type: application/json' \\"
echo "           -d '{\"query\": \"prerequisites for AI course\"}'"
echo ""
