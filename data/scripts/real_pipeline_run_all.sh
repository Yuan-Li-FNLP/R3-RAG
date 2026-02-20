#!/bin/bash
# 真实的知识库创建流程 - 一键运行所有步骤
#
# 使用方法:
#   1. 把你的文件放到一个文件夹（比如 /home/user/my_course_data）
#   2. 运行: bash real_pipeline_run_all.sh /home/user/my_course_data

set -e  # 遇到错误立即退出

echo "╔════════════════════════════════════════════════════╗"
echo "║   TUM课程知识库 - 真实完整流程                    ║"
echo "║   从原始文件到可用知识库                          ║"
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
OUTPUT_DIR="${SCRIPT_DIR}/../output"
CORPUS_FILE="${OUTPUT_DIR}/corpus.jsonl"
INDEX_DIR="${SCRIPT_DIR}/../../indexes/tum_kb"

mkdir -p "${OUTPUT_DIR}"

# 检查输入目录
if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ 输入目录不存在: ${INPUT_DIR}"
    exit 1
fi

# 统计文件
file_count=$(find "${INPUT_DIR}" -type f \( -name "*.pdf" -o -name "*.csv" -o -name "*.txt" \) | wc -l)

if [ "$file_count" -eq 0 ]; then
    echo "❌ 输入目录中没有找到PDF、CSV或TXT文件"
    echo "   目录: ${INPUT_DIR}"
    exit 1
fi

echo "📁 输入目录: ${INPUT_DIR}"
echo "   找到 $file_count 个文件"
echo ""
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "📁 索引目录: ${INDEX_DIR}"
echo ""

read -p "继续吗? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段1/4: 结构化 - 提取文本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "${SCRIPT_DIR}/real_pipeline_step1_extract.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}/extracted_texts.json"

if [ $? -ne 0 ]; then
    echo "❌ 阶段1失败"
    exit 1
fi

echo ""
read -p "阶段1完成，按Enter继续..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段2/4: 分块 - 切成小段"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "${SCRIPT_DIR}/real_pipeline_step2_chunk.py" \
    --input "${OUTPUT_DIR}/extracted_texts.json" \
    --output "${CORPUS_FILE}" \
    --chunk_size 500 \
    --overlap 50

if [ $? -ne 0 ]; then
    echo "❌ 阶段2失败"
    exit 1
fi

echo ""
read -p "阶段2完成，按Enter继续..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段3/4: 构建索引"
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
    --corpus_path "${CORPUS_FILE}" \
    --save_dir "${INDEX_DIR}" \
    ${USE_FP16} \
    --max_length 512 \
    --batch_size ${BATCH_SIZE} \
    --pooling_method mean \
    --faiss_type Flat

if [ $? -ne 0 ]; then
    echo "❌ 阶段3失败"
    exit 1
fi

echo ""
read -p "阶段3完成，按Enter继续..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段4/4: 测试检索"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "${SCRIPT_DIR}/real_pipeline_step3_test.py" \
    --index_path "${INDEX_DIR}" \
    --model_path intfloat/multilingual-e5-base

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   ✅ 知识库创建完成！                             ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "📊 生成的文件:"
echo "   1. 结构化数据: ${OUTPUT_DIR}/extracted_texts.json"
echo "   2. 分块语料库: ${CORPUS_FILE}"
echo "   3. 检索索引: ${INDEX_DIR}/"
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
echo "   3. 启动完整RAG系统:"
echo "      cd ../../startup"
echo "      bash server.sh"
echo ""
