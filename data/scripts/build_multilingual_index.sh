#!/bin/bash
# 构建支持英语查询→德语文档检索的多语言索引

set -e

echo "======================================"
echo "Building Multilingual Index"
echo "English Query → German Documents"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_PATH="${SCRIPT_DIR}/../corpus/tum_courses_corpus.jsonl"
INDEX_DIR="${SCRIPT_DIR}/../../indexes/tum_multilingual"

# 检查corpus是否存在
if [ ! -f "${CORPUS_PATH}" ]; then
    echo "❌ Corpus文件不存在: ${CORPUS_PATH}"
    echo "请先运行: bash build_knowledge_base.sh"
    exit 1
fi

echo ""
echo "📊 配置信息:"
echo "  Corpus: ${CORPUS_PATH}"
echo "  索引目录: ${INDEX_DIR}"
echo ""

# 推荐的多语言模型
echo "🌐 推荐的多语言Embedding模型:"
echo "  1) intfloat/multilingual-e5-large (推荐)"
echo "  2) intfloat/multilingual-e5-base (速度快)"
echo "  3) sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (轻量)"
echo ""
read -p "请选择模型 (1/2/3) [默认: 1]: " model_choice

case ${model_choice} in
    2)
        MODEL_PATH="intfloat/multilingual-e5-base"
        ;;
    3)
        MODEL_PATH="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ;;
    *)
        MODEL_PATH="intfloat/multilingual-e5-large"
        ;;
esac

echo "✅ 选择的模型: ${MODEL_PATH}"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 检测到GPU，将使用GPU加速"
    USE_FP16="--use_fp16"
    BATCH_SIZE=256
else
    echo "💻 未检测到GPU，使用CPU模式"
    USE_FP16=""
    BATCH_SIZE=32
fi

echo ""
echo "🔨 开始构建索引..."
echo "这可能需要几分钟到几小时，取决于课程数量和硬件"
echo ""

# 构建FAISS索引
python3 -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path "${MODEL_PATH}" \
    --corpus_path "${CORPUS_PATH}" \
    --save_dir "${INDEX_DIR}" \
    ${USE_FP16} \
    --max_length 512 \
    --batch_size ${BATCH_SIZE} \
    --pooling_method mean \
    --faiss_type Flat

echo ""
echo "======================================"
echo "✅ 多语言索引构建完成！"
echo "======================================"
echo ""
echo "📁 索引文件位置:"
ls -lh "${INDEX_DIR}"
echo ""
echo "🧪 测试查询:"
echo "  英语: 'What are the prerequisites for the AI course?'"
echo "  应该能检索到德语文档: 'Voraussetzungen für KI-Kurs'"
echo ""
echo "🚀 下一步: 启动检索服务"
echo "  cd ../../benchmark/retriever"
echo "  python src/retrive_server.py --index_path ${INDEX_DIR}"
echo ""
