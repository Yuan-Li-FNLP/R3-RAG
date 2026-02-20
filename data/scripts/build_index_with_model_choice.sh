#!/bin/bash
# TUM知识库索引构建脚本 - 支持选择Embedding模型

set -e

# ============================================
# 配置区域
# ============================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CORPUS_FILE="final_corpus.jsonl"

# 检查语料库文件
if [ ! -f "$CORPUS_FILE" ]; then
    echo "❌ 错误: 找不到 $CORPUS_FILE"
    echo "请先运行: bash quick_process_tum_data.sh"
    exit 1
fi

# ============================================
# 选择模型
# ============================================
echo "╔════════════════════════════════════════════════════╗"
echo "║     选择Embedding模型                             ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "1) multilingual-e5-base (推荐) ⭐⭐⭐⭐"
echo "   - 速度：快"
echo "   - 性能：很好 (82-85%)"
echo "   - 内存：~1.1GB"
echo "   - 适合：快速测试、CPU环境"
echo ""
echo "2) multilingual-e5-large (最佳性能) ⭐⭐⭐⭐⭐"
echo "   - 速度：中等"
echo "   - 性能：优秀 (87-90%)"
echo "   - 内存：~2.2GB"
echo "   - 适合：生产环境、有GPU"
echo ""
echo "3) multilingual-e5-small (轻量级) ⭐⭐⭐"
echo "   - 速度：非常快"
echo "   - 性能：不错 (78-82%)"
echo "   - 内存：~0.5GB"
echo "   - 适合：低配置机器"
echo ""

read -p "请选择 (1/2/3) [默认=1]: " choice
choice=${choice:-1}

case $choice in
    1)
        MODEL_PATH="intfloat/multilingual-e5-base"
        MODEL_NAME="base"
        VECTOR_DIM=768
        ;;
    2)
        MODEL_PATH="intfloat/multilingual-e5-large"
        MODEL_NAME="large"
        VECTOR_DIM=1024
        ;;
    3)
        MODEL_PATH="intfloat/multilingual-e5-small"
        MODEL_NAME="small"
        VECTOR_DIM=384
        ;;
    *)
        echo "无效选择，使用默认: base"
        MODEL_PATH="intfloat/multilingual-e5-base"
        MODEL_NAME="base"
        VECTOR_DIM=768
        ;;
esac

INDEX_DIR="${SCRIPT_DIR}/../../indexes/tum_kb_${MODEL_NAME}"

echo ""
echo "✅ 已选择: $MODEL_PATH"
echo "   索引保存到: $INDEX_DIR"
echo ""

# ============================================
# 检测GPU
# ============================================
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 检测到GPU - 使用FP16加速"
    USE_FP16="--use_fp16"
    BATCH_SIZE=128
else
    echo "💻 CPU模式"
    USE_FP16=""

    # 根据模型调整batch size
    case $MODEL_NAME in
        "small")
            BATCH_SIZE=128
            ;;
        "base")
            BATCH_SIZE=64
            ;;
        "large")
            BATCH_SIZE=32
            ;;
    esac
fi

echo "   Batch size: $BATCH_SIZE"
echo ""

# ============================================
# 构建索引
# ============================================
echo "开始构建索引..."
echo ""

python3 -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path "$MODEL_PATH" \
    --corpus_path "$CORPUS_FILE" \
    --save_dir "$INDEX_DIR" \
    ${USE_FP16} \
    --batch_size ${BATCH_SIZE} \
    --pooling_method mean \
    --faiss_type Flat

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   ✅ 索引构建完成！                               ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "📊 索引信息:"
echo "   - 模型: $MODEL_PATH"
echo "   - 向量维度: $VECTOR_DIM"
echo "   - 保存位置: $INDEX_DIR"
echo ""
echo "🔍 测试索引:"
echo "   python3 test_retrieval.py --index_path $INDEX_DIR"
echo ""
