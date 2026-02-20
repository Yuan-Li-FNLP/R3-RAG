#!/bin/bash
# 傻瓜式知识库构建脚本 - 不需要任何编程知识！
# 作者：Claude
# 使用方法：把你的文件放到 data/my_files/ 然后运行 bash easy_build.sh

set -e

echo "╔════════════════════════════════════════════════════╗"
echo "║   TUM课程知识库 - 傻瓜式构建脚本                  ║"
echo "║   不需要编程知识，跟着提示操作即可                ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MY_FILES_DIR="${SCRIPT_DIR}/../my_files"
OUTPUT_DIR="${SCRIPT_DIR}/../corpus"
INDEX_DIR="${SCRIPT_DIR}/../../indexes/my_tum_kb"

# 步骤1: 检查文件夹
echo "📁 步骤1: 检查你的文件..."
echo "─────────────────────────────────────────────────────"

if [ ! -d "${MY_FILES_DIR}" ]; then
    echo "⚠️  未找到文件夹: ${MY_FILES_DIR}"
    echo ""
    echo "请按以下步骤操作:"
    echo "  1. 创建文件夹: mkdir -p ${MY_FILES_DIR}"
    echo "  2. 把你的所有文件（PDF、CSV、TXT等）放进去"
    echo "  3. 重新运行此脚本"
    echo ""
    read -p "要我帮你创建文件夹吗? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "${MY_FILES_DIR}"
        echo "✅ 文件夹已创建: ${MY_FILES_DIR}"
        echo ""
        echo "现在请:"
        echo "  1. 把你的文件复制到这个文件夹"
        echo "  2. 重新运行: bash $0"
        exit 0
    else
        exit 1
    fi
fi

# 检查是否有文件
file_count=$(find "${MY_FILES_DIR}" -type f | wc -l)
if [ "$file_count" -eq 0 ]; then
    echo "❌ 文件夹是空的！"
    echo "   请把你的文件（PDF、CSV、TXT等）放到:"
    echo "   ${MY_FILES_DIR}"
    exit 1
fi

echo "✅ 找到 $file_count 个文件"
echo ""
echo "文件列表:"
ls -lh "${MY_FILES_DIR}"
echo ""
read -p "这些文件看起来对吗? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请检查文件后重新运行"
    exit 1
fi

# 步骤2: 提取文本
echo ""
echo "📝 步骤2: 提取文本内容..."
echo "─────────────────────────────────────────────────────"

mkdir -p "${OUTPUT_DIR}"
TEMP_CORPUS="${OUTPUT_DIR}/temp_corpus.jsonl"
> "${TEMP_CORPUS}"  # 清空文件

doc_id=0

# 处理每个文件
for file in "${MY_FILES_DIR}"/*; do
    [ -f "$file" ] || continue

    filename=$(basename "$file")
    echo "处理: $filename"

    # 根据文件类型提取文本
    extension="${filename##*.}"
    extension_lower=$(echo "$extension" | tr '[:upper:]' '[:lower:]')

    case "$extension_lower" in
        pdf)
            # 检查是否安装了pdftotext
            if command -v pdftotext &> /dev/null; then
                text=$(pdftotext "$file" - 2>/dev/null || echo "")
            else
                echo "  ⚠️  未安装pdftotext，跳过PDF文件"
                echo "     安装: sudo apt-get install poppler-utils"
                continue
            fi
            ;;
        txt|md)
            text=$(cat "$file")
            ;;
        csv)
            # 简单处理CSV：每行作为一个文档
            while IFS= read -r line; do
                if [ ! -z "$line" ]; then
                    echo "{\"id\": \"$doc_id\", \"contents\": \"$(echo "$line" | sed 's/"/\\"/g')\"}" >> "${TEMP_CORPUS}"
                    doc_id=$((doc_id + 1))
                fi
            done < "$file"
            continue
            ;;
        *)
            echo "  ⚠️  不支持的文件类型: $extension_lower"
            continue
            ;;
    esac

    # 将文本分块（每500字符一块）
    if [ ! -z "$text" ]; then
        echo "$text" | fold -w 500 -s | while IFS= read -r chunk; do
            if [ ! -z "$chunk" ]; then
                # 转义引号
                escaped_chunk=$(echo "$chunk" | sed 's/"/\\"/g' | tr -d '\n')
                echo "{\"id\": \"$doc_id\", \"contents\": \"$escaped_chunk\"}" >> "${TEMP_CORPUS}"
                doc_id=$((doc_id + 1))
            fi
        done
    fi
done

# 统计
total_docs=$(wc -l < "${TEMP_CORPUS}")
echo ""
echo "✅ 提取完成，共 $total_docs 个文档块"

if [ "$total_docs" -eq 0 ]; then
    echo "❌ 没有提取到任何内容！"
    echo "   可能的原因:"
    echo "   - PDF文件是扫描版（需要OCR）"
    echo "   - 文件格式不支持"
    exit 1
fi

# 显示示例
echo ""
echo "示例文档:"
head -n 1 "${TEMP_CORPUS}" | python3 -m json.tool 2>/dev/null || head -n 1 "${TEMP_CORPUS}"
echo ""

# 步骤3: 构建索引
echo ""
echo "🔨 步骤3: 构建检索索引..."
echo "─────────────────────────────────────────────────────"
echo "这一步会下载AI模型（约1-2GB），可能需要几分钟"
echo ""
read -p "继续吗? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消。你可以稍后运行此脚本继续。"
    exit 0
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到Python3！请先安装Python 3.8+"
    exit 1
fi

# 检查flashrag
if ! python3 -c "import flashrag" 2>/dev/null; then
    echo "⚠️  未安装FlashRAG，正在安装..."
    pip install flashrag-pip || {
        echo "❌ 安装失败！请手动运行: pip install flashrag-pip"
        exit 1
    }
fi

echo "正在构建索引..."
python3 -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-base \
    --corpus_path "${TEMP_CORPUS}" \
    --save_dir "${INDEX_DIR}" \
    --batch_size 32 \
    --pooling_method mean \
    --faiss_type Flat \
    2>&1 | tee build_log.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 索引构建成功！"
else
    echo ""
    echo "❌ 索引构建失败，请查看 build_log.txt"
    exit 1
fi

# 步骤4: 测试
echo ""
echo "🧪 步骤4: 测试检索..."
echo "─────────────────────────────────────────────────────"

# 创建测试脚本
cat > /tmp/test_retrieval.py <<'PYTHON'
import sys
sys.path.insert(0, '/home/user/R3-RAG/tool/FlashRAG')

from flashrag.retriever import Retriever

index_path = sys.argv[1]

try:
    retriever = Retriever(
        method="e5",
        index_path=index_path,
        model_path="intfloat/multilingual-e5-base"
    )

    # 测试查询
    queries = [
        "AI course",
        "prerequisites",
        "ECTS credits"
    ]

    print("\n测试查询:")
    for query in queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, top_k=2)
        if results:
            print(f"✅ 找到 {len(results)} 个结果")
            print(f"   第1个结果: {results[0]['contents'][:100]}...")
        else:
            print("⚠️  未找到结果")

    print("\n✅ 测试通过！")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    sys.exit(1)
PYTHON

python3 /tmp/test_retrieval.py "${INDEX_DIR}" || {
    echo "⚠️  测试失败，但索引已构建完成"
}

# 完成
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   ✅ 知识库构建完成！                             ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "📊 统计信息:"
echo "   - 处理文件数: $file_count"
echo "   - 文档块数: $total_docs"
echo "   - 索引位置: ${INDEX_DIR}"
echo ""
echo "🚀 下一步:"
echo "   1. 测试检索:"
echo "      python3 test_my_retrieval.py"
echo ""
echo "   2. 启动RAG服务:"
echo "      cd ../../startup"
echo "      bash server.sh"
echo ""
echo "   3. 问问题:"
echo "      \"What are the prerequisites for AI course?\""
echo "      \"Tell me about machine learning\""
echo ""
echo "💡 提示:"
echo "   - 如果要添加更多文件，把文件放到 ${MY_FILES_DIR}"
echo "   - 然后重新运行: bash $0"
echo ""
