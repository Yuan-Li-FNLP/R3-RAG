"""
真实的知识库创建流程 - 阶段3: 测试检索

作用：
- 测试构建好的索引是否工作正常
- 验证能否检索到相关文档

使用方法：
    python real_pipeline_step3_test.py --index_path indexes/my_kb
"""

import sys
import argparse

# 添加FlashRAG到路径
sys.path.insert(0, '/home/user/R3-RAG/tool/FlashRAG')

try:
    from flashrag.retriever import Retriever
except ImportError:
    print("❌ 未找到FlashRAG！请检查路径或安装flashrag-pip")
    sys.exit(1)

def test_retrieval(index_path, model_path):
    """测试检索功能"""

    print("="*60)
    print("阶段3: 测试检索")
    print("="*60)
    print(f"索引路径: {index_path}")
    print(f"模型: {model_path}")
    print()

    # 加载检索器
    print("加载检索器...")
    try:
        retriever = Retriever(
            method="e5",
            index_path=index_path,
            model_path=model_path
        )
        print("✅ 检索器加载成功")
    except Exception as e:
        print(f"❌ 检索器加载失败: {e}")
        return

    print()

    # 测试查询（德语+英语混合）
    test_queries = [
        {
            "query": "Was sind die Voraussetzungen für den KI-Kurs?",
            "description": "查询课程先修要求（德语）"
        },
        {
            "query": "What is BFS algorithm?",
            "description": "查询BFS算法（英语）"
        },
        {
            "query": "Wie viele ECTS hat der Kurs?",
            "description": "查询学分（德语）"
        },
        {
            "query": "machine learning course content",
            "description": "查询机器学习课程内容（英语）"
        }
    ]

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]

        print("="*60)
        print(f"测试 {i}/{len(test_queries)}: {description}")
        print("="*60)
        print(f"查询: {query}")
        print()

        try:
            results = retriever.search(query, top_k=3)

            if not results:
                print("⚠️  未找到任何结果")
                continue

            print(f"✅ 找到 {len(results)} 个结果:\n")

            for j, result in enumerate(results, 1):
                # 检测语言
                contents = result.get('contents', '')
                has_german = any(char in contents for char in 'äöüßÄÖÜ')
                lang_tag = "🇩🇪" if has_german else "🇬🇧"

                print(f"  结果 {j} (ID: {result.get('id', 'N/A')}) {lang_tag}")
                print(f"  相似度: {result.get('score', 'N/A'):.4f}" if 'score' in result else "  相似度: N/A")

                # 显示内容预览
                preview = contents[:200].replace('\n', ' ')
                print(f"  内容: {preview}...")
                print()

        except Exception as e:
            print(f"  ❌ 检索失败: {e}")

        print()

    # 总结
    print("="*60)
    print("✅ 测试完成！")
    print("="*60)
    print()
    print("如果所有查询都能返回相关结果，说明知识库构建成功！")
    print()
    print("下一步:")
    print("  1. 集成到R3-RAG系统")
    print("  2. 启动检索服务: python ../../benchmark/retriever/src/retrive_server.py")
    print("  3. 测试完整RAG流程")
    print()

def main():
    parser = argparse.ArgumentParser(description='阶段3: 测试检索')
    parser.add_argument('--index_path', required=True, help='索引目录路径')
    parser.add_argument('--model_path', default='intfloat/multilingual-e5-base',
                       help='Embedding模型路径')

    args = parser.parse_args()

    test_retrieval(args.index_path, args.model_path)

if __name__ == "__main__":
    main()
