#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试跨语言检索功能
验证 multilingual-e5 模型在德英混合场景下的表现
"""

import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_model(model_path):
    """测试模型的跨语言能力"""

    print(f"加载模型: {model_path}")
    model = SentenceTransformer(model_path)
    print("✅ 模型加载成功\n")

    # 测试用例：英语查询 vs 德语文档
    test_cases = [
        {
            "name": "测试1: 课程先决条件",
            "query_en": "What are the prerequisites for this course?",
            "doc_de": "Voraussetzungen: Lineare Algebra, Grundlagen der Informatik",
            "doc_en": "Prerequisites: Linear Algebra, Computer Science Fundamentals"
        },
        {
            "name": "测试2: 机器学习",
            "query_en": "machine learning algorithms",
            "doc_de": "maschinelles Lernen und neuronale Netze",
            "doc_en": "machine learning and neural networks"
        },
        {
            "name": "测试3: 学分信息",
            "query_de": "Wie viele ECTS Credits?",
            "doc_de": "Dieser Kurs bietet 6 ECTS Credits",
            "doc_en": "This course offers 6 ECTS credits"
        },
        {
            "name": "测试4: 考试形式",
            "query_en": "exam format",
            "doc_de": "Prüfungsform: Schriftliche Klausur",
            "doc_en": "Exam format: Written examination"
        }
    ]

    print("=" * 70)
    print("跨语言检索测试")
    print("=" * 70)
    print()

    for i, test in enumerate(test_cases, 1):
        print(f"【{test['name']}】")
        print()

        # 获取查询（英语或德语）
        if "query_en" in test:
            query = test["query_en"]
            query_lang = "🇬🇧 英语"
        else:
            query = test["query_de"]
            query_lang = "🇩🇪 德语"

        # 编码查询
        query_emb = model.encode(query, normalize_embeddings=True)

        # 编码文档
        doc_de_emb = model.encode(test["doc_de"], normalize_embeddings=True)
        doc_en_emb = model.encode(test["doc_en"], normalize_embeddings=True)

        # 计算相似度
        sim_de = cosine_similarity(query_emb, doc_de_emb)
        sim_en = cosine_similarity(query_emb, doc_en_emb)

        print(f"  查询 ({query_lang}): \"{query}\"")
        print()
        print(f"  🇩🇪 德语文档: \"{test['doc_de']}\"")
        print(f"     相似度: {sim_de:.4f} {'⭐' * int(sim_de * 5)}")
        print()
        print(f"  🇬🇧 英语文档: \"{test['doc_en']}\"")
        print(f"     相似度: {sim_en:.4f} {'⭐' * int(sim_en * 5)}")
        print()

        # 判断跨语言效果
        if query_lang == "🇬🇧 英语" and sim_de > 0.7:
            print(f"  ✅ 跨语言检索成功！英语查询能找到德语文档 (相似度: {sim_de:.2f})")
        elif query_lang == "🇩🇪 德语" and sim_en > 0.7:
            print(f"  ✅ 跨语言检索成功！德语查询能找到英语文档 (相似度: {sim_en:.2f})")
        elif sim_de > 0.7 or sim_en > 0.7:
            print(f"  ✅ 检索成功！")
        else:
            print(f"  ⚠️  相似度较低，可能需要调整")

        print()
        print("-" * 70)
        print()

    # 总结
    print("=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print()
    print(f"模型: {model_path}")
    print()
    print("✅ 支持的场景:")
    print("  - 英语查询 → 德语文档检索")
    print("  - 德语查询 → 英语文档检索")
    print("  - 英语查询 → 英语文档检索")
    print("  - 德语查询 → 德语文档检索")
    print()
    print("💡 建议:")
    print("  - 相似度 > 0.8: 优秀")
    print("  - 相似度 0.7-0.8: 很好")
    print("  - 相似度 0.6-0.7: 可以")
    print("  - 相似度 < 0.6: 需要改进")
    print()

def main():
    parser = argparse.ArgumentParser(description='测试跨语言检索')
    parser.add_argument(
        '--model',
        default='intfloat/multilingual-e5-base',
        choices=[
            'intfloat/multilingual-e5-small',
            'intfloat/multilingual-e5-base',
            'intfloat/multilingual-e5-large'
        ],
        help='选择模型'
    )

    args = parser.parse_args()

    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        Multilingual-E5 跨语言检索测试工具                        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    test_model(args.model)

if __name__ == '__main__':
    main()
