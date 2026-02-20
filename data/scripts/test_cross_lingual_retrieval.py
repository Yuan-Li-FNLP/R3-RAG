"""
测试跨语言检索：英语查询 → 德语文档
"""

import requests
import json
import argparse
from typing import List, Dict


class CrossLingualRetrieverTest:
    """
    跨语言检索测试器
    """

    def __init__(self, retriever_host: str = "localhost", retriever_port: int = 5000):
        self.base_url = f"http://{retriever_host}:{retriever_port}"

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        发送检索请求

        Args:
            query: 查询文本（英语）
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        url = f"{self.base_url}/search"
        payload = {
            "query": query,
            "top_k": top_k
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ 检索失败: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"❌ 请求错误: {e}")
            return []

    def display_results(self, query: str, results: List[Dict]):
        """
        显示检索结果
        """
        print(f"\n{'='*80}")
        print(f"📝 查询 (英语): {query}")
        print(f"{'='*80}\n")

        if not results:
            print("❌ 没有找到结果")
            return

        for i, result in enumerate(results, 1):
            print(f"🔹 结果 {i} (ID: {result.get('id', 'N/A')})")
            print(f"   相似度: {result.get('score', 0):.4f}")

            contents = result.get('contents', '')

            # 检测是否包含德语（简单判断：包含umlauts）
            has_german = any(char in contents for char in 'äöüßÄÖÜ')
            language_tag = "🇩🇪 德语" if has_german else "🇬🇧 英语"

            print(f"   语言: {language_tag}")
            print(f"   内容:")

            # 显示前200字符
            preview = contents[:200] + "..." if len(contents) > 200 else contents
            for line in preview.split('\n'):
                print(f"      {line}")

            print()

    def run_test_suite(self):
        """
        运行测试用例集
        """
        print("\n" + "="*80)
        print("🧪 跨语言检索测试套件")
        print("   测试: 英语查询 → 德语文档")
        print("="*80)

        test_cases = [
            {
                "query": "What are the prerequisites for the AI course?",
                "expected_de_terms": ["Voraussetzungen", "Lineare Algebra", "Informatik"],
                "description": "查询AI课程的先修要求"
            },
            {
                "query": "How many ECTS credits does the database course have?",
                "expected_de_terms": ["ECTS", "Datenbank"],
                "description": "查询数据库课程的学分"
            },
            {
                "query": "Which courses are offered in winter semester?",
                "expected_de_terms": ["Wintersemester", "WS"],
                "description": "查询冬季学期课程"
            },
            {
                "query": "Tell me about machine learning courses",
                "expected_de_terms": ["Machine Learning", "Maschinelles Lernen"],
                "description": "查询机器学习课程"
            }
        ]

        results_summary = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'─'*80}")
            print(f"测试 {i}/{len(test_cases)}: {test_case['description']}")
            print(f"{'─'*80}")

            results = self.search(test_case['query'], top_k=3)
            self.display_results(test_case['query'], results)

            # 验证是否包含期望的德语术语
            found_german = False
            if results:
                all_contents = ' '.join([r.get('contents', '') for r in results])
                for term in test_case['expected_de_terms']:
                    if term.lower() in all_contents.lower():
                        found_german = True
                        break

            status = "✅ 通过" if found_german and results else "❌ 失败"
            results_summary.append({
                "test": test_case['description'],
                "status": status,
                "found_results": len(results) > 0,
                "found_german": found_german
            })

            print(f"状态: {status}")

        # 打印总结
        print(f"\n{'='*80}")
        print("📊 测试总结")
        print(f"{'='*80}\n")

        passed = sum(1 for r in results_summary if "✅" in r['status'])
        total = len(results_summary)

        for result in results_summary:
            print(f"{result['status']} {result['test']}")

        print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)\n")

        if passed == total:
            print("🎉 所有测试通过！跨语言检索工作正常。")
        else:
            print("⚠️  部分测试失败。可能的原因:")
            print("   1. 检索服务未启动")
            print("   2. 索引未构建或路径错误")
            print("   3. 模型不支持跨语言检索（需要使用multilingual模型）")


def main():
    parser = argparse.ArgumentParser(description='测试跨语言检索（英语→德语）')
    parser.add_argument('--host', default='localhost', help='检索服务host')
    parser.add_argument('--port', type=int, default=5000, help='检索服务端口')
    parser.add_argument('--query', help='单个查询测试')

    args = parser.parse_args()

    tester = CrossLingualRetrieverTest(args.host, args.port)

    if args.query:
        # 单个查询测试
        results = tester.search(args.query, top_k=5)
        tester.display_results(args.query, results)
    else:
        # 运行完整测试套件
        tester.run_test_suite()


if __name__ == "__main__":
    main()
