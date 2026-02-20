#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多格式文本提取器
支持: PDF, HTML, TXT
语言: 德语, 英语（自动检测）
"""

import fitz  # PyMuPDF
import json
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
import re

def detect_language(text):
    """
    简单的语言检测（德语 vs 英语）
    基于常见词汇
    """
    # 德语特征词
    german_words = ['der', 'die', 'das', 'und', 'ist', 'von', 'zu', 'mit', 'für', 'auf',
                    'werden', 'sich', 'auch', 'über', 'nach', 'können', 'werden']

    # 英语特征词
    english_words = ['the', 'and', 'is', 'to', 'of', 'in', 'for', 'with', 'on', 'that',
                     'are', 'this', 'from', 'will', 'can', 'be', 'have']

    text_lower = text.lower()

    german_count = sum(1 for word in german_words if f' {word} ' in text_lower)
    english_count = sum(1 for word in english_words if f' {word} ' in text_lower)

    if german_count > english_count:
        return 'de'
    elif english_count > german_count:
        return 'en'
    else:
        return 'unknown'

def extract_from_pdf(pdf_path):
    """从PDF提取原始文本"""
    print(f"  📄 PDF: {pdf_path.name}")

    try:
        doc = fitz.open(pdf_path)
        raw_text = ""

        for page_num, page in enumerate(doc, 1):
            raw_text += page.get_text()

            if page_num % 10 == 0:
                print(f"    进度: {page_num}/{len(doc)} 页")

        doc.close()

        # 检测语言
        lang = detect_language(raw_text[:2000])  # 用前2000字符检测

        return {
            "source_file": pdf_path.name,
            "format": "pdf",
            "language": lang,
            "raw_text": raw_text,
            "num_pages": len(doc),
            "num_chars": len(raw_text)
        }
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return None

def extract_from_html(html_path):
    """从HTML提取文本"""
    print(f"  🌐 HTML: {html_path.name}")

    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 使用BeautifulSoup解析
        soup = BeautifulSoup(html_content, 'html.parser')

        # 移除script和style标签
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # 提取文本
        raw_text = soup.get_text()

        # 检测语言
        lang = detect_language(raw_text[:2000])

        return {
            "source_file": html_path.name,
            "format": "html",
            "language": lang,
            "raw_text": raw_text,
            "num_chars": len(raw_text)
        }
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return None

def extract_from_txt(txt_path):
    """从TXT提取文本"""
    print(f"  📝 TXT: {txt_path.name}")

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 检测语言
        lang = detect_language(raw_text[:2000])

        return {
            "source_file": txt_path.name,
            "format": "txt",
            "language": lang,
            "raw_text": raw_text,
            "num_chars": len(raw_text)
        }
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='多格式文本提取器（PDF/HTML/TXT）')
    parser.add_argument('--input', required=True, help='输入文件夹')
    parser.add_argument('--output', default='extracted_multiformat.json', help='输出JSON文件')
    parser.add_argument('--formats', default='pdf,html,txt', help='要处理的格式（逗号分隔）')

    args = parser.parse_args()
    input_dir = Path(args.input)
    formats = args.formats.split(',')

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║              多格式文本提取器                                      ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"📁 输入目录: {input_dir}")
    print(f"💾 输出文件: {args.output}")
    print(f"📑 支持格式: {', '.join(formats)}")
    print()

    all_docs = []
    stats = {
        'pdf': {'count': 0, 'de': 0, 'en': 0},
        'html': {'count': 0, 'de': 0, 'en': 0},
        'txt': {'count': 0, 'de': 0, 'en': 0}
    }

    # 处理PDF
    if 'pdf' in formats:
        pdf_files = list(input_dir.glob("*.pdf"))
        if pdf_files:
            print(f"找到 {len(pdf_files)} 个PDF文件")
            print("-" * 70)
            for pdf_file in pdf_files:
                doc_data = extract_from_pdf(pdf_file)
                if doc_data:
                    all_docs.append(doc_data)
                    stats['pdf']['count'] += 1
                    if doc_data['language'] == 'de':
                        stats['pdf']['de'] += 1
                    elif doc_data['language'] == 'en':
                        stats['pdf']['en'] += 1
            print()

    # 处理HTML
    if 'html' in formats:
        html_files = list(input_dir.glob("*.html")) + list(input_dir.glob("*.htm"))
        if html_files:
            print(f"找到 {len(html_files)} 个HTML文件")
            print("-" * 70)
            for html_file in html_files:
                doc_data = extract_from_html(html_file)
                if doc_data:
                    all_docs.append(doc_data)
                    stats['html']['count'] += 1
                    if doc_data['language'] == 'de':
                        stats['html']['de'] += 1
                    elif doc_data['language'] == 'en':
                        stats['html']['en'] += 1
            print()

    # 处理TXT
    if 'txt' in formats:
        txt_files = list(input_dir.glob("*.txt"))
        if txt_files:
            print(f"找到 {len(txt_files)} 个TXT文件")
            print("-" * 70)
            for txt_file in txt_files:
                doc_data = extract_from_txt(txt_file)
                if doc_data:
                    all_docs.append(doc_data)
                    stats['txt']['count'] += 1
                    if doc_data['language'] == 'de':
                        stats['txt']['de'] += 1
                    elif doc_data['language'] == 'en':
                        stats['txt']['en'] += 1
            print()

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   ✅ 提取完成！                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"📊 统计信息:")
    print(f"   总文档数: {len(all_docs)}")
    print()

    for format_type, format_stats in stats.items():
        if format_stats['count'] > 0:
            print(f"   {format_type.upper()}:")
            print(f"     - 总数: {format_stats['count']}")
            print(f"     - 德语 (🇩🇪): {format_stats['de']}")
            print(f"     - 英语 (🇬🇧): {format_stats['en']}")
            print(f"     - 未知: {format_stats['count'] - format_stats['de'] - format_stats['en']}")
            print()

    print(f"💾 保存到: {args.output}")
    print()

if __name__ == "__main__":
    main()
