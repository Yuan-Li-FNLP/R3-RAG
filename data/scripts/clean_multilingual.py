#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言、多格式数据清洗器
支持: PDF, HTML
语言: 德语 (DE), 英语 (EN)
"""

import json
import re
import unicodedata
import argparse

class MultilingualCleaner:
    """多语言文本清洗器"""

    def __init__(self):
        # 德语特有的停用词/噪音
        self.german_noise = [
            r'\bSeite\s+\d+\s+von\s+\d+\b',  # Seite X von Y
            r'\bTechnische Universität München\b',
            r'\bLehrstuhl für\b',
            r'\bWintersemester\s+\d{4}/\d{2,4}\b',
            r'\bSommersemester\s+\d{4}\b',
        ]

        # 英语特有的停用词/噪音
        self.english_noise = [
            r'\bPage\s+\d+\s+of\s+\d+\b',  # Page X of Y
            r'\bTechnical University of Munich\b',
            r'\bDepartment of\b',
            r'\bWinter Semester\s+\d{4}/\d{2,4}\b',
            r'\bSummer Semester\s+\d{4}\b',
        ]

        # 通用噪音
        self.common_noise = [
            r'\bDRAFT\b',
            r'\bCONFIDENTIAL\b',
            r'\bINTERNAL\b',
            r'\bTUM\b',
            r'\d+\s*/\s*\d+',  # 1/20 格式的页码
        ]

    def clean_common(self, text):
        """通用清洗（适用于所有语言和格式）"""

        # 1. 统一Unicode（重要：德语字符）
        text = unicodedata.normalize('NFC', text)

        # 2. 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 3. 去除零宽字符
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)

        # 4. 去除通用噪音
        for pattern in self.common_noise:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text

    def clean_pdf_specific(self, text):
        """PDF特有的清洗"""

        # 1. 修复PDF断词（德语和英语都有）
        # "Infor-\nmation" → "Information"
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

        # 2. 去除页码（PDF特有）
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # 单独一行的数字

        # 3. 去除PDF水印占位符
        text = re.sub(r'�', '', text)  # 无法识别的字符

        return text

    def clean_html_specific(self, text):
        """HTML特有的清洗"""

        # 1. 去除HTML实体残留
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)

        # 2. 去除HTML注释残留
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # 3. 去除多余的导航文本
        navigation_patterns = [
            r'Home\s*>\s*Courses\s*>',
            r'Startseite\s*>\s*Kurse\s*>',
            r'Breadcrumb:',
            r'Navigation:',
        ]
        for pattern in navigation_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 4. 去除HTML中常见的菜单项
        text = re.sub(r'\b(Login|Logout|Sign in|Sign out|Anmelden|Abmelden)\b', '', text)

        return text

    def clean_german(self, text):
        """德语特有的清洗"""

        # 去除德语噪音
        for pattern in self.german_noise:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 标准化德语课程代码
        # "IN 2064" → "IN2064"
        text = re.sub(r'\b([A-Z]{2,3})\s+(\d{4})\b', r'\1\2', text)

        # 标准化德语学分表示
        # "6 ECTS" 保持不变
        text = re.sub(r'(\d+)\s*ECTS\s*Credits?', r'\1 ECTS', text, flags=re.IGNORECASE)

        return text

    def clean_english(self, text):
        """英语特有的清洗"""

        # 去除英语噪音
        for pattern in self.english_noise:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 标准化英语课程代码（同德语）
        text = re.sub(r'\b([A-Z]{2,3})\s+(\d{4})\b', r'\1\2', text)

        # 标准化英语学分表示
        text = re.sub(r'(\d+)\s*ECTS\s*Credits?', r'\1 ECTS', text, flags=re.IGNORECASE)

        return text

    def compress_whitespace(self, text):
        """压缩空白字符"""

        # 多个空格 → 一个空格
        text = re.sub(r' +', ' ', text)

        # 多个换行 → 最多两个（保留段落）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除每行首尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # 去除空行
        lines = [line for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        return text.strip()

    def clean(self, text, language='unknown', format_type='pdf'):
        """
        完整的清洗流程

        Args:
            text: 原始文本
            language: 'de' (德语), 'en' (英语), 'unknown'
            format_type: 'pdf', 'html'
        """

        if not text:
            return ""

        # 第1步: 通用清洗
        text = self.clean_common(text)

        # 第2步: 格式特定清洗
        if format_type == 'pdf':
            text = self.clean_pdf_specific(text)
        elif format_type == 'html':
            text = self.clean_html_specific(text)

        # 第3步: 语言特定清洗
        if language == 'de':
            text = self.clean_german(text)
        elif language == 'en':
            text = self.clean_english(text)

        # 第4步: 压缩空白
        text = self.compress_whitespace(text)

        return text

def main():
    parser = argparse.ArgumentParser(description='多语言、多格式数据清洗')
    parser.add_argument('--input', required=True, help='输入JSON文件（extract_multiformat.py的输出）')
    parser.add_argument('--output', default='cleaned_multilingual.json', help='输出JSON文件')

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          多语言、多格式数据清洗器                                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"📁 输入: {args.input}")
    print(f"💾 输出: {args.output}")
    print()

    # 读取数据
    with open(args.input, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"读取了 {len(documents)} 个文档\n")

    cleaner = MultilingualCleaner()
    cleaned_docs = []
    total_chars_removed = 0

    stats = {
        'de_pdf': 0, 'de_html': 0,
        'en_pdf': 0, 'en_html': 0,
        'unknown': 0
    }

    print("开始清洗...")
    print("-" * 70)

    for i, doc in enumerate(documents, 1):
        source_file = doc['source_file']
        raw_text = doc['raw_text']
        language = doc.get('language', 'unknown')
        format_type = doc.get('format', 'pdf')

        # 语言标签
        lang_emoji = '🇩🇪' if language == 'de' else ('🇬🇧' if language == 'en' else '❓')
        format_emoji = '📄' if format_type == 'pdf' else '🌐'

        print(f"{i}. {format_emoji} {lang_emoji} {source_file}")
        print(f"   格式: {format_type.upper()} | 语言: {language.upper()}")
        print(f"   原始: {len(raw_text):,} 字符")

        # 清洗
        cleaned_text = cleaner.clean(raw_text, language, format_type)
        chars_removed = len(raw_text) - len(cleaned_text)
        total_chars_removed += chars_removed

        print(f"   清洗后: {len(cleaned_text):,} 字符")
        print(f"   去除: {chars_removed:,} 字符 ({chars_removed/len(raw_text)*100:.1f}%)")

        # 显示清洗示例
        if chars_removed > 100:
            print(f"   示例前: {raw_text[:100]}...")
            print(f"   示例后: {cleaned_text[:100]}...")

        cleaned_docs.append({
            "source_file": source_file,
            "format": format_type,
            "language": language,
            "cleaned_text": cleaned_text,
            "num_chars": len(cleaned_text)
        })

        # 统计
        key = f"{language}_{format_type}"
        if key in stats:
            stats[key] += 1
        else:
            stats['unknown'] += 1

        print()

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   ✅ 清洗完成！                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"📊 统计信息:")
    print(f"   总文档数: {len(cleaned_docs)}")
    print(f"   总字符去除: {total_chars_removed:,}")
    print()
    print(f"   分类统计:")
    print(f"     🇩🇪📄 德语PDF: {stats['de_pdf']}")
    print(f"     🇩🇪🌐 德语HTML: {stats['de_html']}")
    print(f"     🇬🇧📄 英语PDF: {stats['en_pdf']}")
    print(f"     🇬🇧🌐 英语HTML: {stats['en_html']}")
    if stats['unknown'] > 0:
        print(f"     ❓ 未知: {stats['unknown']}")
    print()
    print(f"💾 保存到: {args.output}")
    print()

if __name__ == "__main__":
    main()
