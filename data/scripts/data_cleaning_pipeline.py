"""
TUM课程知识库数据清洗与结构化流程
处理来自TUMonline, Moodle, PDF手册, FPSO的原始数据
"""

import re
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import unicodedata

class TUMDataCleaner:
    """
    TUM课程数据清洗器
    """

    # 常见的TUM课程代码格式
    COURSE_CODE_PATTERN = re.compile(r'\b[A-Z]{2,3}\d{4}[A-Z]?\b')

    # ECTS学分模式
    ECTS_PATTERN = re.compile(r'(\d+)\s*ECTS')

    # 学期模式
    SEMESTER_PATTERN = re.compile(r'(Wintersemester|Sommersemester|WS|SS)')

    def __init__(self):
        self.statistics = {
            'processed': 0,
            'cleaned': 0,
            'errors': []
        }

    def clean_text(self, text: str) -> str:
        """
        清洗文本：去除PDF提取的噪音、统一空白符

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 1. 统一Unicode字符（处理德语特殊字符）
        text = unicodedata.normalize('NFC', text)

        # 2. 去除PDF常见的噪音字符
        # - 去除零宽字符
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)

        # - 去除页码（如 "Page 1 of 10"）
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)

        # - 去除水印文本
        text = re.sub(r'DRAFT|CONFIDENTIAL', '', text, flags=re.IGNORECASE)

        # 3. 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 4. 去除多余的空白
        # - 多个空格压缩为一个
        text = re.sub(r' +', ' ', text)

        # - 多个换行压缩为最多两个
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 5. 去除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # 6. 修复常见的PDF提取错误
        # - 修复连字符断词（德语常见）
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

        return text.strip()

    def normalize_course_code(self, code: str) -> Optional[str]:
        """
        标准化课程代码

        Examples:
            "in2064" -> "IN2064"
            "IN 2064" -> "IN2064"
            "IN2064-01" -> "IN2064"
        """
        if not code:
            return None

        # 转大写并去除空格
        code = code.upper().replace(' ', '')

        # 去除后缀（如 -01, -W24）
        code = re.sub(r'-\d+$', '', code)
        code = re.sub(r'-[A-Z]\d{2}$', '', code)

        # 验证格式
        match = self.COURSE_CODE_PATTERN.match(code)
        if match:
            return match.group(0)

        return None

    def extract_ects(self, text: str) -> Optional[int]:
        """
        从文本中提取ECTS学分

        Examples:
            "6 ECTS" -> 6
            "ECTS: 6" -> 6
            "Credits: 6 ECTS" -> 6
        """
        match = self.ECTS_PATTERN.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    def extract_semesters(self, text: str) -> List[str]:
        """
        提取学期信息

        Returns:
            List of semesters (e.g., ["WS", "SS"])
        """
        semesters = []
        matches = self.SEMESTER_PATTERN.findall(text)

        for match in matches:
            if match in ['Wintersemester', 'WS']:
                if 'WS' not in semesters:
                    semesters.append('WS')
            elif match in ['Sommersemester', 'SS']:
                if 'SS' not in semesters:
                    semesters.append('SS')

        return semesters

    def extract_prerequisites(self, text: str) -> Dict[str, List[str]]:
        """
        提取先修课程

        Returns:
            {
                'mandatory': ['IN0001', 'MA0001'],
                'recommended': ['IN2222']
            }
        """
        prerequisites = {
            'mandatory': [],
            'recommended': []
        }

        # 查找课程代码
        course_codes = self.COURSE_CODE_PATTERN.findall(text)

        # 区分必修和推荐
        text_lower = text.lower()

        for code in course_codes:
            normalized_code = self.normalize_course_code(code)
            if not normalized_code:
                continue

            # 检查上下文判断是否为必修
            # 查找课程代码周围的文本
            pattern = re.compile(rf'.{{0,50}}{re.escape(code)}.{{0,50}}', re.IGNORECASE)
            context_match = pattern.search(text)

            if context_match:
                context = context_match.group(0).lower()

                if any(word in context for word in ['mandatory', 'required', 'pflicht', 'voraussetzung']):
                    if normalized_code not in prerequisites['mandatory']:
                        prerequisites['mandatory'].append(normalized_code)
                elif any(word in context for word in ['recommended', 'empfohlen', 'hilfreich']):
                    if normalized_code not in prerequisites['recommended']:
                        prerequisites['recommended'].append(normalized_code)
                else:
                    # 默认为推荐
                    if normalized_code not in prerequisites['recommended']:
                        prerequisites['recommended'].append(normalized_code)

        return prerequisites

    def detect_language(self, text: str) -> str:
        """
        检测文本语言

        Returns:
            'de', 'en', or 'mixed'
        """
        if not text:
            return 'unknown'

        # 德语特征词
        german_words = [
            'der', 'die', 'das', 'und', 'mit', 'für', 'von', 'zu', 'im', 'am',
            'vorlesung', 'übung', 'prüfung', 'wintersemester', 'sommersemester',
            'modulbeschreibung', 'lernziele'
        ]

        # 英语特征词
        english_words = [
            'the', 'and', 'with', 'for', 'from', 'to', 'in', 'at',
            'lecture', 'exercise', 'exam', 'module', 'learning', 'outcomes'
        ]

        text_lower = text.lower()

        german_count = sum(1 for word in german_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)

        if german_count > english_count * 1.5:
            return 'de'
        elif english_count > german_count * 1.5:
            return 'en'
        else:
            return 'mixed'

    def clean_course_data(self, raw_data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        清洗单个课程的数据

        Args:
            raw_data: 原始课程数据
            source: 数据源 (tumonline|moodle|handbook|fpso)

        Returns:
            清洗后的结构化数据
        """
        cleaned = {
            'source': source,
            'raw_data': raw_data  # 保留原始数据用于调试
        }

        try:
            # 1. 清洗课程代码
            if 'course_code' in raw_data:
                cleaned['course_code'] = self.normalize_course_code(raw_data['course_code'])

            # 2. 清洗标题
            for lang in ['de', 'en']:
                key = f'title_{lang}'
                if key in raw_data:
                    cleaned[key] = self.clean_text(raw_data[key])

            # 3. 提取ECTS
            if 'ects' in raw_data:
                if isinstance(raw_data['ects'], int):
                    cleaned['ects'] = raw_data['ects']
                else:
                    cleaned['ects'] = self.extract_ects(str(raw_data['ects']))

            # 4. 清洗讲师名称
            if 'lecturer' in raw_data:
                lecturer = self.clean_text(raw_data['lecturer'])
                # 移除职称前缀如 "Prof. Dr. Dr."
                lecturer = re.sub(r'^(Prof\.\s*)?(Dr\.\s*)+', '', lecturer).strip()
                cleaned['lecturer'] = lecturer

            # 5. 提取学期
            if 'semester' in raw_data:
                if isinstance(raw_data['semester'], list):
                    cleaned['semester'] = raw_data['semester']
                else:
                    cleaned['semester'] = self.extract_semesters(str(raw_data['semester']))

            # 6. 处理先修课程
            if 'prerequisites' in raw_data:
                if isinstance(raw_data['prerequisites'], dict):
                    cleaned['prerequisites'] = raw_data['prerequisites']
                else:
                    cleaned['prerequisites'] = self.extract_prerequisites(str(raw_data['prerequisites']))

            # 7. 清洗长文本字段
            for field in ['description', 'learning_outcomes', 'syllabus']:
                if field in raw_data:
                    cleaned[field] = self.clean_text(raw_data[field])

                    # 检测语言
                    if cleaned[field]:
                        cleaned[f'{field}_language'] = self.detect_language(cleaned[field])

            # 8. 处理材料文件路径
            if 'materials' in raw_data and isinstance(raw_data['materials'], list):
                cleaned['materials'] = []
                for material in raw_data['materials']:
                    if isinstance(material, dict):
                        cleaned_material = {
                            'title': self.clean_text(material.get('title', '')),
                            'file_path': material.get('file_path', ''),
                            'file_type': material.get('file_type', '').lower()
                        }
                        cleaned['materials'].append(cleaned_material)

            self.statistics['cleaned'] += 1

        except Exception as e:
            self.statistics['errors'].append({
                'course_code': raw_data.get('course_code', 'unknown'),
                'error': str(e)
            })

        self.statistics['processed'] += 1
        return cleaned

    def validate_course_data(self, course_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        验证课程数据的完整性和正确性

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # 必需字段检查
        required_fields = ['course_code']
        for field in required_fields:
            if field not in course_data or not course_data[field]:
                issues.append(f"Missing required field: {field}")

        # 课程代码格式检查
        if 'course_code' in course_data:
            if not self.COURSE_CODE_PATTERN.match(course_data['course_code']):
                issues.append(f"Invalid course code format: {course_data['course_code']}")

        # ECTS合理性检查
        if 'ects' in course_data:
            ects = course_data['ects']
            if not isinstance(ects, int) or ects < 1 or ects > 30:
                issues.append(f"Invalid ECTS value: {ects}")

        # 至少有一个标题（德语或英语）
        if not course_data.get('title_de') and not course_data.get('title_en'):
            issues.append("Missing both German and English titles")

        # 学期格式检查
        if 'semester' in course_data:
            semesters = course_data['semester']
            if not isinstance(semesters, list):
                issues.append("Semester should be a list")
            elif not all(s in ['WS', 'SS'] for s in semesters):
                issues.append(f"Invalid semester values: {semesters}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取清洗统计信息
        """
        return {
            'total_processed': self.statistics['processed'],
            'successfully_cleaned': self.statistics['cleaned'],
            'error_count': len(self.statistics['errors']),
            'errors': self.statistics['errors']
        }


# 使用示例
if __name__ == "__main__":
    cleaner = TUMDataCleaner()

    # 示例：清洗TUMonline的原始数据
    raw_tumonline_data = {
        'course_code': 'in 2064',  # 需要标准化
        'title_en': '  Introduction to   AI  \n(with noise)',  # 需要清洗
        'title_de': 'Einführung in die künstliche Intelligenz',
        'ects': '6 ECTS',  # 需要提取数字
        'lecturer': 'Prof. Dr. Dr. Max Müller',  # 需要去除职称
        'semester': 'Wintersemester and Sommersemester',  # 需要解析
        'prerequisites': 'Prerequisites: IN0001 (mandatory), MA0001 (recommended)'
    }

    cleaned = cleaner.clean_course_data(raw_tumonline_data, 'tumonline')

    print("Cleaned data:")
    print(json.dumps(cleaned, indent=2, ensure_ascii=False))

    # 验证数据
    is_valid, issues = cleaner.validate_course_data(cleaned)
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")

    # 统计信息
    print("\nStatistics:")
    print(json.dumps(cleaner.get_statistics(), indent=2, ensure_ascii=False))
