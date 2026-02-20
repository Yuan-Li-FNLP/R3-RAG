"""
TUM课程手册PDF解析器
专门处理课程手册PDF，提取结构化信息
"""

import re
import json
from typing import Dict, List, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("警告: PyMuPDF未安装。请运行: pip install PyMuPDF")
    fitz = None


class HandbookPDFParser:
    """
    解析TUM课程手册PDF
    """

    def __init__(self):
        if fitz is None:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

        self.courses = []
        self.current_course = None

        # 课程块开始标志（匹配课程代码）
        self.course_start_pattern = re.compile(r'^([A-Z]{2,3}\d{4}[A-Z]?)\s*[-–]\s*(.+)$', re.MULTILINE)

        # 字段模式
        self.field_patterns = {
            'ects': re.compile(r'ECTS[:\s]+(\d+)', re.IGNORECASE),
            'sws': re.compile(r'SWS[:\s]+(\d+)', re.IGNORECASE),
            'semester': re.compile(r'Semester[:\s]+(WS|SS|Wintersemester|Sommersemester)', re.IGNORECASE),
            'lecturer': re.compile(r'(?:Lecturer|Dozent)[:\s]+(.+?)(?:\n|$)', re.IGNORECASE),
            'language': re.compile(r'(?:Language|Sprache)[:\s]+(English|German|Deutsch|Englisch)', re.IGNORECASE),
            'module_type': re.compile(r'(?:Module Type|Modultyp)[:\s]+(Pflicht|Wahl|Mandatory|Elective)', re.IGNORECASE),
        }

        # 章节标题模式
        self.section_patterns = {
            'prerequisites': re.compile(r'(?:Prerequisites|Voraussetzungen)[:\s]*', re.IGNORECASE),
            'learning_outcomes': re.compile(r'(?:Learning Outcomes?|Lernziele?)[:\s]*', re.IGNORECASE),
            'description': re.compile(r'(?:Description|Beschreibung)[:\s]*', re.IGNORECASE),
            'assessment': re.compile(r'(?:Assessment|Prüfung|Examination)[:\s]*', re.IGNORECASE),
            'content': re.compile(r'(?:Content|Inhalt)[:\s]*', re.IGNORECASE),
            'literature': re.compile(r'(?:Literature|Literatur|Recommended Books)[:\s]*', re.IGNORECASE),
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF提取文本

        Args:
            pdf_path: PDF文件路径

        Returns:
            提取的文本
        """
        doc = fitz.open(pdf_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()

        doc.close()
        return text

    def split_into_course_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        将整个手册文本分割成单个课程块

        Returns:
            List of {course_code, course_title, content}
        """
        courses = []
        matches = list(self.course_start_pattern.finditer(text))

        for i, match in enumerate(matches):
            course_code = match.group(1)
            course_title = match.group(2).strip()

            # 确定当前课程块的结束位置
            start_pos = match.start()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)

            content = text[start_pos:end_pos]

            courses.append({
                'course_code': course_code,
                'course_title': course_title,
                'content': content
            })

        return courses

    def extract_field_value(self, text: str, field_name: str) -> Optional[str]:
        """
        从文本中提取特定字段的值
        """
        pattern = self.field_patterns.get(field_name)
        if not pattern:
            return None

        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        return None

    def extract_section_content(self, text: str, section_name: str) -> Optional[str]:
        """
        提取章节内容（如Prerequisites, Learning Outcomes等）

        Args:
            text: 课程块文本
            section_name: 章节名称

        Returns:
            章节内容
        """
        pattern = self.section_patterns.get(section_name)
        if not pattern:
            return None

        # 查找章节开始位置
        match = pattern.search(text)
        if not match:
            return None

        start_pos = match.end()

        # 查找下一个章节的开始位置（作为当前章节的结束）
        end_pos = len(text)
        for other_section, other_pattern in self.section_patterns.items():
            if other_section == section_name:
                continue

            other_match = other_pattern.search(text, start_pos)
            if other_match and other_match.start() < end_pos:
                end_pos = other_match.start()

        content = text[start_pos:end_pos].strip()
        return content if content else None

    def parse_course_block(self, course_block: Dict[str, str]) -> Dict[str, any]:
        """
        解析单个课程块，提取所有字段

        Args:
            course_block: {course_code, course_title, content}

        Returns:
            结构化的课程数据
        """
        content = course_block['content']

        course_data = {
            'course_code': course_block['course_code'],
            'title_raw': course_block['course_title'],
        }

        # 提取简单字段
        for field_name in self.field_patterns.keys():
            value = self.extract_field_value(content, field_name)
            if value:
                course_data[field_name] = value

        # 提取章节内容
        for section_name in self.section_patterns.keys():
            section_content = self.extract_section_content(content, section_name)
            if section_content:
                course_data[section_name] = section_content

        # 语言检测：判断标题是德语还是英语
        title = course_block['course_title']
        if any(char in title for char in 'äöüßÄÖÜ'):
            course_data['title_de'] = title
        else:
            # 假设是英语或混合，尝试从内容中提取另一种语言的标题
            # 这里可以添加更复杂的逻辑
            course_data['title_en'] = title

        return course_data

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        解析整个PDF手册

        Args:
            pdf_path: PDF文件路径

        Returns:
            List of course data dictionaries
        """
        print(f"正在解析PDF: {pdf_path}")

        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)
        print(f"提取文本长度: {len(text)} 字符")

        # 分割成课程块
        course_blocks = self.split_into_course_blocks(text)
        print(f"找到 {len(course_blocks)} 个课程")

        # 解析每个课程块
        courses = []
        for block in course_blocks:
            try:
                course_data = self.parse_course_block(block)
                courses.append(course_data)
            except Exception as e:
                print(f"解析课程 {block['course_code']} 时出错: {e}")

        return courses

    def save_to_jsonl(self, courses: List[Dict], output_path: str):
        """
        保存为JSONL格式
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for course in courses:
                json.dump(course, f, ensure_ascii=False)
                f.write('\n')

        print(f"已保存 {len(courses)} 个课程到 {output_path}")


# 使用示例
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdf_handbook_parser.py <pdf_file_path> [output_jsonl_path]")
        print("\n示例:")
        print("  python pdf_handbook_parser.py handbook_ws2024.pdf handbook_data.jsonl")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "handbook_courses.jsonl"

    # 检查文件是否存在
    if not Path(pdf_path).exists():
        print(f"错误: 文件不存在: {pdf_path}")
        sys.exit(1)

    # 解析PDF
    parser = HandbookPDFParser()
    courses = parser.parse_pdf(pdf_path)

    # 保存结果
    parser.save_to_jsonl(courses, output_path)

    # 显示示例
    if courses:
        print("\n第一个课程示例:")
        print(json.dumps(courses[0], indent=2, ensure_ascii=False))
