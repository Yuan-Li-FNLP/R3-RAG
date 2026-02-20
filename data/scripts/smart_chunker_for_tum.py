"""
针对TUM课程数据的智能切分策略
根据文档类型采用不同的切分方式
"""

import json
import re
from pathlib import Path
from typing import List, Dict

class TUMDocumentChunker:
    """
    TUM课程文档智能切分器
    根据文档类型采用不同策略
    """

    def __init__(self):
        self.chunks = []
        self.chunk_id = 0

    def chunk_course_intro(self, text: str, course_code: str) -> List[str]:
        """
        切分课程介绍（200门简短介绍）

        策略：不切分！直接作为一个块
        原因：课程介绍通常很短（<1000字符），切分会破坏信息完整性

        示例：
        "IN2064 - Introduction to AI
         ECTS: 6, Lecturer: Prof. Müller
         Prerequisites: IN0001, MA0001
         Description: This course..."
        """
        # 不切分，直接返回整个文档
        return [text]

    def chunk_module_plan(self, text: str, module_name: str) -> List[str]:
        """
        切分模块规划（8个模块的学分要求）

        策略：按模块切分，每个模块一个块
        原因：模块信息相对独立，按模块切分便于检索

        示例：
        "Modul 1: Kernmodule (60 ECTS)
         - IN2064 (6 ECTS)
         - IN2118 (6 ECTS)
         ..."
        """
        # 如果文本不长（<2000字符），不切分
        if len(text) < 2000:
            return [text]

        # 按模块标题切分
        # 匹配 "Modul X:", "Module X:", "X. " 等
        module_pattern = re.compile(r'(?:Modul|Module)\s+\d+:|^\d+\.\s+', re.MULTILINE)

        splits = module_pattern.split(text)
        chunks = []

        for i, chunk in enumerate(splits):
            if chunk.strip():
                # 保留模块标题
                if i > 0:
                    match = module_pattern.search(text)
                    if match:
                        chunk = match.group(0) + chunk
                chunks.append(chunk.strip())

        return chunks if chunks else [text]

    def chunk_lecture_pdf(self, text: str, course_code: str, chunk_size=1500, overlap=150) -> List[str]:
        """
        切分课程讲义PDF（30门长文档）

        策略：
        1. 优先按章节结构切分（Chapter, Kapitel, Section等）
        2. 如果单个章节太长（>3000字符），在章节内部二次切分
        3. 如果没有章节结构，回退到标准字符切分

        示例输入：
        "Kapitel 1: Suchverfahren
         1.1 Uninformierte Suche
         BFS (Breitensuche) ist ein Algorithmus...
         Kapitel 2: Heuristiken
         ..."

        示例输出：
        块1: "Kapitel 1: Suchverfahren\n1.1 Uninformierte Suche\n..."
        块2: "Kapitel 2: Heuristiken\n..."
        """
        # 尝试识别章节结构
        # 匹配常见章节标记（德语/英语）
        chapter_patterns = [
            r'^(?:Chapter|Kapitel|CHAPTER|KAPITEL)\s+\d+',  # Chapter 1, Kapitel 1
            r'^\d+\.\s+[A-Z]',  # 1. Introduction
            r'^#{1,3}\s+',  # Markdown格式: # Chapter
            r'^(?:Lecture|Vorlesung)\s+\d+',  # Lecture 1
        ]

        combined_pattern = re.compile('|'.join(chapter_patterns), re.MULTILINE)
        matches = list(combined_pattern.finditer(text))

        if len(matches) >= 2:  # 至少找到2个章节标记
            # 按章节切分
            chunks = []
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i+1].start() if i+1 < len(matches) else len(text)
                chapter_text = text[start:end].strip()

                # 如果单个章节太长，在章节内部二次切分
                if len(chapter_text) > 3000:
                    sub_chunks = self._smart_chunk(chapter_text, chunk_size, overlap)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chapter_text)

            return chunks if chunks else [text]
        else:
            # 没有找到章节结构，使用标准切分
            return self._smart_chunk(text, chunk_size, overlap)

    def chunk_exercise(self, text: str, course_code: str) -> List[str]:
        """
        切分练习题

        策略：按题切分
        原因：每道题是独立的，学生可能查询"第3题怎么做"

        示例输入：
        "Aufgabe 1: Implementieren Sie BFS...
         Aufgabe 2: Zeigen Sie dass...
         Aufgabe 3: ..."

        示例输出：
        块1: "Aufgabe 1: Implementieren Sie BFS..."
        块2: "Aufgabe 2: Zeigen Sie dass..."
        块3: "Aufgabe 3: ..."
        """
        # 匹配题目标记
        # Aufgabe, Exercise, Problem, Question 等
        problem_pattern = re.compile(
            r'(?:Aufgabe|Exercise|Problem|Question|Übung)\s+\d+',
            re.IGNORECASE | re.MULTILINE
        )

        # 找到所有题目位置
        matches = list(problem_pattern.finditer(text))

        if not matches:
            # 没有找到题目标记，按标准方式切分
            return self._smart_chunk(text, chunk_size=1500, overlap=150)

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)

            chunk_text = text[start:end].strip()

            # 如果单个题目太长（>2000字符），进一步切分
            if len(chunk_text) > 2000:
                sub_chunks = self._smart_chunk(chunk_text, chunk_size=1500, overlap=150)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return chunks

    def chunk_script(self, text: str, course_code: str, chunk_size=1500, overlap=150) -> List[str]:
        """
        切分Script/课程笔记

        策略：标准分块（1500字符，150重叠）
        原因：Script通常是结构化的长文档，类似讲义
        """
        return self._smart_chunk(text, chunk_size, overlap)

    def chunk_references(self, text: str, course_code: str) -> List[str]:
        """
        切分参考文献

        策略：不切分，或按参考文献条目切分
        原因：参考文献是列表，通常不长

        示例：
        "Literatur:
         [1] Russell, S. & Norvig, P. (2020). Artificial Intelligence...
         [2] Mitchell, T. (1997). Machine Learning..."
        """
        # 如果不长（<2000字符），不切分
        if len(text) < 2000:
            return [text]

        # 按参考文献编号切分
        ref_pattern = re.compile(r'\[\d+\]', re.MULTILINE)

        matches = list(ref_pattern.finditer(text))
        if not matches:
            return [text]

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def chunk_project_requirements(self, text: str, project_name: str) -> List[str]:
        """
        切分项目要求文档（2个项目）

        策略：按章节切分
        原因：项目要求通常有明确的章节结构

        示例：
        "1. Projektziele
         2. Anforderungen
         3. Abgabetermine
         4. Bewertungskriterien"
        """
        # 按章节标题切分
        # 匹配 "1.", "2.", "1.1", "Chapter 1", "Section 1" 等
        section_pattern = re.compile(r'^\d+\.(?:\d+\.)?\s+\w+', re.MULTILINE)

        matches = list(section_pattern.finditer(text))

        if not matches or len(matches) < 2:
            # 没有明显的章节结构，按标准方式切分
            return self._smart_chunk(text, chunk_size=1500, overlap=150)

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)

            chunk_text = text[start:end].strip()

            # 如果单个章节太长（>2000字符），进一步切分
            if len(chunk_text) > 2000:
                sub_chunks = self._smart_chunk(chunk_text, chunk_size=1500, overlap=150)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return chunks

    def _smart_chunk(self, text: str, chunk_size=1500, overlap=150) -> List[str]:
        """
        标准智能分块（1500字符，150重叠）
        按段落切分，避免切断句子
        """
        paragraphs = re.split(r'\n\n+', text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 超长段落强制切分
            if len(para) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                for i in range(0, len(para), chunk_size - overlap):
                    sub_chunk = para[i:i + chunk_size]
                    if sub_chunk.strip():
                        chunks.append(sub_chunk.strip())
                continue

            # 检查加上这段是否超限
            if len(current_chunk) + len(para) + 2 < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # 包含重叠
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + "\n\n" + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def process_document(self, text: str, doc_type: str, doc_id: str) -> None:
        """
        根据文档类型处理文档

        Args:
            text: 文档文本
            doc_type: 文档类型
                - 'course_intro': 课程介绍
                - 'module_plan': 模块规划
                - 'lecture': 课程讲义
                - 'exercise': 练习题
                - 'script': 课程笔记
                - 'references': 参考文献
                - 'project': 项目要求
            doc_id: 文档标识（如课程代码）
        """
        # 根据类型选择切分策略
        if doc_type == 'course_intro':
            chunks = self.chunk_course_intro(text, doc_id)
        elif doc_type == 'module_plan':
            chunks = self.chunk_module_plan(text, doc_id)
        elif doc_type == 'lecture':
            chunks = self.chunk_lecture_pdf(text, doc_id)
        elif doc_type == 'exercise':
            chunks = self.chunk_exercise(text, doc_id)
        elif doc_type == 'script':
            chunks = self.chunk_script(text, doc_id)
        elif doc_type == 'references':
            chunks = self.chunk_references(text, doc_id)
        elif doc_type == 'project':
            chunks = self.chunk_project_requirements(text, doc_id)
        else:
            # 默认：标准分块
            chunks = self._smart_chunk(text, chunk_size=1500, overlap=150)

        # 添加到总列表
        for chunk_text in chunks:
            self.chunks.append({
                "id": str(self.chunk_id),
                "contents": chunk_text,
                "_metadata": {
                    "doc_type": doc_type,
                    "doc_id": doc_id,
                    "chunk_length": len(chunk_text)
                }
            })
            self.chunk_id += 1

    def export_to_jsonl(self, output_path: str):
        """导出为FlashRAG格式的JSONL"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                flashrag_doc = {
                    "id": chunk["id"],
                    "contents": chunk["contents"]
                }
                f.write(json.dumps(flashrag_doc, ensure_ascii=False) + '\n')

        print(f"✅ 导出完成: {len(self.chunks)} 个块")
        print(f"   保存到: {output_path}")

    def print_statistics(self):
        """打印统计信息"""
        print("\n📊 切分统计:")
        print(f"   总块数: {len(self.chunks)}")

        # 按类型统计
        type_counts = {}
        type_avg_lengths = {}

        for chunk in self.chunks:
            doc_type = chunk['_metadata']['doc_type']
            chunk_length = chunk['_metadata']['chunk_length']

            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            if doc_type not in type_avg_lengths:
                type_avg_lengths[doc_type] = []
            type_avg_lengths[doc_type].append(chunk_length)

        print("\n   按类型统计:")
        for doc_type, count in type_counts.items():
            avg_len = sum(type_avg_lengths[doc_type]) // len(type_avg_lengths[doc_type])
            print(f"   - {doc_type}: {count} 块，平均 {avg_len} 字符")


# 使用示例
if __name__ == "__main__":
    chunker = TUMDocumentChunker()

    # 示例1: 课程介绍（200门，不切分）
    course_intro = """
    IN2064 - Introduction to Artificial Intelligence
    ECTS: 6
    Lecturer: Prof. Dr. Müller
    Prerequisites: IN0001, MA0001
    Description: This course introduces fundamental concepts...
    """
    chunker.process_document(course_intro, 'course_intro', 'IN2064')

    # 示例2: 课程讲义（30门，标准切分）
    lecture_text = """
    Kapitel 1: Suchverfahren

    1.1 Uninformierte Suche
    BFS (Breitensuche) ist ein Algorithmus, der...
    ... (假设有10000字符)
    """
    chunker.process_document(lecture_text, 'lecture', 'IN2064_lecture')

    # 示例3: 练习题（按题切分）
    exercise_text = """
    Aufgabe 1: Implementieren Sie den BFS Algorithmus...

    Aufgabe 2: Zeigen Sie, dass der A* Algorithmus optimal ist...

    Aufgabe 3: ...
    """
    chunker.process_document(exercise_text, 'exercise', 'IN2064_ex01')

    # 导出
    chunker.export_to_jsonl('tum_corpus.jsonl')
    chunker.print_statistics()
