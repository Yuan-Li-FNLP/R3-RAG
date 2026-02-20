"""
将TUM课程数据转换为FlashRAG所需的JSONL格式
支持多语言（德语/英语）和多粒度的文档切分
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from data_cleaning_pipeline import TUMDataCleaner
from data_deduplication import CourseDataMerger


class FlashRAGConverter:
    """
    将结构化的TUM课程数据转换为FlashRAG格式
    """

    def __init__(self, granularity: str = 'course'):
        """
        Args:
            granularity: 文档粒度
                - 'course': 一个课程一条记录
                - 'section': 按章节切分（如Prerequisites, Learning Outcomes分别为独立记录)
                - 'material': 按课程材料切分（每个PDF/PPT一条记录）
        """
        self.granularity = granularity
        self.doc_id_counter = 0

    def create_flashrag_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        创建FlashRAG格式的文档

        FlashRAG要求的格式:
        {
            "id": "唯一标识",
            "contents": "文档内容"
        }

        Args:
            content: 文档内容
            doc_id: 文档ID（如果不提供则自动生成）
            metadata: 额外的元数据

        Returns:
            FlashRAG格式的文档字典
        """
        if doc_id is None:
            doc_id = str(self.doc_id_counter)
            self.doc_id_counter += 1

        doc = {
            "id": doc_id,
            "contents": content
        }

        # FlashRAG支持额外字段，但id和contents是必需的
        if metadata:
            doc.update(metadata)

        return doc

    def course_to_content(self, course: Dict[str, Any], language: str = 'mixed') -> str:
        """
        将课程数据转换为适合检索的文本内容

        Args:
            course: 课程数据字典
            language: 目标语言 ('de', 'en', 'mixed')

        Returns:
            格式化的文本内容
        """
        lines = []

        # 标题行（FlashRAG推荐使用 "title\ntext" 格式）
        if language == 'de' and course.get('title_de'):
            title = f"{course['course_code']} - {course['title_de']}"
        elif language == 'en' and course.get('title_en'):
            title = f"{course['course_code']} - {course['title_en']}"
        else:  # mixed
            title_de = course.get('title_de', '')
            title_en = course.get('title_en', '')
            if title_de and title_en:
                title = f"{course['course_code']} - {title_en} / {title_de}"
            elif title_en:
                title = f"{course['course_code']} - {title_en}"
            elif title_de:
                title = f"{course['course_code']} - {title_de}"
            else:
                title = course['course_code']

        lines.append(title)

        # 基本信息
        if course.get('ects'):
            lines.append(f"ECTS: {course['ects']}")

        if course.get('sws'):
            lines.append(f"SWS: {course['sws']}")

        if course.get('semester'):
            semesters = ', '.join(course['semester']) if isinstance(course['semester'], list) else course['semester']
            lines.append(f"Semester: {semesters}")

        if course.get('lecturer'):
            lines.append(f"Lecturer: {course['lecturer']}")

        if course.get('language'):
            langs = ', '.join(course['language']) if isinstance(course['language'], list) else course['language']
            lines.append(f"Language: {langs}")

        if course.get('module_type'):
            lines.append(f"Module Type: {course['module_type']}")

        # 先修课程
        if course.get('prerequisites'):
            prereqs = course['prerequisites']
            if isinstance(prereqs, dict):
                if prereqs.get('mandatory'):
                    lines.append(f"Mandatory Prerequisites: {', '.join(prereqs['mandatory'])}")
                if prereqs.get('recommended'):
                    lines.append(f"Recommended Prerequisites: {', '.join(prereqs['recommended'])}")
            else:
                lines.append(f"Prerequisites: {prereqs}")

        # 详细信息
        if course.get('description'):
            lines.append(f"\nDescription:\n{course['description']}")

        if course.get('learning_outcomes'):
            lines.append(f"\nLearning Outcomes:\n{course['learning_outcomes']}")

        if course.get('content'):
            lines.append(f"\nContent:\n{course['content']}")

        if course.get('assessment'):
            lines.append(f"\nAssessment:\n{course['assessment']}")

        if course.get('literature'):
            lines.append(f"\nRecommended Literature:\n{course['literature']}")

        return '\n'.join(lines)

    def convert_course_level(self, course: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        课程级别的转换：一个课程一条FlashRAG记录

        Args:
            course: 课程数据

        Returns:
            FlashRAG文档列表（通常只有一个元素）
        """
        documents = []

        # 生成混合语言版本（最全面）
        content_mixed = self.course_to_content(course, language='mixed')
        doc = self.create_flashrag_document(
            content=content_mixed,
            doc_id=course['course_code'],
            metadata={
                'course_code': course['course_code'],
                'language': 'mixed',
                'ects': course.get('ects'),
                'semester': course.get('semester')
            }
        )
        documents.append(doc)

        return documents

    def convert_section_level(self, course: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        章节级别的转换：将课程的不同部分切分为独立的FlashRAG记录

        适用场景：用户查询更细粒度的信息（如"IN2064的先修课程是什么？"）

        Returns:
            多个FlashRAG文档
        """
        documents = []
        course_code = course['course_code']

        # 1. 基本信息文档
        basic_info = []
        basic_info.append(f"{course_code} - {course.get('title_en', course.get('title_de', ''))}")
        if course.get('ects'):
            basic_info.append(f"ECTS: {course['ects']}")
        if course.get('lecturer'):
            basic_info.append(f"Lecturer: {course['lecturer']}")
        if course.get('semester'):
            basic_info.append(f"Semester: {', '.join(course['semester'])}")

        doc = self.create_flashrag_document(
            content='\n'.join(basic_info),
            doc_id=f"{course_code}_basic",
            metadata={'course_code': course_code, 'section': 'basic_info'}
        )
        documents.append(doc)

        # 2. 先修课程文档
        if course.get('prerequisites'):
            prereq_content = [f"{course_code} - Prerequisites"]
            prereqs = course['prerequisites']
            if isinstance(prereqs, dict):
                if prereqs.get('mandatory'):
                    prereq_content.append(f"Mandatory: {', '.join(prereqs['mandatory'])}")
                if prereqs.get('recommended'):
                    prereq_content.append(f"Recommended: {', '.join(prereqs['recommended'])}")
            else:
                prereq_content.append(str(prereqs))

            doc = self.create_flashrag_document(
                content='\n'.join(prereq_content),
                doc_id=f"{course_code}_prerequisites",
                metadata={'course_code': course_code, 'section': 'prerequisites'}
            )
            documents.append(doc)

        # 3. 学习成果文档
        if course.get('learning_outcomes'):
            content = f"{course_code} - Learning Outcomes\n{course['learning_outcomes']}"
            doc = self.create_flashrag_document(
                content=content,
                doc_id=f"{course_code}_learning_outcomes",
                metadata={'course_code': course_code, 'section': 'learning_outcomes'}
            )
            documents.append(doc)

        # 4. 课程描述文档
        if course.get('description'):
            content = f"{course_code} - Description\n{course['description']}"
            doc = self.create_flashrag_document(
                content=content,
                doc_id=f"{course_code}_description",
                metadata={'course_code': course_code, 'section': 'description'}
            )
            documents.append(doc)

        # 5. 考核方式文档
        if course.get('assessment'):
            content = f"{course_code} - Assessment\n{course['assessment']}"
            doc = self.create_flashrag_document(
                content=content,
                doc_id=f"{course_code}_assessment",
                metadata={'course_code': course_code, 'section': 'assessment'}
            )
            documents.append(doc)

        return documents

    def convert_material_level(self, course: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        材料级别的转换：将课程的每个材料文件作为独立记录

        适用场景：有大量课程材料（lecture slides, exercises）需要索引

        Returns:
            每个材料一个FlashRAG文档
        """
        documents = []
        course_code = course['course_code']

        if not course.get('materials'):
            # 如果没有材料，退化为课程级别
            return self.convert_course_level(course)

        for idx, material in enumerate(course['materials']):
            title = material.get('title', f'Material {idx+1}')
            extracted_text = material.get('extracted_text', '')

            if not extracted_text:
                continue

            # 构建内容：标题 + 提取的文本
            content = f"{course_code} - {title}\n{extracted_text}"

            doc = self.create_flashrag_document(
                content=content,
                doc_id=f"{course_code}_material_{idx}",
                metadata={
                    'course_code': course_code,
                    'section': 'material',
                    'material_title': title,
                    'material_type': material.get('file_type', 'unknown')
                }
            )
            documents.append(doc)

        return documents

    def convert_courses(self, courses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量转换课程数据

        Args:
            courses: 课程数据列表

        Returns:
            FlashRAG格式的文档列表
        """
        all_documents = []

        for course in courses:
            if self.granularity == 'course':
                docs = self.convert_course_level(course)
            elif self.granularity == 'section':
                docs = self.convert_section_level(course)
            elif self.granularity == 'material':
                docs = self.convert_material_level(course)
            else:
                raise ValueError(f"Unknown granularity: {self.granularity}")

            all_documents.extend(docs)

        return all_documents

    def save_to_jsonl(self, documents: List[Dict[str, Any]], output_path: str):
        """
        保存为JSONL格式
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                json.dump(doc, f, ensure_ascii=False)
                f.write('\n')

        print(f"✅ 已保存 {len(documents)} 个文档到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description='将TUM课程数据转换为FlashRAG格式')
    parser.add_argument('--input', required=True, help='输入的课程数据文件（JSONL格式）')
    parser.add_argument('--output', required=True, help='输出的FlashRAG格式文件')
    parser.add_argument(
        '--granularity',
        choices=['course', 'section', 'material'],
        default='course',
        help='文档粒度: course(课程级), section(章节级), material(材料级)'
    )
    parser.add_argument('--clean', action='store_true', help='是否先进行数据清洗')

    args = parser.parse_args()

    # 读取输入数据
    print(f"📖 读取输入文件: {args.input}")
    courses = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                courses.append(json.loads(line))

    print(f"找到 {len(courses)} 个课程")

    # 数据清洗（可选）
    if args.clean:
        print("🧹 开始数据清洗...")
        cleaner = TUMDataCleaner()
        cleaned_courses = []
        for course in courses:
            cleaned = cleaner.clean_course_data(course, source='unknown')
            cleaned_courses.append(cleaned)
        courses = cleaned_courses
        print(f"✅ 清洗完成")

    # 转换为FlashRAG格式
    print(f"🔄 转换为FlashRAG格式 (粒度: {args.granularity})...")
    converter = FlashRAGConverter(granularity=args.granularity)
    documents = converter.convert_courses(courses)

    # 保存
    converter.save_to_jsonl(documents, args.output)

    print(f"\n📊 转换统计:")
    print(f"  输入课程数: {len(courses)}")
    print(f"  输出文档数: {len(documents)}")
    print(f"  平均每课程: {len(documents)/len(courses):.1f} 个文档")


if __name__ == "__main__":
    main()
