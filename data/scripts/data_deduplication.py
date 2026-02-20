"""
数据去重与冲突解决脚本
处理来自多个数据源的课程信息
"""

import json
from datetime import datetime
from typing import List, Dict, Any

class CourseDataMerger:
    """
    合并来自多个数据源的课程信息，处理冲突
    """

    # 数据源优先级（数字越大优先级越高）
    SOURCE_PRIORITY = {
        'tumonline': 4,      # 最权威
        'handbook': 3,       # 官方文档
        'moodle': 2,         # 课程材料
        'fpso': 1            # 学位规定
    }

    def __init__(self):
        self.courses = {}  # course_id -> merged_data

    def add_course_data(self, course_id: str, source: str, data: Dict[str, Any]):
        """
        添加课程数据，自动处理冲突

        Args:
            course_id: 课程代码 (e.g., IN2064)
            source: 数据源 (tumonline|moodle|handbook|fpso)
            data: 课程数据字典
        """
        if course_id not in self.courses:
            self.courses[course_id] = {
                'id': course_id,
                'sources': {},
                'merged': {},
                'conflicts': []
            }

        # 保存原始数据
        self.courses[course_id]['sources'][source] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }

        # 合并数据
        self._merge_data(course_id)

    def _merge_data(self, course_id: str):
        """
        合并单个课程的所有数据源
        """
        course = self.courses[course_id]
        sources = course['sources']
        merged = {}

        # 按字段合并
        all_keys = set()
        for source_data in sources.values():
            all_keys.update(source_data['data'].keys())

        for key in all_keys:
            values_by_source = {}
            for source, source_data in sources.items():
                if key in source_data['data']:
                    values_by_source[source] = source_data['data'][key]

            # 冲突检测
            unique_values = set(str(v) for v in values_by_source.values())

            if len(unique_values) > 1:
                # 存在冲突，使用优先级最高的数据源
                best_source = max(
                    values_by_source.keys(),
                    key=lambda s: self.SOURCE_PRIORITY.get(s, 0)
                )
                merged[key] = values_by_source[best_source]

                # 记录冲突
                course['conflicts'].append({
                    'field': key,
                    'values': values_by_source,
                    'resolved_with': best_source
                })
            else:
                # 无冲突，直接使用
                merged[key] = list(values_by_source.values())[0]

        course['merged'] = merged

    def get_merged_course(self, course_id: str) -> Dict[str, Any]:
        """
        获取合并后的课程数据
        """
        return self.courses.get(course_id, {}).get('merged', {})

    def export_conflicts_report(self, output_path: str):
        """
        导出冲突报告，用于人工审核
        """
        conflicts_report = []
        for course_id, course in self.courses.items():
            if course['conflicts']:
                conflicts_report.append({
                    'course_id': course_id,
                    'conflicts': course['conflicts']
                })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conflicts_report, f, ensure_ascii=False, indent=2)

        print(f"Conflicts report exported to {output_path}")
        print(f"Total courses with conflicts: {len(conflicts_report)}")


# 示例使用
if __name__ == "__main__":
    merger = CourseDataMerger()

    # 示例：从不同数据源添加IN2064课程数据

    # TUMonline数据
    merger.add_course_data('IN2064', 'tumonline', {
        'title_en': 'Introduction to Artificial Intelligence',
        'ects': 6,
        'lecturer': 'Prof. Dr. Müller',
        'semester': ['WS']
    })

    # Handbook数据（可能有冲突）
    merger.add_course_data('IN2064', 'handbook', {
        'title_en': 'Introduction to AI',  # 标题略有不同
        'ects': 6,
        'prerequisites': ['IN0001', 'MA0001']
    })

    # Moodle数据
    merger.add_course_data('IN2064', 'moodle', {
        'title_en': 'Introduction to Artificial Intelligence',
        'materials': ['lecture_01.pdf', 'exercise_01.pdf']
    })

    # 获取合并结果
    merged = merger.get_merged_course('IN2064')
    print("Merged course data:")
    print(json.dumps(merged, indent=2, ensure_ascii=False))

    # 导出冲突报告
    merger.export_conflicts_report('conflicts_report.json')
