"""
增量更新脚本
定期检查数据源变化，只更新修改的部分
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Set
import os

class IncrementalUpdater:
    """
    管理知识库的增量更新
    """

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(cache_dir, "update_metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """加载更新元数据"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'last_update': None,
            'course_hashes': {},  # course_id -> content_hash
            'update_history': []
        }

    def _save_metadata(self):
        """保存更新元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def compute_hash(self, data: Dict) -> str:
        """计算数据的哈希值"""
        # 排序后序列化，确保一致性
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def check_updates(self, course_id: str, new_data: Dict) -> bool:
        """
        检查课程数据是否有更新

        Returns:
            True if data has changed, False otherwise
        """
        new_hash = self.compute_hash(new_data)
        old_hash = self.metadata['course_hashes'].get(course_id)

        if old_hash != new_hash:
            print(f"[UPDATE DETECTED] Course {course_id} has been modified")
            return True
        else:
            print(f"[NO CHANGE] Course {course_id} is up to date")
            return False

    def update_course(self, course_id: str, new_data: Dict, source: str):
        """
        更新课程数据并记录变更历史
        """
        if self.check_updates(course_id, new_data):
            new_hash = self.compute_hash(new_data)

            # 更新哈希值
            self.metadata['course_hashes'][course_id] = new_hash

            # 记录更新历史
            self.metadata['update_history'].append({
                'course_id': course_id,
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'hash': new_hash
            })

            # 保存元数据
            self._save_metadata()

            return True
        return False

    def get_stale_courses(self, days: int = 90) -> List[str]:
        """
        获取超过指定天数未更新的课程列表

        Args:
            days: 天数阈值

        Returns:
            List of course IDs that haven't been updated recently
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        stale_courses = []

        for entry in self.metadata['update_history']:
            update_time = datetime.fromisoformat(entry['timestamp'])
            if update_time < cutoff_date:
                course_id = entry['course_id']
                if course_id not in stale_courses:
                    stale_courses.append(course_id)

        return stale_courses

    def generate_update_report(self) -> str:
        """
        生成更新报告
        """
        total_courses = len(self.metadata['course_hashes'])
        recent_updates = [
            entry for entry in self.metadata['update_history']
            if datetime.fromisoformat(entry['timestamp']) > datetime.now() - timedelta(days=7)
        ]

        report = f"""
# Knowledge Base Update Report

**Last Update**: {self.metadata.get('last_update', 'Never')}
**Total Courses**: {total_courses}
**Updates in Last 7 Days**: {len(recent_updates)}

## Recent Updates
"""
        for entry in recent_updates[-10:]:  # 显示最近10条
            report += f"- {entry['course_id']} ({entry['source']}) at {entry['timestamp']}\n"

        return report


# 示例使用
if __name__ == "__main__":
    updater = IncrementalUpdater()

    # 模拟新数据
    new_course_data = {
        'id': 'IN2064',
        'title_en': 'Introduction to AI',
        'ects': 6,
        'lecturer': 'Prof. Dr. Müller'
    }

    # 检查并更新
    if updater.update_course('IN2064', new_course_data, 'tumonline'):
        print("Course data updated successfully!")
    else:
        print("No changes detected.")

    # 生成报告
    print(updater.generate_update_report())

    # 检查过期课程
    stale = updater.get_stale_courses(days=90)
    print(f"\nCourses not updated in 90 days: {stale}")
