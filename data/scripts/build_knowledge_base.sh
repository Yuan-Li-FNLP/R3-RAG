#!/bin/bash
# TUM课程知识库构建完整流程
# 整合所有数据源，完成清洗、合并、转换

set -e  # 遇到错误立即退出

echo "======================================"
echo "TUM Course Knowledge Base Builder"
echo "======================================"

# 配置参数
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/.."
RAW_DATA_DIR="${DATA_DIR}/raw"
PROCESSED_DATA_DIR="${DATA_DIR}/processed"
FINAL_OUTPUT_DIR="${DATA_DIR}/corpus"

# 创建目录
mkdir -p "${RAW_DATA_DIR}"
mkdir -p "${PROCESSED_DATA_DIR}"
mkdir -p "${FINAL_OUTPUT_DIR}"

echo ""
echo "📁 工作目录:"
echo "  原始数据: ${RAW_DATA_DIR}"
echo "  处理数据: ${PROCESSED_DATA_DIR}"
echo "  最终输出: ${FINAL_OUTPUT_DIR}"
echo ""

# ======================================
# 阶段1: 数据采集
# ======================================
echo "======================================"
echo "阶段1: 数据采集"
echo "======================================"

# 注意: 以下步骤需要手动运行爬虫脚本
echo "⚠️  请确保已完成以下数据采集:"
echo "  1. TUMonline数据 -> ${RAW_DATA_DIR}/tumonline_courses.jsonl"
echo "  2. Moodle材料   -> ${RAW_DATA_DIR}/moodle_materials/"
echo "  3. PDF手册      -> ${RAW_DATA_DIR}/handbook_ws2024.pdf"
echo "  4. FPSO文档     -> ${RAW_DATA_DIR}/fpso_documents/"
echo ""
read -p "是否已完成数据采集? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 请先完成数据采集后再运行此脚本"
    exit 1
fi

# ======================================
# 阶段2: PDF手册解析
# ======================================
echo ""
echo "======================================"
echo "阶段2: 解析PDF手册"
echo "======================================"

if [ -f "${RAW_DATA_DIR}/handbook_ws2024.pdf" ]; then
    echo "📖 解析课程手册..."
    python3 "${SCRIPT_DIR}/pdf_handbook_parser.py" \
        "${RAW_DATA_DIR}/handbook_ws2024.pdf" \
        "${PROCESSED_DATA_DIR}/handbook_parsed.jsonl"
    echo "✅ 手册解析完成"
else
    echo "⚠️  未找到PDF手册，跳过此步骤"
fi

# ======================================
# 阶段3: 数据清洗
# ======================================
echo ""
echo "======================================"
echo "阶段3: 数据清洗"
echo "======================================"

# 清洗TUMonline数据
if [ -f "${RAW_DATA_DIR}/tumonline_courses.jsonl" ]; then
    echo "🧹 清洗TUMonline数据..."
    python3 - <<EOF
import json
import sys
sys.path.append('${SCRIPT_DIR}')
from data_cleaning_pipeline import TUMDataCleaner

cleaner = TUMDataCleaner()
cleaned_courses = []

with open('${RAW_DATA_DIR}/tumonline_courses.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            raw = json.loads(line)
            cleaned = cleaner.clean_course_data(raw, source='tumonline')
            cleaned_courses.append(cleaned)

with open('${PROCESSED_DATA_DIR}/tumonline_cleaned.jsonl', 'w', encoding='utf-8') as f:
    for course in cleaned_courses:
        json.dump(course, f, ensure_ascii=False)
        f.write('\n')

print(f"✅ 清洗完成: {len(cleaned_courses)} 个课程")
print(json.dumps(cleaner.get_statistics(), indent=2, ensure_ascii=False))
EOF
fi

# 清洗手册数据
if [ -f "${PROCESSED_DATA_DIR}/handbook_parsed.jsonl" ]; then
    echo "🧹 清洗手册数据..."
    python3 - <<EOF
import json
import sys
sys.path.append('${SCRIPT_DIR}')
from data_cleaning_pipeline import TUMDataCleaner

cleaner = TUMDataCleaner()
cleaned_courses = []

with open('${PROCESSED_DATA_DIR}/handbook_parsed.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            raw = json.loads(line)
            cleaned = cleaner.clean_course_data(raw, source='handbook')
            cleaned_courses.append(cleaned)

with open('${PROCESSED_DATA_DIR}/handbook_cleaned.jsonl', 'w', encoding='utf-8') as f:
    for course in cleaned_courses:
        json.dump(course, f, ensure_ascii=False)
        f.write('\n')

print(f"✅ 清洗完成: {len(cleaned_courses)} 个课程")
EOF
fi

# ======================================
# 阶段4: 数据合并与去重
# ======================================
echo ""
echo "======================================"
echo "阶段4: 数据合并与去重"
echo "======================================"

echo "🔀 合并多个数据源..."
python3 - <<EOF
import json
import sys
sys.path.append('${SCRIPT_DIR}')
from data_deduplication import CourseDataMerger

merger = CourseDataMerger()

# 加载TUMonline数据
try:
    with open('${PROCESSED_DATA_DIR}/tumonline_cleaned.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                course = json.loads(line)
                course_id = course.get('course_code')
                if course_id:
                    merger.add_course_data(course_id, 'tumonline', course)
except FileNotFoundError:
    print("⚠️  TUMonline数据文件不存在")

# 加载手册数据
try:
    with open('${PROCESSED_DATA_DIR}/handbook_cleaned.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                course = json.loads(line)
                course_id = course.get('course_code')
                if course_id:
                    merger.add_course_data(course_id, 'handbook', course)
except FileNotFoundError:
    print("⚠️  手册数据文件不存在")

# 导出合并结果
merged_courses = []
for course_id in merger.courses.keys():
    merged = merger.get_merged_course(course_id)
    if merged:
        merged_courses.append(merged)

with open('${PROCESSED_DATA_DIR}/merged_courses.jsonl', 'w', encoding='utf-8') as f:
    for course in merged_courses:
        json.dump(course, f, ensure_ascii=False)
        f.write('\n')

print(f"✅ 合并完成: {len(merged_courses)} 个课程")

# 导出冲突报告
merger.export_conflicts_report('${PROCESSED_DATA_DIR}/conflicts_report.json')
EOF

# ======================================
# 阶段5: 数据验证
# ======================================
echo ""
echo "======================================"
echo "阶段5: 数据验证"
echo "======================================"

echo "✔️  验证数据完整性..."
python3 - <<EOF
import json
import sys
sys.path.append('${SCRIPT_DIR}')
from data_cleaning_pipeline import TUMDataCleaner

cleaner = TUMDataCleaner()
validation_report = []

with open('${PROCESSED_DATA_DIR}/merged_courses.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            course = json.loads(line)
            is_valid, issues = cleaner.validate_course_data(course)

            if not is_valid:
                validation_report.append({
                    'course_code': course.get('course_code'),
                    'issues': issues
                })

if validation_report:
    print(f"⚠️  发现 {len(validation_report)} 个课程存在问题:")
    with open('${PROCESSED_DATA_DIR}/validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)

    for item in validation_report[:5]:  # 只显示前5个
        print(f"  - {item['course_code']}: {', '.join(item['issues'])}")
else:
    print("✅ 所有课程数据验证通过")
EOF

# ======================================
# 阶段6: 转换为FlashRAG格式
# ======================================
echo ""
echo "======================================"
echo "阶段6: 转换为FlashRAG格式"
echo "======================================"

echo "🔄 转换为FlashRAG格式..."

# 提供三种粒度选择
echo "请选择文档粒度:"
echo "  1) course  - 课程级别（一个课程一条记录）"
echo "  2) section - 章节级别（按Prerequisites, Learning Outcomes等切分）"
echo "  3) material- 材料级别（每个PDF/PPT一条记录）"
read -p "请输入选择 (1/2/3) [默认: 1]: " granularity_choice

case ${granularity_choice} in
    2)
        GRANULARITY="section"
        ;;
    3)
        GRANULARITY="material"
        ;;
    *)
        GRANULARITY="course"
        ;;
esac

python3 "${SCRIPT_DIR}/convert_to_flashrag_format.py" \
    --input "${PROCESSED_DATA_DIR}/merged_courses.jsonl" \
    --output "${FINAL_OUTPUT_DIR}/tum_courses_corpus.jsonl" \
    --granularity "${GRANULARITY}"

echo "✅ FlashRAG格式转换完成"

# ======================================
# 完成
# ======================================
echo ""
echo "======================================"
echo "✅ 知识库构建完成！"
echo "======================================"
echo ""
echo "📊 输出文件:"
echo "  主corpus: ${FINAL_OUTPUT_DIR}/tum_courses_corpus.jsonl"
echo "  冲突报告: ${PROCESSED_DATA_DIR}/conflicts_report.json"
echo "  验证报告: ${PROCESSED_DATA_DIR}/validation_report.json"
echo ""
echo "🚀 下一步: 构建检索索引"
echo "  运行: bash build_index.sh"
echo ""
