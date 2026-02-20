# TUM课程知识库 - 数据清洗与结构化完整指南

## 📋 目录

1. [流程概览](#流程概览)
2. [环境准备](#环境准备)
3. [详细步骤](#详细步骤)
4. [脚本说明](#脚本说明)
5. [常见问题](#常见问题)
6. [示例与最佳实践](#示例与最佳实践)

---

## 流程概览

```
┌─────────────────┐
│  数据采集       │  TUMonline, Moodle, PDF手册, FPSO
└────────┬────────┘
         │
         v
┌─────────────────┐
│  格式转换       │  PDF解析, HTML清洗
└────────┬────────┘
         │
         v
┌─────────────────┐
│  数据清洗       │  文本规范化, 字段提取, 语言检测
└────────┬────────┘
         │
         v
┌─────────────────┐
│  数据合并       │  多源合并, 冲突解决, 去重
└────────┬────────┘
         │
         v
┌─────────────────┐
│  数据验证       │  完整性检查, 格式验证
└────────┬────────┘
         │
         v
┌─────────────────┐
│ FlashRAG转换    │  生成检索语料库
└────────┬────────┘
         │
         v
┌─────────────────┐
│  索引构建       │  Embedding + FAISS
└─────────────────┘
```

---

## 环境准备

### 1. Python依赖安装

```bash
# 基础依赖
pip install PyMuPDF  # PDF解析
pip install requests  # API调用
pip install beautifulsoup4  # HTML解析
pip install selenium  # 网页爬取
pip install python-docx  # Word文档处理

# 可选：OCR支持（处理扫描版PDF）
pip install pytesseract
sudo apt-get install tesseract-ocr tesseract-ocr-deu  # 德语OCR
```

### 2. 目录结构

```
R3-RAG/data/
├── scripts/                           # 脚本目录
│   ├── data_cleaning_pipeline.py      # 数据清洗主脚本
│   ├── pdf_handbook_parser.py         # PDF解析器
│   ├── data_deduplication.py          # 数据去重与合并
│   ├── convert_to_flashrag_format.py  # FlashRAG格式转换
│   ├── incremental_update.py          # 增量更新管理
│   ├── build_knowledge_base.sh        # 端到端构建脚本
│   └── README_DATA_CLEANING.md        # 本文档
├── raw/                               # 原始数据
│   ├── tumonline_courses.jsonl
│   ├── moodle_materials/
│   ├── handbook_ws2024.pdf
│   └── fpso_documents/
├── processed/                         # 处理后的数据
│   ├── tumonline_cleaned.jsonl
│   ├── handbook_cleaned.jsonl
│   ├── merged_courses.jsonl
│   ├── conflicts_report.json
│   └── validation_report.json
└── corpus/                            # 最终语料库
    └── tum_courses_corpus.jsonl
```

---

## 详细步骤

### 阶段1: 数据采集

#### 1.1 TUMonline爬取

```python
# 示例：使用Selenium爬取TUMonline
from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import time

driver = webdriver.Chrome()
driver.get('https://campus.tum.de/tumonline/...')

# 登录流程（需要TUM账号）
username_field = driver.find_element(By.ID, 'username')
username_field.send_keys('YOUR_TUM_ID')
# ... 完整登录逻辑

# 爬取课程列表
courses = []
# ... 爬取逻辑

# 保存为JSONL
with open('raw/tumonline_courses.jsonl', 'w', encoding='utf-8') as f:
    for course in courses:
        json.dump(course, f, ensure_ascii=False)
        f.write('\n')
```

**⚠️ 注意事项**：
- 遵守TUMonline使用条款
- 添加延迟避免过载服务器（建议2-5秒/请求）
- 实现断点续爬功能
- 保存原始HTML用于调试

#### 1.2 Moodle材料下载

```python
# 示例：使用Moodle API下载课程材料
import requests

MOODLE_URL = "https://www.moodle.tum.de"
TOKEN = "your_moodle_token"  # 从 Preferences -> Security Keys 获取

# 获取已注册课程
response = requests.get(
    f"{MOODLE_URL}/webservice/rest/server.php",
    params={
        'wstoken': TOKEN,
        'wsfunction': 'core_enrol_get_users_courses',
        'userid': 'YOUR_USER_ID',
        'moodlewsrestformat': 'json'
    }
)

courses = response.json()

# 下载课程材料
for course in courses:
    # ... 下载逻辑
    pass
```

#### 1.3 PDF手册下载

```bash
# 从TUM官网下载课程手册
wget https://www.in.tum.de/fileadmin/.../module_handbook_ws2024.pdf \
     -O raw/handbook_ws2024.pdf
```

---

### 阶段2: 数据清洗

#### 2.1 PDF手册解析

```bash
# 使用pdf_handbook_parser.py解析PDF
python3 scripts/pdf_handbook_parser.py \
    raw/handbook_ws2024.pdf \
    processed/handbook_parsed.jsonl
```

**输出示例**：
```json
{
  "course_code": "IN2064",
  "title_en": "Introduction to Artificial Intelligence",
  "ects": "6",
  "lecturer": "Prof. Dr. Müller",
  "prerequisites": "Prerequisites: IN0001 (mandatory), MA0001 (recommended)",
  "learning_outcomes": "Students will learn...",
  "assessment": "90-minute written exam"
}
```

#### 2.2 数据清洗

```bash
# 使用data_cleaning_pipeline.py清洗数据
python3 -c "
import json
from scripts.data_cleaning_pipeline import TUMDataCleaner

cleaner = TUMDataCleaner()

with open('raw/tumonline_courses.jsonl', 'r') as f_in, \
     open('processed/tumonline_cleaned.jsonl', 'w') as f_out:
    for line in f_in:
        raw = json.loads(line)
        cleaned = cleaner.clean_course_data(raw, source='tumonline')
        json.dump(cleaned, f_out, ensure_ascii=False)
        f_out.write('\n')

print(cleaner.get_statistics())
"
```

**清洗操作包括**：
- ✅ 统一Unicode字符（德语umlauts）
- ✅ 去除PDF噪音（页码、水印）
- ✅ 标准化课程代码（IN 2064 → IN2064）
- ✅ 提取ECTS学分（"6 ECTS" → 6）
- ✅ 解析学期信息（Wintersemester → WS）
- ✅ 提取先修课程
- ✅ 检测语言（de/en/mixed）

---

### 阶段3: 数据合并与去重

```bash
# 使用data_deduplication.py合并多个数据源
python3 -c "
import json
from scripts.data_deduplication import CourseDataMerger

merger = CourseDataMerger()

# 加载TUMonline数据（高优先级）
with open('processed/tumonline_cleaned.jsonl', 'r') as f:
    for line in f:
        course = json.loads(line)
        merger.add_course_data(course['course_code'], 'tumonline', course)

# 加载手册数据（次优先级）
with open('processed/handbook_cleaned.jsonl', 'r') as f:
    for line in f:
        course = json.loads(line)
        merger.add_course_data(course['course_code'], 'handbook', course)

# 导出合并结果
with open('processed/merged_courses.jsonl', 'w') as f:
    for course_id in merger.courses.keys():
        merged = merger.get_merged_course(course_id)
        json.dump(merged, f, ensure_ascii=False)
        f.write('\n')

# 导出冲突报告
merger.export_conflicts_report('processed/conflicts_report.json')
"
```

**冲突解决策略**：
- TUMonline优先级最高（最权威）
- 手册数据次之（官方文档）
- Moodle材料最后（可能过时）

---

### 阶段4: 数据验证

```bash
# 验证数据完整性
python3 -c "
import json
from scripts.data_cleaning_pipeline import TUMDataCleaner

cleaner = TUMDataCleaner()
validation_report = []

with open('processed/merged_courses.jsonl', 'r') as f:
    for line in f:
        course = json.loads(line)
        is_valid, issues = cleaner.validate_course_data(course)

        if not is_valid:
            validation_report.append({
                'course_code': course['course_code'],
                'issues': issues
            })

print(f'验证结果: {len(validation_report)} 个课程存在问题')

with open('processed/validation_report.json', 'w') as f:
    json.dump(validation_report, f, ensure_ascii=False, indent=2)
"
```

**验证检查项**：
- ✔️ 必需字段完整性（course_code必须存在）
- ✔️ 课程代码格式（符合 [A-Z]{2,3}\d{4} 模式）
- ✔️ ECTS合理性（1-30之间）
- ✔️ 至少有一种语言的标题
- ✔️ 学期格式正确（WS/SS）

---

### 阶段5: 转换为FlashRAG格式

```bash
# 转换为FlashRAG所需的JSONL格式
python3 scripts/convert_to_flashrag_format.py \
    --input processed/merged_courses.jsonl \
    --output corpus/tum_courses_corpus.jsonl \
    --granularity course  # 可选: course/section/material
```

**三种粒度对比**：

| 粒度 | 说明 | 适用场景 | 示例文档数 |
|------|------|----------|-----------|
| **course** | 一个课程一条记录 | 通用查询（"IN2064是什么课程？"） | 1个/课程 |
| **section** | 按章节切分 | 细粒度查询（"IN2064的先修课程？"） | 5-7个/课程 |
| **material** | 按材料切分 | 材料检索（"第3章讲了什么？"） | 10-50个/课程 |

**FlashRAG格式示例**：
```json
{
  "id": "IN2064",
  "contents": "IN2064 - Introduction to Artificial Intelligence / Einführung in die Künstliche Intelligenz\nECTS: 6\nSemester: WS\nPrerequisites: IN0001, MA0001\n\nDescription:\nThis course introduces fundamental concepts of AI..."
}
```

---

## 脚本说明

### 1. `data_cleaning_pipeline.py`

**功能**：核心数据清洗逻辑

**主要类**：
- `TUMDataCleaner`: 清洗器主类
  - `clean_text()`: 文本清洗
  - `normalize_course_code()`: 课程代码标准化
  - `extract_prerequisites()`: 先修课程提取
  - `detect_language()`: 语言检测
  - `validate_course_data()`: 数据验证

**使用方法**：
```python
from data_cleaning_pipeline import TUMDataCleaner

cleaner = TUMDataCleaner()
cleaned = cleaner.clean_course_data(raw_data, source='tumonline')
is_valid, issues = cleaner.validate_course_data(cleaned)
```

---

### 2. `pdf_handbook_parser.py`

**功能**：解析PDF课程手册

**主要方法**：
- `extract_text_from_pdf()`: PDF文本提取
- `split_into_course_blocks()`: 按课程切分
- `parse_course_block()`: 解析单个课程

**命令行用法**：
```bash
python3 pdf_handbook_parser.py handbook.pdf output.jsonl
```

---

### 3. `data_deduplication.py`

**功能**：合并多个数据源，处理冲突

**优先级**：tumonline(4) > handbook(3) > moodle(2) > fpso(1)

**使用方法**：
```python
from data_deduplication import CourseDataMerger

merger = CourseDataMerger()
merger.add_course_data('IN2064', 'tumonline', data1)
merger.add_course_data('IN2064', 'handbook', data2)

merged = merger.get_merged_course('IN2064')
merger.export_conflicts_report('conflicts.json')
```

---

### 4. `convert_to_flashrag_format.py`

**功能**：转换为FlashRAG检索格式

**命令行参数**：
- `--input`: 输入文件
- `--output`: 输出文件
- `--granularity`: 粒度 (course/section/material)
- `--clean`: 是否先清洗

**示例**：
```bash
python3 convert_to_flashrag_format.py \
    --input merged.jsonl \
    --output corpus.jsonl \
    --granularity section \
    --clean
```

---

### 5. `build_knowledge_base.sh`

**功能**：端到端自动化脚本

**使用方法**：
```bash
# 交互式运行（会提示确认每个步骤）
bash build_knowledge_base.sh

# 自动运行所有步骤
bash build_knowledge_base.sh --auto
```

---

## 常见问题

### Q1: PDF解析失败，提取的文本乱码怎么办？

**A**: 可能是扫描版PDF，需要OCR：
```bash
# 安装Tesseract OCR
sudo apt-get install tesseract-ocr tesseract-ocr-deu

# 使用OCR模式解析
python3 pdf_handbook_parser.py handbook.pdf output.jsonl --ocr
```

### Q2: 如何处理德语特殊字符（ä, ö, ü, ß）？

**A**: `data_cleaning_pipeline.py`已自动处理：
```python
# Unicode规范化
text = unicodedata.normalize('NFC', text)
```

### Q3: 数据源冲突如何解决？

**A**: 查看`conflicts_report.json`，手动审核：
```json
{
  "course_id": "IN2064",
  "conflicts": [
    {
      "field": "ects",
      "values": {"tumonline": 6, "handbook": 5},
      "resolved_with": "tumonline"
    }
  ]
}
```

### Q4: 如何更新已有知识库？

**A**: 使用增量更新脚本：
```python
from incremental_update import IncrementalUpdater

updater = IncrementalUpdater()
if updater.update_course('IN2064', new_data, 'tumonline'):
    print("数据已更新，需要重建索引")
```

### Q5: 如何支持更多语言（如中文课程描述）？

**A**: 修改`detect_language()`添加中文检测：
```python
def detect_language(self, text: str) -> str:
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chinese_chars) > 10:
        return 'zh'
    # ... 原有逻辑
```

---

## 示例与最佳实践

### 完整流程示例

```bash
# 1. 准备原始数据
mkdir -p data/raw
# ... 将采集的数据放入 data/raw/

# 2. 运行自动化脚本
cd data/scripts
bash build_knowledge_base.sh

# 3. 检查输出
cat ../corpus/tum_courses_corpus.jsonl | head -5
cat ../processed/conflicts_report.json
cat ../processed/validation_report.json

# 4. 构建索引（下一步）
bash build_index.sh
```

### 最佳实践

#### ✅ 推荐做法

1. **保留原始数据**：永远不要删除`raw/`目录
2. **版本控制**：为每学期的数据创建版本（如`ws2024`, `ss2025`）
3. **增量更新**：使用`incremental_update.py`避免重复处理
4. **人工审核**：检查`conflicts_report.json`中的冲突
5. **测试驱动**：先用少量数据测试流程

#### ❌ 避免的错误

1. ❌ 直接修改原始数据文件
2. ❌ 跳过数据验证步骤
3. ❌ 忽略冲突报告
4. ❌ 使用过细的粒度（material级）导致索引过大
5. ❌ 忘记处理德语特殊字符

---

## 下一步

数据清洗完成后，继续：

1. **构建检索索引**：
   ```bash
   bash data/scripts/build_index.sh
   ```

2. **测试RAG系统**：
   ```bash
   python3 benchmark/R3-RAG/src/inference.py
   ```

3. **部署可视化界面**：
   ```bash
   cd startup
   bash startup_visualize.sh
   ```

---

## 联系与支持

如有问题，请查阅：
- R3-RAG主文档: `/home/user/R3-RAG/readme.md`
- FlashRAG文档: `/home/user/R3-RAG/tool/FlashRAG/docs/`

---

**最后更新**: 2025-12-16
