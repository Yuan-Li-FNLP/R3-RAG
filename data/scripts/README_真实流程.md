# TUM课程知识库 - 真实创建流程（最详细版）

## 🎯 核心概念澄清

### 问题1: 结构化 vs 分块，有什么区别？

#### **结构化（Structuring）= 格式转换**

```
作用：把不同格式的文件转成统一的文本数据

输入：各种格式的文件
├── AI课程手册.pdf        (PDF格式，二进制)
├── TUMonline导出.csv     (表格格式)
└── 讲义.docx             (Word格式)

处理：提取文本
├── PyMuPDF读取PDF，提取文本
├── Pandas读取CSV，转成文本
└── python-docx读取Word，提取文本

输出：纯文本数据（JSON格式）
{
  "source_file": "AI课程手册.pdf",
  "full_text": "IN2064 - Einführung in die KI\n\n本课程介绍...",
  "length": 10000
}
```

**为什么需要结构化？**
- 计算机无法直接理解PDF、Word等二进制格式
- 必须提取出纯文本才能进行后续处理

---

#### **分块（Chunking）= 切小段**

```
作用：把长文档切成小块，方便检索

输入：一篇长文档（结构化后）
"IN2064 - Einführung in die KI\n\n
Modulbeschreibung:\n
Dieses Modul führt in die Grundlagen der KI ein...\n\n
Voraussetzungen:\n
- IN0001 Informatik Grundlagen\n
- MA0001 Lineare Algebra\n\n
Kapitel 1: Suchverfahren\n
Uninformierte Suche umfasst Algorithmen wie BFS und DFS...\n\n
Kapitel 2: Wissensrepräsentation\n
..."
（共10,000字）

处理：按段落切分
- 每块约500字
- 块之间有50字重叠

输出：多个小块
块0 (500字):
"IN2064 - Einführung in die KI\n\n
Modulbeschreibung:\n
Dieses Modul führt in die Grundlagen der KI ein..."

块1 (500字):
"Voraussetzungen:\n
- IN0001 Informatik Grundlagen\n
- MA0001 Lineare Algebra\n\n
Lernziele:..."

块2 (500字):
"Kapitel 1: Suchverfahren\n
Uninformierte Suche umfasst Algorithmen wie BFS..."

...
块19 (500字)
```

**为什么需要分块？**
1. **AI模型限制**：embedding模型一次只能处理512字符
2. **检索精度**：太长的文档包含太多主题，降低检索准确度
3. **返回内容**：用户查询"BFS算法"，只需返回相关段落，不需要整本书

---

### 对比表

| 操作 | 结构化 | 分块 |
|------|--------|------|
| **英文名** | Structuring | Chunking |
| **输入** | PDF、CSV、Word等文件 | 长文本 |
| **输出** | 纯文本 | 小文本块 |
| **目的** | 格式转换 | 长度控制 |
| **工具** | PyMuPDF, pandas | 字符串分割 |
| **比喻** | 把书扫描成电子文本 | 把整本书切成章节段落 |

---

## 📚 真实流程（4个阶段）

### 阶段0: 准备数据

把你的所有文件放到一个文件夹：

```bash
mkdir -p /home/user/my_course_data

# 示例文件结构
/home/user/my_course_data/
├── AI课程手册.pdf          (20页，来自官网下载)
├── AI讲义第1-3章.pdf       (50页，Moodle下载)
├── 数据库课程手册.pdf       (15页)
├── TUMonline导出.csv       (100行，课程列表)
└── 其他任何文件...
```

---

### 阶段1: 结构化 - 提取文本

**脚本**：`real_pipeline_step1_extract.py`

**做什么**：
- 读取PDF，提取文本
- 读取CSV，转成文本
- 统一保存为JSON

**运行**：
```bash
python3 real_pipeline_step1_extract.py \
    --input /home/user/my_course_data \
    --output extracted_texts.json
```

**输入示例**：
```
my_course_data/AI课程手册.pdf
```

**输出示例**：`extracted_texts.json`
```json
[
  {
    "source_file": "AI课程手册.pdf",
    "file_type": "pdf",
    "full_text": "IN2064 - Einführung in die KI\n\nModulbeschreibung:\nDieses Modul...(10000字)",
    "length": 10000,
    "pages": 20
  },
  {
    "source_file": "AI讲义第1-3章.pdf",
    "file_type": "pdf",
    "full_text": "Kapitel 1: Suchverfahren\nUninformierte Suche...(25000字)",
    "length": 25000,
    "pages": 50
  },
  ...
]
```

**关键点**：
- ✅ PDF二进制 → 纯文本
- ✅ 所有文件 → 统一JSON格式
- ❌ 还没有切分，每个文档是完整的

---

### 阶段2: 分块 - 切成小段

**脚本**：`real_pipeline_step2_chunk.py`

**做什么**：
- 读取阶段1的JSON
- 把长文档切成500字的小块
- 块之间保留50字重叠
- 转换成FlashRAG所需的JSONL格式

**运行**：
```bash
python3 real_pipeline_step2_chunk.py \
    --input extracted_texts.json \
    --output corpus.jsonl \
    --chunk_size 500 \
    --overlap 50
```

**输入**：`extracted_texts.json`（来自阶段1）

**输出**：`corpus.jsonl`
```jsonl
{"id": "0", "contents": "IN2064 - Einführung in die KI\n\nModulbeschreibung:\nDieses Modul führt..."}
{"id": "1", "contents": "Voraussetzungen:\n- IN0001 Informatik Grundlagen\n- MA0001 Lineare Algebra..."}
{"id": "2", "contents": "Kapitel 1: Suchverfahren\nUninformierte Suche umfasst Algorithmen wie BFS..."}
...
{"id": "150", "contents": "Referenzen:\n[1] Russell, S. & Norvig, P. (2020) Artificial Intelligence..."}
```

**关键点**：
- ✅ 长文档（10000字）→ 小块（500字 × 20块）
- ✅ 重叠避免信息丢失
- ✅ JSONL格式（每行一个JSON对象）
- ❌ 还没有向量化，只是文本

---

### 阶段3: 构建索引 - 向量化

**工具**：FlashRAG的`index_builder`

**做什么**：
- 读取corpus.jsonl
- 用AI模型把每个文本块转成向量（数字数组）
- 保存到FAISS索引（可以快速搜索）

**运行**：
```bash
python3 -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-base \
    --corpus_path corpus.jsonl \
    --save_dir indexes/my_kb \
    --batch_size 32 \
    --pooling_method mean \
    --faiss_type Flat
```

**处理过程**：
```
读取: {"id": "0", "contents": "IN2064 - Einführung in die KI..."}
  ↓
AI模型 (multilingual-e5-base)
  ↓
向量: [0.234, -0.456, 0.678, ..., 0.123]  (768维)
  ↓
保存到FAISS索引
```

**输出**：`indexes/my_kb/`目录
```
indexes/my_kb/
├── index.faiss         (FAISS索引文件)
├── docid.json          (文档ID映射)
└── config.json         (配置信息)
```

**关键点**：
- ✅ 文本 → 向量（数字）
- ✅ 可以快速搜索相似向量
- ✅ 支持跨语言（英语查询 → 德语文档）

---

### 阶段4: 测试检索

**脚本**：`real_pipeline_step3_test.py`

**做什么**：
- 加载索引
- 测试几个查询
- 验证检索结果

**运行**：
```bash
python3 real_pipeline_step3_test.py \
    --index_path indexes/my_kb \
    --model_path intfloat/multilingual-e5-base
```

**测试查询**：
```
查询1: "Was sind die Voraussetzungen für den KI-Kurs?"
期望: 找到包含"Voraussetzungen: IN0001, MA0001"的块

查询2: "What is BFS algorithm?"
期望: 找到包含"Breitensuche (BFS)"的块
```

**输出示例**：
```
===========================================================
测试 1/4: 查询课程先修要求（德语）
===========================================================
查询: Was sind die Voraussetzungen für den KI-Kurs?

✅ 找到 3 个结果:

  结果 1 (ID: 1) 🇩🇪
  相似度: 0.8523
  内容: Voraussetzungen:
- IN0001 Informatik Grundlagen
- MA0001 Lineare Algebra

Lernziele:
1. Verständnis von Suchverfahren...

  结果 2 (ID: 0) 🇩🇪
  相似度: 0.7234
  内容: IN2064 - Einführung in die Künstliche Intelligenz

Modulbeschreibung:
Dieses Modul führt in die Grundlagen der KI ein...
```

---

## 🚀 一键运行所有步骤

**脚本**：`real_pipeline_run_all.sh`

**使用方法**：
```bash
# 1. 准备数据
mkdir -p /home/user/my_course_data
# 把你的PDF、CSV文件放进去

# 2. 运行一键脚本
cd /home/user/R3-RAG/data/scripts
bash real_pipeline_run_all.sh /home/user/my_course_data

# 3. 跟着提示操作（每个阶段完成后会暂停）
```

**脚本会自动执行**：
1. ✅ 阶段1: 结构化（提取文本）
2. ✅ 阶段2: 分块（切小段）
3. ✅ 阶段3: 构建索引（向量化）
4. ✅ 阶段4: 测试检索（验证）

---

## 📊 数据流示意图

```
【原始文件】
├── AI课程手册.pdf (20页)
├── AI讲义.pdf (50页)
└── TUMonline.csv (100行)
         ↓

【阶段1: 结构化】real_pipeline_step1_extract.py
输入: PDF、CSV文件
处理: PyMuPDF提取文本
输出: extracted_texts.json
      [
        {"source_file": "...", "full_text": "10000字"},
        ...
      ]
         ↓

【阶段2: 分块】real_pipeline_step2_chunk.py
输入: extracted_texts.json
处理: 按500字切分，50字重叠
输出: corpus.jsonl
      {"id": "0", "contents": "500字块"}
      {"id": "1", "contents": "500字块"}
      ...
      （共150个块）
         ↓

【阶段3: 构建索引】flashrag.retriever.index_builder
输入: corpus.jsonl
处理: AI模型转向量 → FAISS索引
输出: indexes/my_kb/
      ├── index.faiss
      ├── docid.json
      └── config.json
         ↓

【阶段4: 检索】real_pipeline_step3_test.py
输入: 用户查询 "prerequisites?"
处理: 查询 → 向量 → 搜索 → 返回top-k
输出: 最相关的3个文本块
```

---

## 🎯 常见疑问解答

### Q1: 为什么要分块？整个文档不行吗？

**A**: 不行，因为：
1. **AI模型限制**：embedding模型最多处理512字符
2. **检索精度**：10000字的文档包含很多主题，查询"BFS"会匹配到整个文档，但用户只需要BFS那一段
3. **效率**：返回500字比返回10000字快得多

### Q2: 块太小会不会信息不完整？

**A**: 有重叠机制：
```
块1: "...BFS是一种图搜索算法，它从根节点开始..."
块2: "...从根节点开始，逐层遍历所有邻居节点..."
      ↑ 50字重叠，避免切断关键信息
```

### Q3: 课程信息和授课内容要分开吗？

**A**: 不需要！
- 都会被切成小块
- 查询"先修课程"自动匹配到课程信息块
- 查询"BFS算法"自动匹配到授课内容块
- RAG系统根据语义相似度自动路由

### Q4: 块大小如何选择？

| 块大小 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| 200字 | 精确匹配 | 信息不完整 | 问答型（单句回答）|
| **500字** | 平衡 | - | **推荐** |
| 1000字 | 上下文丰富 | 检索精度下降 | 长文档生成 |

**推荐500字**，因为：
- 通常包含1-2个完整段落
- 足够完整，又不会太长
- 符合大多数查询需求

### Q5: 重叠多少合适？

**推荐50-100字（10-20%）**：
- 太少（10字）：关键信息可能被切断
- 太多（200字）：浪费存储空间，降低多样性
- **50字**：平衡信息完整性和效率

---

## 📝 文件说明

| 文件 | 作用 |
|------|------|
| `real_pipeline_step1_extract.py` | 阶段1: 结构化 - 从PDF/CSV提取文本 |
| `real_pipeline_step2_chunk.py` | 阶段2: 分块 - 切成小段 |
| `real_pipeline_step3_test.py` | 阶段4: 测试检索 |
| `real_pipeline_run_all.sh` | 一键运行所有阶段 |
| `README_真实流程.md` | 本文档 |

---

## 🎓 总结

### 记住3个关键点：

1. **结构化 ≠ 分块**
   - 结构化 = 格式转换（PDF → 文本）
   - 分块 = 长度控制（10000字 → 500字×20）

2. **流程是线性的**
   ```
   原始文件 → 结构化 → 分块 → 索引 → 检索
   ```

3. **不需要手动分类**
   - 不需要区分"这是课程信息""这是授课内容"
   - RAG系统根据查询自动找到最相关的块

---

## 🚀 现在开始！

```bash
# 复制这些命令，按顺序运行：

# 1. 准备数据文件夹
mkdir -p /home/user/my_course_data

# 2. 把你的PDF、CSV文件放进去（用文件管理器或cp命令）

# 3. 运行一键脚本
cd /home/user/R3-RAG/data/scripts
bash real_pipeline_run_all.sh /home/user/my_course_data

# 4. 等待完成（5-15分钟）

# 5. 测试！
```

**就这么简单！** 🎉

---

**有任何问题，随时问我！** 💬
