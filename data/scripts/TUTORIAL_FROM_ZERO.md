# 从零开始创建TUM课程知识库 - 小白教程

## 🎯 目标
创建一个能回答TUM课程问题的RAG系统，比如：
- "IN2064的先修课程是什么？"（课程信息）
- "人工智能第3章讲了什么？"（授课内容）

---

## 📦 你需要的数据（举例）

假设你有以下几种文件：

### 1. 课程目录（来自TUMonline）
```
IN2064 - Introduction to AI
ECTS: 6
讲师: Prof. Müller
先修课程: IN0001, MA0001
```

### 2. 课程大纲（PDF手册）
```
IN2064 - Einführung in die KI

Lernziele:
- Verständnis von Suchalgorithmen
- Kenntnisse über maschinelles Lernen
- ...

Prüfung: 90分钟笔试
```

### 3. 讲义内容（Lecture Slides）
```
Kapitel 1: Suchverfahren
- Breiten-suche (BFS)
- Tiefen-suche (DFS)
- A*-Algorithmus

Kapitel 2: Wissensrepräsentation
- Logik
- Ontologien
- ...
```

---

## 🤔 问题：这些不同的内容需要分开吗？

### **答案：NO！统一处理即可**

**原因**：
1. **用户不关心数据来自哪里**
   - 用户问："IN2064的先修课程？"
   - 他不在乎答案来自TUMonline还是PDF手册

2. **RAG系统会自动找到最相关的内容**
   - 用户问："第1章讲了什么？"
   - 系统自动检索到"Kapitel 1: Suchverfahren"
   - 用户问："先修课程是什么？"
   - 系统自动检索到"先修课程: IN0001, MA0001"

3. **统一处理更简单**
   - 不需要判断"这是课程信息还是授课内容"
   - 不需要维护多个索引
   - 不需要复杂的路由逻辑

---

## 🛠️ 实际操作（5步走）

### 第0步：准备你的原始数据

把你收集的所有文件放到一个文件夹：
```bash
mkdir -p /home/user/R3-RAG/data/my_raw_data

# 放入你的文件
data/my_raw_data/
├── tumonline_export.csv       # TUMonline导出的课程列表
├── module_handbook.pdf         # 课程手册PDF
├── ai_lecture_slides.pdf       # AI课程讲义
├── database_lecture_slides.pdf # 数据库课程讲义
└── ...
```

---

### 第1步：手动创建一个最小示例

**不要用复杂的脚本！先手动创建一个最小的示例corpus**

创建文件：`data/my_raw_data/sample_corpus.jsonl`

```jsonl
{"id": "0", "contents": "IN2064 - Introduction to Artificial Intelligence\nECTS: 6\nPrerequisites: IN0001 (Informatik Grundlagen), MA0001 (Lineare Algebra)\nLecturer: Prof. Dr. Müller\nSemester: Wintersemester"}
{"id": "1", "contents": "IN2064 - Kapitel 1: Suchverfahren\nInhalt:\n- Breitensuche (BFS)\n- Tiefensuche (DFS)\n- A*-Algorithmus\n- Heuristiken"}
{"id": "2", "contents": "IN2064 - Kapitel 2: Wissensrepräsentation\nInhalt:\n- Aussagenlogik\n- Prädikatenlogik\n- Ontologien\n- Semantische Netze"}
{"id": "3", "contents": "IN2118 - Datenbanksysteme\nECTS: 6\nPrerequisites: IN0001\nLecturer: Prof. Dr. Schmidt\nSemester: Sommersemester"}
{"id": "4", "contents": "IN2118 - Kapitel 1: Relationale Datenmodelle\nInhalt:\n- ER-Diagramme\n- Normalisierung\n- SQL Grundlagen"}
```

**解释**：
- 每一行是一个独立的"文档块"
- `id`：唯一标识符（任意字符串）
- `contents`：文档内容（课程信息或授课内容都可以）

**看到了吗？课程信息和授课内容混在一起，完全没问题！** ✅

---

### 第2步：构建索引（真正的RAG开始了）

```bash
cd /home/user/R3-RAG

# 安装FlashRAG（如果还没装）
pip install flashrag-pip

# 构建索引（使用多语言模型支持英语查询→德语文档）
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-base \
    --corpus_path data/my_raw_data/sample_corpus.jsonl \
    --save_dir indexes/my_first_index \
    --batch_size 32 \
    --pooling_method mean \
    --faiss_type Flat
```

**这一步做了什么？**
- 读取你的`sample_corpus.jsonl`
- 用AI模型把每个文档转换成向量（数字）
- 保存成FAISS索引（一个可以快速搜索的数据库）

**预期输出**：
```
Building index...
Processing 5 documents...
Index saved to indexes/my_first_index
✅ Done!
```

---

### 第3步：测试检索（看看能不能找到文档）

创建测试脚本：`test_my_retrieval.py`

```python
from flashrag.retriever import Retriever

# 加载你刚构建的索引
retriever = Retriever(
    method="e5",
    index_path="indexes/my_first_index",
    model_path="intfloat/multilingual-e5-base"
)

# 测试1: 英语查询 → 课程信息
print("=== 测试1: 查询先修课程 ===")
query1 = "What are the prerequisites for AI course?"
results1 = retriever.search(query1, top_k=2)

for i, result in enumerate(results1, 1):
    print(f"\n结果 {i}:")
    print(result['contents'][:200])  # 只显示前200字符

# 测试2: 德语查询 → 授课内容
print("\n\n=== 测试2: 查询第1章内容 ===")
query2 = "Was ist in Kapitel 1 über Suchverfahren?"
results2 = retriever.search(query2, top_k=2)

for i, result in enumerate(results2, 1):
    print(f"\n结果 {i}:")
    print(result['contents'][:200])
```

运行：
```bash
python test_my_retrieval.py
```

**预期输出**：
```
=== 测试1: 查询先修课程 ===

结果 1:
IN2064 - Introduction to Artificial Intelligence
ECTS: 6
Prerequisites: IN0001 (Informatik Grundlagen), MA0001 (Lineare Algebra)
...

=== 测试2: 查询第1章内容 ===

结果 1:
IN2064 - Kapitel 1: Suchverfahren
Inhalt:
- Breitensuche (BFS)
- Tiefensuche (DFS)
...
```

**看！不同类型的内容（课程信息 vs 授课内容）都能被正确检索到！** 🎉

---

### 第4步：整合到RAG系统（生成答案）

现在加上LLM来生成答案（而不只是检索文档）

创建：`test_my_rag.py`

```python
from flashrag import RAG
from flashrag.config import Config

# 配置
config = {
    'retrieval_method': 'e5',
    'index_path': 'indexes/my_first_index',
    'model_path': 'intfloat/multilingual-e5-base',
    'generator_model': 'meta-llama/Llama-3.1-8B-Instruct',  # 或其他LLM
    'retrieval_topk': 3
}

# 创建RAG系统
rag = RAG(config=Config(config))

# 测试问答
questions = [
    "What are the prerequisites for the AI course IN2064?",
    "Tell me about search algorithms in chapter 1",
    "What topics are covered in the database course?"
]

for question in questions:
    print(f"\n问题: {question}")
    answer = rag.generate(question)
    print(f"答案: {answer}\n")
    print("-" * 80)
```

**预期输出**：
```
问题: What are the prerequisites for the AI course IN2064?
答案: The prerequisites for IN2064 (Introduction to Artificial Intelligence) are:
- IN0001 (Informatik Grundlagen / Computer Science Fundamentals)
- MA0001 (Lineare Algebra / Linear Algebra)

问题: Tell me about search algorithms in chapter 1
答案: Chapter 1 covers search algorithms including:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Algorithm
- Heuristics
...
```

---

### 第5步：扩展到真实数据

现在你理解了原理，可以处理真实数据了：

```bash
# 使用之前我写的脚本来处理你的真实数据

# 1. 解析PDF手册
python data/scripts/pdf_handbook_parser.py \
    data/my_raw_data/module_handbook.pdf \
    data/processed/handbook.jsonl

# 2. 处理TUMonline数据（你需要先把CSV转成JSONL）
python -c "
import csv, json
with open('data/my_raw_data/tumonline_export.csv', 'r') as f_in, \
     open('data/processed/tumonline.jsonl', 'w') as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        # 构建contents字段
        contents = f\"{row['course_code']} - {row['title']}\\n\"
        contents += f\"ECTS: {row['ects']}\\n\"
        contents += f\"Lecturer: {row['lecturer']}\\n\"
        # ...
        doc = {'id': row['course_code'], 'contents': contents}
        f_out.write(json.dumps(doc, ensure_ascii=False) + '\\n')
"

# 3. 合并所有数据
cat data/processed/handbook.jsonl \
    data/processed/tumonline.jsonl \
    > data/corpus/final_corpus.jsonl

# 4. 重新构建索引
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-base \
    --corpus_path data/corpus/final_corpus.jsonl \
    --save_dir indexes/tum_full_index
```

---

## 🎓 关键理解

### ✅ **统一处理的优势**

| 分开处理 ❌ | 统一处理 ✅ |
|------------|-----------|
| 需要判断文档类型 | 不需要判断，一视同仁 |
| 维护多个索引 | 只有一个索引 |
| 查询时需要路由逻辑 | 自动找到最相关内容 |
| 实现复杂 | 实现简单 |

### 🧠 **为什么能自动区分？**

RAG系统的工作原理：
```
用户查询: "先修课程是什么？"
    ↓
转换成向量: [0.23, -0.45, 0.67, ...]
    ↓
在索引中搜索最相似的向量
    ↓
找到: "Prerequisites: IN0001, MA0001" （课程信息文档）
    ↓
生成答案
```

```
用户查询: "第1章讲了什么？"
    ↓
转换成向量: [0.12, 0.34, -0.56, ...]
    ↓
在索引中搜索最相似的向量
    ↓
找到: "Kapitel 1: Suchverfahren..." （授课内容文档）
    ↓
生成答案
```

**系统根据语义相似度自动匹配，不需要你手动区分！**

---

## 🤷 **什么时候需要分开？**

只有在以下**极少数**情况下才需要分开：

### 1. 完全不同的检索策略
```
课程信息 → 用精确匹配（课程代码）
授课内容 → 用语义搜索（模糊匹配）
```
但这不常见，统一用语义搜索就够了。

### 2. 不同的访问权限
```
课程信息 → 所有人可见
授课内容 → 只有选课学生可见
```
这需要权限控制，确实要分开索引。

### 3. 更新频率完全不同
```
课程信息 → 每学期更新一次
授课内容 → 每周更新
```
可以分开索引，方便增量更新。

**但对于你的场景，统一处理就足够了！** ✅

---

## 📝 总结：你只需要3个文件

1. **corpus.jsonl** - 所有文档（课程信息+授课内容混在一起）
2. **index/** - FAISS索引（用上面的命令构建）
3. **test.py** - 测试脚本（验证检索效果）

就这么简单！

---

## 🆘 还是不会？我帮你写个一键脚本

如果你还是觉得复杂，告诉我：
1. 你现在手头有什么文件？（PDF? CSV? 文本文件？）
2. 大概有多少个课程？
3. 文件在哪个文件夹？

我给你写一个**专属的、一键运行的脚本**，你只需要：
```bash
bash MY_EASY_SCRIPT.sh
```
就能创建好知识库！

**不要放弃，我会帮你搞定的！** 💪
