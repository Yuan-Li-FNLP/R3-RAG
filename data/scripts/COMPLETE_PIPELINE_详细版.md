# TUM课程知识库建立 - 完整详细步骤（每一步都拆解）

## 📋 目录

1. [流程概览](#流程概览)
2. [阶段0: 准备](#阶段0-准备)
3. [阶段1: 结构化（提取文本）](#阶段1-结构化提取文本)
4. [阶段2: 数据清洗](#阶段2-数据清洗)
5. [阶段3: 分块](#阶段3-分块)
6. [阶段4: 构建索引](#阶段4-构建索引)
7. [阶段5: 测试](#阶段5-测试)
8. [参数调优](#参数调优)

---

## 流程概览

```
┌─────────────────┐
│ 阶段0: 准备     │  收集PDF、CSV等文件
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 阶段1: 结构化   │  PDF → 文本（格式转换）
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 阶段2: 数据清洗 │  去除噪音、标准化（独立步骤！）
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 阶段3: 分块     │  长文本 → 小段（长度控制）
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 阶段4: 构建索引 │  文本 → 向量（FAISS）
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 阶段5: 测试     │  验证检索效果
└─────────────────┘
```

---

## 阶段0: 准备

### 0.1 创建工作目录

```bash
# 创建主目录
mkdir -p /home/user/tum_kb_project

# 创建子目录
cd /home/user/tum_kb_project
mkdir -p raw_files      # 原始文件
mkdir -p step1_extracted   # 阶段1输出
mkdir -p step2_cleaned     # 阶段2输出
mkdir -p step3_chunked     # 阶段3输出
mkdir -p step4_index       # 阶段4输出
mkdir -p logs              # 日志
```

### 0.2 收集原始文件

把你的所有文件放到 `raw_files/`：

```bash
/home/user/tum_kb_project/raw_files/
├── AI课程手册_WS2024.pdf         (官网下载)
├── AI讲义_Kapitel1-3.pdf         (Moodle)
├── 数据库课程手册_SS2024.pdf
├── TUMonline课程列表.csv
└── ... (更多文件)
```

### 0.3 安装依赖

```bash
pip install PyMuPDF pandas flashrag-pip
```

---

## 阶段1: 结构化（提取文本）

### 1.1 目标

把不同格式的文件转成统一的**纯文本**：

```
输入: AI课程手册.pdf (PDF二进制)
输出: {"source": "AI课程手册.pdf", "text": "IN2064 - Einführung..."}
```

### 1.2 处理PDF

**脚本**: `step1_extract_pdf.py`

```python
import fitz  # PyMuPDF
import json
from pathlib import Path

def extract_pdf(pdf_path):
    """提取PDF文本"""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return {
        "source_file": pdf_path.name,
        "raw_text": text,
        "num_pages": len(doc),
        "num_chars": len(text)
    }

# 使用
input_dir = Path("raw_files")
output_file = "step1_extracted/all_documents.json"

all_docs = []
for pdf_file in input_dir.glob("*.pdf"):
    print(f"处理: {pdf_file.name}")
    doc_data = extract_pdf(pdf_file)
    all_docs.append(doc_data)
    print(f"  提取了 {doc_data['num_chars']} 字符")

# 保存
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_docs, f, ensure_ascii=False, indent=2)

print(f"✅ 完成！共 {len(all_docs)} 个文档")
```

**运行**:
```bash
python step1_extract_pdf.py
```

**输出**: `step1_extracted/all_documents.json`
```json
[
  {
    "source_file": "AI课程手册_WS2024.pdf",
    "raw_text": "IN2064 - Einführung in die KI\n\nModulbeschreibung:\n...",
    "num_pages": 20,
    "num_chars": 12500
  },
  ...
]
```

**关键点**:
- ✅ PDF二进制 → 纯文本
- ❌ **还没有清洗**，文本可能有噪音（页码、乱码等）
- ❌ **还没有分块**，每个文档是完整的

---

## 阶段2: 数据清洗

### 2.1 目标

清除文本中的噪音，标准化格式：

```
输入: "Page 1 of 20\nIN2064 - Einführung   in die KI\n\n\n本课程..."
输出: "IN2064 - Einführung in die KI\n本课程..."
```

### 2.2 清洗内容

需要清洗的内容：

1. **去除PDF噪音**
   - 页码: "Page 1 of 20"
   - 页眉页脚
   - 水印: "DRAFT", "CONFIDENTIAL"

2. **统一空白符**
   - 多个空格 → 一个空格
   - 多个换行 → 最多两个换行

3. **修复PDF提取错误**
   - 连字符断词: "Infor-\nmation" → "Information"
   - 德语特殊字符: 确保 ä, ö, ü, ß 正确

4. **标准化格式**
   - 课程代码: "in 2064" → "IN2064"
   - 统一换行符: \r\n → \n

### 2.3 清洗脚本

**脚本**: `step2_clean_text.py`

```python
import json
import re
import unicodedata

def clean_text(text):
    """
    清洗文本
    """
    if not text:
        return ""

    # 1. 统一Unicode（处理德语字符）
    text = unicodedata.normalize('NFC', text)

    # 2. 去除页码
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Seite \d+ von \d+', '', text, flags=re.IGNORECASE)

    # 3. 去除水印
    text = re.sub(r'\bDRAFT\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCONFIDENTIAL\b', '', text, flags=re.IGNORECASE)

    # 4. 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 5. 修复连字符断词
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # 6. 压缩多余空白
    # 多个空格 → 一个空格
    text = re.sub(r' +', ' ', text)
    # 多个换行 → 最多两个换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. 去除行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 8. 标准化课程代码
    text = re.sub(r'\b([A-Z]{2,3})\s+(\d{4})\b', r'\1\2', text)

    return text.strip()

# 读取阶段1的输出
with open('step1_extracted/all_documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

print(f"开始清洗 {len(documents)} 个文档...")

cleaned_docs = []
for doc in documents:
    print(f"清洗: {doc['source_file']}")

    raw_text = doc['raw_text']
    cleaned_text = clean_text(raw_text)

    # 统计清洗效果
    chars_removed = len(raw_text) - len(cleaned_text)

    cleaned_docs.append({
        "source_file": doc['source_file'],
        "cleaned_text": cleaned_text,
        "num_chars": len(cleaned_text),
        "chars_removed": chars_removed
    })

    print(f"  原始: {len(raw_text)} 字符")
    print(f"  清洗后: {len(cleaned_text)} 字符")
    print(f"  去除了: {chars_removed} 字符")

# 保存
with open('step2_cleaned/cleaned_documents.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)

print(f"\n✅ 清洗完成！")
```

**运行**:
```bash
python step2_clean_text.py
```

**输出**: `step2_cleaned/cleaned_documents.json`
```json
[
  {
    "source_file": "AI课程手册_WS2024.pdf",
    "cleaned_text": "IN2064 - Einführung in die KI\n\nModulbeschreibung:\nDieses Modul...",
    "num_chars": 11800,
    "chars_removed": 700
  },
  ...
]
```

**效果对比**:

```
【清洗前】
Page 1 of 20

IN2064  -   Einführung
in die KI


Modulbeschreibung:
Dieses Modul   behandelt...

DRAFT


【清洗后】
IN2064 - Einführung in die KI

Modulbeschreibung:
Dieses Modul behandelt...
```

**关键点**:
- ✅ 去除了PDF噪音
- ✅ 统一了格式
- ✅ 修复了提取错误
- ❌ **还没有分块**

---

## 阶段3: 分块

### 3.1 目标

把长文档切成小块，方便检索：

```
输入: 11800字符的长文档
输出: 8个块，每块约1500字符
```

### 3.2 分块参数选择 ⭐ **重要**

**推荐参数**:
- **块大小**: 1500字符（不是500！）
- **重叠**: 150字符（10%）

**为什么1500字符？**

```
1500字符 ≈ 240-300个英文单词
         ≈ 210-270个德语单词
         ≈ 2-3个完整段落
```

**不同块大小对比**:

| 块大小 | 单词数 | 段落数 | 优点 | 缺点 |
|--------|--------|--------|------|------|
| 500 | 80-100词 | <1段 | 精确匹配 | ❌ 太短，信息不完整 |
| **1000** | **160-200词** | **1-2段** | ✅ 平衡 | 适合问答 |
| **1500** | **240-300词** | **2-3段** | ✅ **推荐** | 适合课程内容 |
| 2000 | 320-400词 | 3-4段 | 上下文丰富 | 检索精度略降 |

### 3.3 分块脚本

**脚本**: `step3_chunk_text.py`

```python
import json
import re

def smart_chunk(text, chunk_size=1500, overlap=150):
    """
    智能分块

    Args:
        text: 要分块的文本
        chunk_size: 块大小（字符数）【推荐1500】
        overlap: 重叠大小（字符数）【推荐150】
    """
    # 按段落分割（两个换行符）
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果单个段落超长，强制切分
        if len(para) > chunk_size:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 强制切分超长段落
            for i in range(0, len(para), chunk_size - overlap):
                sub_chunk = para[i:i + chunk_size]
                if sub_chunk.strip():
                    chunks.append(sub_chunk.strip())

            continue

        # 如果加上这段不超限制
        if len(current_chunk) + len(para) + 2 < chunk_size:
            current_chunk += para + "\n\n"
        else:
            # 保存当前块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # 开始新块（包含重叠）
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para + "\n\n"
            else:
                current_chunk = para + "\n\n"

    # 保存最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# 读取清洗后的数据
with open('step2_cleaned/cleaned_documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

print(f"开始分块 {len(documents)} 个文档...")
print(f"块大小: 1500 字符")
print(f"重叠: 150 字符")
print()

all_chunks = []
chunk_id = 0

for doc in documents:
    source_file = doc['source_file']
    text = doc['cleaned_text']

    print(f"分块: {source_file}")
    print(f"  原始长度: {len(text):,} 字符")

    # 分块
    chunks = smart_chunk(text, chunk_size=1500, overlap=150)

    print(f"  生成块数: {len(chunks)}")

    # 统计块大小
    chunk_sizes = [len(c) for c in chunks]
    avg_size = sum(chunk_sizes) // len(chunks) if chunks else 0
    print(f"  平均块大小: {avg_size:,} 字符")
    print(f"  最小块: {min(chunk_sizes):,} 字符")
    print(f"  最大块: {max(chunk_sizes):,} 字符")

    # 添加到总列表
    for chunk_text in chunks:
        all_chunks.append({
            "id": str(chunk_id),
            "contents": chunk_text,
            "source_file": source_file
        })
        chunk_id += 1

    print()

# 保存为JSONL（FlashRAG格式）
output_file = 'step3_chunked/corpus.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for chunk in all_chunks:
        # FlashRAG只需要id和contents
        flashrag_doc = {
            "id": chunk["id"],
            "contents": chunk["contents"]
        }
        f.write(json.dumps(flashrag_doc, ensure_ascii=False) + '\n')

print("="*60)
print(f"✅ 分块完成！")
print(f"   总块数: {len(all_chunks)}")
print(f"   保存到: {output_file}")
print("="*60)

# 显示示例块
print(f"\n示例块（前2个）:")
for i in range(min(2, len(all_chunks))):
    chunk = all_chunks[i]
    preview = chunk["contents"][:200].replace('\n', ' ')
    print(f"\n[块{chunk['id']}] 来源: {chunk['source_file']}")
    print(f"长度: {len(chunk['contents'])} 字符")
    print(f"内容: {preview}...")
```

**运行**:
```bash
python step3_chunk_text.py
```

**输出**: `step3_chunked/corpus.jsonl`
```jsonl
{"id": "0", "contents": "IN2064 - Einführung in die Künstliche Intelligenz\n\nModulbeschreibung:\nDieses Modul führt in die Grundlagen der KI ein...(1500字符)"}
{"id": "1", "contents": "Voraussetzungen:\n- IN0001 Informatik Grundlagen\n- MA0001 Lineare Algebra\n\nLernziele:\n1. Verständnis von Suchverfahren...(1500字符)"}
...
```

**效果示例**:

```
原始文档: 11800字符

分块后:
├── 块0 (1500字符): 课程介绍
├── 块1 (1500字符): 先修课程 + 学习目标
├── 块2 (1500字符): 第1章 搜索算法
├── 块3 (1500字符): 第2章 知识表示
├── 块4 (1500字符): 第3章 机器学习
├── 块5 (1500字符): 考核方式 + 参考文献
└── 块6 (800字符): 剩余内容
```

**关键点**:
- ✅ 长文档 → 小块
- ✅ 每块约1500字符（2-3段落）
- ✅ 重叠150字符避免信息丢失
- ✅ JSONL格式（FlashRAG要求）

---

## 阶段4: 构建索引

### 4.1 目标

把文本块转成向量，存入FAISS索引：

```
输入: {"id": "0", "contents": "IN2064 - Einführung..."}
输出: 向量 [0.234, -0.456, ..., 0.123] (768维)
```

### 4.2 运行命令

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-base \
    --corpus_path step3_chunked/corpus.jsonl \
    --save_dir step4_index/faiss_index \
    --batch_size 32 \
    --max_length 512 \
    --pooling_method mean \
    --faiss_type Flat
```

**参数说明**:
- `--model_path`: 多语言embedding模型（支持英语查询→德语文档）
- `--corpus_path`: 分块后的语料库
- `--save_dir`: 索引保存位置
- `--batch_size`: 批量处理大小（GPU用128，CPU用32）
- `--max_length`: 最大长度512（模型限制）

**输出**:
```
Building index...
Loading model: intfloat/multilingual-e5-base
Processing 150 documents...
[====================] 100%
Index saved to: step4_index/faiss_index
✅ Done!
```

**索引文件**:
```
step4_index/faiss_index/
├── index.faiss         (向量索引)
├── docid.json          (ID映射)
└── config.json         (配置)
```

---

## 阶段5: 测试

### 5.1 测试脚本

```python
import sys
sys.path.insert(0, '/home/user/R3-RAG/tool/FlashRAG')

from flashrag.retriever import Retriever

# 加载索引
retriever = Retriever(
    method="e5",
    index_path="step4_index/faiss_index",
    model_path="intfloat/multilingual-e5-base"
)

# 测试查询
queries = [
    "Was sind die Voraussetzungen für den KI-Kurs?",  # 德语
    "What topics are covered in chapter 1?",  # 英语
    "Wie viele ECTS hat IN2064?",  # 德语
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print('='*60)

    results = retriever.search(query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"ID: {result['id']}")
        print(f"相似度: {result.get('score', 'N/A'):.4f}")
        print(f"内容: {result['contents'][:200]}...")
```

---

## 参数调优

### 块大小（chunk_size）

根据你的数据特点选择：

| 数据类型 | 推荐块大小 | 原因 |
|---------|-----------|------|
| **课程目录**（简短信息） | 1000字符 | 每条信息较短 |
| **课程手册**（中等长度） | **1500字符** | ✅ **推荐** |
| **讲义内容**（长文档） | 2000字符 | 需要更多上下文 |

### 重叠（overlap）

**推荐公式**: overlap = chunk_size × 10%

```
chunk_size = 1500 → overlap = 150
chunk_size = 2000 → overlap = 200
```

### 批量大小（batch_size）

```
GPU (NVIDIA): 128-256
CPU: 32-64
```

---

## 📊 完整流程总结

```
【你的文件】
├── AI课程手册.pdf (20页)
├── AI讲义.pdf (50页)
└── TUMonline.csv
         ↓

【阶段1: 结构化】step1_extract_pdf.py
输出: all_documents.json
      [{"source": "...", "raw_text": "12500字符"}, ...]
         ↓

【阶段2: 数据清洗】step2_clean_text.py
输出: cleaned_documents.json
      [{"source": "...", "cleaned_text": "11800字符"}, ...]
         ↓

【阶段3: 分块】step3_chunk_text.py
参数: chunk_size=1500, overlap=150
输出: corpus.jsonl
      {"id": "0", "contents": "1500字符块"}
      {"id": "1", "contents": "1500字符块"}
      ...（约8个块/文档）
         ↓

【阶段4: 构建索引】flashrag.retriever.index_builder
输出: faiss_index/
      ├── index.faiss
      ├── docid.json
      └── config.json
         ↓

【阶段5: 测试】test_retrieval.py
输入: "先修课程是什么？"
输出: 最相关的3个文本块
```

---

## ✅ 检查清单

在每个阶段完成后，检查：

- [ ] **阶段1完成**: `all_documents.json`存在，包含所有PDF的文本
- [ ] **阶段2完成**: `cleaned_documents.json`存在，文本没有页码/噪音
- [ ] **阶段3完成**: `corpus.jsonl`存在，块大小约1500字符
- [ ] **阶段4完成**: `faiss_index/index.faiss`文件存在
- [ ] **阶段5完成**: 测试查询能返回相关结果

---

## 🚀 快速开始命令

```bash
# 1. 创建工作目录
mkdir -p /home/user/tum_kb_project && cd /home/user/tum_kb_project
mkdir -p raw_files step1_extracted step2_cleaned step3_chunked step4_index

# 2. 把文件放到raw_files/

# 3. 运行阶段1
python step1_extract_pdf.py

# 4. 运行阶段2
python step2_clean_text.py

# 5. 运行阶段3（注意：1500字符！）
python step3_chunk_text.py

# 6. 运行阶段4
python -m flashrag.retriever.index_builder \
    --corpus_path step3_chunked/corpus.jsonl \
    --save_dir step4_index/faiss_index \
    --chunk_size 1500

# 7. 测试
python test_retrieval.py
```

---

**每一步都清楚了吗？** 有任何问题随时问我！💬
