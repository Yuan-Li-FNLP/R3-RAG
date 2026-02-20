# 跨语言检索指南：英语查询 → 德语文档

## 📖 问题描述

你希望实现：
- ✅ **用户输入**：英语查询（"What are the prerequisites for AI course?"）
- ✅ **系统检索**：德语文档（"Voraussetzungen: Lineare Algebra, Informatik"）
- ✅ **系统回答**：英语答案（"The prerequisites are Linear Algebra and Computer Science"）

这是典型的**跨语言检索（Cross-lingual Retrieval）**场景。

---

## 🎯 三种实现方案

### 方案1：多语言Embedding模型 ⭐ **推荐**

#### 原理
使用**多语言Embedding模型**（如`multilingual-e5-large`），将不同语言的文本映射到**同一个语义空间**。

```
英语查询: "prerequisites for AI"
    ↓ [Embedding]
Vector: [0.23, -0.45, 0.67, ...]

德语文档: "Voraussetzungen für KI"
    ↓ [Embedding]
Vector: [0.21, -0.43, 0.65, ...]  ← 向量相似！
```

#### 优势
- ✅ **无需翻译**：直接检索，速度快
- ✅ **语义理解**：能理解同义词、近义词
- ✅ **实现简单**：只需更换Embedding模型

#### 实现步骤

**Step 1: 准备双语语料库**

确保你的文档包含德英混合内容：
```json
{
  "id": "IN2064",
  "contents": "IN2064 - Introduction to Artificial Intelligence / Einführung in die Künstliche Intelligenz\nECTS: 6\nSemester: WS\nPrerequisites / Voraussetzungen: Linear Algebra (MA0001), Introduction to Computer Science (IN0001)\n\nDescription:\nThis course covers the fundamentals of AI...\n\nBeschreibung:\nDieser Kurs behandelt die Grundlagen der KI..."
}
```

这已经由`convert_to_flashrag_format.py`的`language='mixed'`模式自动完成！

**Step 2: 构建多语言索引**

```bash
# 运行脚本
cd /home/user/R3-RAG/data/scripts
bash build_multilingual_index.sh
```

或手动构建：
```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-large \
    --corpus_path ../corpus/tum_courses_corpus.jsonl \
    --save_dir ../../indexes/tum_multilingual \
    --use_fp16 \
    --max_length 512 \
    --batch_size 128 \
    --pooling_method mean \
    --faiss_type Flat
```

**Step 3: 启动检索服务**

```bash
cd ../../benchmark/retriever
python src/retrive_server.py \
    --index_path ../../indexes/tum_multilingual \
    --model_path intfloat/multilingual-e5-large \
    --port 5000
```

**Step 4: 测试跨语言检索**

```bash
cd ../../data/scripts
python test_cross_lingual_retrieval.py
```

示例输出：
```
📝 查询 (英语): What are the prerequisites for the AI course?

🔹 结果 1 (ID: IN2064)
   相似度: 0.8523
   语言: 🇩🇪 德语
   内容:
      IN2064 - Introduction to AI / Einführung in die KI
      Voraussetzungen: Lineare Algebra (MA0001)
      Informatik Grundlagen (IN0001)
      ...

✅ 通过 - 成功检索到德语文档！
```

#### 推荐的多语言模型

| 模型 | 性能 | 速度 | 推荐场景 |
|------|------|------|----------|
| **intfloat/multilingual-e5-large** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 生产环境（最佳） |
| **intfloat/multilingual-e5-base** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 平衡性能和速度 |
| **paraphrase-multilingual-MiniLM-L12-v2** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 测试/轻量级应用 |

---

### 方案2：查询翻译

#### 原理
将英语查询翻译成德语，然后用德语查询检索德语文档。

```
英语查询: "prerequisites for AI"
    ↓ [翻译]
德语查询: "Voraussetzungen für KI"
    ↓ [检索]
德语文档: "Voraussetzungen: Lineare Algebra"
```

#### 优势
- ✅ 可以使用普通的德语Embedding模型
- ✅ 检索精度可能更高（语言匹配）

#### 劣势
- ❌ 需要额外的翻译步骤（延迟增加）
- ❌ 翻译错误会影响检索质量
- ❌ 实现复杂度更高

#### 实现（仅供参考）

```python
from transformers import MarianMTModel, MarianTokenizer

# 加载翻译模型（英语→德语）
model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_en_to_de(query: str) -> str:
    """将英语查询翻译为德语"""
    inputs = tokenizer(query, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 使用
en_query = "What are the prerequisites for AI course?"
de_query = translate_en_to_de(en_query)  # "Was sind die Voraussetzungen für KI-Kurs?"

# 然后用德语查询检索
results = retriever.search(de_query)
```

**不推荐此方案**，因为方案1更简单高效。

---

### 方案3：双语索引（冗余存储）

#### 原理
为每个课程创建**两条**索引记录：
- 一条纯英语
- 一条纯德语

```json
// 英语版本
{"id": "IN2064_en", "contents": "IN2064 - Introduction to AI\nPrerequisites: Linear Algebra..."}

// 德语版本
{"id": "IN2064_de", "contents": "IN2064 - Einführung in die KI\nVoraussetzungen: Lineare Algebra..."}
```

#### 优势
- ✅ 可以使用单语言模型
- ✅ 检索速度快（无需跨语言）

#### 劣势
- ❌ 索引大小翻倍
- ❌ 需要双语文档（你可能只有德语）
- ❌ 维护复杂

**不推荐此方案**，除非你有完整的双语平行语料。

---

## 🚀 快速开始（推荐方案1）

### 一键部署

```bash
# 1. 构建多语言索引
cd /home/user/R3-RAG/data/scripts
bash build_multilingual_index.sh

# 2. 启动检索服务
cd ../../benchmark/retriever
python src/retrive_server.py \
    --index_path ../../indexes/tum_multilingual \
    --model_path intfloat/multilingual-e5-large \
    --port 5000 &

# 3. 测试跨语言检索
cd ../../data/scripts
python test_cross_lingual_retrieval.py

# 4. 启动完整RAG系统
cd ../../startup
bash server.sh  # 使用config_english_query_german_docs.yaml配置
```

---

## 🧪 测试示例

### 单个查询测试

```bash
python test_cross_lingual_retrieval.py \
    --query "What are the prerequisites for machine learning?"
```

### 完整测试套件

```bash
python test_cross_lingual_retrieval.py
```

测试用例包括：
1. ✅ "What are the prerequisites for the AI course?"
2. ✅ "How many ECTS credits does the database course have?"
3. ✅ "Which courses are offered in winter semester?"
4. ✅ "Tell me about machine learning courses"

---

## 📊 性能对比

| 方案 | 检索速度 | 精度 | 实现难度 | 推荐指数 |
|------|---------|------|----------|---------|
| **方案1: 多语言Embedding** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 方案2: 查询翻译 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 方案3: 双语索引 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## ⚙️ 配置说明

### R3-RAG配置文件

使用提供的配置文件：
```bash
cp config_english_query_german_docs.yaml ../../startup/config.yaml
```

关键配置项：
```yaml
retriever_config:
  model_path: "intfloat/multilingual-e5-large"  # 必须使用多语言模型！
  query_instruction: "query: "  # E5模型要求

generator_config:
  system_prompt: |
    You are a TUM course advisor.
    Answer in English, even if documents are in German.
```

---

## ❓ 常见问题

### Q1: 为什么不能用普通的E5模型？

**A**: 普通的`intfloat/e5-base-v2`是**单语言**（英语）模型，只能理解英语。必须使用`multilingual-e5-*`系列。

### Q2: 检索结果全是英语文档，没有德语怎么办？

**A**: 检查以下几点：
1. 语料库中是否包含德语内容？
   ```bash
   grep "ä\|ö\|ü\|ß" data/corpus/tum_courses_corpus.jsonl
   ```
2. 是否使用了正确的多语言模型？
3. 索引是否正确构建？

### Q3: 我的文档只有德语，没有英语标题怎么办？

**A**: 没问题！多语言模型可以直接匹配：
```
英语查询: "AI course" → 德语文档: "KI-Kurs" ✅
```

不需要文档中有英语翻译（但有的话更好）。

### Q4: 能同时支持德语查询吗？

**A**: 当然可以！多语言模型支持双向：
- 英语查询 → 德语文档 ✅
- 德语查询 → 德语文档 ✅
- 德语查询 → 英语文档 ✅
- 英语查询 → 英语文档 ✅

### Q5: 性能影响有多大？

**A**: multilingual-e5-large比普通e5-base-v2略大：
- 模型大小: ~2.2GB vs ~1.1GB
- 推理速度: 慢约20%
- 检索精度: 跨语言场景显著提升

---

## 📈 优化建议

### 1. 文档优化

**在文档中同时包含关键术语的德英版本**：
```json
{
  "contents": "IN2064 - Introduction to AI / Einführung in die KI\nPrerequisites / Voraussetzungen: Linear Algebra / Lineare Algebra (MA0001)"
}
```

### 2. 查询扩展

如果检索效果不佳，可以添加关键词：
```python
# 扩展查询
original_query = "prerequisites for AI"
expanded_query = "prerequisites requirements for AI artificial intelligence course"
```

### 3. 混合检索

结合BM25（关键词匹配）和多语言Embedding（语义匹配）：
```python
# BM25检索（快速，精确匹配）
bm25_results = bm25_retriever.search(query, top_k=10)

# 语义检索（跨语言）
semantic_results = multilingual_retriever.search(query, top_k=10)

# 混合重排
final_results = rerank(bm25_results + semantic_results)
```

---

## 🎓 进阶主题

### 自定义多语言模型微调

如果检索效果不理想，可以在TUM课程数据上微调模型：

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 准备训练数据（英语查询 - 德语文档配对）
train_examples = [
    InputExample(texts=[
        "prerequisites for AI course",
        "Voraussetzungen: Lineare Algebra, Informatik Grundlagen"
    ], label=1.0),
    # 更多示例...
]

# 加载基础模型
model = SentenceTransformer('intfloat/multilingual-e5-base')

# 定义损失函数
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 微调
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# 保存
model.save('tum_multilingual_e5_finetuned')
```

---

## 📚 参考资源

- [Multilingual-E5 论文](https://arxiv.org/abs/2402.05672)
- [FlashRAG文档](../../../tool/FlashRAG/README.md)
- [R3-RAG论文](https://arxiv.org/abs/2505.23794)

---

## 🎯 总结

**推荐方案**：使用多语言Embedding模型（方案1）

**核心步骤**：
1. ✅ 使用`multilingual-e5-large`构建索引
2. ✅ 确保文档包含德英混合内容（`convert_to_flashrag_format.py`已支持）
3. ✅ 启动检索服务并测试
4. ✅ 配置R3-RAG使用多语言检索器

**下一步**：
```bash
bash build_multilingual_index.sh  # 立即开始！
```

---

**最后更新**: 2025-12-16
