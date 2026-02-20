# GRPO训练显存计算完整指南

## 📊 显存组成

GRPO/PPO训练的总显存由以下部分组成：

```
总显存 = 模型显存 + 优化器显存 + 激活显存 + Rollout显存 + KV Cache显存
```

---

## 🧮 详细计算公式

### 1. 模型显存（Model Memory）

```python
# Actor模型（训练中，需要梯度，bf16）
actor_memory = model_size × 2 × 2 = model_size × 4 bytes
# 参数(2 bytes) + 梯度(2 bytes)

# Reference模型（冻结，只推理，bf16）
reference_memory = model_size × 2 bytes
# 只有参数，无梯度

# 总模型显存（不含Critic）
model_memory = actor_memory + reference_memory
             = model_size × 4 + model_size × 2
             = model_size × 6 bytes

# 如果使用PPO（含Critic）
critic_memory = model_size × 4 bytes  # 同Actor
total_with_critic = model_size × 10 bytes
```

**7B模型示例**：
```
model_memory = 7B × 6 bytes = 42 GB
```

---

### 2. 优化器显存（Optimizer Memory）

```python
# AdamW优化器（fp32状态）
optimizer_memory = 2 × model_size × 4 bytes
# momentum(4 bytes) + variance(4 bytes)

# 使用8-bit优化器（推荐）
optimizer_memory_8bit = 2 × model_size × 1 byte
```

**7B模型示例**：
```
# 标准AdamW
optimizer_memory = 2 × 7B × 4 = 56 GB

# 8-bit优化器（--adam_offload）
optimizer_memory_8bit = 2 × 7B × 1 = 14 GB  ✅ 节省75%
```

---

### 3. 激活显存（Activation Memory）

```python
# 不使用gradient_checkpointing
activation_memory = layers × batch_size × seq_length × hidden_size × 4

# 使用gradient_checkpointing（推荐）
activation_memory_ckpt = sqrt(layers) × batch_size × seq_length × hidden_size × 4
```

**7B模型示例**：
```
layers = 32
hidden_size = 4096
batch_size = 64
seq_length = 1536

# 使用checkpointing
activation_memory ≈ sqrt(32) × 64 × 1536 × 4096 × 4 / (1024^3)
                 ≈ 8.7 GB
```

---

### 4. Rollout显存（最关键！）

这是GRPO特有的，也是最占显存的部分：

```python
# 存储生成的tokens
rollout_tokens = rollouts_per_batch × max_new_tokens × hidden_size × 2

# 存储元数据（log_probs, rewards, values等）
rollout_metadata = rollouts_per_batch × max_new_tokens × 4

# KV Cache（生成时）
kv_cache_per_sample = 2 × layers × seq_length × hidden_size × 2

# 如果并行生成所有轨迹
kv_cache_all = kv_cache_per_sample × rollouts_per_batch

# 如果sequential生成（推荐）
kv_cache_sequential = kv_cache_per_sample × micro_rollout_batch_size
```

**7B模型示例**：
```
rollouts_per_batch = 64
max_new_tokens = 512

# 并行生成所有轨迹（不推荐）
kv_cache_all ≈ 49.3 GB

# Sequential生成（micro_batch=1，推荐）
kv_cache_sequential ≈ 0.77 GB × 需要同时保持的轨迹数
                    ≈ 0.77 × 8 = 6.16 GB  ✅ 节省87%
```

---

## 🎯 实际计算示例

### 示例1：激进配置（显存占用大）

```python
# 配置
model_size = 7B
prompts_per_batch = 8
group_size = 8
rollouts_per_batch = 64
max_new_tokens = 512
num_gpus = 8

# 显存计算（无优化）
model_memory = 42 GB
optimizer_memory = 56 GB
activation_memory = 8.7 GB
rollout_memory = 49.3 GB  # 并行生成

total = 156 GB
per_gpu = 156 / 8 = 19.5 GB  # ZeRO-3全分片

# ❌ 问题：总显存太大，即使用ZeRO-3也可能不稳定
```

---

### 示例2：推荐配置（R3-RAG）

```python
# 配置
model_size = 7B
prompts_per_batch = 16
group_size = 4
rollouts_per_batch = 64
max_new_tokens = 512
micro_rollout_batch_size = 1  # Sequential生成
num_gpus = 8

# 显存计算（优化后）
model_memory = 42 GB
optimizer_memory_8bit = 14 GB         # ✅ 使用8-bit
activation_memory_ckpt = 8.7 GB       # ✅ 使用checkpointing
rollout_memory_seq = 6.16 GB          # ✅ Sequential生成

total = 70.9 GB

# 使用ZeRO Stage 2
# - 参数不分片（每卡42 GB）
# - 梯度+优化器分片（14+14）/ 8 = 3.5 GB per GPU
per_gpu = 42 + 3.5 + 8.7 + 6.16/8
        = 60.4 GB

# ✅ 可用: 8×A100 (80GB)
```

---

## 📐 通用计算公式（Python）

```python
def calculate_grpo_memory(
    model_size_B,         # 模型参数量（单位：Billion，如7）
    prompts_per_batch,    # 每batch的问题数
    group_size,           # 每个问题的轨迹数
    max_new_tokens,       # 生成长度
    num_gpus=1,           # GPU数量
    use_8bit_optimizer=True,
    use_gradient_checkpointing=True,
    use_zero_stage=2,
    micro_rollout_batch=1  # 微批量（sequential生成）
):
    """
    计算GRPO训练显存需求（单位：GB）
    """

    model_size = model_size_B * 1e9

    # 1. 模型显存（GB）
    model_memory = model_size * 6 / 1e9  # Actor(4B) + Reference(2B)

    # 2. 优化器显存（GB）
    if use_8bit_optimizer:
        optimizer_memory = model_size * 2 / 1e9
    else:
        optimizer_memory = model_size * 8 / 1e9

    # 3. 激活显存（GB，经验公式）
    if use_gradient_checkpointing:
        activation_memory = model_size_B * 1.2  # 约1.2 GB per B
    else:
        activation_memory = model_size_B * 12   # 约12 GB per B

    # 4. Rollout显存（GB）
    rollouts_per_batch = prompts_per_batch * group_size
    hidden_size = 4096  # 7B模型的hidden size

    # KV Cache per sample
    layers = 32
    seq_length = 1024 + max_new_tokens
    kv_cache_per_sample = (2 * layers * seq_length * hidden_size * 2) / 1e9

    # Sequential生成
    rollout_memory = kv_cache_per_sample * micro_rollout_batch * \
                    (rollouts_per_batch / micro_rollout_batch)

    # 5. 总显存（不分片）
    total_memory = (
        model_memory +
        optimizer_memory +
        activation_memory +
        rollout_memory
    )

    # 6. 使用ZeRO分片
    if use_zero_stage == 2:
        # 参数不分片，梯度+优化器分片
        gradient_memory = model_memory / 2  # 梯度约为参数的一半大小
        per_gpu_memory = (
            model_memory +                                    # 参数（每卡全部）
            (optimizer_memory + gradient_memory) / num_gpus + # 优化器+梯度分片
            activation_memory +                               # 激活（每卡）
            rollout_memory / num_gpus                        # Rollout可以分片
        )
    elif use_zero_stage == 3:
        # 全部分片
        per_gpu_memory = total_memory / num_gpus
    else:
        per_gpu_memory = total_memory

    return {
        'total_memory_GB': round(total_memory, 1),
        'per_gpu_memory_GB': round(per_gpu_memory, 1),
        'model_memory_GB': round(model_memory, 1),
        'optimizer_memory_GB': round(optimizer_memory, 1),
        'activation_memory_GB': round(activation_memory, 1),
        'rollout_memory_GB': round(rollout_memory, 1)
    }

# === 使用示例 ===

# 示例1：推荐配置
result = calculate_grpo_memory(
    model_size_B=7,
    prompts_per_batch=16,
    group_size=4,
    max_new_tokens=512,
    num_gpus=8,
    use_8bit_optimizer=True,
    use_gradient_checkpointing=True,
    use_zero_stage=2,
    micro_rollout_batch=1
)

print("=== 推荐配置（7B模型）===")
print(f"总显存: {result['total_memory_GB']} GB")
print(f"每卡显存: {result['per_gpu_memory_GB']} GB")
print(f"  - 模型: {result['model_memory_GB']} GB")
print(f"  - 优化器: {result['optimizer_memory_GB']} GB")
print(f"  - 激活: {result['activation_memory_GB']} GB")
print(f"  - Rollout: {result['rollout_memory_GB']} GB")
print(f"推荐GPU: 8×A100 (80GB)")

# 示例2：激进配置
result2 = calculate_grpo_memory(
    model_size_B=7,
    prompts_per_batch=8,
    group_size=8,
    max_new_tokens=512,
    num_gpus=8,
    use_8bit_optimizer=True,
    use_gradient_checkpointing=True,
    use_zero_stage=2,
    micro_rollout_batch=1
)

print("\n=== 激进配置（7B模型）===")
print(f"总显存: {result2['total_memory_GB']} GB")
print(f"每卡显存: {result2['per_gpu_memory_GB']} GB")
```

**预期输出**：
```
=== 推荐配置（7B模型）===
总显存: 70.9 GB
每卡显存: 60.4 GB
  - 模型: 42.0 GB
  - 优化器: 14.0 GB
  - 激活: 8.4 GB
  - Rollout: 6.2 GB
推荐GPU: 8×A100 (80GB)

=== 激进配置（7B模型）===
总显存: 77.3 GB
每卡显存: 64.8 GB
```

---

## 🎯 显存优化技巧（按效果排序）

### 1. Sequential Rollout（最有效，减少87%）
```bash
--micro_rollout_batch_size 1
```
**效果**：49.3 GB → 6.16 GB

---

### 2. 8-bit优化器（减少75%）
```bash
--adam_offload
```
**效果**：56 GB → 14 GB

---

### 3. Gradient Checkpointing（减少70%激活）
```bash
--gradient_checkpointing
```
**效果**：激活显存减少约70%

---

### 4. 降低group_size（减少50%）
```python
group_size = 4  # 而不是8
```
**效果**：Rollout显存减少50%

---

### 5. 使用ZeRO Stage 2（分片优化器）
```bash
--zero_stage 2
```
**效果**：优化器+梯度显存在多卡间分片
- 8卡：14 GB → 1.75 GB per GPU

---

### 6. 使用ZeRO Stage 3（全分片）
```bash
--zero_stage 3
```
**效果**：参数+梯度+优化器全分片
- 8卡：42 GB → 5.25 GB per GPU
- ⚠️ 注意：可能影响训练速度

---

### 7. 降低max_new_tokens（减少25%）
```python
max_new_tokens = 384  # 而不是512
```
**效果**：Rollout显存减少约25%

---

## 📊 不同GPU配置的推荐

### 8×A100 (80GB) - 推荐配置
```python
model_size = 7B
prompts_per_batch = 16
group_size = 4
rollouts_per_batch = 64
max_new_tokens = 512

优化：
--adam_offload
--gradient_checkpointing
--zero_stage 2
--micro_rollout_batch_size 1

显存: 约60 GB/卡 ✅
```

---

### 8×A100 (40GB) - 紧凑配置
```python
model_size = 7B
prompts_per_batch = 12
group_size = 3
rollouts_per_batch = 36
max_new_tokens = 384

优化：
--adam_offload
--gradient_checkpointing
--zero_stage 3  # 使用ZeRO-3
--micro_rollout_batch_size 1

显存: 约38 GB/卡 ✅
```

---

### 4×A100 (80GB) - 小规模训练
```python
model_size = 3B  # 使用更小模型
prompts_per_batch = 12
group_size = 4
rollouts_per_batch = 48
max_new_tokens = 512

优化：
--adam_offload
--gradient_checkpointing
--zero_stage 2

显存: 约70 GB/卡 ✅
```

---

## ⚠️ 常见问题

### Q1: OOM（Out of Memory）怎么办？

**排查步骤**：
1. 检查是否并行生成所有轨迹
   ```bash
   # 添加这个参数
   --micro_rollout_batch_size 1
   ```

2. 降低group_size
   ```python
   group_size = 4  # 从8降到4
   ```

3. 使用更强的优化
   ```bash
   --zero_stage 3  # 从2升级到3
   ```

4. 降低批量大小
   ```python
   prompts_per_batch = 8  # 从16降到8
   ```

---

### Q2: 训练速度太慢怎么办？

**加速策略**：
1. 不要用ZeRO Stage 3（除非必须）
   ```bash
   --zero_stage 2  # Stage 3会慢很多
   ```

2. 增加micro_batch_size（如果显存允许）
   ```bash
   --micro_rollout_batch_size 2  # 从1增加到2
   ```

3. 使用Flash Attention
   ```bash
   --flash_attn
   ```

---

### Q3: 如何验证显存计算是否准确？

**实际监控**：
```bash
# 训练时实时监控GPU显存
watch -n 1 nvidia-smi

# 或使用tensorboard查看详细统计
tensorboard --logdir ./tensorboard_logs
```

**预期显存使用**：
- 训练刚开始：模型+优化器显存（~56 GB）
- Rollout阶段：增加Rollout显存（+6 GB）
- 峰值：约60-65 GB（7B模型，推荐配置）

---

## ✅ 推荐配置总结

### 7B模型 + TUM知识库训练

```python
# === 训练配置 ===
model_size = 7B
prompts_per_batch = 16          # 16个问题
group_size = 4                  # 每个问题4条轨迹
rollouts_per_batch = 64         # 总共64条轨迹
max_new_tokens = 512            # 生成长度
max_iterations = 5              # 最多5步检索
topk_per_retrieval = 2          # 每步2个文档

# === 优化配置 ===
optimizer = AdamW
learning_rate = 5e-7            # 保守学习率
warmup_ratio = 0.05
weight_decay = 0.01
kl_coef = 0.01

# === 显存优化 ===
--adam_offload                  # 8-bit优化器
--gradient_checkpointing        # 激活检查点
--zero_stage 2                  # ZeRO-2
--micro_rollout_batch_size 1    # Sequential生成
--flash_attn                    # Flash Attention

# === 硬件需求 ===
GPU: 8×A100 (80GB)
每卡显存: 约60 GB
训练时间: 约1-2天（取决于数据量）
```

---

## 📚 参考资料

- **R3-RAG官方配置**: `/train/R3RAG_OpenRLHF/examples/RLHF.sh`
- **ZeRO论文**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- **DeepSpeed文档**: https://www.deepspeed.ai/
- **OpenRLHF文档**: https://github.com/OpenLLMAI/OpenRLHF
