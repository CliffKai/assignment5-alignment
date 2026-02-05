# 01 - 基础工具函数

## 概述

这些是后续 SFT 和 GRPO 实现中会用到的基础工具函数。

---

## 1. Masked Mean

### 目标
计算张量在某个维度上的均值，但只考虑 mask=1 的元素。

### 数学公式

$$\text{masked\_mean}(x, m, \text{dim}) = \frac{\sum_i x_i \cdot m_i}{\sum_i m_i}$$

### 为什么需要？
在语言模型训练中，我们通常只想在 **response tokens** 上计算 loss，忽略：
- Prompt tokens（不应该影响 loss）
- Padding tokens（填充用的无意义 token）

### 实现要点
```python
def run_masked_mean(tensor, mask, dim=None):
    # 1. 将 mask 转换为与 tensor 相同的 dtype
    # 2. 计算 masked 元素的和: (tensor * mask).sum(dim)
    # 3. 计算 mask 的和: mask.sum(dim)
    # 4. 返回比值，注意处理 dim=None 的情况
    pass
```

### 测试
```bash
uv run pytest tests/test_metrics.py::test_masked_mean -v
```

---

## 2. Masked Normalize

### 目标
对 masked 元素求和，然后除以一个常数（而不是 mask 的数量）。

### 数学公式

$$\text{masked\_normalize}(x, m, \text{dim}, c) = \frac{\sum_i x_i \cdot m_i}{c}$$

### 为什么需要？
在 Dr. GRPO 中，loss 需要按固定常数归一化，而不是按实际 token 数量。这与标准的 masked mean 不同。

### 实现要点
```python
def run_masked_normalize(tensor, mask, dim=None, normalize_constant=1.0):
    # 1. 计算 masked 元素的和: (tensor * mask).sum(dim)
    # 2. 除以 normalize_constant
    pass
```

### 测试
```bash
uv run pytest tests/test_metrics.py::test_masked_normalize -v
```

---

## 3. Entropy 计算

### 目标
计算 logits 分布的熵。

### 数学公式

对于 logits $z$，先计算概率分布 $p = \text{softmax}(z)$，然后：

$$H(p) = -\sum_i p_i \log p_i$$

### 为什么需要？
- 熵是衡量分布"不确定性"的指标
- 在 RL 训练中，常用 entropy bonus 来鼓励探索
- 监控训练过程中模型输出的多样性

### 实现要点
```python
def run_compute_entropy(logits):
    # logits: (batch_size, seq_length, vocab_size)
    # 1. 计算 log_softmax: log_probs = F.log_softmax(logits, dim=-1)
    # 2. 计算 softmax: probs = F.softmax(logits, dim=-1)
    # 3. 计算熵: entropy = -(probs * log_probs).sum(dim=-1)
    # 返回: (batch_size, seq_length)
    pass
```

### 数值稳定性
使用 `log_softmax` 而不是 `log(softmax(x))` 来避免数值问题。

### 测试
```bash
uv run pytest tests/test_grpo.py::test_compute_entropy -v
```

---

## 笔记区域

（在这里记录你的学习笔记和遇到的问题）


