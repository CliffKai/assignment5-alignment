# 02 - Supervised Fine-Tuning (SFT)

## 概述

SFT 是 LLM alignment 的第一步：在高质量的 (prompt, response) 数据上进行监督学习。

---

## 1. Tokenize Prompt and Output

### 目标
将 prompt 和 output 字符串 tokenize，并创建 response mask。

### 输入输出
```
输入:
  prompt_strs: ["What is 2+2?", "Hello"]
  output_strs: ["The answer is 4", "Hi there!"]
  tokenizer: HuggingFace tokenizer

输出:
  input_ids: [batch_size, seq_len-1]  # 去掉最后一个 token
  labels: [batch_size, seq_len-1]     # 去掉第一个 token (shifted)
  response_mask: [batch_size, seq_len-1]  # 1 表示 response token
```

### 关键概念：为什么要 shift？

语言模型是 **next token prediction**：
```
input:  [A, B, C, D]
         ↓  ↓  ↓  ↓
output: [B, C, D, E]  (预测下一个 token)
```

所以：
- `input_ids` = 原始序列去掉最后一个 token
- `labels` = 原始序列去掉第一个 token

### Response Mask

只在 response 部分计算 loss：
```
Prompt:   "What is 2+2?"
Response: "The answer is 4"

Tokens:   [What, is, 2, +, 2, ?, The, answer, is, 4]
Mask:     [0,    0,  0, 0, 0, 0, 1,   1,      1,  1]
```

### 实现要点
```python
def run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # 1. 分别 tokenize prompt 和 output
    # 2. 拼接得到完整序列
    # 3. Padding 到相同长度
    # 4. 构建 response_mask (prompt 和 padding 位置为 0)
    # 5. 创建 shifted input_ids 和 labels
    pass
```

### 测试
```bash
uv run pytest tests/test_sft.py::test_tokenize_prompt_and_output -v
```

---

## 2. Get Response Log Probabilities

### 目标
计算模型对每个 token 的 log 概率。

### 数学公式

给定 input_ids $x_{1:T}$，模型输出 logits $z_t$，计算：

$$\log p(x_{t+1} | x_{1:t}) = \log \text{softmax}(z_t)[x_{t+1}]$$

### 实现要点
```python
def run_get_response_log_probs(model, input_ids, labels, return_token_entropy):
    # 1. 前向传播得到 logits: logits = model(input_ids).logits
    # 2. 计算 log_softmax: log_probs = F.log_softmax(logits, dim=-1)
    # 3. 用 gather 取出 labels 对应位置的 log_prob
    # 4. 可选：计算 token entropy

    # 返回 dict:
    #   "log_probs": [batch_size, seq_length]
    #   "token_entropy": [batch_size, seq_length] (如果 return_token_entropy=True)
    pass
```

### 关键：gather 操作
```python
# logits: [batch, seq, vocab]
# labels: [batch, seq]
# 我们要取出每个位置上 label token 的 log prob

log_probs = F.log_softmax(logits, dim=-1)
# log_probs: [batch, seq, vocab]

token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
# token_log_probs: [batch, seq]
```

### 测试
```bash
uv run pytest tests/test_grpo.py::test_get_response_log_probs -v
```

---

## 3. SFT Microbatch Train Step

### 目标
执行一个 SFT 训练步骤，支持 gradient accumulation。

### SFT Loss

标准的 cross-entropy loss，但只在 response tokens 上计算：

$$\mathcal{L}_{SFT} = -\frac{1}{N} \sum_{t \in \text{response}} \log p(x_t | x_{<t})$$

### Gradient Accumulation

当 batch size 太大无法放入 GPU 时，分成多个 microbatch：
```
Total batch = 32
Microbatch = 8
Gradient accumulation steps = 4

每个 microbatch 的 loss 要除以 4，这样累加后等价于完整 batch 的 loss
```

### 实现要点
```python
def run_sft_microbatch_train_step(
    policy_log_probs,      # [batch, seq]
    response_mask,         # [batch, seq]
    gradient_accumulation_steps,
    normalize_constant=1.0
):
    # 1. 计算 negative log likelihood: nll = -policy_log_probs
    # 2. 应用 mask 并求均值（或用 normalize_constant）
    # 3. 除以 gradient_accumulation_steps
    # 4. 调用 loss.backward()
    # 5. 返回 loss 值和 metadata
    pass
```

### 测试
```bash
uv run pytest tests/test_sft.py::test_sft_microbatch_train_step -v
```

---

## 完整 SFT 训练流程

```python
# 伪代码
for batch in dataloader:
    optimizer.zero_grad()

    for microbatch in split(batch, gradient_accumulation_steps):
        # 1. Tokenize
        tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)

        # 2. 获取 log probs
        log_probs = get_response_log_probs(model, tokenized["input_ids"], tokenized["labels"])

        # 3. 计算 loss 并 backward
        sft_microbatch_train_step(log_probs, tokenized["response_mask"], grad_accum_steps)

    optimizer.step()
```

---

## 笔记区域

（在这里记录你的学习笔记和遇到的问题）


