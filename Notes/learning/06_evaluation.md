# 06 - 评估方法

## 概述

本作业涉及多个评估基准，用于测试模型的不同能力。

**注意：这是可选作业的内容。**

---

## 1. MMLU (Massive Multitask Language Understanding)

### 简介
- 57 个学科的多选题
- 测试模型的知识和推理能力
- 每题 4 个选项 (A, B, C, D)

### 数据格式
```python
{
    "subject": "abstract_algebra",
    "question": "Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q.",
    "options": ["2", "4", "6", "8"],
    "answer": "B"  # 正确答案是 "4"
}
```

### 解析模型输出

模型可能输出各种格式：
```
"The answer is B"
"B"
"B) 4"
"I think the answer is B because..."
```

需要从中提取出 A/B/C/D。

### 实现要点
```python
def run_parse_mmlu_response(mmlu_example, model_output):
    """
    从模型输出中解析出预测的选项 (A/B/C/D)
    如果无法解析，返回 None
    """
    # 策略 1: 查找明确的 "answer is X" 模式
    # 策略 2: 查找独立的 A/B/C/D
    # 策略 3: 查找选项内容匹配

    # 常见模式:
    patterns = [
        r"answer is\s*([A-D])",
        r"^([A-D])[\.\)\s]",
        r"\b([A-D])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None
```

### 测试
```bash
uv run pytest tests/test_data.py::test_parse_mmlu_response -v
```

---

## 2. GSM8K (Grade School Math 8K)

### 简介
- 8.5K 小学数学应用题
- 测试多步推理能力
- 答案是数字

### 数据格式
```python
{
    "question": "Janet's ducks lay 16 eggs per day...",
    "answer": "18"  # 最终数字答案
}
```

### 解析模型输出

模型通常会展示推理过程，最后给出答案：
```
Let me solve this step by step.
First, the ducks lay 16 eggs per day.
Janet eats 3 for breakfast...
...
Therefore, the answer is 18.
```

需要提取最后出现的数字。

### 实现要点
```python
def run_parse_gsm8k_response(model_output):
    """
    从模型输出中提取最后一个数字作为答案
    如果没有数字，返回 None
    """
    # 查找所有数字（包括负数、小数）
    numbers = re.findall(r'-?\d+\.?\d*', model_output)

    if numbers:
        # 返回最后一个数字
        return numbers[-1]

    return None
```

### 注意事项
- 处理逗号分隔的数字：`1,000` -> `1000`
- 处理百分比：`50%` -> `50`
- 处理分数：可能需要计算

### 测试
```bash
uv run pytest tests/test_data.py::test_parse_gsm8k_response -v
```

---

## 3. AlpacaEval

### 简介
- 评估指令遵循能力
- 使用 LLM 作为评判者
- 比较模型输出与参考输出

### 评估流程
```
1. 给模型一个指令
2. 模型生成回复
3. LLM 评判者比较回复质量
4. 计算胜率
```

### 本作业中的使用
- 使用 Llama 3.3 70B Instruct 作为评判者
- 配置在 `scripts/alpaca_eval_vllm_llama3_3_70b_fn/`

---

## 4. Simple Safety Tests

### 简介
- 测试模型的安全性
- 包含有害请求
- 检查模型是否拒绝

### 数据位置
`data/simple_safety_tests/`

### 评估脚本
`scripts/evaluate_safety.py`

---

## 5. Packed SFT Dataset

### 为什么要 Packing？

标准方法：每个样本 padding 到相同长度
```
Sample 1: [tokens...][PAD][PAD][PAD][PAD]
Sample 2: [tokens...........][PAD][PAD]
Sample 3: [tokens.][PAD][PAD][PAD][PAD][PAD]
```
问题：大量计算浪费在 PAD tokens 上。

Packing 方法：将多个样本拼接
```
Packed: [sample1 tokens][EOS][sample2 tokens][EOS][sample3...]
```
优势：没有 padding，100% 利用率。

### 实现要点
```python
def get_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle):
    """
    1. 加载数据集
    2. Tokenize 所有样本
    3. 拼接成一个长序列
    4. 切分成固定长度的 chunks
    """
    # 加载并 tokenize
    all_tokens = []
    for example in dataset:
        tokens = tokenizer(example["text"]).input_ids
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)

    # 切分成 seq_length 的 chunks
    chunks = []
    for i in range(0, len(all_tokens) - seq_length, seq_length):
        chunk = all_tokens[i:i+seq_length]
        chunks.append({
            "input_ids": torch.tensor(chunk[:-1]),
            "labels": torch.tensor(chunk[1:])
        })

    return Dataset(chunks)
```

### 测试
```bash
uv run pytest tests/test_data.py::test_packed_sft_dataset -v
```

---

## 笔记区域

（在这里记录你的学习笔记和遇到的问题）


