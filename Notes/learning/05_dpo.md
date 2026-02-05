# 05 - DPO (Direct Preference Optimization)

## 概述

DPO 是一种不需要显式 reward model 的 alignment 方法。直接从偏好数据学习，避免了 RL 训练的复杂性。

**注意：这是可选作业的内容。**

---

## 1. 从 RLHF 到 DPO

### 传统 RLHF 流程

```
1. 收集偏好数据: (prompt, chosen, rejected)
2. 训练 Reward Model: r(prompt, response)
3. 用 RL (PPO) 优化: max E[r(x,y)] - β·KL(π||π_ref)
```

问题：
- 需要训练单独的 reward model
- RL 训练不稳定
- 计算成本高

### DPO 的洞察

DPO 发现 RLHF 的最优解有闭式形式：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

反解出 reward：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

---

## 2. DPO Loss

### Bradley-Terry 模型

人类偏好可以建模为：

$$p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

其中 $\sigma$ 是 sigmoid 函数。

### DPO Loss 推导

将 reward 的闭式解代入 Bradley-Terry 模型：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### 简化形式

定义 log ratio：
$$h_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

则：
$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \cdot (h_\theta(x, y_w) - h_\theta(x, y_l))\right)$$

---

## 3. 实现

### 计算 Response Log Probability

```python
def get_response_log_prob(model, tokenizer, prompt, response):
    """计算 p(response | prompt) 的 log probability"""
    # 1. Tokenize prompt + response
    full_text = prompt + response
    tokens = tokenizer(full_text, return_tensors="pt")

    # 2. 前向传播
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    # 3. 计算 log probs
    log_probs = F.log_softmax(logits, dim=-1)

    # 4. 只取 response 部分的 log prob 并求和
    prompt_len = len(tokenizer(prompt).input_ids)
    response_log_prob = log_probs[prompt_len:].sum()

    return response_log_prob
```

### DPO Loss 实现

```python
def run_compute_per_instance_dpo_loss(
    lm,           # 正在训练的模型
    lm_ref,       # 参考模型（frozen）
    tokenizer,
    beta,
    prompt,
    response_chosen,
    response_rejected,
):
    # 1. 计算 policy 的 log probs
    log_prob_chosen = get_response_log_prob(lm, tokenizer, prompt, response_chosen)
    log_prob_rejected = get_response_log_prob(lm, tokenizer, prompt, response_rejected)

    # 2. 计算 reference 的 log probs
    with torch.no_grad():
        ref_log_prob_chosen = get_response_log_prob(lm_ref, tokenizer, prompt, response_chosen)
        ref_log_prob_rejected = get_response_log_prob(lm_ref, tokenizer, prompt, response_rejected)

    # 3. 计算 log ratios
    log_ratio_chosen = log_prob_chosen - ref_log_prob_chosen
    log_ratio_rejected = log_prob_rejected - ref_log_prob_rejected

    # 4. 计算 DPO loss
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))

    return loss
```

### 测试
```bash
uv run pytest tests/test_dpo.py -v
```

---

## 4. DPO vs RLHF

| 方面 | RLHF | DPO |
|------|------|-----|
| 需要 Reward Model | 是 | 否 |
| 训练稳定性 | 较差 | 好 |
| 计算成本 | 高 | 低 |
| 超参数敏感度 | 高 | 低 |
| 理论基础 | RL | 闭式解 |

### DPO 的优势
- 实现简单，只需要监督学习
- 不需要 reward model
- 训练稳定

### DPO 的局限
- 需要成对的偏好数据
- 不能利用 reward model 的泛化能力
- 对数据质量敏感

---

## 5. β 参数的作用

$\beta$ 控制与参考模型的偏离程度：
- $\beta$ 大：更保守，接近参考模型
- $\beta$ 小：更激进，可能偏离参考模型较多

典型值：$\beta \in [0.1, 0.5]$

---

## 笔记区域

（在这里记录你的学习笔记和遇到的问题）


