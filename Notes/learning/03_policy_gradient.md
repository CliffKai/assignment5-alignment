# 03 - 策略梯度基础

## 概述

策略梯度是强化学习的核心方法，用于优化策略以最大化期望奖励。GRPO 建立在这些基础之上。

---

## 1. REINFORCE 算法

### 目标
最大化期望奖励：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

### 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

在语言模型中：
- $\pi_\theta$ = 语言模型
- $a_t$ = 生成的 token
- $s_t$ = 之前生成的所有 tokens
- $R(\tau)$ = 整个 response 的奖励

### Loss 函数

$$\mathcal{L}_{PG} = -R \cdot \sum_t \log p_\theta(x_t | x_{<t})$$

注意负号：我们要 **最大化** 奖励，但优化器是 **最小化** loss。

### 实现
```python
def run_compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    # raw_rewards_or_advantages: [batch, 1]
    # policy_log_probs: [batch, seq]

    # Loss = -reward * log_prob (per token)
    loss = -raw_rewards_or_advantages * policy_log_probs
    # loss: [batch, seq]

    return loss
```

### 测试
```bash
uv run pytest tests/test_grpo.py::test_compute_naive_policy_gradient_loss -v
```

---

## 2. Baseline 与 Advantage

### 问题：高方差

原始 REINFORCE 的方差很高，因为：
- 奖励可能都是正的（或都是负的）
- 即使是"差"的 action，如果奖励为正，也会被强化

### 解决方案：Baseline

引入 baseline $b$，使用 **advantage** 代替 raw reward：

$$A = R - b$$

常见的 baseline：
- 均值：$b = \mathbb{E}[R]$
- Value function：$b = V(s)$

### 为什么有效？

$$\mathbb{E}[A \cdot \nabla \log \pi] = \mathbb{E}[(R-b) \cdot \nabla \log \pi] = \mathbb{E}[R \cdot \nabla \log \pi]$$

Baseline 不改变梯度的期望，但降低方差！

### GRPO 中的 Group Baseline

GRPO 使用同一个 prompt 的多个 rollout 的平均奖励作为 baseline：

$$A_i = R_i - \frac{1}{G}\sum_{j=1}^{G} R_j$$

其中 $G$ 是 group size。

---

## 3. PPO Clipping

### 问题：策略更新过大

如果一步更新太大，可能导致：
- 策略崩溃
- 训练不稳定

### 解决方案：Clipped Objective

PPO 限制新旧策略的比率：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

Clipped loss：

$$\mathcal{L}^{CLIP} = -\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)$$

### 直觉理解

- 如果 $A > 0$（好的 action）：允许增加概率，但不超过 $1+\epsilon$ 倍
- 如果 $A < 0$（差的 action）：允许减少概率，但不低于 $1-\epsilon$ 倍

### 实现
```python
def run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    # 计算 log ratio
    log_ratio = policy_log_probs - old_log_probs

    # 计算 ratio
    ratio = torch.exp(log_ratio)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    # 两个 loss
    loss1 = -advantages * ratio
    loss2 = -advantages * clipped_ratio

    # 取 max（因为我们要最小化 loss，所以取 max 是保守的）
    loss = torch.max(loss1, loss2)

    # Metadata: 用于计算 clip fraction
    clipped = (ratio < 1 - cliprange) | (ratio > 1 + cliprange)

    return loss, {"clipped": clipped}
```

### 测试
```bash
uv run pytest tests/test_grpo.py::test_compute_grpo_clip_loss -v
```

---

## 4. 三种 Loss 类型对比

| Loss Type | 公式 | 特点 |
|-----------|------|------|
| `no_baseline` | $-R \cdot \log p$ | 最简单，高方差 |
| `reinforce_with_baseline` | $-A \cdot \log p$ | 使用 advantage，低方差 |
| `grpo_clip` | $-\min(rA, \text{clip}(r)A)$ | PPO 风格，稳定训练 |

### 实现 Wrapper
```python
def run_compute_policy_gradient_loss(
    policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
):
    if loss_type == "no_baseline":
        return run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return run_compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
```

---

## 笔记区域

（在这里记录你的学习笔记和遇到的问题）


