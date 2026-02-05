# 04 - GRPO (Group Relative Policy Optimization)

## 概述

GRPO 是 DeepSeekMath 和 DeepSeek-R1 中使用的强化学习方法。核心思想是：
1. 对同一个 prompt 生成多个 rollout
2. 在 group 内计算相对优势
3. 使用 PPO-style clipping 稳定训练

---

## 1. Group Normalized Rewards

### 核心思想

对于同一个 prompt，生成 $G$ 个 response，然后在 group 内归一化：

$$A_i = \frac{R_i - \mu_g}{\sigma_g + \epsilon}$$

其中：
- $\mu_g = \frac{1}{G}\sum_{j=1}^G R_j$ （group 内均值）
- $\sigma_g = \sqrt{\frac{1}{G}\sum_{j=1}^G (R_j - \mu_g)^2}$ （group 内标准差）

### 为什么要 Group Normalization？

1. **相对比较**：不同 prompt 的难度不同，绝对奖励没有可比性
2. **降低方差**：在 group 内比较，消除 prompt 难度的影响
3. **自动 baseline**：group 均值自然成为 baseline

### 数据组织

```
假设 batch 有 2 个 prompt，每个 prompt 生成 4 个 rollout (group_size=4)

rollout_responses = [
    "response_0_for_prompt_0",  # group 0
    "response_1_for_prompt_0",  # group 0
    "response_2_for_prompt_0",  # group 0
    "response_3_for_prompt_0",  # group 0
    "response_0_for_prompt_1",  # group 1
    "response_1_for_prompt_1",  # group 1
    "response_2_for_prompt_1",  # group 1
    "response_3_for_prompt_1",  # group 1
]

# 需要 reshape 成 [n_prompts, group_size] 来计算 group 统计量
```

### 实现要点
```python
def run_compute_group_normalized_rewards(
    reward_fn, rollout_responses, repeated_ground_truths,
    group_size, advantage_eps, normalize_by_std
):
    # 1. 计算每个 response 的 reward
    rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, gt)
        rewards.append(reward_dict["reward"])
    rewards = torch.tensor(rewards)

    # 2. Reshape 成 [n_prompts, group_size]
    n_prompts = len(rewards) // group_size
    rewards_grouped = rewards.view(n_prompts, group_size)

    # 3. 计算 group 统计量
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)
    group_std = rewards_grouped.std(dim=1, keepdim=True)

    # 4. 归一化
    if normalize_by_std:
        advantages = (rewards_grouped - group_mean) / (group_std + advantage_eps)
    else:
        advantages = rewards_grouped - group_mean

    # 5. Flatten 回原始形状
    advantages = advantages.view(-1)

    # 6. 返回 advantages, raw_rewards, metadata
    return advantages, rewards, {"mean_reward": rewards.mean().item(), ...}
```

### 测试
```bash
uv run pytest tests/test_grpo.py::test_compute_group_normalized_rewards -v
```

---

## 2. GRPO Training Step

### 完整流程

```
1. Rollout Phase:
   - 对每个 prompt 生成 G 个 response
   - 计算每个 response 的 reward
   - 计算 group normalized advantages
   - 记录 old_log_probs（用于 PPO clipping）

2. Training Phase (多个 epochs):
   - 计算当前 policy 的 log_probs
   - 计算 GRPO loss
   - Backward & update
```

### Microbatch Train Step

```python
def run_grpo_microbatch_train_step(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    loss_type,  # "no_baseline", "reinforce_with_baseline", "grpo_clip"
    raw_rewards=None,
    advantages=None,
    old_log_probs=None,
    cliprange=None,
):
    # 1. 计算 policy gradient loss
    loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    # 2. 应用 response mask
    # 方法 A: masked mean
    # masked_loss = run_masked_mean(loss, response_mask, dim=1).mean()

    # 方法 B: masked normalize (Dr. GRPO style)
    # masked_loss = run_masked_normalize(loss, response_mask, dim=1, normalize_constant).mean()

    # 3. 除以 gradient accumulation steps
    masked_loss = masked_loss / gradient_accumulation_steps

    # 4. Backward
    masked_loss.backward()

    # 5. 返回 loss 和 metadata
    return masked_loss.detach(), metadata
```

### 测试
```bash
uv run pytest tests/test_grpo.py::test_grpo_microbatch_train_step -v
```

---

## 3. GRPO vs PPO

| 方面 | PPO | GRPO |
|------|-----|------|
| Baseline | Value function $V(s)$ | Group mean |
| 需要 Critic | 是 | 否 |
| 样本效率 | 较高 | 需要多个 rollout |
| 实现复杂度 | 高（需要训练 Critic） | 低 |

GRPO 的优势：
- 不需要训练额外的 value network
- 实现简单
- 在 LLM 场景下效果好

---

## 4. Verified Rewards (Math)

### 什么是 Verified Rewards？

对于数学问题，可以通过验证答案是否正确来获得 **确定性** 的奖励：
- 答案正确：reward = 1
- 答案错误：reward = 0

这比使用 reward model 更可靠，因为没有 reward hacking 的问题。

### Math Grader

`cs336_alignment/drgrpo_grader.py` 提供了数学答案的验证功能：
- 支持多种格式（LaTeX, 纯文本等）
- 高召回率的答案匹配
- 来自 understand-r1-zero 项目

---

## 5. 完整 GRPO 训练伪代码

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        prompts = batch["prompts"]
        ground_truths = batch["ground_truths"]

        # === Rollout Phase ===
        rollout_responses = []
        for prompt in prompts:
            for _ in range(group_size):
                response = model.generate(prompt)
                rollout_responses.append(response)

        # 计算 rewards 和 advantages
        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            reward_fn, rollout_responses, repeated_ground_truths,
            group_size, advantage_eps, normalize_by_std=True
        )

        # 记录 old log probs
        with torch.no_grad():
            old_log_probs = get_response_log_probs(model, ...)

        # === Training Phase ===
        for ppo_epoch in range(ppo_epochs):
            optimizer.zero_grad()

            for microbatch in split(batch, grad_accum_steps):
                policy_log_probs = get_response_log_probs(model, ...)

                grpo_microbatch_train_step(
                    policy_log_probs, response_mask, grad_accum_steps,
                    loss_type="grpo_clip",
                    advantages=advantages,
                    old_log_probs=old_log_probs,
                    cliprange=0.2
                )

            optimizer.step()
```

---

## 笔记区域

（在这里记录你的学习笔记和遇到的问题）


