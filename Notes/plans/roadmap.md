# 学习路线图

## 总体目标

完成 CS336 Assignment 5: Alignment，掌握 SFT、GRPO 和 DPO 的原理与实现。

---

## 阶段一：基础准备

### 1.1 环境搭建
- [ ] 安装依赖 (`uv sync`)
- [ ] 运行测试确认环境正常
- [ ] 阅读作业 PDF 了解要求

### 1.2 代码结构熟悉
- [ ] 理解 `tests/adapters.py` 的作用
- [ ] 了解测试框架和 snapshot testing
- [ ] 浏览 `cs336_alignment/` 目录

---

## 阶段二：主要作业

### 2.1 基础工具 (Day 1)
```
learning/01_basics.md
├── masked_mean
├── masked_normalize
└── entropy
```
测试：`uv run pytest tests/test_metrics.py tests/test_grpo.py::test_compute_entropy -v`

### 2.2 SFT 实现 (Day 1-2)
```
learning/02_sft.md
├── tokenize_prompt_and_output
├── get_response_log_probs
└── sft_microbatch_train_step
```
测试：`uv run pytest tests/test_sft.py tests/test_grpo.py::test_get_response_log_probs -v`

### 2.3 策略梯度 (Day 2-3)
```
learning/03_policy_gradient.md
├── naive_policy_gradient_loss
├── grpo_clip_loss
└── policy_gradient_loss (wrapper)
```
测试：`uv run pytest tests/test_grpo.py::test_compute_naive_policy_gradient_loss tests/test_grpo.py::test_compute_grpo_clip_loss -v`

### 2.4 GRPO 完整实现 (Day 3-4)
```
learning/04_grpo.md
├── compute_group_normalized_rewards
└── grpo_microbatch_train_step
```
测试：`uv run pytest tests/test_grpo.py -v`

---

## 阶段三：可选作业

### 3.1 DPO (Optional)
```
learning/05_dpo.md
└── compute_per_instance_dpo_loss
```
测试：`uv run pytest tests/test_dpo.py -v`

### 3.2 数据处理与评估 (Optional)
```
learning/06_evaluation.md
├── packed_sft_dataset
├── iterate_batches
├── parse_mmlu_response
└── parse_gsm8k_response
```
测试：`uv run pytest tests/test_data.py -v`

---

## 阶段四：实验与报告

### 4.1 训练实验
- [ ] SFT 训练
- [ ] Expert Iteration
- [ ] GRPO 训练

### 4.2 评估
- [ ] MMLU 评估
- [ ] GSM8K 评估
- [ ] AlpacaEval (可选)

### 4.3 报告撰写
- [ ] 实验结果整理
- [ ] 分析与讨论

---

## 依赖关系图

```
                    ┌─────────────────┐
                    │  masked_mean    │
                    │ masked_normalize│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌───────────┐  ┌───────────┐
        │ entropy  │  │ log_probs │  │ tokenize  │
        └────┬─────┘  └─────┬─────┘  └─────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ naive_pg │  │ grpo_clip│  │ sft_step │
        └────┬─────┘  └────┬─────┘  └──────────┘
             │             │
             └──────┬──────┘
                    │
                    ▼
            ┌──────────────┐
            │ grpo_step    │
            └──────────────┘
```

---

## 参考时间分配

| 阶段 | 内容 | 建议时间 |
|------|------|----------|
| 阶段一 | 基础准备 | 0.5 天 |
| 阶段二 | 主要作业 | 3-4 天 |
| 阶段三 | 可选作业 | 1-2 天 |
| 阶段四 | 实验报告 | 2-3 天 |

**总计：约 1 周**
