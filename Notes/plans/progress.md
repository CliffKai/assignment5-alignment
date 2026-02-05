# 进度追踪

## 总体进度

- [ ] 阶段一：基础准备
- [ ] 阶段二：主要作业
- [ ] 阶段三：可选作业
- [ ] 阶段四：实验与报告

---

## 详细进度

### 基础工具
| 函数 | 学习 | 实现 | 测试通过 | 备注 |
|------|:----:|:----:|:--------:|------|
| `run_masked_mean` | [ ] | [ ] | [ ] | |
| `run_masked_normalize` | [ ] | [ ] | [ ] | |
| `run_compute_entropy` | [ ] | [ ] | [ ] | |

### SFT
| 函数 | 学习 | 实现 | 测试通过 | 备注 |
|------|:----:|:----:|:--------:|------|
| `run_tokenize_prompt_and_output` | [ ] | [ ] | [ ] | |
| `run_get_response_log_probs` | [ ] | [ ] | [ ] | |
| `run_sft_microbatch_train_step` | [ ] | [ ] | [ ] | |

### 策略梯度
| 函数 | 学习 | 实现 | 测试通过 | 备注 |
|------|:----:|:----:|:--------:|------|
| `run_compute_naive_policy_gradient_loss` | [ ] | [ ] | [ ] | |
| `run_compute_grpo_clip_loss` | [ ] | [ ] | [ ] | |
| `run_compute_policy_gradient_loss` | [ ] | [ ] | [ ] | |

### GRPO
| 函数 | 学习 | 实现 | 测试通过 | 备注 |
|------|:----:|:----:|:--------:|------|
| `run_compute_group_normalized_rewards` | [ ] | [ ] | [ ] | |
| `run_grpo_microbatch_train_step` | [ ] | [ ] | [ ] | |

### 可选：DPO
| 函数 | 学习 | 实现 | 测试通过 | 备注 |
|------|:----:|:----:|:--------:|------|
| `run_compute_per_instance_dpo_loss` | [ ] | [ ] | [ ] | |

### 可选：数据处理
| 函数 | 学习 | 实现 | 测试通过 | 备注 |
|------|:----:|:----:|:--------:|------|
| `get_packed_sft_dataset` | [ ] | [ ] | [ ] | |
| `run_iterate_batches` | [ ] | [ ] | [ ] | |
| `run_parse_mmlu_response` | [ ] | [ ] | [ ] | |
| `run_parse_gsm8k_response` | [ ] | [ ] | [ ] | |

---

## 测试结果记录

### 最近一次测试
```bash
# 运行时间：
# 命令：uv run pytest -v
# 结果：
```

### 测试历史
| 日期 | 通过 | 失败 | 备注 |
|------|------|------|------|
| | | | |

---

## 里程碑

- [ ] **M1**: 所有基础工具测试通过
- [ ] **M2**: SFT 相关测试通过
- [ ] **M3**: GRPO 相关测试通过
- [ ] **M4**: 所有主要作业测试通过
- [ ] **M5**: 可选作业完成
- [ ] **M6**: 实验完成
- [ ] **M7**: 提交作业
