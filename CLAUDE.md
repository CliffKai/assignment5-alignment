# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS336 Spring 2025 Assignment 5: Alignment. Implements SFT (Supervised Fine-Tuning), Expert Iteration, and GRPO (Group Relative Policy Optimization) with verified rewards for math problems. Optional supplement covers safety alignment, instruction tuning, and RLHF/DPO.

## Build and Test Commands

```bash
# Install dependencies (two-step for flash-attn compatibility)
uv sync --no-install-package flash-attn
uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_grpo.py

# Run specific test
uv run pytest tests/test_grpo.py::test_compute_group_normalized_rewards

# Run tests with verbose output
uv run pytest -v

# Run tests and create submission
./test_and_make_submission.sh
```

## Architecture

### Implementation Pattern
All implementations go in `cs336_alignment/` module. Tests connect to implementations via adapter functions in `tests/adapters.py`. Each adapter function raises `NotImplementedError` initially - implement the actual logic and have adapters call your implementations.

### Core Components to Implement

**SFT (test_sft.py)**
- `run_tokenize_prompt_and_output`: Tokenize prompts/outputs with response masking
- `run_sft_microbatch_train_step`: SFT training step with gradient accumulation
- `get_packed_sft_dataset`: Create packed dataset for efficient training
- `run_iterate_batches`: Batch iteration over datasets

**GRPO (test_grpo.py)**
- `run_compute_group_normalized_rewards`: Group-normalize rewards (DeepSeekMath/R1 style)
- `run_compute_entropy`: Entropy of logits distribution
- `run_get_response_log_probs`: Log-probs of responses given prompts
- `run_compute_naive_policy_gradient_loss`: Basic REINFORCE loss
- `run_compute_grpo_clip_loss`: PPO-style clipped loss for GRPO
- `run_grpo_microbatch_train_step`: GRPO training step

**Utilities (test_metrics.py)**
- `run_masked_mean`: Mean over masked elements
- `run_masked_normalize`: Sum and normalize with masking

**Evaluation (test_data.py)**
- `run_parse_mmlu_response`: Parse MMLU model outputs to A/B/C/D
- `run_parse_gsm8k_response`: Extract numeric answers from GSM8K outputs

**DPO (test_dpo.py, optional)**
- `run_compute_per_instance_dpo_loss`: Direct Preference Optimization loss

### Key Files
- `cs336_alignment/drgrpo_grader.py`: Math answer grading with high recall (from understand-r1-zero)
- `cs336_alignment/prompts/`: Prompt templates (alpaca_sft, r1_zero, zero_shot_system_prompt)
- `tests/conftest.py`: Pytest fixtures including model/tokenizer setup
- `tests/_snapshots/`: Expected outputs for snapshot testing

### Data
- `data/mmlu/`: MMLU benchmark (dev/val/test splits)
- `data/gsm8k/`: GSM8K math problems (train/test)
- `data/alpaca_eval/`: AlpacaEval benchmark
- `data/simple_safety_tests/`: Safety evaluation data

### Models
Tests expect models at `/data/a5-alignment/models/Qwen2.5-Math-1.5B` (configurable via fixtures).

## Key Concepts

- **GRPO**: Group Relative Policy Optimization - normalizes rewards within groups of rollouts for the same prompt
- **Response masking**: Loss computed only on response tokens, not prompt or padding
- **Gradient accumulation**: Divide loss by accumulation steps before backward pass
- **Snapshot testing**: Tests compare against pre-computed expected outputs in `tests/_snapshots/`
