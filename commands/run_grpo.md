---
description: Run GRPO reinforcement learning with verifiable rewards
argument-hint: "[config_path]"
allowed-tools: bash, python_user_visible, read, write
---

# GRPO Training Workflow

Group Relative Policy Optimization - memory-efficient RL without value model.

## Best For

- Math reasoning (verifiable answers)
- Code generation (test case verification)
- Any task with objective success criteria

## Pre-flight Checks

1. **Verify reward function** - must return scalar reward for (prompt, response)
2. **Check group size** - K samples per prompt (default 8)
3. **Confirm prompts file** - one prompt per line in JSONL

## Execution

```bash
cd sota-ml-researcher/scripts
python grpo_trainer.py --config "$1"
```

## Key GRPO Insight

From DeepSeek research:

> Eliminates the value/critic model by using group mean as baseline

This means:
- **No value model** to train or store
- **Memory efficient** - only one model in memory
- **Simpler training** - fewer hyperparameters

## LoRA for RL

From Thinking Machines:

> "LoRA fully matches FullFT for policy gradient algorithms, even with rank as low as 1"

Why? RL absorbs ~1000x less info per token than SFT.

| Model Size | SFT Rank | RL Rank |
|------------|----------|---------|
| 8B | 64 | 8 |
| 70B | 256 | 32 |

## Reward Functions

Built-in options:
- `math`: Checks numerical answer correctness
- `code`: Runs test cases
- `format`: Checks structural compliance
- `custom`: Load from script

## Metrics to Monitor

- **Mean reward**: Should increase over training
- **Max reward**: Upper bound on performance
- **Reward std**: Diversity of solutions

## Output

- Model checkpoint at `checkpoint_dir/final/`
- Ready for evaluation
