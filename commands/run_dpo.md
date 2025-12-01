---
description: Run DPO preference optimization on preference pairs
argument-hint: "[config_path]"
allowed-tools: bash, python_user_visible, read, write
---

# DPO Training Workflow

Direct Preference Optimization - learns from preference pairs without a reward model.

## Prerequisites

Your data must be in preference format:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

## Pre-flight Checks

1. **Validate data format** - each line has prompt, chosen, rejected
2. **Check reference model** - using base model as frozen reference
3. **Confirm beta value** - KL penalty coefficient (default 0.1)

## Execution

```bash
cd sota-ml-researcher/scripts
python dpo_trainer.py --config "$1"
```

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| beta | 0.1 | Higher = more conservative |
| lora_rank | 32 | Lower rank OK for preference learning |
| learning_rate | 1e-5 | Standard for DPO |

## Metrics to Monitor

- **Loss**: Should decrease steadily
- **Reward margin**: Gap between chosen/rejected rewards (should increase)
- **Accuracy**: How often chosen > rejected (should approach 1.0)

## When to Use DPO vs GRPO

| Scenario | Method |
|----------|--------|
| Have preference pairs | DPO |
| Have verifiable rewards (math/code) | GRPO |
| Limited compute | DPO (simpler) |
| Need nuanced alignment | DPO |

## Output

- Model checkpoint at `checkpoint_dir/final/`
- Ready for LLM-as-judge evaluation
