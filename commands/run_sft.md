---
description: Run supervised fine-tuning with research-backed LoRA defaults
argument-hint: "[config_path]"
allowed-tools: bash, python_user_visible, read, write
---

# Supervised Fine-Tuning Workflow

Execute SFT training with automatic capacity validation and research-backed hyperparameters.

## Pre-flight Checks

Before training, I will:

1. **Load and validate config** from `$1` (or create default)
2. **Count dataset tokens** to estimate information content
3. **Validate LoRA capacity** - ensure `LoRA_params × 2 >= dataset_tokens`
4. **Auto-derive learning rate** if not specified (10x FullFT rule)
5. **Confirm layers** - LoRA should apply to ALL layers (mlp, attn, unembed)

## Execution

```bash
# Navigate to plugin directory
cd sota-ml-researcher/scripts

# Run with capacity validation first
python sft_trainer.py --config "$1" --validate-only

# If capacity OK, proceed with training
python sft_trainer.py --config "$1"
```

## Key Research-Backed Defaults

From Thinking Machines "LoRA Without Regret":

| Setting | Value | Rationale |
|---------|-------|-----------|
| Learning rate | 10x FullFT | Empirically optimal |
| LoRA layers | mlp + attn + unembed | Attention-only underperforms |
| Capacity | params×2 >= tokens | ~2 bits per parameter storage |

## Monitoring

During training, watch for:
- Loss decreasing steadily (log scale with steps)
- Validation loss tracking training loss
- No sudden spikes (learning rate too high)

## Output

- Checkpoints saved to `checkpoint_dir/`
- Best model at `checkpoint_dir/best/`
- Training logs at `log_dir/`
