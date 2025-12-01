---
description: Analyze training run results and identify failure modes
argument-hint: "[checkpoint_dir]"
allowed-tools: bash, python_user_visible, read, write
---

# Post-Training Analysis Workflow

Analyze completed training run to identify issues and next steps.

## Analysis Steps

### 1. Training Metrics Review

```python
import json
from pathlib import Path

checkpoint_dir = "$1"
metrics_file = Path(checkpoint_dir) / "metrics.json"

if metrics_file.exists():
    metrics = json.load(open(metrics_file))

    # Check for convergence
    train_losses = metrics.get("train_losses", [])
    if len(train_losses) > 100:
        early_loss = sum(train_losses[:10]) / 10
        late_loss = sum(train_losses[-10:]) / 10
        improvement = (early_loss - late_loss) / early_loss * 100
        print(f"Loss improvement: {improvement:.1f}%")

    # Check for overfitting
    val_losses = metrics.get("val_losses", [])
    if val_losses:
        if val_losses[-1] > val_losses[len(val_losses)//2]:
            print("WARNING: Possible overfitting detected")
```

### 2. Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss plateaus early | Undercapacity | Increase LoRA rank |
| Loss spikes | LR too high | Reduce learning rate |
| Val loss increases | Overfitting | Reduce steps, add data |
| No improvement | Wrong layers | Apply LoRA to MLP |
| Slow convergence | LR too low | Use 10x FullFT rule |

### 3. Capacity Check

```python
from sft_trainer import validate_capacity

result = validate_capacity(
    rank=64,  # from config
    hidden_dim=4096,
    num_layers=32,
    dataset_tokens=1_000_000  # from data
)

if result["status"] == "UNDERCAPACITY":
    print(f"Issue: {result['message']}")
    print(f"Suggested rank: {result['suggested_rank']}")
```

### 4. Sample Quality Review

Generate samples from best checkpoint and review:

```python
from tinker import SamplingClient

sampler = SamplingClient(f"{checkpoint_dir}/best")

test_prompts = [
    "Explain quantum computing",
    "Write a Python function to sort a list",
    "What are the benefits of exercise?"
]

for prompt in test_prompts:
    response = sampler.sample(prompt, temperature=0.7)
    print(f"Prompt: {prompt}")
    print(f"Response: {response[:500]}...")
    print("---")
```

### 5. Recommendations

Based on analysis:

1. **If undercapacity**: Increase rank, re-run training
2. **If overfitting**: Reduce steps to point before val loss increase
3. **If poor quality**: Review data, check for formatting issues
4. **If good metrics but poor samples**: Run LLM-as-judge eval

## Next Steps

After successful training:
1. Run `/run_eval` to compare against baseline
2. If win rate > 55%: Success, consider deployment
3. If win rate < 50%: Investigate failure modes, iterate

## Output

Analysis report written to `{checkpoint_dir}/analysis.md`
