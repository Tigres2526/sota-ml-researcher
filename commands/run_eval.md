---
description: Run LLM-as-judge evaluation comparing baseline to finetuned model
argument-hint: "[config_path]"
allowed-tools: bash, python_user_visible, read, write
---

# LLM-as-Judge Evaluation Workflow

Rigorous pairwise evaluation using Claude as judge with position debiasing.

## What This Does

1. Loads test prompts
2. Generates responses from baseline and finetuned models
3. Runs pairwise LLM-as-judge comparison
4. Applies position debiasing (runs both orderings)
5. Computes win rates with confidence intervals

## Pre-flight Checks

1. **Verify test set** - JSONL with prompt field
2. **Confirm models** - baseline and finetuned paths
3. **Set rubric** - evaluation criteria

## Execution

```bash
cd sota-ml-researcher/scripts
python llm_judge.py --config "$1"
```

Or run async:
```python
import asyncio
from llm_judge import evaluate_models, EvalConfig

config = EvalConfig.from_yaml("$1")
results = asyncio.run(evaluate_models(config))
print(results["metrics"])
```

## Position Debiasing

**Problem**: Judges often prefer first or second response regardless of quality.

**Solution**: Run both orderings (A,B) and (B,A):
- If both agree → confident result
- If they disagree → declare TIE (position bias detected)

## Metrics Output

| Metric | Description |
|--------|-------------|
| `finetuned_win_rate` | % where finetuned beats baseline |
| `baseline_win_rate` | % where baseline beats finetuned |
| `tie_rate` | % of ties (including disagreements) |
| `confidence_rate` | % of confident (non-positional) judgments |
| `finetuned_win_rate_95ci` | Win rate with 95% confidence interval |

## Rubric Best Practices

1. **Explicit criteria** with score anchors
2. **Chain-of-thought** - judge explains before scoring
3. **Pairwise comparison** - more reliable than absolute
4. **Specific to task** - customize for your domain

## Output Files

- `eval_results/judgments.jsonl` - detailed per-item judgments
- `eval_results/metrics.json` - aggregate statistics
