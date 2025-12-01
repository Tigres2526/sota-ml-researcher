# Evaluation Doctrine

Best practices for rigorous LLM evaluation with focus on LLM-as-judge methods.

---

## 3-Tier Evaluation Framework

### Overview

| Tier | Purpose | When | Metrics |
|------|---------|------|---------|
| 1 | Training health | Every run | Loss, perplexity, gradients |
| 2 | Task capability | Before deploy | Benchmark accuracy |
| 3 | Quality judgment | Final selection | LLM-as-judge win rate |

### Tier 1: Training Metrics

Monitor during every training run:

```python
# Essential metrics
metrics = {
    "train_loss": [],     # Should decrease
    "val_loss": [],       # Should track train_loss
    "gradient_norm": [],  # Should be stable
    "learning_rate": [],  # For debugging
}

# Warning signs
if val_loss[-1] > val_loss[-10]:
    print("WARNING: Possible overfitting")

if gradient_norm[-1] > 10 * gradient_norm[0]:
    print("WARNING: Gradient explosion")
```

### Tier 2: Standard Benchmarks

Choose based on use case:

| Task Type | Benchmarks |
|-----------|-----------|
| General reasoning | MMLU-Pro, BBH, ARC |
| Math | GSM8K, MATH, AIME |
| Code | HumanEval, MBPP, APPS |
| Instruction following | IFEval, AlpacaEval |
| Safety | TruthfulQA, HarmBench |

**Note**: MMLU is saturated; use MMLU-Pro for modern models.

### Tier 3: LLM-as-Judge

For final model selection:

```python
from llm_judge import LLMJudge, evaluate_models

judge = LLMJudge(model="claude-sonnet-4-20250514")
results = await evaluate_models(config)
print(f"Win rate: {results['metrics']['finetuned_win_rate']}")
```

---

## LLM-as-Judge Best Practices

### 1. Pairwise Comparison

More reliable than absolute scoring:

```
Instead of: "Rate this response 1-5"
Use: "Which response is better: A or B?"
```

### 2. Position Debiasing

**Problem**: Judges often prefer first or second response.

**Solution**: Run both orderings:

```python
async def debiased_judge(prompt, resp_a, resp_b):
    # Order 1: A then B
    result_ab = await judge(prompt, resp_a, resp_b)

    # Order 2: B then A
    result_ba = await judge(prompt, resp_b, resp_a)

    # Aggregate
    if result_ab.winner == "A" and result_ba.winner == "B":
        return "A"  # Both agree A is better
    elif result_ab.winner == "B" and result_ba.winner == "A":
        return "B"  # Both agree B is better
    else:
        return "TIE"  # Disagreement = position bias
```

### 3. Structured Rubrics

Explicit criteria with score anchors:

```markdown
Evaluate on:

1. **Correctness** (0-5)
   - 0: Completely wrong
   - 3: Mostly correct with minor errors
   - 5: Perfectly accurate

2. **Helpfulness** (0-5)
   - 0: Doesn't address the question
   - 3: Partially addresses the question
   - 5: Fully addresses with useful detail

3. **Clarity** (0-5)
   - 0: Confusing, poorly organized
   - 3: Understandable but could be clearer
   - 5: Crystal clear, well-structured
```

### 4. Chain-of-Thought Judging

Judge explains reasoning before scoring:

```markdown
For each criterion:
1. Quote relevant parts of each response
2. Explain strengths and weaknesses
3. Give a score

Then state the overall winner.
```

### 5. Use Stronger Judge

Judge should be stronger than the models being evaluated:

| Evaluated Models | Recommended Judge |
|-----------------|-------------------|
| 7-8B fine-tuned | Claude Sonnet |
| 70B fine-tuned | Claude Opus |
| Claude Sonnet | Claude Opus |

---

## Metrics Interpretation

### Win Rate

```
win_rate = finetuned_wins / total_comparisons
```

| Win Rate | Interpretation |
|----------|----------------|
| < 45% | Regression - finetuned is worse |
| 45-55% | No significant difference |
| 55-65% | Modest improvement |
| 65-75% | Strong improvement |
| > 75% | Major improvement (verify not overfitting) |

### Confidence Intervals

Always report with CI:

```python
import math

def compute_ci(win_rate, n, confidence=0.95):
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    se = math.sqrt(win_rate * (1 - win_rate) / n)
    return z * se

# Example
win_rate = 0.62
n = 200
ci = compute_ci(win_rate, n)
print(f"Win rate: {win_rate:.1%} +/- {ci:.1%}")
# Output: Win rate: 62.0% +/- 6.7%
```

### Confidence Rate

Percentage of judgments where both orderings agree:

```
confidence_rate = confident_judgments / total_comparisons
```

| Confidence Rate | Interpretation |
|-----------------|----------------|
| < 60% | High position bias, results unreliable |
| 60-80% | Moderate bias, interpret cautiously |
| > 80% | Low bias, results reliable |

---

## Common Pitfalls

### 1. Benchmark Overfitting

**Problem**: Model memorized benchmark answers.

**Signs**:
- Perfect benchmark scores
- Poor on novel test cases
- Scores dropped when benchmark was updated

**Fix**: Use held-out test sets, dynamic benchmarks.

### 2. Evaluation Data Leakage

**Problem**: Test data in training set.

**Prevention**:
- Explicit data splits
- Check for overlaps
- Use post-training test sets

### 3. Judge Model Bias

**Problem**: Judge has systematic preferences.

**Examples**:
- Prefers longer responses
- Prefers certain formats
- Has knowledge cutoff issues

**Fix**: Position debiasing, multiple judges, human validation.

### 4. Insufficient Sample Size

**Problem**: Win rate estimate has high variance.

**Rule of thumb**:
- Minimum 100 comparisons for rough estimate
- 200+ for publication-quality results
- 500+ for detecting small differences

---

## Evaluation Checklist

Before reporting results:

- [ ] Ran position debiasing
- [ ] Used appropriate judge model
- [ ] Sample size >= 200
- [ ] Reported confidence intervals
- [ ] Checked for data leakage
- [ ] Saved detailed judgment logs
- [ ] Spot-checked random judgments
- [ ] Reported confidence rate

---

## Tools and Frameworks

### Inspect AI

UK AI Safety Institute's evaluation framework:

```python
from inspect_ai import Task, eval
from inspect_ai.scorer import model_graded_qa

task = Task(
    dataset="my_dataset.jsonl",
    solver=...,
    scorer=model_graded_qa(model="claude-3-sonnet")
)

results = eval(task, model="my_model")
```

### Custom LLM-as-Judge

Our implementation in `scripts/llm_judge.py`:

```python
from llm_judge import LLMJudge, EvalConfig

config = EvalConfig(
    judge_model="claude-sonnet-4-20250514",
    test_set="data/test.jsonl",
    position_debias=True,
    num_samples=200
)

results = await evaluate_models(config)
```

---

## Research References

1. **Judging LLM-as-Judge** (Zheng et al., 2023)
   - Analysis of judge biases
   - MT-Bench and Chatbot Arena

2. **AlpacaEval** (Li et al., 2023)
   - Automated evaluation benchmark
   - Length-controlled win rates

3. **Inspect AI** (UK AISI, 2024)
   - Open-source evaluation framework
   - Multi-modal, agent-capable

4. **MMLU-Pro** (2024)
   - Harder version of MMLU
   - 10 choices, reasoning-focused
