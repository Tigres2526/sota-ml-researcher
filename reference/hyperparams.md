# Hyperparameter Quick Reference

Fast lookup tables for common training configurations.

---

## LoRA Configuration

### Rank by Model Size (SFT)

| Model Size | Rank | Alpha | Params (approx) |
|------------|------|-------|-----------------|
| 1-3B | 16-32 | 32 | 5-20M |
| 7-8B | 32-64 | 32 | 20-80M |
| 13B | 64-128 | 32 | 40-160M |
| 34B | 128-192 | 32 | 100-300M |
| 70B | 128-256 | 32 | 200-500M |
| MoE | rank/active_experts | 32 | varies |

### Rank by Model Size (RL)

RL needs much lower rank - absorbs ~1000x less info per token.

| Model Size | Rank | Alpha | Params (approx) |
|------------|------|-------|-----------------|
| 7-8B | 4-16 | 16 | 3-12M |
| 13B | 8-32 | 16 | 5-25M |
| 70B | 16-64 | 32 | 15-60M |

### Target Modules

**Always apply to ALL layers:**

```yaml
# Llama / Mistral
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
  - lm_head  # Unembed

# Qwen
target_modules:
  - c_attn      # Combined QKV
  - c_proj      # Attention output
  - w1          # MLP up
  - w2          # MLP down
  - lm_head
```

---

## Learning Rates

### Base Learning Rates

| Method | FullFT LR | LoRA LR (10x) |
|--------|-----------|---------------|
| SFT | 1e-5 to 5e-5 | 1e-4 to 5e-4 |
| DPO | 5e-7 to 5e-6 | 5e-6 to 5e-5 |
| GRPO | 1e-6 to 5e-6 | 1e-5 to 5e-5 |
| PPO | 1e-6 | 1e-5 |

### By Model Size (SFT + LoRA)

```python
# Rule of thumb
lr = 10 * (2000 / hidden_size) * 1e-4
```

| Model | Hidden Size | LoRA LR |
|-------|-------------|---------|
| 7B | 4096 | ~5e-4 |
| 8B | 4096 | ~5e-4 |
| 13B | 5120 | ~4e-4 |
| 34B | 8192 | ~2.5e-4 |
| 70B | 8192 | ~2.5e-4 |

---

## Training Duration

### Steps vs Epochs

| Dataset Size | Epochs | Approx Steps (batch=16) |
|--------------|--------|-------------------------|
| <10K | 1-3 | 600-1800 |
| 10K-50K | 1-2 | 600-6000 |
| 50K-100K | 1 | 3000-6000 |
| >100K | 0.5-1 | 3000-10000 |

### Eval Frequency

| Training Steps | Eval Every |
|----------------|------------|
| <1000 | 50-100 |
| 1000-5000 | 100-500 |
| 5000-20000 | 500-1000 |
| >20000 | 1000-2000 |

---

## Batch Sizes

### By GPU Memory (LoRA, 8B model)

| GPU Memory | Max Batch Size | Effective with Grad Accum |
|------------|----------------|---------------------------|
| 16GB | 1-2 | 16 (8 accum) |
| 24GB | 2-4 | 16 (4 accum) |
| 40GB | 4-8 | 32 (4 accum) |
| 80GB | 8-16 | 64 (4 accum) |

### Recommendations

- **SFT**: 16-64 effective batch size
- **DPO**: 8-32 (pairs = 2x memory)
- **GRPO**: 4-16 prompts × group_size samples

---

## DPO Specific

### Beta (KL Coefficient)

| Beta | Behavior |
|------|----------|
| 0.01 | Very aggressive, may overfit |
| 0.05 | Aggressive |
| 0.1 | Standard (recommended) |
| 0.2 | Moderate |
| 0.5 | Conservative |
| 1.0 | Very conservative |

### Choosing Beta

- **New domain**: Start with 0.1
- **Safety alignment**: Use 0.2-0.5
- **Strong baseline**: Use 0.05-0.1
- **Weak baseline**: Use 0.2+

---

## GRPO Specific

### Group Size

| Samples per Prompt | Trade-off |
|-------------------|-----------|
| 4 | Fast, higher variance |
| 8 | Standard (recommended) |
| 16 | Lower variance, 2x cost |
| 32 | Research only, 4x cost |

### Temperature

| Temperature | Use Case |
|-------------|----------|
| 0.5 | Less exploration, faster convergence |
| 0.7 | Standard (recommended) |
| 1.0 | More exploration |
| 1.2+ | Maximum exploration, may be unstable |

---

## Optimizer Settings

### AdamW (Standard)

```yaml
optimizer: adamw
betas: [0.9, 0.999]
eps: 1e-8
weight_decay: 0.0  # Usually 0 for LoRA
```

### For Stability

```yaml
max_grad_norm: 1.0  # Gradient clipping
warmup_ratio: 0.03  # 3% of steps
lr_scheduler: cosine  # Or constant
```

---

## Context Lengths

### Maximum Sequence Length

| Model | Native | Extended |
|-------|--------|----------|
| Llama 3 | 8K | 128K (RoPE scaling) |
| Qwen2.5 | 32K | 128K |
| Mistral | 8K | 32K |

### Recommendations

| Use Case | Max Seq Len |
|----------|-------------|
| Short-form QA | 2048 |
| General chat | 4096 |
| Long documents | 8192+ |
| Code | 4096-8192 |

---

## Common Configurations

### SFT (General Purpose)

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
lora_rank: 64
lora_alpha: 32
learning_rate: 5e-4
batch_size: 16
num_steps: 5000
max_seq_len: 4096
target_modules: [mlp, attn, lm_head]
```

### DPO (Preference Alignment)

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
lora_rank: 32
lora_alpha: 32
beta: 0.1
learning_rate: 1e-5
batch_size: 8
num_steps: 3000
```

### GRPO (Math Reasoning)

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
lora_rank: 8
lora_alpha: 16
group_size: 8
temperature: 0.7
learning_rate: 1e-5
batch_size: 4
num_steps: 2000
```

---

## Troubleshooting Quick Reference

| Symptom | Check | Fix |
|---------|-------|-----|
| Loss not decreasing | LR too low | Increase LR 2-5x |
| Loss exploding | LR too high | Reduce LR 2-5x |
| Early plateau | Undercapacity | Increase rank |
| Overfitting | Too many steps | Reduce steps |
| Poor generation | Missing unembed | Add lm_head to targets |
| Slow training | Batch too small | Increase batch/accum |
