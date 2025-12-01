# LoRA/QLoRA Doctrine

**Source: Thinking Machines "LoRA Without Regret" (2025)**

This document synthesizes key findings from extensive LoRA vs Full Fine-Tuning experiments.

---

## Executive Summary

> When done correctly, LoRA matches Full Fine-Tuning performance with ~2/3 the compute.

Key requirements:
1. Apply LoRA to ALL layers (especially MLP)
2. Use 10x higher learning rate than FullFT
3. Ensure capacity: `LoRA_params × 2 >= dataset_info_bits`

---

## Critical Findings

### 1. Which Layers to Apply LoRA

**Finding**: Apply LoRA to ALL layers, not just attention.

| Configuration | Performance |
|---------------|-------------|
| Attention-only | Significantly underperforms |
| MLP-only | Strong performance |
| MLP + Attention | Best results |
| MLP + Attention + Unembed | Optimal |

**Evidence**: Attention-only with rank 256 (0.25B params) underperformed MLP-only with rank 128 (0.24B params).

**Conclusion**: The original LoRA paper recommendation of attention-only is suboptimal for modern use cases.

### 2. Learning Rate

**Finding**: Optimal LoRA LR is consistently 10x the FullFT LR.

This ratio holds for:
- Supervised learning
- Reinforcement learning
- All model sizes tested
- All ranks tested

**Formula**:
```
LR_LoRA = 10 × LR_FullFT
LR_FullFT ≈ (2000 / hidden_size) × 1e-4
```

For Llama-3.1-8B (hidden_size=4096):
- FullFT LR: ~5e-5
- LoRA LR: ~5e-4

### 3. Capacity Requirements

**Finding**: LoRA needs sufficient parameters to store the information in the dataset.

**Key insight**: Neural networks can store ~2 bits per parameter.

**Rule of thumb**:
```
LoRA_params × 2 >= dataset_tokens × target_loss
```

Where target_loss ≈ 1 bit/token for typical LLM training.

**Example**:
- Dataset: 1M tokens
- Required capacity: 1M bits
- Minimum LoRA params: 500K
- For rank-64 on 8B model: ~50M params ✓

### 4. Batch Size Effects

**Finding**: At large batch sizes, LoRA shows a persistent performance gap vs FullFT.

- This gap is NOT mitigated by increasing rank
- It's inherent to the product-of-matrices parametrization
- Keep batch sizes moderate (16-64) for best results

---

## Hyperparameter Reference

### Default Configuration

```yaml
# LoRA settings
lora_rank: 64          # For 8B model, adjust by size
lora_alpha: 32         # Scaling factor
dropout: 0.0           # Usually not needed

# Which layers
apply_to:
  - mlp.up_proj
  - mlp.down_proj
  - mlp.gate_proj     # If SwiGLU
  - attn.q_proj
  - attn.k_proj
  - attn.v_proj
  - attn.o_proj
  - lm_head           # Unembed

# Initialization
init_A: "uniform"      # Scale 1/sqrt(d_in)
init_B: "zero"         # Critical for stable start

# Learning rate
lr_multiplier: 10      # vs FullFT baseline
```

### Rank Selection by Model Size

| Model Size | SFT Rank | RL Rank | Params (approx) |
|------------|----------|---------|-----------------|
| 1-3B | 16-32 | 4-8 | 5-20M |
| 7-8B | 32-64 | 8-16 | 20-80M |
| 13B | 64-128 | 16-32 | 40-160M |
| 34B | 128-192 | 32-48 | 100-300M |
| 70B | 128-256 | 32-64 | 200-500M |

### MoE Model Handling

For Mixture of Experts models:
- Apply separate LoRA to each expert
- Divide rank by number of active experts

Example for Qwen3-30B-A3B (8 experts, 2 active):
```yaml
lora_rank_per_expert: 32  # Instead of rank 64 for dense
# Total params similar to dense model
```

---

## Capacity Calculation

### Quick Formula

```python
def min_rank_for_dataset(dataset_tokens, hidden_dim, num_layers):
    """Calculate minimum LoRA rank for dataset size."""
    # Required capacity in bits
    required_bits = dataset_tokens  # ~1 bit/token

    # Params per rank (approximate for all-layer LoRA)
    params_per_rank = hidden_dim * 2 * 10 * num_layers
    # 10 = approximate number of LoRA-adapted projections per layer

    # Bits per param
    bits_per_param = 2

    # Minimum params
    min_params = required_bits / bits_per_param

    # Minimum rank
    min_rank = min_params / params_per_rank

    return int(min_rank) + 1
```

### Capacity Validation

Always run before training:

```python
from sft_trainer import validate_capacity

result = validate_capacity(
    rank=64,
    hidden_dim=4096,
    num_layers=32,
    dataset_tokens=count_tokens("data/train.jsonl")
)

if result["status"] == "UNDERCAPACITY":
    print(f"WARNING: {result['message']}")
    print(f"Suggested rank: {result['suggested_rank']}")
```

---

## Common Pitfalls

### 1. Attention-Only LoRA
**Symptom**: Poor performance despite large rank
**Fix**: Add MLP layers

### 2. Learning Rate Too Low
**Symptom**: Very slow convergence, loss barely decreasing
**Fix**: Use 10x FullFT LR

### 3. Undercapacity
**Symptom**: Loss plateaus early, poor generalization
**Fix**: Increase rank, verify capacity calculation

### 4. Missing Unembed
**Symptom**: Poor generation quality despite low loss
**Fix**: Add lm_head/unembed to LoRA layers

### 5. Wrong Alpha
**Symptom**: Unstable training
**Fix**: Use alpha=32 or alpha=rank

---

## Research Citations

1. **LoRA Without Regret** (Thinking Machines, 2025)
   - Primary source for these guidelines
   - Extensive Llama 3 / Qwen3 experiments

2. **LoRA: Low-Rank Adaptation** (Hu et al., 2021)
   - Original LoRA paper
   - Note: Attention-only recommendation is outdated

3. **QLoRA: Efficient Finetuning** (Dettmers et al., 2023)
   - 4-bit quantization + LoRA
   - Memory efficiency techniques

4. **DoRA: Weight-Decomposed LoRA** (Liu et al., 2024)
   - Alternative decomposition
   - Marginal improvements in some cases
