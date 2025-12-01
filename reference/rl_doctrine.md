# Reinforcement Learning Doctrine

Comprehensive guide to RL methods for LLM post-training: DPO, GRPO, PPO, RLAIF.

---

## Method Selection

### Decision Tree

```
START
│
├─ Have preference pairs (chosen/rejected)?
│   └─ YES → DPO
│
├─ Have verifiable rewards (math/code test cases)?
│   └─ YES → GRPO
│
├─ Need nuanced alignment, complex preferences?
│   └─ YES → PPO + Reward Model
│
├─ Want to scale with AI feedback?
│   └─ YES → RLAIF (Constitutional AI pattern)
│
└─ Limited compute?
    └─ YES → DPO (simplest)
```

### Comparison Table

| Method | Requires | Complexity | Best For |
|--------|----------|------------|----------|
| DPO | Preference pairs | Low | General alignment |
| GRPO | Verifiable rewards | Medium | Math, code |
| PPO | Reward model + value model | High | Complex alignment |
| RLAIF | Strong AI labeler | Medium | Scaling feedback |

---

## DPO (Direct Preference Optimization)

### Core Idea

DPO eliminates the reward model by showing there's a closed-form solution:
> The optimal policy can be derived directly from preferences without explicit reward modeling.

### Loss Function

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    L_DPO = -log(sigmoid(β * (log π(y_w|x)/π_ref(y_w|x) -
                              log π(y_l|x)/π_ref(y_l|x))))
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| beta | 0.1 | Higher = more conservative, closer to reference |
| learning_rate | 1e-5 to 1e-6 | Lower than SFT |
| lora_rank | 32 | Lower rank often sufficient |

### When Beta Matters

- **beta=0.05**: Aggressive learning, may overfit to preferences
- **beta=0.1**: Standard, good balance
- **beta=0.5**: Conservative, preserves reference behavior
- **beta=1.0+**: Very conservative, minimal change from reference

### Data Format

```json
{"prompt": "Explain quantum computing", "chosen": "...", "rejected": "..."}
```

Generating preference pairs:
1. Sample N responses per prompt
2. Score with reward model or human annotation
3. Pair highest (chosen) with lowest (rejected)

---

## GRPO (Group Relative Policy Optimization)

### Core Innovation

From DeepSeek:
> Eliminates the critic/value model by using group mean as baseline.

Traditional RL requires:
- Policy model (trainable)
- Value model (for advantage estimation)
- Reward model

GRPO only requires:
- Policy model
- Reward function

### Algorithm

```python
def grpo_step(policy, prompts, reward_fn, group_size=8):
    for prompt in prompts:
        # 1. Generate K responses
        responses = [policy.sample(prompt) for _ in range(group_size)]

        # 2. Score each
        rewards = [reward_fn(prompt, r) for r in responses]

        # 3. Compute group-relative advantage
        baseline = mean(rewards)
        advantages = [(r - baseline) / (std(rewards) + 1e-8) for r in rewards]

        # 4. Policy gradient update
        loss = policy_gradient_loss(responses, advantages)
        loss.backward()
        policy.step()
```

### Memory Efficiency

| Configuration | GPU Memory (8B model) |
|--------------|----------------------|
| PPO (policy + value) | ~48 GB |
| GRPO (policy only) | ~24 GB |
| GRPO + LoRA | ~16 GB |

### LoRA for RL

Key finding from Thinking Machines:
> "LoRA fully matches FullFT for policy gradient algorithms, even with rank as low as 1"

Why? RL absorbs ~1000x less information per token than SFT.

| Model | SFT Rank | RL Rank |
|-------|----------|---------|
| 8B | 64 | 8 |
| 70B | 256 | 32 |

### Reward Functions

Built-in:
```python
# Math - checks numerical answer
def math_reward(prompt, response):
    target = extract_target(prompt)
    answer = extract_answer(response)
    return 1.0 if target == answer else 0.0

# Code - runs test cases
def code_reward(prompt, response):
    code = extract_code(response)
    tests = extract_tests(prompt)
    return 1.0 if run_tests(code, tests) else 0.0
```

---

## PPO (Proximal Policy Optimization)

### When to Use

- Complex alignment requiring nuanced preferences
- Production systems needing fine-grained control
- When you have budget for reward model training

### Components

1. **Reward Model**: Trained on human preferences
2. **Value Model**: Estimates expected return (for advantage)
3. **Policy Model**: The LLM being trained
4. **Reference Model**: Frozen copy for KL penalty

### PPO Objective

```
L_PPO = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A) - β*KL(π||π_ref)]

Where:
- r(θ) = π(a|s) / π_old(a|s)  # probability ratio
- A = advantage estimate
- ε = clip ratio (typically 0.2)
- β = KL penalty coefficient
```

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| clip_ratio | 0.2 | PPO clipping |
| kl_coef | 0.1 | KL penalty |
| value_coef | 0.5 | Value loss weight |
| entropy_coef | 0.01 | Exploration bonus |
| gae_lambda | 0.95 | GAE parameter |

---

## RLAIF (RL from AI Feedback)

### Core Idea

From Anthropic's Constitutional AI:
> Use a strong AI model to generate preference labels instead of humans.

### Pipeline

1. **Generate responses**: Sample multiple from policy
2. **AI labeling**: Use Claude/GPT-4 to pick preferred
3. **Train reward model** (or use direct RLAIF)
4. **RL optimization**: Standard PPO/DPO

### Direct RLAIF

Skip reward model training:
```python
def direct_rlaif_reward(prompt, response, judge_model):
    """Get reward directly from AI judge."""
    score = judge_model.evaluate(prompt, response)
    return score
```

### Constitutional AI Pattern

```python
def constitutional_critique(response, principles):
    """Self-critique and revision loop."""
    critique = model.critique(response, principles)
    revised = model.revise(response, critique)
    return revised

# Example principles
principles = [
    "Is this response helpful and harmless?",
    "Does this encourage violence?",
    "Is this factually accurate?"
]
```

---

## Reward Hacking

### What It Is

Model finds ways to maximize reward without achieving the intended goal.

### Examples

- Length hacking: Longer responses get higher reward
- Format hacking: Adding "I hope this helps!" boosts score
- Sycophancy: Agreeing with user regardless of truth

### Mitigations

1. **KL penalty**: Prevent large divergence from reference
2. **Diverse reward signals**: Multiple criteria
3. **Human oversight**: Periodic quality checks
4. **Adversarial evaluation**: Test on edge cases

---

## Training Tips

### General

1. **Start with DPO** - Simplest, works for most cases
2. **Use lower LR for RL** - 1e-6 instead of 1e-5
3. **Monitor reward vs KL** - Balance exploration and stability
4. **Checkpoint frequently** - RL can be unstable

### Debugging

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Reward not improving | LR too low, bad reward fn | Increase LR, check reward |
| Reward spikes then crashes | LR too high | Reduce LR |
| KL exploding | Weak KL penalty | Increase beta/kl_coef |
| High reward, bad outputs | Reward hacking | Review reward function |

---

## Research References

1. **DPO** (Rafailov et al., 2023)
   - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

2. **GRPO** (DeepSeek, 2024)
   - Used for DeepSeek-Math and DeepSeek-R1

3. **PPO** (Schulman et al., 2017)
   - Original PPO paper, adapted for LLMs by OpenAI

4. **Constitutional AI** (Anthropic, 2022)
   - RLAIF and self-improvement patterns

5. **REINFORCE++** (2024)
   - Simpler, more stable than GRPO in some cases
