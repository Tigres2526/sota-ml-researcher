# SOTA ML Researcher - Claude Code Plugin

Expert ML/RL/SL researcher plugin that thinks like a real scientist: evidence-based, focused, rigorous evals without overthinking.

## Installation

```bash
# Clone the repo
git clone https://github.com/Tigres2526/sota-ml-researcher.git

# Install dependencies
pip install tinker tinker-cookbook anthropic torch pyyaml
```

## What This Does

Transforms Claude into an expert ML researcher for LLM post-training:

- **SFT** with LoRA capacity validation (10x LR rule, all layers)
- **DPO** preference optimization (no reward model needed)
- **GRPO** memory-efficient RL (DeepSeek's approach)
- **LLM-as-Judge** evaluation with position debiasing

## Research Embedded

Key findings from Thinking Machines "LoRA Without Regret":

```
✓ Learning rate: 10x higher than FullFT for LoRA
✓ Which layers: ALL layers (MLP provides most benefit)
✓ Capacity: LoRA_params × 2 >= dataset_tokens
✓ RL rank: Even rank-1 works for policy gradient
```

## Quick Start

### 1. Configure

```yaml
# templates/sft_config.yaml
base_model: "meta-llama/Llama-3.1-8B-Instruct"
train_file: "data/train.jsonl"
lora_rank: 64
learning_rate: null  # Auto-derived (10x FullFT)
```

### 2. Validate Capacity

```bash
python scripts/sft_trainer.py --config templates/sft_config.yaml --validate-only
```

### 3. Train

```bash
python scripts/sft_trainer.py --config templates/sft_config.yaml
```

### 4. Evaluate

```bash
python scripts/llm_judge.py --config templates/eval_config.yaml
```

## Commands (when used as Claude Code plugin)

| Command | Description |
|---------|-------------|
| `/run_sft` | Supervised fine-tuning |
| `/run_dpo` | DPO preference optimization |
| `/run_grpo` | GRPO reinforcement learning |
| `/run_eval` | LLM-as-judge evaluation |
| `/analyze_run` | Post-training analysis |

## Training Methods

### SFT (Supervised Fine-Tuning)
```python
from scripts.sft_trainer import SFTTrainer, SFTConfig

config = SFTConfig.from_yaml("config.yaml")
trainer = SFTTrainer(config)
trainer.validate_capacity()  # Check before training!
trainer.train()
```

### DPO (Direct Preference Optimization)
```python
from scripts.dpo_trainer import DPOTrainer, DPOConfig

config = DPOConfig(beta=0.1, lora_rank=32)
trainer = DPOTrainer(config)
trainer.train(preference_data)
```

### GRPO (Group Relative Policy Optimization)
```python
from scripts.grpo_trainer import GRPOTrainer, GRPOConfig, math_reward

config = GRPOConfig(group_size=8, lora_rank=8)
trainer = GRPOTrainer(config, reward_fn=math_reward)
trainer.train(prompts)
```

## Evaluation

Position-debiased LLM-as-judge:

```python
from scripts.llm_judge import LLMJudge

judge = LLMJudge(model="claude-sonnet-4-20250514")
result = await judge.pairwise_judge(prompt, response_a, response_b, debias=True)
# Runs both orderings to eliminate position bias
```

## Reference Documentation

- `reference/lora_doctrine.md` - LoRA/QLoRA best practices
- `reference/rl_doctrine.md` - DPO/GRPO/PPO guide
- `reference/eval_doctrine.md` - Evaluation methodology
- `reference/hyperparams.md` - Quick lookup tables

## Requirements

- Python 3.10+
- tinker, tinker-cookbook (Thinking Machines)
- anthropic (for LLM-as-judge)
- torch >= 2.0
- pyyaml

## License

MIT

## Credits

Research synthesized from:
- [Thinking Machines - LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [DeepSeek - GRPO](https://github.com/deepseek-ai)
- [Anthropic - Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [UK AISI - Inspect](https://github.com/UKGovernmentBEIS/inspect_ai)
