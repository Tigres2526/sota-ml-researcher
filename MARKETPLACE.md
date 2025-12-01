# SOTA ML Researcher

**Expert ML/RL/SL researcher plugin for Claude Code** - combines deep SOTA research knowledge with practical Tinker-based implementation for fine-tuning LLMs.

## Overview

This plugin transforms Claude into an expert ML researcher specializing in:
- **LoRA/QLoRA fine-tuning** with research-backed defaults
- **DPO, GRPO, PPO** preference and reinforcement learning
- **LLM-as-judge evaluation** with position debiasing
- **Tinker platform** integration

Think like a real researcher: evidence-based decisions, capacity validation before training, rigorous evals without overthinking.

## Features

### Training Methods
- **SFT (Supervised Fine-Tuning)** - LoRA with automatic capacity validation
- **DPO (Direct Preference Optimization)** - Learn from preference pairs without reward model
- **GRPO (Group Relative Policy Optimization)** - Memory-efficient RL for math/code tasks

### Evaluation
- **LLM-as-Judge** with Claude as evaluator
- **Position debiasing** - runs both orderings to eliminate bias
- **Structured rubrics** with score anchors
- **Win rate metrics** with confidence intervals

### Research Doctrine
Embedded knowledge from:
- Thinking Machines "LoRA Without Regret" (2025)
- DeepSeek GRPO methodology
- Anthropic Constitutional AI / RLAIF
- UK AISI Inspect evaluation patterns

## Key Research Insights

| Finding | Value | Source |
|---------|-------|--------|
| LoRA learning rate | 10x higher than FullFT | Thinking Machines |
| Which layers | ALL layers (MLP + attn) | Thinking Machines |
| RL LoRA rank | Even rank-1 works | Thinking Machines |
| Capacity rule | params×2 >= tokens | Thinking Machines |

## Commands

| Command | Description |
|---------|-------------|
| `/run_sft [config]` | Run supervised fine-tuning |
| `/run_dpo [config]` | Run DPO preference optimization |
| `/run_grpo [config]` | Run GRPO reinforcement learning |
| `/run_eval [config]` | Run LLM-as-judge evaluation |
| `/analyze_run [dir]` | Analyze training results |

## Quick Start

1. Copy config template from `templates/`
2. Edit with your data paths and model
3. Run capacity validation: `/run_sft config.yaml --validate-only`
4. Execute training: `/run_sft config.yaml`
5. Evaluate: `/run_eval eval_config.yaml`

## Requirements

- Python 3.10+
- Tinker SDK (`pip install tinker tinker-cookbook`)
- Anthropic SDK for evaluation (`pip install anthropic`)
- PyTorch 2.0+

## Structure

```
sota-ml-researcher/
├── skills/researcher.md      # Core research persona
├── commands/                 # Slash commands
├── scripts/                  # Python implementations
├── reference/                # Doctrine documentation
└── templates/                # YAML config templates
```

## License

MIT

## Author

Built with research from Thinking Machines, Anthropic, DeepSeek, and UK AISI.
