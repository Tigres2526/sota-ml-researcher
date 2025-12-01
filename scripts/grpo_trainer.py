"""
Group Relative Policy Optimization (GRPO) Trainer

GRPO is a memory-efficient RL algorithm developed by DeepSeek.
Key innovation: Eliminates the value/critic model by using group-relative advantages.

Used to train DeepSeek-Math and DeepSeek-R1 models.

Best for:
- Math reasoning (verifiable rewards)
- Code generation (test case rewards)
- Any task with objective success criteria

Usage:
    trainer = GRPOTrainer(config, reward_fn=math_reward)
    trainer.train(prompts)
"""

import json
import yaml
import logging
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tinker import TrainingClient, SamplingClient
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    run_name: str = "grpo-experiment"

    # GRPO-specific parameters
    group_size: int = 8  # K samples per prompt
    temperature: float = 0.7  # Sampling temperature

    # Reward configuration
    reward_type: str = "verifiable"  # verifiable, model, or custom
    reward_script: Optional[str] = None  # Path to custom reward script

    # LoRA config
    # Key finding: LoRA with rank as low as 1 works for RL!
    # RL absorbs ~1000x less info per token than SFT
    lora_rank: int = 8
    lora_alpha: int = 16
    apply_to: List[str] = field(default_factory=lambda: ["mlp", "attn"])

    # Training
    batch_size: int = 4  # Number of prompts per batch
    learning_rate: float = 1e-6
    num_steps: int = 2000
    max_response_tokens: int = 2048

    # PPO-like parameters (optional, for stability)
    clip_ratio: float = 0.2  # PPO clipping
    entropy_coef: float = 0.01  # Entropy bonus
    kl_coef: float = 0.1  # KL penalty to reference

    # Paths
    prompt_file: str = "data/prompts.jsonl"
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


def grpo_advantage(
    rewards: "torch.Tensor",
    normalize: bool = True
) -> "torch.Tensor":
    """
    Compute GRPO advantages using group-relative baseline.

    Key insight from DeepSeek:
    Instead of training a separate value model to estimate baselines,
    GRPO uses the mean reward within each group as the baseline.

    This eliminates:
    - Value model training
    - Value model memory overhead
    - Complexity of advantage estimation (GAE, etc.)

    Args:
        rewards: Shape [batch_size, group_size] - rewards for each sample
        normalize: Whether to normalize advantages

    Returns:
        advantages: Shape [batch_size, group_size]
    """
    # Group mean as baseline
    baseline = rewards.mean(dim=1, keepdim=True)

    # Advantage = reward - baseline
    advantages = rewards - baseline

    if normalize:
        # Normalize across the batch for stability
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std

    return advantages


def policy_gradient_loss(
    log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    old_log_probs: Optional["torch.Tensor"] = None,
    clip_ratio: float = 0.2
) -> "torch.Tensor":
    """
    Compute policy gradient loss, optionally with PPO clipping.

    Args:
        log_probs: Log probs of actions under current policy
        advantages: Advantage estimates
        old_log_probs: Log probs under old policy (for PPO clipping)
        clip_ratio: PPO clipping parameter

    Returns:
        Scalar loss
    """
    if old_log_probs is not None:
        # PPO-style clipping
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        # Take minimum of clipped and unclipped
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        loss = -torch.min(surrogate1, surrogate2).mean()
    else:
        # Standard REINFORCE
        loss = -(log_probs * advantages).mean()

    return loss


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def math_reward(prompt: str, response: str) -> float:
    """
    Reward function for math problems.

    Checks if the response contains a correct numerical answer.
    Works with GSM8K, MATH, and similar datasets.
    """
    import re

    # Try to extract the target answer from prompt (if provided)
    # Common format: "The answer is X" or "#### X"
    target_match = re.search(r'(?:answer is|####)\s*([-\d.,]+)', prompt, re.IGNORECASE)
    if not target_match:
        # No target in prompt, return 0 (need target for verification)
        return 0.0

    target = target_match.group(1).replace(',', '').strip()

    # Extract answer from response
    # Look for boxed answers: \boxed{X}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        answer = boxed_match.group(1).strip()
    else:
        # Look for "The answer is X" pattern
        answer_match = re.search(r'(?:answer is|=)\s*([-\d.,]+)', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).replace(',', '').strip()
        else:
            # Try to find the last number in the response
            numbers = re.findall(r'[-\d.,]+', response)
            answer = numbers[-1].replace(',', '') if numbers else ""

    # Compare
    try:
        target_val = float(target)
        answer_val = float(answer)
        return 1.0 if abs(target_val - answer_val) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if target.lower() == answer.lower() else 0.0


def code_reward(prompt: str, response: str) -> float:
    """
    Reward function for code generation.

    Extracts code from response and runs test cases.
    """
    import re
    import subprocess
    import tempfile

    # Extract code block
    code_match = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    if not code_match:
        return 0.0

    code = code_match.group(1)

    # Extract test cases from prompt (if provided)
    test_match = re.search(r'Test cases?:(.*?)(?:$|Example)', prompt, re.DOTALL | re.IGNORECASE)
    if not test_match:
        # No tests provided, can't verify
        return 0.0

    tests = test_match.group(1).strip()

    # Write code + tests to temp file and run
    full_code = f"{code}\n\n# Tests\n{tests}"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            timeout=10
        )

        # Success if no errors
        return 1.0 if result.returncode == 0 else 0.0

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0


def format_reward(prompt: str, response: str) -> float:
    """
    Reward function for format compliance.

    Checks if response follows expected format (e.g., JSON, specific structure).
    """
    import json as json_module

    # Check for JSON format
    if "json" in prompt.lower():
        try:
            json_module.loads(response)
            return 1.0
        except json_module.JSONDecodeError:
            return 0.0

    # Check for specific markers
    required_markers = []
    if "step by step" in prompt.lower():
        required_markers.append(r'step \d')
    if "provide sources" in prompt.lower():
        required_markers.append(r'source|reference|citation')

    if required_markers:
        import re
        found = sum(1 for m in required_markers if re.search(m, response, re.IGNORECASE))
        return found / len(required_markers)

    return 0.5  # Default partial reward


class GRPOTrainer:
    """
    GRPO Trainer implementing DeepSeek's memory-efficient RL algorithm.

    Key innovations:
    1. No value model - uses group mean as baseline
    2. Generates K responses per prompt, computes relative advantages
    3. Memory efficient - only one model in memory
    4. Works with very low LoRA ranks (even rank-1)
    """

    def __init__(
        self,
        config: GRPOConfig,
        reward_fn: Optional[Callable[[str, str], float]] = None
    ):
        self.config = config
        self.reward_fn = reward_fn or self._get_default_reward_fn()
        self.policy = None
        self.ref = None
        self.step = 0

        if TINKER_AVAILABLE:
            # Policy model (trainable)
            self.policy = TrainingClient(
                config.base_model,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                learning_rate=config.learning_rate
            )

            # Reference model for KL penalty (optional)
            if config.kl_coef > 0:
                self.ref = SamplingClient(config.base_model)

            logger.info(f"Initialized GRPO trainer")
            logger.info(f"  Policy: {config.base_model} (LoRA rank={config.lora_rank})")
            logger.info(f"  Group size: {config.group_size}")
            logger.info(f"  Temperature: {config.temperature}")

    def _get_default_reward_fn(self) -> Callable:
        """Get reward function based on config."""
        reward_type = self.config.reward_type.lower()

        if reward_type == "math":
            return math_reward
        elif reward_type == "code":
            return code_reward
        elif reward_type == "format":
            return format_reward
        elif reward_type == "custom" and self.config.reward_script:
            # Load custom reward function
            import importlib.util
            spec = importlib.util.spec_from_file_location("reward", self.config.reward_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.reward
        else:
            # Default: simple length-based (not recommended for production)
            logger.warning("Using length-based reward. Provide proper reward_fn for real training.")
            return lambda p, r: min(len(r) / 500, 1.0)

    def load_prompts(self, file_path: str) -> List[str]:
        """Load prompts from JSONL file."""
        prompts = []
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("input") or item.get("question")
                if prompt:
                    prompts.append(prompt)
        return prompts

    def generate_group(self, prompt: str) -> List[str]:
        """Generate K responses for a prompt."""
        if not TINKER_AVAILABLE or not self.policy:
            raise RuntimeError("Tinker not available")

        responses = []
        for _ in range(self.config.group_size):
            response = self.policy.sample(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_response_tokens
            )
            responses.append(response)

        return responses

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Execute one GRPO training step.

        For each prompt:
        1. Generate K responses
        2. Score with reward function
        3. Compute group-relative advantages
        4. Update policy with policy gradient
        """
        all_log_probs = []
        all_advantages = []
        all_rewards = []

        for prompt in prompts:
            # Generate group of responses
            responses = self.generate_group(prompt)

            # Score each response
            rewards = [self.reward_fn(prompt, r) for r in responses]
            all_rewards.extend(rewards)

            # Get log probs for each response
            log_probs = []
            for response in responses:
                lp = self.policy.get_logprobs(prompt + response)
                # Sum log probs for response tokens
                prompt_len = len(self.policy.tokenize(prompt))
                response_lp = sum(lp[prompt_len:])
                log_probs.append(response_lp)

            all_log_probs.extend(log_probs)

        # Convert to tensors
        if TORCH_AVAILABLE:
            rewards_tensor = torch.tensor(all_rewards).view(len(prompts), self.config.group_size)
            log_probs_tensor = torch.tensor(all_log_probs)

            # Compute GRPO advantages
            advantages = grpo_advantage(rewards_tensor, normalize=True)
            advantages_flat = advantages.view(-1)

            # Policy gradient loss
            loss = policy_gradient_loss(
                log_probs_tensor,
                advantages_flat,
                clip_ratio=self.config.clip_ratio
            )

            # Add entropy bonus for exploration
            if self.config.entropy_coef > 0:
                entropy = -log_probs_tensor.mean()
                loss = loss - self.config.entropy_coef * entropy

            # Backward and update
            loss.backward()
            self.policy.optim_step()

            return {
                "loss": loss.item(),
                "mean_reward": sum(all_rewards) / len(all_rewards),
                "max_reward": max(all_rewards),
                "reward_std": float(rewards_tensor.std()),
                "advantage_std": float(advantages.std())
            }
        else:
            # NumPy fallback
            import numpy as np
            rewards_array = np.array(all_rewards).reshape(len(prompts), self.config.group_size)
            advantages = rewards_array - rewards_array.mean(axis=1, keepdims=True)

            return {
                "mean_reward": float(np.mean(all_rewards)),
                "max_reward": float(np.max(all_rewards)),
                "reward_std": float(np.std(all_rewards))
            }

    def train(self, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute GRPO training loop.
        """
        if prompts is None:
            prompts = self.load_prompts(self.config.prompt_file)

        logger.info(f"GRPO training on {len(prompts)} prompts")
        logger.info(f"  Group size: {self.config.group_size}")
        logger.info(f"  Total samples per epoch: {len(prompts) * self.config.group_size}")

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        metrics_history = {
            "loss": [],
            "mean_reward": [],
            "max_reward": [],
            "steps": []
        }

        for step in range(self.config.num_steps):
            self.step = step

            # Sample batch of prompts
            batch_prompts = random.sample(
                prompts,
                min(self.config.batch_size, len(prompts))
            )

            # Training step
            metrics = self.train_step(batch_prompts)

            # Log
            for key, value in metrics.items():
                if key in metrics_history:
                    metrics_history[key].append(value)
            metrics_history["steps"].append(step)

            if step % 20 == 0:
                logger.info(
                    f"Step {step}: reward={metrics['mean_reward']:.3f} "
                    f"(max={metrics['max_reward']:.3f}), "
                    f"loss={metrics.get('loss', 0):.4f}"
                )

            # Checkpoint
            if step % 500 == 0 and step > 0:
                self.save_checkpoint(f"step_{step}")

        self.save_checkpoint("final")

        return {
            "metrics": metrics_history,
            "final_mean_reward": metrics_history["mean_reward"][-1] if metrics_history["mean_reward"] else 0
        }

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        if not self.policy:
            return

        path = f"{self.config.checkpoint_dir}/{name}"
        self.policy.save(path)
        logger.info(f"Saved checkpoint: {path}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Trainer")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = GRPOConfig.from_yaml(args.config)

    # Select reward function based on config
    if config.reward_type == "math":
        reward_fn = math_reward
    elif config.reward_type == "code":
        reward_fn = code_reward
    else:
        reward_fn = None

    trainer = GRPOTrainer(config, reward_fn=reward_fn)

    print(f"\nGRPO Training Configuration:")
    print(f"  Model: {config.base_model}")
    print(f"  LoRA rank: {config.lora_rank} (low rank OK for RL)")
    print(f"  Group size: {config.group_size}")
    print(f"  Reward type: {config.reward_type}")

    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Final mean reward: {results['final_mean_reward']:.3f}")


if __name__ == "__main__":
    main()
