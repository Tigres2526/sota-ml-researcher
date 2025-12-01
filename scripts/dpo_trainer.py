"""
Direct Preference Optimization (DPO) Trainer for Tinker Platform

DPO learns directly from preference pairs without a separate reward model.
Key insight: There's a closed-form mapping between reward functions and optimal policies.

Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

Usage:
    trainer = DPOTrainer(config)
    trainer.train(preference_data)
"""

import json
import yaml
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

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
class DPOConfig:
    """Configuration for DPO training."""

    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    run_name: str = "dpo-experiment"

    # Reference model (frozen)
    ref_model: Optional[str] = None  # Uses base_model if None

    # Data (preference format: prompt, chosen, rejected)
    train_file: str = "data/preferences.jsonl"
    val_file: Optional[str] = None

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient
    # Higher beta = more conservative, closer to reference
    # Lower beta = more aggressive preference learning

    # LoRA config (lower rank often sufficient for DPO)
    lora_rank: int = 32
    lora_alpha: int = 32
    apply_to: List[str] = field(default_factory=lambda: ["mlp", "attn"])

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_steps: int = 5000
    eval_every: int = 200
    checkpoint_every: int = 500

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, path: str) -> "DPOConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


def dpo_loss(
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    ref_chosen_logps: "torch.Tensor",
    ref_rejected_logps: "torch.Tensor",
    beta: float = 0.1
) -> "torch.Tensor":
    """
    Compute DPO loss.

    The DPO loss directly optimizes the policy to prefer chosen over rejected,
    without needing a separate reward model.

    Math:
        L_DPO = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) -
                                      log(pi(y_l|x)/pi_ref(y_l|x)))))

    Where:
        y_w = chosen (winner) response
        y_l = rejected (loser) response
        pi = policy model
        pi_ref = reference model (frozen)
        beta = KL penalty coefficient

    Args:
        policy_chosen_logps: Log probs of chosen responses under policy
        policy_rejected_logps: Log probs of rejected responses under policy
        ref_chosen_logps: Log probs of chosen responses under reference
        ref_rejected_logps: Log probs of rejected responses under reference
        beta: Temperature/KL penalty (default 0.1)

    Returns:
        Scalar loss tensor
    """
    # Compute log ratios (implicit rewards)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss is negative log sigmoid of reward difference
    # This pushes chosen > rejected in the implicit reward space
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss


def compute_log_probs(
    client: Any,  # TrainingClient or SamplingClient
    prompts: List[str],
    responses: List[str]
) -> List[float]:
    """
    Compute log probabilities of responses given prompts.

    This is the key operation for DPO - we need P(response|prompt) from both
    the policy and reference models.
    """
    log_probs = []

    for prompt, response in zip(prompts, responses):
        # Combine prompt + response for full sequence
        full_text = prompt + response

        # Get token-level log probs
        token_logps = client.get_logprobs(full_text)

        # Sum log probs for response tokens only
        # (skip prompt tokens in the sum)
        prompt_len = len(client.tokenize(prompt))
        response_logps = token_logps[prompt_len:]
        total_logp = sum(response_logps)

        log_probs.append(total_logp)

    return log_probs


class DPOTrainer:
    """
    DPO Trainer implementing preference optimization without reward model.

    Key advantages over RLHF/PPO:
    - No separate reward model training
    - No value model needed
    - More stable training
    - Simpler implementation
    """

    def __init__(self, config: DPOConfig):
        self.config = config
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

            # Reference model (frozen) - same architecture, no training
            ref_model = config.ref_model or config.base_model
            self.ref = SamplingClient(ref_model)

            logger.info(f"Initialized DPO trainer")
            logger.info(f"  Policy: {config.base_model} (LoRA rank={config.lora_rank})")
            logger.info(f"  Reference: {ref_model} (frozen)")
            logger.info(f"  Beta (KL penalty): {config.beta}")
        else:
            logger.warning("Tinker not available. Running in reference mode.")

    def load_preferences(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load preference data.

        Expected format (JSONL):
        {"prompt": "...", "chosen": "...", "rejected": "..."}
        """
        data = []
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                if "prompt" in item and "chosen" in item and "rejected" in item:
                    data.append(item)
                else:
                    logger.warning(f"Skipping malformed item: {item.keys()}")
        return data

    def compute_dpo_loss(self, batch: List[Dict[str, str]]) -> Tuple[float, Dict]:
        """
        Compute DPO loss for a batch.

        Returns:
            (loss_value, metrics_dict)
        """
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]

        # Get log probs from policy (trainable)
        policy_chosen_logps = compute_log_probs(self.policy, prompts, chosen)
        policy_rejected_logps = compute_log_probs(self.policy, prompts, rejected)

        # Get log probs from reference (frozen)
        ref_chosen_logps = compute_log_probs(self.ref, prompts, chosen)
        ref_rejected_logps = compute_log_probs(self.ref, prompts, rejected)

        # Convert to tensors
        if TORCH_AVAILABLE:
            policy_chosen = torch.tensor(policy_chosen_logps)
            policy_rejected = torch.tensor(policy_rejected_logps)
            ref_chosen = torch.tensor(ref_chosen_logps)
            ref_rejected = torch.tensor(ref_rejected_logps)

            # Compute DPO loss
            loss = dpo_loss(
                policy_chosen, policy_rejected,
                ref_chosen, ref_rejected,
                beta=self.config.beta
            )

            # Compute metrics
            with torch.no_grad():
                chosen_rewards = self.config.beta * (policy_chosen - ref_chosen)
                rejected_rewards = self.config.beta * (policy_rejected - ref_rejected)
                reward_margin = (chosen_rewards - rejected_rewards).mean().item()
                accuracy = ((chosen_rewards > rejected_rewards).float().mean().item())

            metrics = {
                "loss": loss.item(),
                "reward_margin": reward_margin,
                "accuracy": accuracy,  # How often chosen > rejected in reward
                "chosen_reward_mean": chosen_rewards.mean().item(),
                "rejected_reward_mean": rejected_rewards.mean().item()
            }

            return loss, metrics
        else:
            # Numpy fallback for reference
            import numpy as np

            policy_chosen = np.array(policy_chosen_logps)
            policy_rejected = np.array(policy_rejected_logps)
            ref_chosen = np.array(ref_chosen_logps)
            ref_rejected = np.array(ref_rejected_logps)

            chosen_rewards = self.config.beta * (policy_chosen - ref_chosen)
            rejected_rewards = self.config.beta * (policy_rejected - ref_rejected)

            # Log sigmoid loss
            margin = chosen_rewards - rejected_rewards
            loss = -np.mean(np.log(1 / (1 + np.exp(-margin))))

            metrics = {
                "loss": float(loss),
                "reward_margin": float(np.mean(margin)),
                "accuracy": float(np.mean(margin > 0))
            }

            return loss, metrics

    def train(self, train_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute DPO training loop.
        """
        if not TINKER_AVAILABLE:
            raise RuntimeError("Tinker not available for training")

        if train_data is None:
            train_data = self.load_preferences(self.config.train_file)

        logger.info(f"Training DPO on {len(train_data)} preference pairs")
        logger.info(f"Beta={self.config.beta}, LR={self.config.learning_rate}")

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        metrics_history = {
            "loss": [],
            "reward_margin": [],
            "accuracy": [],
            "steps": []
        }

        import random

        for step in range(self.config.num_steps):
            self.step = step

            # Sample batch
            batch_indices = random.sample(
                range(len(train_data)),
                min(self.config.batch_size, len(train_data))
            )
            batch = [train_data[i] for i in batch_indices]

            # Compute loss and backprop
            loss, metrics = self.compute_dpo_loss(batch)

            if TORCH_AVAILABLE and hasattr(loss, 'backward'):
                loss.backward()
                self.policy.optim_step()

            # Log metrics
            for key, value in metrics.items():
                if key in metrics_history:
                    metrics_history[key].append(value)
            metrics_history["steps"].append(step)

            if step % 50 == 0:
                logger.info(
                    f"Step {step}: loss={metrics['loss']:.4f}, "
                    f"margin={metrics['reward_margin']:.3f}, "
                    f"acc={metrics['accuracy']:.3f}"
                )

            # Checkpointing
            if step % self.config.checkpoint_every == 0 and step > 0:
                self.save_checkpoint(f"step_{step}")

        # Final checkpoint
        self.save_checkpoint("final")

        return {
            "metrics": metrics_history,
            "final_accuracy": metrics_history["accuracy"][-1] if metrics_history["accuracy"] else 0
        }

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        if not self.policy:
            return

        path = f"{self.config.checkpoint_dir}/{name}"
        self.policy.save(path)
        logger.info(f"Saved checkpoint: {path}")


class DPODataGenerator:
    """
    Utility to generate DPO training data from a model.

    Common approach: Generate multiple responses, use reward model or
    human annotation to pick winners/losers.
    """

    def __init__(self, model: str, reward_fn=None):
        """
        Args:
            model: Model to generate responses
            reward_fn: Optional reward function to score responses
        """
        self.model = model
        self.reward_fn = reward_fn

        if TINKER_AVAILABLE:
            self.sampler = SamplingClient(model)

    def generate_pairs(
        self,
        prompts: List[str],
        samples_per_prompt: int = 4,
        temperature: float = 0.8
    ) -> List[Dict[str, str]]:
        """
        Generate preference pairs via rejection sampling.

        For each prompt:
        1. Generate N responses
        2. Score with reward function
        3. Pair highest scoring (chosen) with lower scoring (rejected)
        """
        pairs = []

        for prompt in prompts:
            # Generate multiple responses
            responses = [
                self.sampler.sample(prompt, temperature=temperature)
                for _ in range(samples_per_prompt)
            ]

            # Score responses
            if self.reward_fn:
                scores = [self.reward_fn(prompt, r) for r in responses]
            else:
                # Fallback: use response length as proxy (not recommended)
                scores = [len(r) for r in responses]

            # Sort by score
            scored = sorted(zip(responses, scores), key=lambda x: x[1], reverse=True)

            # Create pairs (best vs worst, second best vs second worst, etc.)
            n = len(scored)
            for i in range(n // 2):
                pairs.append({
                    "prompt": prompt,
                    "chosen": scored[i][0],
                    "rejected": scored[n - 1 - i][0]
                })

        return pairs


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="DPO Trainer")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = DPOConfig.from_yaml(args.config)
    trainer = DPOTrainer(config)

    print(f"\nDPO Training Configuration:")
    print(f"  Model: {config.base_model}")
    print(f"  Beta (KL penalty): {config.beta}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Learning rate: {config.learning_rate}")

    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Final accuracy: {results['final_accuracy']:.3f}")


if __name__ == "__main__":
    main()
