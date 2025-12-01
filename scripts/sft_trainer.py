"""
Supervised Fine-Tuning Trainer for Tinker Platform

Implements research-backed defaults from Thinking Machines "LoRA Without Regret":
- 10x learning rate vs FullFT
- LoRA on ALL layers (MLP + attention + unembed)
- Capacity validation before training
- Automatic hyperparameter derivation

Usage:
    trainer = SFTTrainer(config)
    trainer.validate_capacity()
    trainer.train(train_data, val_data)
"""

import json
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import asyncio

# Tinker imports (install via: pip install tinker tinker-cookbook)
try:
    from tinker import TrainingClient, SamplingClient
    from tinker_cookbook import hyperparam_utils
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    logging.warning("Tinker not installed. Running in reference mode.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training with research-backed defaults."""

    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    run_name: str = "sft-experiment"

    # Data
    train_file: str = "data/train.jsonl"
    val_file: str = "data/val.jsonl"

    # LoRA config (all layers per Thinking Machines research)
    lora_rank: int = 64
    lora_alpha: int = 32
    apply_to: List[str] = field(default_factory=lambda: ["mlp", "attn", "unembed"])

    # Training
    max_seq_len: int = 4096
    batch_size: int = 16
    learning_rate: Optional[float] = None  # Auto-derived if None
    weight_decay: float = 0.0
    num_steps: int = 10000
    num_epochs: Optional[int] = None  # Alternative to num_steps
    eval_every: int = 500
    checkpoint_every: int = 1000

    # Model architecture (for capacity calculation)
    hidden_dim: int = 4096
    num_layers: int = 32

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


def estimate_lora_capacity(
    rank: int,
    hidden_dim: int,
    num_layers: int,
    apply_mlp: bool = True,
    apply_attn: bool = True,
    apply_unembed: bool = True
) -> tuple[int, int]:
    """
    Calculate LoRA parameter count and information capacity.

    Returns:
        (total_params, capacity_bits)

    Based on Thinking Machines research:
    - Neural networks can store ~2 bits per parameter (long training limit)
    - LLM datasets have loss ~1 bit per token
    """
    params_per_layer = 0

    if apply_mlp:
        # up_proj, down_proj, gate_proj (if SwiGLU)
        # Each has rank*hidden_dim*2 params (A and B matrices)
        mlp_intermediate = hidden_dim * 4  # typical ratio
        params_per_layer += rank * hidden_dim * 2  # up_proj
        params_per_layer += rank * mlp_intermediate * 2  # down_proj
        params_per_layer += rank * hidden_dim * 2  # gate_proj

    if apply_attn:
        # q, k, v, o projections
        params_per_layer += 4 * rank * hidden_dim * 2

    total_params = params_per_layer * num_layers

    if apply_unembed:
        vocab_size = 128000  # typical for modern LLMs
        total_params += rank * vocab_size * 2

    capacity_bits = total_params * 2  # ~2 bits per parameter

    return total_params, capacity_bits


def validate_capacity(
    rank: int,
    hidden_dim: int,
    num_layers: int,
    dataset_tokens: int,
    apply_mlp: bool = True,
    apply_attn: bool = True
) -> Dict[str, Any]:
    """
    Check if LoRA has sufficient capacity for dataset.

    Args:
        rank: LoRA rank
        hidden_dim: Model hidden dimension
        num_layers: Number of transformer layers
        dataset_tokens: Total tokens in training dataset

    Returns:
        dict with status and recommendation
    """
    params, capacity = estimate_lora_capacity(
        rank, hidden_dim, num_layers, apply_mlp, apply_attn
    )

    # Assume ~1 bit per token information content
    required_bits = dataset_tokens

    if capacity < required_bits:
        deficit = required_bits - capacity
        suggested_rank = int(rank * (required_bits / capacity)) + 1
        return {
            "status": "UNDERCAPACITY",
            "params": params,
            "capacity_bits": capacity,
            "required_bits": required_bits,
            "deficit_bits": deficit,
            "suggested_rank": suggested_rank,
            "message": f"Insufficient capacity. Increase rank from {rank} to {suggested_rank}+"
        }

    headroom = (capacity / required_bits - 1) * 100
    return {
        "status": "OK",
        "params": params,
        "capacity_bits": capacity,
        "required_bits": required_bits,
        "headroom_percent": headroom,
        "message": f"Capacity OK with {headroom:.1f}% headroom"
    }


def get_lora_learning_rate(
    base_model: str,
    lora_rank: int,
    hidden_size: int = 4096
) -> float:
    """
    Calculate optimal LoRA learning rate.

    Key finding from Thinking Machines:
    - Optimal LoRA LR is consistently 10x the FullFT LR
    - This ratio holds for both SFT and RL

    Formula: LR = M_LoRA * (2000/hidden_size)^power
    Where M_LoRA ≈ 9.8x the FullFT multiplier
    """
    if TINKER_AVAILABLE:
        try:
            return hyperparam_utils.get_lr(base_model, lora_rank, method="lora")
        except Exception:
            pass

    # Fallback calculation
    # Base FullFT LR scales with hidden size
    fullft_lr = (2000 / hidden_size) * 1e-4
    # LoRA uses 10x
    lora_lr = fullft_lr * 10

    return lora_lr


def count_dataset_tokens(file_path: str, max_seq_len: int = 4096) -> int:
    """
    Estimate total tokens in a JSONL dataset.

    Assumes format: {"input": "...", "output": "..."}
    or {"prompt": "...", "completion": "..."}
    """
    total_chars = 0

    with open(file_path) as f:
        for line in f:
            item = json.loads(line)
            # Try different field names
            text = item.get("input", "") + item.get("output", "")
            if not text:
                text = item.get("prompt", "") + item.get("completion", "")
            if not text:
                text = item.get("text", "")
            total_chars += len(text)

    # Rough estimate: 4 chars per token
    estimated_tokens = total_chars // 4

    return estimated_tokens


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer implementing research-backed practices.

    Key features:
    - Automatic capacity validation
    - 10x LR scaling for LoRA
    - All-layer LoRA application
    - Checkpointing and validation
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.client = None
        self.best_val_loss = float('inf')
        self.step = 0

        # Auto-derive learning rate if not specified
        if config.learning_rate is None:
            self.config.learning_rate = get_lora_learning_rate(
                config.base_model,
                config.lora_rank,
                config.hidden_dim
            )
            logger.info(f"Auto-derived LR: {self.config.learning_rate:.2e} (10x FullFT rule)")

        # Initialize Tinker client if available
        if TINKER_AVAILABLE:
            self.client = TrainingClient(
                config.base_model,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                learning_rate=config.learning_rate
            )
        else:
            logger.warning("Running without Tinker. Use for config validation only.")

    def validate_capacity(self, train_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate LoRA capacity against dataset size.

        CRITICAL: Run this before training to avoid undercapacity issues.
        """
        train_file = train_file or self.config.train_file
        dataset_tokens = count_dataset_tokens(train_file, self.config.max_seq_len)

        result = validate_capacity(
            rank=self.config.lora_rank,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dataset_tokens=dataset_tokens,
            apply_mlp="mlp" in self.config.apply_to,
            apply_attn="attn" in self.config.apply_to
        )

        if result["status"] == "UNDERCAPACITY":
            logger.warning(f"CAPACITY WARNING: {result['message']}")
        else:
            logger.info(f"Capacity check: {result['message']}")

        return result

    def load_data(self, file_path: str) -> List[Dict[str, str]]:
        """Load JSONL training data."""
        data = []
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                # Normalize field names
                if "input" in item and "output" in item:
                    data.append({"prompt": item["input"], "completion": item["output"]})
                elif "prompt" in item and "completion" in item:
                    data.append(item)
                elif "text" in item:
                    # Split on common delimiters
                    data.append({"prompt": "", "completion": item["text"]})
        return data

    def get_batch(self, data: List[Dict], batch_size: int) -> List[Dict]:
        """Get a random batch from data."""
        import random
        indices = random.sample(range(len(data)), min(batch_size, len(data)))
        return [data[i] for i in indices]

    def train(
        self,
        train_data: Optional[List[Dict]] = None,
        val_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Execute training loop.

        Returns:
            Training metrics and best checkpoint path
        """
        if not TINKER_AVAILABLE:
            raise RuntimeError("Tinker not available. Install with: pip install tinker")

        # Load data if not provided
        if train_data is None:
            train_data = self.load_data(self.config.train_file)
        if val_data is None and Path(self.config.val_file).exists():
            val_data = self.load_data(self.config.val_file)

        logger.info(f"Training on {len(train_data)} examples")
        if val_data:
            logger.info(f"Validation on {len(val_data)} examples")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        metrics = {
            "train_losses": [],
            "val_losses": [],
            "steps": []
        }

        for step in range(self.config.num_steps):
            self.step = step

            # Get batch and train
            batch = self.get_batch(train_data, self.config.batch_size)
            loss = self.client.forward_backward(batch)
            self.client.optim_step()

            metrics["train_losses"].append(loss)
            metrics["steps"].append(step)

            # Logging
            if step % 100 == 0:
                logger.info(f"Step {step}: loss={loss:.4f}")

            # Validation
            if val_data and step % self.config.eval_every == 0:
                val_loss = self.evaluate(val_data)
                metrics["val_losses"].append(val_loss)
                logger.info(f"Step {step}: val_loss={val_loss:.4f}")

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")

            # Periodic checkpointing
            if step % self.config.checkpoint_every == 0 and step > 0:
                self.save_checkpoint(f"step_{step}")

        # Final checkpoint
        self.save_checkpoint("final")

        return {
            "metrics": metrics,
            "best_val_loss": self.best_val_loss,
            "best_checkpoint": f"{self.config.checkpoint_dir}/best"
        }

    def evaluate(self, val_data: List[Dict]) -> float:
        """Evaluate on validation set."""
        if not TINKER_AVAILABLE or not self.client:
            raise RuntimeError("Tinker client not available")

        total_loss = 0
        num_batches = 0

        for i in range(0, len(val_data), self.config.batch_size):
            batch = val_data[i:i + self.config.batch_size]
            loss = self.client.forward(batch)  # Forward only, no backward
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if not TINKER_AVAILABLE or not self.client:
            return

        path = f"{self.config.checkpoint_dir}/{name}"
        self.client.save(path)
        logger.info(f"Saved checkpoint: {path}")

        # Also save config
        config_path = f"{path}/config.yaml"
        self.config.to_yaml(config_path)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--validate-only", action="store_true", help="Only validate capacity")
    args = parser.parse_args()

    config = SFTConfig.from_yaml(args.config)
    trainer = SFTTrainer(config)

    # Always validate capacity first
    capacity_result = trainer.validate_capacity()
    print(f"\nCapacity Check: {capacity_result['status']}")
    print(f"  LoRA params: {capacity_result['params']:,}")
    print(f"  Capacity: {capacity_result['capacity_bits']:,} bits")
    print(f"  Required: {capacity_result['required_bits']:,} bits")

    if capacity_result["status"] == "UNDERCAPACITY":
        print(f"\n  RECOMMENDATION: {capacity_result['message']}")
        if not args.validate_only:
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                return

    if args.validate_only:
        return

    # Train
    print("\nStarting training...")
    results = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Best val loss: {results['best_val_loss']:.4f}")
    print(f"  Best checkpoint: {results['best_checkpoint']}")


if __name__ == "__main__":
    main()
