"""
LLM-as-Judge Evaluator with Position Debiasing

Uses a stronger model (Claude) to evaluate fine-tuned model outputs.
Implements best practices from evaluation research:
- Pairwise comparison (more reliable than absolute scoring)
- Position debiasing (run both orderings)
- Structured rubrics
- Chain-of-thought judging

Usage:
    judge = LLMJudge()
    result = await judge.evaluate_models(baseline, finetuned, test_set)
"""

import json
import yaml
import logging
import asyncio
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Winner(Enum):
    A = "A"
    B = "B"
    TIE = "TIE"


@dataclass
class JudgmentResult:
    """Result of a single pairwise judgment."""
    winner: Winner
    confident: bool
    reasoning: str
    scores: Dict[str, int] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Configuration for LLM-as-judge evaluation."""

    # Judge model
    judge_model: str = "claude-sonnet-4-20250514"

    # Test data
    test_set: str = "data/test.jsonl"

    # Models to compare
    baseline_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    finetuned_model: str = "checkpoints/best"

    # Evaluation settings
    num_samples: int = 200
    position_debias: bool = True  # Run both orderings
    parallel_judges: int = 10  # Concurrent judge calls

    # Rubric
    rubric: str = """Evaluate AI assistant responses on these criteria:

1. CORRECTNESS (0-5): Is the answer factually accurate and complete?
   0: Completely wrong  |  3: Mostly correct  |  5: Perfectly accurate

2. HELPFULNESS (0-5): Does it address the user's actual need?
   0: Misses the point  |  3: Partially helpful  |  5: Fully addresses need

3. CLARITY (0-5): Is the response well-structured and easy to understand?
   0: Confusing  |  3: Understandable  |  5: Crystal clear

For each criterion, explain your reasoning briefly, then give a score.
Finally, state which response is better overall: "A", "B", or "TIE"."""

    # Output
    output_dir: str = "eval_results"
    save_judgments: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


DEFAULT_RUBRIC = """You are an expert evaluator comparing AI assistant responses.

Evaluate each response on:

1. **Correctness** (0-5): Is the information factually accurate?
2. **Helpfulness** (0-5): Does it fully address the user's request?
3. **Clarity** (0-5): Is it well-organized and easy to understand?

For each criterion, provide brief reasoning then a score.
At the end, state which response is better overall: "A", "B", or "TIE".
"""


class LLMJudge:
    """
    LLM-as-Judge evaluator using Claude as the judge.

    Key features:
    - Position debiasing: Runs both orderings to eliminate position bias
    - Structured rubrics: Explicit criteria with score anchors
    - Async execution: Parallel evaluation for speed
    - Detailed logging: Full reasoning saved for analysis
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        rubric: Optional[str] = None
    ):
        self.model = model
        self.rubric = rubric or DEFAULT_RUBRIC

        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.AsyncAnthropic()
        else:
            raise ImportError("anthropic package required: pip install anthropic")

    async def _single_judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> Tuple[Winner, str, Dict[str, int]]:
        """
        Run a single judgment (one ordering).

        Returns:
            (winner, reasoning, scores)
        """
        judge_prompt = f"""{self.rubric}

---

**User Prompt:**
{prompt}

---

**Response A:**
{response_a}

---

**Response B:**
{response_b}

---

Evaluate both responses. For each criterion, explain your reasoning then give a score (0-5).
Finally, state the overall winner: "A", "B", or "TIE".
"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": judge_prompt}]
            )

            text = response.content[0].text

            # Parse winner
            winner = self._parse_winner(text)

            # Parse scores (optional)
            scores = self._parse_scores(text)

            return winner, text, scores

        except Exception as e:
            logger.error(f"Judge call failed: {e}")
            return Winner.TIE, f"Error: {e}", {}

    def _parse_winner(self, text: str) -> Winner:
        """Extract winner from judge response."""
        # Look for explicit winner statement
        patterns = [
            r'(?:winner|better|prefer)\s*(?:is|:)?\s*["\']?([AB])["\']?',
            r'(?:Response\s+)?([AB])\s+(?:is|wins|better)',
            r'([AB])\s*$',  # Last character if A or B
            r'"([AB])"',
        ]

        text_upper = text.upper()

        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                winner_char = match.group(1)
                return Winner.A if winner_char == "A" else Winner.B

        # Check for TIE
        if re.search(r'\bTIE\b', text_upper):
            return Winner.TIE

        # Fallback: count mentions
        a_count = text_upper.count("RESPONSE A") + text_upper.count('"A"')
        b_count = text_upper.count("RESPONSE B") + text_upper.count('"B"')

        if a_count > b_count:
            return Winner.A
        elif b_count > a_count:
            return Winner.B
        else:
            return Winner.TIE

    def _parse_scores(self, text: str) -> Dict[str, int]:
        """Extract scores from judge response."""
        scores = {}

        # Look for patterns like "Correctness: 4" or "Correctness (4)"
        criteria = ["correctness", "helpfulness", "clarity"]

        for criterion in criteria:
            pattern = rf'{criterion}[:\s]*\(?(\d)[/\)\s]'
            match = re.search(pattern, text.lower())
            if match:
                scores[criterion] = int(match.group(1))

        return scores

    async def pairwise_judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        debias: bool = True
    ) -> JudgmentResult:
        """
        Run pairwise evaluation with optional position debiasing.

        Position bias is a known issue where judges prefer the first or second
        response regardless of quality. We mitigate this by:
        1. Running both orderings (A,B) and (B,A)
        2. Only declaring a winner if both orderings agree
        3. Otherwise declaring a TIE

        Args:
            prompt: The user prompt
            response_a: First response
            response_b: Second response
            debias: Whether to run position debiasing

        Returns:
            JudgmentResult with winner, confidence, and reasoning
        """
        if not debias:
            winner, reasoning, scores = await self._single_judge(
                prompt, response_a, response_b
            )
            return JudgmentResult(
                winner=winner,
                confident=True,
                reasoning=reasoning,
                scores=scores
            )

        # Run both orderings in parallel
        result_ab, result_ba = await asyncio.gather(
            self._single_judge(prompt, response_a, response_b),
            self._single_judge(prompt, response_b, response_a)
        )

        winner_ab, reasoning_ab, scores_ab = result_ab
        winner_ba, reasoning_ba, scores_ba = result_ba

        # Flip the BA result (since we swapped the order)
        if winner_ba == Winner.A:
            winner_ba_normalized = Winner.B
        elif winner_ba == Winner.B:
            winner_ba_normalized = Winner.A
        else:
            winner_ba_normalized = Winner.TIE

        # Aggregate results
        if winner_ab == winner_ba_normalized:
            # Both orderings agree - confident result
            return JudgmentResult(
                winner=winner_ab,
                confident=True,
                reasoning=f"AB: {reasoning_ab}\n\nBA: {reasoning_ba}",
                scores=scores_ab
            )
        else:
            # Orderings disagree - likely position bias, declare TIE
            return JudgmentResult(
                winner=Winner.TIE,
                confident=False,
                reasoning=f"Position bias detected. AB said {winner_ab.value}, "
                         f"BA said {winner_ba.value} (normalized: {winner_ba_normalized.value})\n\n"
                         f"AB: {reasoning_ab}\n\nBA: {reasoning_ba}",
                scores={}
            )

    async def evaluate_batch(
        self,
        test_items: List[Dict[str, str]],
        model_a_responses: List[str],
        model_b_responses: List[str],
        debias: bool = True,
        max_concurrent: int = 10
    ) -> List[JudgmentResult]:
        """
        Evaluate a batch of comparisons with concurrency control.

        Args:
            test_items: List of {"prompt": ...} dicts
            model_a_responses: Responses from model A
            model_b_responses: Responses from model B
            debias: Whether to run position debiasing
            max_concurrent: Max parallel judge calls

        Returns:
            List of JudgmentResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_judge(prompt, resp_a, resp_b):
            async with semaphore:
                return await self.pairwise_judge(prompt, resp_a, resp_b, debias)

        tasks = [
            bounded_judge(item["prompt"], resp_a, resp_b)
            for item, resp_a, resp_b in zip(test_items, model_a_responses, model_b_responses)
        ]

        results = await asyncio.gather(*tasks)
        return results

    def compute_metrics(
        self,
        results: List[JudgmentResult],
        model_a_name: str = "baseline",
        model_b_name: str = "finetuned"
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics from judgment results.

        Returns:
            Dict with win rates, confidence metrics, etc.
        """
        total = len(results)
        a_wins = sum(1 for r in results if r.winner == Winner.A)
        b_wins = sum(1 for r in results if r.winner == Winner.B)
        ties = sum(1 for r in results if r.winner == Winner.TIE)

        confident = sum(1 for r in results if r.confident)

        # Win rates
        a_rate = a_wins / total if total > 0 else 0
        b_rate = b_wins / total if total > 0 else 0
        tie_rate = ties / total if total > 0 else 0

        # Confidence interval (simple binomial)
        import math
        n = total
        if n > 0:
            se = math.sqrt(b_rate * (1 - b_rate) / n)
            ci_95 = 1.96 * se
        else:
            ci_95 = 0

        return {
            "total_comparisons": total,
            f"{model_a_name}_wins": a_wins,
            f"{model_b_name}_wins": b_wins,
            "ties": ties,
            f"{model_a_name}_win_rate": a_rate,
            f"{model_b_name}_win_rate": b_rate,
            "tie_rate": tie_rate,
            f"{model_b_name}_win_rate_95ci": f"{b_rate:.3f} +/- {ci_95:.3f}",
            "confident_judgments": confident,
            "confidence_rate": confident / total if total > 0 else 0
        }


async def evaluate_models(
    config: EvalConfig,
    baseline_sampler=None,
    finetuned_sampler=None
) -> Dict[str, Any]:
    """
    Full evaluation pipeline: load data, generate responses, judge.

    Args:
        config: Evaluation configuration
        baseline_sampler: Optional pre-initialized sampler for baseline
        finetuned_sampler: Optional pre-initialized sampler for finetuned

    Returns:
        Metrics and detailed results
    """
    # Load test set
    test_items = []
    with open(config.test_set) as f:
        for line in f:
            item = json.loads(line)
            test_items.append(item)

    # Limit to num_samples
    if config.num_samples < len(test_items):
        import random
        test_items = random.sample(test_items, config.num_samples)

    logger.info(f"Evaluating {len(test_items)} test items")

    # Generate responses (or use pre-generated)
    baseline_responses = []
    finetuned_responses = []

    for item in test_items:
        prompt = item.get("prompt") or item.get("input")

        if baseline_sampler:
            baseline_resp = baseline_sampler.sample(prompt)
        else:
            baseline_resp = item.get("baseline_response", "[BASELINE RESPONSE PLACEHOLDER]")

        if finetuned_sampler:
            finetuned_resp = finetuned_sampler.sample(prompt)
        else:
            finetuned_resp = item.get("finetuned_response", "[FINETUNED RESPONSE PLACEHOLDER]")

        baseline_responses.append(baseline_resp)
        finetuned_responses.append(finetuned_resp)

    # Run LLM-as-judge evaluation
    judge = LLMJudge(model=config.judge_model, rubric=config.rubric)

    results = await judge.evaluate_batch(
        test_items,
        baseline_responses,
        finetuned_responses,
        debias=config.position_debias,
        max_concurrent=config.parallel_judges
    )

    # Compute metrics
    metrics = judge.compute_metrics(results, "baseline", "finetuned")

    # Save detailed results
    if config.save_judgments:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        output_path = Path(config.output_dir) / "judgments.jsonl"
        with open(output_path, "w") as f:
            for item, base_resp, ft_resp, result in zip(
                test_items, baseline_responses, finetuned_responses, results
            ):
                record = {
                    "prompt": item.get("prompt") or item.get("input"),
                    "baseline_response": base_resp,
                    "finetuned_response": ft_resp,
                    "winner": result.winner.value,
                    "confident": result.confident,
                    "reasoning": result.reasoning,
                    "scores": result.scores
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved judgments to {output_path}")

        # Save metrics
        metrics_path = Path(config.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_path}")

    return {
        "metrics": metrics,
        "results": results
    }


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluator")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)

    print(f"\nLLM-as-Judge Evaluation")
    print(f"  Judge: {config.judge_model}")
    print(f"  Test set: {config.test_set}")
    print(f"  Position debiasing: {config.position_debias}")
    print(f"  Samples: {config.num_samples}")

    # Run evaluation
    results = asyncio.run(evaluate_models(config))

    print(f"\n=== Results ===")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
