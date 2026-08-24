"""Create a fixed direct-solver baseline configuration.

The search loop discovers new configurations automatically. This script creates
a stable reference configuration so the final report can compare discovered
configs against a simple single-agent baseline.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from research_agent.interfaces import ExperimentConfig
from research_agent.model_factory import PROJECT_ROOT
from research_agent.search.hashing import attach_config_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a direct-solver baseline ExperimentConfig to JSON.",
    )
    parser.add_argument(
        "--output",
        default="results/baseline_direct_solver.json",
        help="Path where the baseline configuration JSON should be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed recorded in the baseline configuration.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the baseline solver.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum output tokens for the baseline solver.",
    )

    parser.add_argument(
        "--kind",
        choices=['careful', 'naive'],
        default='careful',
        help='Baseline type to create'
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    if args.kind == "careful":
        hypothesis = "A single direct solver provides the baseline performance."
        rationale = (
            "This reference configuration measures how well the base model "
            "performs without memory-guided search, critique, revision, or "
            "multi-agent role structure."
        )
        system_prompt = (
            "You are a careful reasoning assistant. Analyze the problem step by "
            "step, verify your reasoning against the provided information, and "
            "return the answer using the requested final-answer format."
        )
        max_tokens = args.max_tokens
    elif args.kind == "naive":
        hypothesis = "A naive direct solver provides a weak baseline."
        rationale = (
            "This baseline measures how the model performs with a minimal system "
            "prompt and without explicit step-by-step reasoning, verification, "
            "memory, critique, revision, or multi-agent structure."
        )
        system_prompt = "You are an assistant. Answer the question."
        max_tokens = args.max_tokens


    config = ExperimentConfig(
        hypothesis=hypothesis,
        rationale=rationale,
        system_prompt=system_prompt,
        reasoning_mode="direct",
        agent_count=1,
        roles=["solver"],
        communication_order=["solver"],
        aggregation="single_answer",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        condition="no_memory",
        parent_ids=[],
        search_mode="exploration",
        config_hash="",
        model_id=os.environ["MODEL_NAME"],
    )
    config = attach_config_hash(config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    print(f"Wrote baseline config to {output_path}")
    print(config.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
