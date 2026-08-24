"""Evaluate a saved ExperimentConfig on one benchmark split."""

import argparse
import asyncio
import json
from pathlib import Path

from research_agent.evaluation.runner import (
    MAX_EXAMPLES,
    SUPPORTED_BENCHMARKS,
    Evaluator,
)
from research_agent.interfaces import ExperimentConfig


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ExperimentConfig on one benchmark.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an ExperimentConfig JSON file.",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(SUPPORTED_BENCHMARKS),
        help="Benchmark to evaluate.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["development", "test", "transfer"],
        help="Benchmark split to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Evaluation sampling seed. Defaults to the seed stored in the "
            "configuration."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the EvaluationResult JSON should be written.",
    )
    parser.add_argument(
        "--details-output",
        default=None,
        help=(
            "Optional JSON path for per-item expected/predicted labels. Raw "
            "model reasoning is deliberately not stored."
        ),
    )

    example_group = parser.add_mutually_exclusive_group()
    example_group.add_argument(
        "--max-examples",
        type=positive_integer,
        default=MAX_EXAMPLES,
        help="Maximum number of benchmark examples to evaluate.",
    )
    example_group.add_argument(
        "--all-examples",
        action="store_true",
        help="Evaluate every example in the selected benchmark split.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.split == "transfer" and not args.benchmark.startswith("mmlu_"):
        raise SystemExit(
            "The transfer split is currently defined only for MMLU benchmarks.",
        )

    config_path = Path(args.config)
    config = ExperimentConfig.model_validate_json(
        config_path.read_text(encoding="utf-8"),
    )
    seed = config.seed if args.seed is None else args.seed

    evaluator = Evaluator(
        max_examples=None if args.all_examples else args.max_examples,
    )
    result = await evaluator.evaluate(
        config=config,
        benchmark=args.benchmark,
        split=args.split,
        seed=seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    if args.details_output is not None:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details = {
            "config_hash": config.config_hash,
            "benchmark": args.benchmark,
            "split": args.split,
            "seed": seed,
            "items": evaluator.last_item_results,
        }
        details_path.write_text(
            json.dumps(details, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote per-item details to {details_path}")

    print(result.model_dump_json(indent=2))
    print(f"Wrote evaluation result to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
