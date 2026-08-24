"""Create validated summaries for the frozen MMLU transfer study."""

import argparse
import json
from pathlib import Path

from research_agent.evaluation.transfer_analysis import (
    DEFAULT_BENCHMARKS,
    generate_transfer_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen MMLU transfer artifacts and write statistical "
            "JSON, CSV, and Markdown summaries."
        ),
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Root containing configs/, evaluations/, and item_details/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Summary destination. Defaults to <experiment-root>/analysis."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else experiment_root / "analysis"
    )
    report = generate_transfer_report(
        experiment_root=experiment_root,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "benchmarks": list(DEFAULT_BENCHMARKS),
                "summary_rows": len(report["summary"]),
                "comparison_rows": len(report["comparisons"]),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
