"""Select the final configuration from full-development confirmations."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev

from research_agent.interfaces import EvaluationResult, ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank baseline and searched configurations using repeated full-"
            "development evaluations. Accuracy is primary; total tokens and "
            "latency break exact ties."
        ),
    )
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Directory containing configurations named <label>.json.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing results named <label>_seed_<seed>.json.",
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--output-config",
        required=True,
        help="Path for the frozen winning ExperimentConfig.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for JSON and CSV selection summaries.",
    )
    return parser.parse_args()


def _load_results(
    results_dir: Path,
    label: str,
    config: ExperimentConfig,
    benchmark: str,
) -> list[EvaluationResult]:
    paths = sorted(results_dir.glob(f"{label}_seed_*.json"))
    if not paths:
        raise ValueError(f"No confirmation results found for {label}")

    results = []
    for path in paths:
        result = EvaluationResult.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
        if result.config_hash != config.config_hash:
            raise ValueError(f"Configuration hash mismatch in {path}")
        if result.benchmark.lower() != benchmark.lower():
            raise ValueError(f"Unexpected benchmark in {path}")
        if result.split != "development":
            raise ValueError(f"Confirmation result is not development: {path}")
        if result.status != "success":
            raise ValueError(f"Confirmation result failed: {path}")
        results.append(result)
    return results


def _candidate_row(
    label: str,
    config: ExperimentConfig,
    results: list[EvaluationResult],
) -> dict[str, object]:
    accuracies = [result.accuracy for result in results]
    input_tokens = [result.input_tokens for result in results]
    output_tokens = [result.output_tokens for result in results]
    total_tokens = [
        result.input_tokens + result.output_tokens for result in results
    ]
    latencies = [result.latency_seconds for result in results]
    return {
        "rank": None,
        "selected": False,
        "label": label,
        "source": "baseline" if label in {"naive", "careful"} else "search",
        "config_hash": config.config_hash,
        "runs": len(results),
        "mean_accuracy": fmean(accuracies),
        "accuracy_std": pstdev(accuracies),
        "mean_input_tokens": fmean(input_tokens),
        "mean_output_tokens": fmean(output_tokens),
        "mean_total_tokens": fmean(total_tokens),
        "mean_latency_seconds": fmean(latencies),
        "reasoning_mode": config.reasoning_mode,
        "agent_count": config.agent_count,
        "aggregation": config.aggregation,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "hypothesis": config.hypothesis,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs: dict[str, ExperimentConfig] = {}
    rows = []
    for config_path in sorted(config_dir.glob("*.json")):
        label = config_path.stem
        config = ExperimentConfig.model_validate_json(
            config_path.read_text(encoding="utf-8"),
        )
        configs[label] = config
        results = _load_results(
            results_dir,
            label,
            config,
            benchmark=args.benchmark,
        )
        rows.append(_candidate_row(label, config, results))

    if not rows:
        raise SystemExit(f"No configurations found in {config_dir}")

    rows.sort(
        key=lambda row: (
            -float(row["mean_accuracy"]),
            float(row["mean_total_tokens"]),
            float(row["mean_latency_seconds"]),
            str(row["label"]),
        ),
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    rows[0]["selected"] = True

    winner_label = str(rows[0]["label"])
    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        configs[winner_label].model_dump_json(indent=2),
        encoding="utf-8",
    )

    (output_dir / "final_selection.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "final_selection.csv", rows)

    print(f"Selected final configuration: {winner_label}")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
