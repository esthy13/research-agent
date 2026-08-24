"""Summarize development-search experiences for the final report."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from research_agent.interfaces import ExperienceRecord
from research_agent.memory.library import ExperienceLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank successful development experiments and export tables.",
    )
    parser.add_argument(
        "--library-path",
        default="data/experiences.jsonl",
        help="ExperienceLibrary JSONL file to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_report",
        help="Directory where summary files should be written.",
    )
    parser.add_argument(
        "--benchmark",
        default="gsm8k",
        help="Benchmark to summarize.",
    )
    parser.add_argument(
        "--split",
        default="development",
        choices=["development", "test"],
        help="Split to summarize.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Number of distinct searched configurations to export for "
            "full-development confirmation."
        ),
    )
    return parser.parse_args()


def _record_row(record: ExperienceRecord) -> dict[str, object]:
    return {
        "rank": None,
        "experiment_id": record.experiment_id,
        "cycle": record.cycle,
        "condition": record.config.condition,
        "seed": record.config.seed,
        "search_mode": record.config.search_mode,
        "config_hash": record.config.config_hash,
        "accuracy": record.result.accuracy,
        "fitness_score": record.fitness_score,
        "input_tokens": record.result.input_tokens,
        "output_tokens": record.result.output_tokens,
        "latency_seconds": record.result.latency_seconds,
        "reasoning_mode": record.config.reasoning_mode,
        "agent_count": record.config.agent_count,
        "roles": " | ".join(record.config.roles),
        "communication_order": " | ".join(record.config.communication_order),
        "aggregation": record.config.aggregation,
        "temperature": record.config.temperature,
        "max_tokens": record.config.max_tokens,
        "hypothesis": record.config.hypothesis,
        "critic_recommendation": record.critic_review.future_recommendation,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = ExperienceLibrary(args.library_path).load_all()
    selected = [
        record
        for record in records
        if record.result.benchmark.lower() == args.benchmark.lower()
        and record.result.split == args.split
        and record.result.status == "success"
    ]
    selected.sort(key=lambda record: record.fitness_score, reverse=True)

    if not selected:
        raise SystemExit(
            "No successful records found for "
            f"benchmark={args.benchmark}, split={args.split}.",
        )

    ranked_rows = []
    for rank, record in enumerate(selected, start=1):
        row = _record_row(record)
        row["rank"] = rank
        ranked_rows.append(row)

    _write_csv(output_dir / "ranked_development_experiments.csv", ranked_rows)
    (output_dir / "ranked_development_experiments.json").write_text(
        json.dumps(ranked_rows, indent=2),
        encoding="utf-8",
    )

    best_overall = selected[0]
    best_searched_json = best_overall.config.model_dump_json(indent=2)
    (output_dir / "best_config_overall.json").write_text(
        best_searched_json,
        encoding="utf-8",
    )
    (output_dir / "best_searched_config_100.json").write_text(
        best_searched_json,
        encoding="utf-8",
    )

    distinct_records = []
    seen_hashes: set[str] = set()
    for record in selected:
        if record.config.config_hash in seen_hashes:
            continue
        distinct_records.append(record)
        seen_hashes.add(record.config.config_hash)
        if len(distinct_records) == args.top_k:
            break

    top_config_dir = output_dir / "top_candidate_configs"
    top_config_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in top_config_dir.glob("candidate_*.json"):
        stale_path.unlink()

    top_candidate_rows = []
    for candidate_number, record in enumerate(distinct_records, start=1):
        label = f"candidate_{candidate_number:02d}"
        config_path = top_config_dir / f"{label}.json"
        config_path.write_text(
            record.config.model_dump_json(indent=2),
            encoding="utf-8",
        )
        top_candidate_rows.append(
            {
                "label": label,
                "search_rank": selected.index(record) + 1,
                "experiment_id": record.experiment_id,
                "config_hash": record.config.config_hash,
                "accuracy": record.result.accuracy,
                "total_tokens": (
                    record.result.input_tokens + record.result.output_tokens
                ),
                "latency_seconds": record.result.latency_seconds,
                "condition": record.config.condition,
                "config_path": str(
                    Path("top_candidate_configs") / config_path.name,
                ),
            },
        )
    (output_dir / "top_candidates.json").write_text(
        json.dumps(top_candidate_rows, indent=2),
        encoding="utf-8",
    )

    grouped: dict[str, list[ExperienceRecord]] = defaultdict(list)
    for record in selected:
        grouped[record.config.condition].append(record)

    condition_rows = []
    best_by_condition = {}
    for condition, condition_records in sorted(grouped.items()):
        accuracies = [record.result.accuracy for record in condition_records]
        best_record = max(
            condition_records,
            key=lambda record: record.fitness_score,
        )
        best_by_condition[condition] = best_record.config.model_dump()
        condition_rows.append(
            {
                "condition": condition,
                "num_records": len(condition_records),
                "best_accuracy": best_record.result.accuracy,
                "mean_accuracy": sum(accuracies) / len(accuracies),
                "best_config_hash": best_record.config.config_hash,
                "best_experiment_id": best_record.experiment_id,
                "best_reasoning_mode": best_record.config.reasoning_mode,
                "best_agent_count": best_record.config.agent_count,
                "best_aggregation": best_record.config.aggregation,
            }
        )

    _write_csv(output_dir / "condition_summary.csv", condition_rows)
    (output_dir / "condition_summary.json").write_text(
        json.dumps(condition_rows, indent=2),
        encoding="utf-8",
    )
    (output_dir / "best_configs_by_condition.json").write_text(
        json.dumps(best_by_condition, indent=2),
        encoding="utf-8",
    )

    print("Best searched configuration on the development screen:")
    print(best_overall.config.model_dump_json(indent=2))
    print(f"\nWrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
