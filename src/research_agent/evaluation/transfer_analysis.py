"""Validate and summarize frozen cross-benchmark transfer evaluations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent.interfaces import EvaluationResult, ExperimentConfig


DEFAULT_BENCHMARKS = (
    "mmlu_college_chemistry",
    "mmlu_college_physics",
)
DEFAULT_LABELS = ("naive", "careful", "selected")
SELECTED_LABEL = "selected"
BASELINE_LABELS = ("naive", "careful")
COMBINED_BENCHMARK = "mmlu_science_combined"
BOOTSTRAP_SAMPLES = 10_000
WILSON_Z_95 = 1.959963984540054


def wilson_interval(
    successes: int,
    total: int,
    z: float = WILSON_Z_95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successful trials.
        total: Total number of independent trials.
        z: Normal distribution quantile for confidence level (default 95%).

    Returns:
        Tuple of (lower_bound, upper_bound) floats in [0, 1].

    Raises:
        ValueError: If total <= 0 or successes is outside [0, total].
    """

    if total < 1:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")

    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total),
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar_p_value(
    selected_only_correct: int,
    baseline_only_correct: int,
) -> float:
    """Return the two-sided exact McNemar p-value.

    Args:
        selected_only_correct: Count of items where selected model succeeded but baseline failed.
        baseline_only_correct: Count of items where baseline succeeded but selected model failed.

    Returns:
        Exact two-sided McNemar p-value in [0, 1].

    Raises:
        ValueError: If any discordant count is negative.
    """

    if selected_only_correct < 0 or baseline_only_correct < 0:
        raise ValueError("discordant counts cannot be negative")

    discordant = selected_only_correct + baseline_only_correct
    if discordant == 0:
        return 1.0

    smaller = min(selected_only_correct, baseline_only_correct)
    lower_tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def paired_bootstrap_interval(
    differences: list[int],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 2026,
) -> tuple[float, float]:
    """Bootstrap the paired accuracy difference in proportion units.

    Args:
        differences: List of paired differences (-1, 0, or 1) per item.
        samples: Number of bootstrap iterations (default 10,000).
        seed: Random seed for bootstrap sampling.

    Returns:
        Tuple of (2.5th_percentile, 97.5th_percentile) accuracy difference bounds.

    Raises:
        ValueError: If differences list is empty or contains non-difference values.
    """

    if not differences:
        raise ValueError("differences cannot be empty")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if any(value not in {-1, 0, 1} for value in differences):
        raise ValueError("paired correctness differences must be -1, 0, or 1")

    rng = random.Random(seed)
    item_count = len(differences)
    estimates = []
    for _ in range(samples):
        estimate = sum(
            differences[rng.randrange(item_count)]
            for _ in range(item_count)
        ) / item_count
        estimates.append(estimate)
    estimates.sort()
    return (
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    )


def generate_transfer_report(
    experiment_root: Path,
    output_dir: Path,
    *,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    labels: tuple[str, ...] = DEFAULT_LABELS,
) -> dict[str, Any]:
    """Validate transfer artifacts and write JSON, CSV, and Markdown reports.

    Args:
        experiment_root: Path containing frozen configs and evaluation results.
        output_dir: Directory where summary reports will be written.
        benchmarks: Tuple of transfer benchmark names.
        labels: Tuple of experiment condition labels.

    Returns:
        Dictionary containing full validated transfer report structure.

    Raises:
        ValueError: If required labels or baseline configurations are missing.
    """

    if SELECTED_LABEL not in labels:
        raise ValueError(f"labels must include {SELECTED_LABEL!r}")
    missing_baselines = set(BASELINE_LABELS) - set(labels)
    if missing_baselines:
        raise ValueError(
            "labels are missing required baselines: "
            + ", ".join(sorted(missing_baselines)),
        )

    configs = {
        label: _load_config(experiment_root / "configs" / f"{label}.json")
        for label in labels
    }
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        for label in labels:
            artifact = _load_artifact(
                experiment_root=experiment_root,
                benchmark=benchmark,
                label=label,
                config=configs[label],
            )
            artifacts[(benchmark, label)] = artifact
            summary_rows.append(
                _summary_row(
                    benchmark=benchmark,
                    label=label,
                    config=configs[label],
                    artifact=artifact,
                ),
            )

    for label in labels:
        summary_rows.append(
            _combined_summary_row(
                benchmarks=benchmarks,
                label=label,
                config=configs[label],
                artifacts=artifacts,
            ),
        )

    comparison_rows = []
    for benchmark in (*benchmarks, COMBINED_BENCHMARK):
        for baseline_label in BASELINE_LABELS:
            comparison_rows.append(
                _comparison_row(
                    benchmark=benchmark,
                    baseline_label=baseline_label,
                    benchmarks=benchmarks,
                    artifacts=artifacts,
                ),
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "study": "frozen PubMedQA configuration transfer to MMLU",
            "split": "transfer",
            "selection_rule": (
                "All configurations were frozen before MMLU evaluation. "
                "No MMLU result is used for search, adaptation, or reselection."
            ),
            "benchmarks": list(benchmarks),
            "dataset_revision": (
                "c30699e8356da336a370243923dbaf21066bb9fe"
            ),
            "labels": list(labels),
            "combined_row": COMBINED_BENCHMARK,
            "confidence_interval": (
                "95% Wilson interval for each accuracy; deterministic paired "
                "bootstrap interval for selected-minus-baseline differences"
            ),
            "paired_test": "two-sided exact McNemar test",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "configurations": {
            label: {
                "config_hash": config.config_hash,
                "model_id": config.model_id,
                "reasoning_mode": config.reasoning_mode,
                "agent_count": config.agent_count,
                "aggregation": config.aggregation,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            for label, config in configs.items()
        },
        "summary": summary_rows,
        "comparisons": comparison_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transfer_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "transfer_summary.csv", summary_rows)
    _write_csv(output_dir / "transfer_comparisons.csv", comparison_rows)
    (output_dir / "TRANSFER_REPORT.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    return report


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values cannot be empty")
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _load_config(path: Path) -> ExperimentConfig:
    if not path.is_file():
        raise ValueError(f"Missing frozen configuration: {path}")
    return ExperimentConfig.model_validate_json(
        path.read_text(encoding="utf-8"),
    )


def _load_artifact(
    *,
    experiment_root: Path,
    benchmark: str,
    label: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    result_path = (
        experiment_root
        / "evaluations"
        / "transfer"
        / f"{label}_{benchmark}.json"
    )
    details_path = (
        experiment_root
        / "item_details"
        / "transfer"
        / f"{label}_{benchmark}.json"
    )
    if not result_path.is_file():
        raise ValueError(f"Missing aggregate result: {result_path}")
    if not details_path.is_file():
        raise ValueError(f"Missing item details: {details_path}")

    result = EvaluationResult.model_validate_json(
        result_path.read_text(encoding="utf-8"),
    )
    details = json.loads(details_path.read_text(encoding="utf-8"))

    if result.config_hash != config.config_hash:
        raise ValueError(f"Configuration hash mismatch in {result_path}")
    if result.benchmark != benchmark:
        raise ValueError(f"Unexpected benchmark in {result_path}")
    if result.split != "transfer":
        raise ValueError(f"Result is not a transfer evaluation: {result_path}")
    if result.status != "success":
        raise ValueError(f"Evaluation did not succeed: {result_path}")

    for field, expected in (
        ("config_hash", config.config_hash),
        ("benchmark", benchmark),
        ("split", "transfer"),
        ("seed", result.seed),
    ):
        if details.get(field) != expected:
            raise ValueError(
                f"Unexpected {field!r} in {details_path}: "
                f"{details.get(field)!r}",
            )

    items = details.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"No item records found in {details_path}")

    items_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed item record in {details_path}")
        raw_item_id = item.get("item_id")
        if raw_item_id is None or not str(raw_item_id).strip():
            raise ValueError(f"Missing item_id in {details_path}")
        item_id = str(raw_item_id)
        if item_id in items_by_id:
            raise ValueError(f"Duplicate item_id {item_id!r} in {details_path}")
        if not isinstance(item.get("correct"), bool):
            raise ValueError(
                f"Item {item_id!r} has non-boolean correctness in {details_path}",
            )
        items_by_id[item_id] = item

    measured_accuracy = (
        sum(bool(item["correct"]) for item in items) / len(items)
    )
    if not math.isclose(
        measured_accuracy,
        result.accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Aggregate accuracy does not match item details in {result_path}",
        )

    return {
        "result": result,
        "items": items,
        "items_by_id": items_by_id,
    }


def _summary_row(
    *,
    benchmark: str,
    label: str,
    config: ExperimentConfig,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    result: EvaluationResult = artifact["result"]
    items: list[dict[str, Any]] = artifact["items"]
    correct = sum(bool(item["correct"]) for item in items)
    ci_low, ci_high = wilson_interval(correct, len(items))
    extraction_failures = sum(
        "model did not produce an extractable answer" in str(item.get("error"))
        for item in items
    )
    other_failures = sum(
        bool(item.get("error"))
        and "model did not produce an extractable answer"
        not in str(item.get("error"))
        for item in items
    )
    return {
        "benchmark": benchmark,
        "label": label,
        "config_hash": config.config_hash,
        "n_items": len(items),
        "correct": correct,
        "accuracy": result.accuracy,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
        "latency_seconds": result.latency_seconds,
        "extraction_failures": extraction_failures,
        "other_failures": other_failures,
        "reasoning_mode": config.reasoning_mode,
        "agent_count": config.agent_count,
        "aggregation": config.aggregation,
        "max_tokens": config.max_tokens,
    }


def _combined_summary_row(
    *,
    benchmarks: tuple[str, ...],
    label: str,
    config: ExperimentConfig,
    artifacts: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    items = [
        item
        for benchmark in benchmarks
        for item in artifacts[(benchmark, label)]["items"]
    ]
    results = [
        artifacts[(benchmark, label)]["result"]
        for benchmark in benchmarks
    ]
    correct = sum(bool(item["correct"]) for item in items)
    accuracy = correct / len(items)
    ci_low, ci_high = wilson_interval(correct, len(items))
    extraction_failures = sum(
        "model did not produce an extractable answer" in str(item.get("error"))
        for item in items
    )
    other_failures = sum(
        bool(item.get("error"))
        and "model did not produce an extractable answer"
        not in str(item.get("error"))
        for item in items
    )
    return {
        "benchmark": COMBINED_BENCHMARK,
        "label": label,
        "config_hash": config.config_hash,
        "n_items": len(items),
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "input_tokens": sum(result.input_tokens for result in results),
        "output_tokens": sum(result.output_tokens for result in results),
        "total_tokens": sum(
            result.input_tokens + result.output_tokens
            for result in results
        ),
        "latency_seconds": sum(
            result.latency_seconds
            for result in results
        ),
        "extraction_failures": extraction_failures,
        "other_failures": other_failures,
        "reasoning_mode": config.reasoning_mode,
        "agent_count": config.agent_count,
        "aggregation": config.aggregation,
        "max_tokens": config.max_tokens,
    }


def _comparison_row(
    *,
    benchmark: str,
    baseline_label: str,
    benchmarks: tuple[str, ...],
    artifacts: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    selected_items, selected_results = _comparison_inputs(
        benchmark=benchmark,
        label=SELECTED_LABEL,
        benchmarks=benchmarks,
        artifacts=artifacts,
    )
    baseline_items, baseline_results = _comparison_inputs(
        benchmark=benchmark,
        label=baseline_label,
        benchmarks=benchmarks,
        artifacts=artifacts,
    )

    selected_ids = set(selected_items)
    baseline_ids = set(baseline_items)
    if selected_ids != baseline_ids:
        missing_selected = sorted(baseline_ids - selected_ids)
        missing_baseline = sorted(selected_ids - baseline_ids)
        raise ValueError(
            f"Paired item mismatch for {benchmark}/{baseline_label}; "
            f"missing selected={missing_selected[:5]}, "
            f"missing baseline={missing_baseline[:5]}",
        )

    differences = []
    both_correct = 0
    both_wrong = 0
    selected_only = 0
    baseline_only = 0
    for item_id in sorted(selected_ids):
        selected_item = selected_items[item_id]
        baseline_item = baseline_items[item_id]
        selected_expected = selected_item.get("expected_answer")
        baseline_expected = baseline_item.get("expected_answer")
        if (
            selected_expected is not None
            and baseline_expected is not None
            and selected_expected != baseline_expected
        ):
            raise ValueError(
                f"Expected-answer mismatch for {benchmark}/{item_id}",
            )

        selected_correct = bool(selected_item["correct"])
        baseline_correct = bool(baseline_item["correct"])
        differences.append(int(selected_correct) - int(baseline_correct))
        if selected_correct and baseline_correct:
            both_correct += 1
        elif selected_correct:
            selected_only += 1
        elif baseline_correct:
            baseline_only += 1
        else:
            both_wrong += 1

    selected_accuracy = (
        sum(bool(item["correct"]) for item in selected_items.values())
        / len(selected_items)
    )
    baseline_accuracy = (
        sum(bool(item["correct"]) for item in baseline_items.values())
        / len(baseline_items)
    )
    bootstrap_low, bootstrap_high = paired_bootstrap_interval(
        differences,
        seed=_stable_seed(benchmark, baseline_label),
    )
    selected_tokens = sum(
        result.input_tokens + result.output_tokens
        for result in selected_results
    )
    baseline_tokens = sum(
        result.input_tokens + result.output_tokens
        for result in baseline_results
    )
    selected_latency = sum(
        result.latency_seconds for result in selected_results
    )
    baseline_latency = sum(
        result.latency_seconds for result in baseline_results
    )

    return {
        "benchmark": benchmark,
        "selected_label": SELECTED_LABEL,
        "baseline_label": baseline_label,
        "n_items": len(differences),
        "selected_accuracy": selected_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_difference": selected_accuracy - baseline_accuracy,
        "accuracy_difference_ci_low": bootstrap_low,
        "accuracy_difference_ci_high": bootstrap_high,
        "both_correct": both_correct,
        "selected_only_correct": selected_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(
            selected_only,
            baseline_only,
        ),
        "selected_total_tokens": selected_tokens,
        "baseline_total_tokens": baseline_tokens,
        "token_ratio": (
            selected_tokens / baseline_tokens
            if baseline_tokens
            else None
        ),
        "selected_latency_seconds": selected_latency,
        "baseline_latency_seconds": baseline_latency,
        "latency_ratio": (
            selected_latency / baseline_latency
            if baseline_latency
            else None
        ),
    }


def _comparison_inputs(
    *,
    benchmark: str,
    label: str,
    benchmarks: tuple[str, ...],
    artifacts: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[EvaluationResult]]:
    included_benchmarks = (
        benchmarks
        if benchmark == COMBINED_BENCHMARK
        else (benchmark,)
    )
    items: dict[str, dict[str, Any]] = {}
    results = []
    for subject in included_benchmarks:
        artifact = artifacts[(subject, label)]
        results.append(artifact["result"])
        for item_id, item in artifact["items_by_id"].items():
            combined_id = f"{subject}:{item_id}"
            items[combined_id] = item
    return items, results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(report: dict[str, Any]) -> str:
    summary_rows = report["summary"]
    comparison_rows = report["comparisons"]
    lines = [
        "# Frozen MMLU transfer evaluation",
        "",
        (
            "The PubMedQA-selected configuration and both predefined baselines "
            "were frozen before evaluation. MMLU results were not used for "
            "search, adaptation, or configuration selection."
        ),
        "",
        (
            "College Chemistry and College Physics are two subjects from the "
            "same MMLU benchmark family, not two independent datasets."
        ),
        "",
        "## Accuracy and cost",
        "",
        "| Benchmark | Configuration | n | Accuracy (95% CI) | Total tokens | Latency (s) | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {benchmark} | {label} | {n_items} | "
            "{accuracy:.3f} [{accuracy_ci_low:.3f}, {accuracy_ci_high:.3f}] | "
            "{total_tokens} | {latency_seconds:.1f} | {errors} |".format(
                **row,
                errors=row["extraction_failures"] + row["other_failures"],
            ),
        )

    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Benchmark | Baseline | Accuracy difference (pp, 95% CI) | Selected-only correct | Baseline-only correct | Exact McNemar p | Token ratio | Latency ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for row in comparison_rows:
        display = {
            **row,
            "difference": 100.0 * row["accuracy_difference"],
            "ci_low": 100.0 * row["accuracy_difference_ci_low"],
            "ci_high": 100.0 * row["accuracy_difference_ci_high"],
            "token_ratio": (
                f"{row['token_ratio']:.2f}x"
                if row["token_ratio"] is not None
                else "n/a"
            ),
            "latency_ratio": (
                f"{row['latency_ratio']:.2f}x"
                if row["latency_ratio"] is not None
                else "n/a"
            ),
        }
        lines.append(
            "| {benchmark} | {baseline_label} | "
            "{difference:.1f} [{ci_low:.1f}, {ci_high:.1f}] | "
            "{selected_only_correct} | {baseline_only_correct} | "
            "{mcnemar_exact_p_value:.4f} | {token_ratio} | {latency_ratio} |".format(
                **display,
            ),
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- The combined row pools item-level outcomes across both MMLU subjects.",
            "- Accuracy intervals are Wilson intervals.",
            "- Difference intervals use a deterministic paired bootstrap with 10,000 resamples.",
            "- McNemar p-values are exact, two-sided, and exploratory; they are not corrected for multiple comparisons.",
            "- A transfer failure does not invalidate the search system; it indicates that the PubMedQA-selected configuration is domain-specific.",
            "",
        ],
    )
    return "\n".join(lines)
