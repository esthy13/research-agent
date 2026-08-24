"""Tests for validated MMLU transfer summaries."""

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.evaluation.transfer_analysis import (
    COMBINED_BENCHMARK,
    exact_mcnemar_p_value,
    generate_transfer_report,
    paired_bootstrap_interval,
    wilson_interval,
)
from research_agent.interfaces import EvaluationResult, ExperimentConfig
from research_agent.search.hashing import attach_config_hash


class TransferStatisticsTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_proportion(self) -> None:
        low, high = wilson_interval(7, 10)
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)

    def test_exact_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p_value(2, 0), 0.5)

    def test_paired_bootstrap_is_reproducible(self) -> None:
        differences = [1, 1, 0, -1, 0]
        first = paired_bootstrap_interval(
            differences,
            samples=200,
            seed=17,
        )
        second = paired_bootstrap_interval(
            differences,
            samples=200,
            seed=17,
        )
        self.assertEqual(first, second)


class TransferReportTests(unittest.TestCase):
    def test_report_validates_and_combines_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_fixture(root)

            report = generate_transfer_report(
                experiment_root=root,
                output_dir=root / "analysis",
            )

            self.assertEqual(len(report["summary"]), 9)
            self.assertEqual(len(report["comparisons"]), 6)
            selected_combined = next(
                row
                for row in report["summary"]
                if row["benchmark"] == COMBINED_BENCHMARK
                and row["label"] == "selected"
            )
            self.assertEqual(selected_combined["n_items"], 7)
            self.assertEqual(selected_combined["correct"], 5)
            self.assertAlmostEqual(selected_combined["accuracy"], 5 / 7)

            comparison = next(
                row
                for row in report["comparisons"]
                if row["benchmark"] == COMBINED_BENCHMARK
                and row["baseline_label"] == "naive"
            )
            self.assertEqual(comparison["selected_only_correct"], 2)
            self.assertEqual(comparison["baseline_only_correct"], 0)
            self.assertEqual(comparison["mcnemar_exact_p_value"], 0.5)

            self.assertTrue(
                (root / "analysis" / "transfer_report.json").is_file(),
            )
            self.assertTrue(
                (root / "analysis" / "transfer_summary.csv").is_file(),
            )
            self.assertTrue(
                (root / "analysis" / "transfer_comparisons.csv").is_file(),
            )
            self.assertTrue(
                (root / "analysis" / "TRANSFER_REPORT.md").is_file(),
            )

    def _write_fixture(self, root: Path) -> None:
        configs = {
            "naive": self._config(
                prompt="Answer the question.",
                roles=["solver"],
                aggregation="single_answer",
            ),
            "careful": self._config(
                prompt="Reason carefully.",
                roles=["solver"],
                aggregation="single_answer",
            ),
            "selected": self._config(
                prompt="Reason carefully.",
                roles=["reasoner", "critic"],
                aggregation="final_revision",
            ),
        }
        correctness = {
            "mmlu_college_chemistry": {
                "naive": [True, False, False, True],
                "careful": [True, True, True, False],
                "selected": [True, True, False, True],
            },
            "mmlu_college_physics": {
                "naive": [False, False, True],
                "careful": [True, True, False],
                "selected": [True, False, True],
            },
        }

        config_dir = root / "configs"
        evaluation_dir = root / "evaluations" / "transfer"
        details_dir = root / "item_details" / "transfer"
        config_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)
        details_dir.mkdir(parents=True)

        for label, config in configs.items():
            (config_dir / f"{label}.json").write_text(
                config.model_dump_json(indent=2),
                encoding="utf-8",
            )

        for benchmark, benchmark_results in correctness.items():
            for label, values in benchmark_results.items():
                config = configs[label]
                items = [
                    {
                        "dataset_index": index,
                        "item_id": str(index),
                        "expected_answer": "A",
                        "predicted_answer": "A" if correct else "B",
                        "correct": correct,
                        "error": None,
                    }
                    for index, correct in enumerate(values)
                ]
                result = EvaluationResult(
                    config_hash=config.config_hash,
                    benchmark=benchmark,
                    split="transfer",
                    seed=2026,
                    accuracy=sum(values) / len(values),
                    input_tokens=100 * len(values),
                    output_tokens=10 * len(values),
                    latency_seconds=float(len(values)),
                    status="success",
                    errors=[],
                )
                (evaluation_dir / f"{label}_{benchmark}.json").write_text(
                    result.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                details = {
                    "config_hash": config.config_hash,
                    "benchmark": benchmark,
                    "split": "transfer",
                    "seed": 2026,
                    "items": items,
                }
                (details_dir / f"{label}_{benchmark}.json").write_text(
                    json.dumps(details, indent=2),
                    encoding="utf-8",
                )

    @staticmethod
    def _config(
        *,
        prompt: str,
        roles: list[str],
        aggregation: str,
    ) -> ExperimentConfig:
        config = ExperimentConfig(
            hypothesis="fixture",
            rationale="fixture",
            system_prompt=prompt,
            reasoning_mode=(
                "direct" if aggregation == "single_answer" else "chain_of_thought"
            ),
            agent_count=len(roles),
            roles=roles,
            communication_order=roles,
            aggregation=aggregation,
            temperature=0.0,
            max_tokens=1024,
            seed=2026,
            condition="no_memory",
            parent_ids=[],
            search_mode="exploration",
            config_hash="",
            model_id="fixture-model",
        )
        return attach_config_hash(config)


if __name__ == "__main__":
    unittest.main()
