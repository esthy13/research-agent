"""Tests for the frozen MMLU transfer data path."""

import unittest
from unittest.mock import patch

from datasets import Dataset

from research_agent.evaluation import runner


class MMLUTransferSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset.from_dict(
            {
                "question": [
                    "Question one",
                    "Question two",
                    "Exact duplicate",
                    "Exact duplicate",
                    "Stem variant",
                    "Stem variant",
                ],
                "choices": [
                    ["a", "b", "c", "d"],
                    ["e", "f", "g", "h"],
                    ["i", "j", "k", "l"],
                    ["i", "j", "k", "l"],
                    ["m", "n", "o", "p"],
                    ["m2", "n2", "o2", "p2"],
                ],
                "answer": [0, 1, 2, 2, 3, 0],
            },
        )

    @patch("research_agent.evaluation.runner.load_dataset")
    def test_transfer_uses_every_unique_official_test_item(
        self,
        load_dataset_mock,
    ) -> None:
        load_dataset_mock.return_value = self.dataset

        transfer = runner._load_mmlu_subject_split(
            "college_physics",
            "transfer",
        )

        self.assertEqual(len(transfer), 5)
        self.assertEqual(
            sorted(transfer["_source_index"]),
            [0, 1, 2, 4, 5],
        )
        load_dataset_mock.assert_called_once_with(
            runner.MMLU_DATASET,
            "college_physics",
            split="test",
            revision=runner.MMLU_REVISION,
        )

    @patch("research_agent.evaluation.runner.load_dataset")
    def test_internal_partitions_are_disjoint_and_group_isolated(
        self,
        load_dataset_mock,
    ) -> None:
        load_dataset_mock.return_value = self.dataset

        development = runner._load_mmlu_subject_split(
            "college_chemistry",
            "development",
        )
        test = runner._load_mmlu_subject_split(
            "college_chemistry",
            "test",
        )
        transfer = runner._load_mmlu_subject_split(
            "college_chemistry",
            "transfer",
        )

        development_ids = set(development["_source_index"])
        test_ids = set(test["_source_index"])
        transfer_ids = set(transfer["_source_index"])
        self.assertFalse(development_ids & test_ids)
        self.assertEqual(development_ids | test_ids, transfer_ids)
        self.assertTrue(
            {4, 5}.issubset(development_ids)
            or {4, 5}.issubset(test_ids),
        )

    def test_evaluator_rejects_nonpositive_sample_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive or None"):
            runner.Evaluator(max_examples=0)


if __name__ == "__main__":
    unittest.main()
