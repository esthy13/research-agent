"""Structural interface implemented by mock and production evaluators."""

from typing import Protocol

from research_agent.interfaces import (
    BenchmarkSplit,
    EvaluationResult,
    ExperimentConfig,
)


class ExperimentEvaluator(Protocol):
    """Contract required by the research-cycle orchestrator."""

    async def evaluate(
        self,
        config: ExperimentConfig,
        benchmark: str,
        split: BenchmarkSplit,
        seed: int,
    ) -> EvaluationResult:
        """Execute one configuration and return aggregate measurements.

        Args:
            config: The experiment configuration to execute.
            benchmark: Target benchmark name.
            split: Dataset split to evaluate against.
            seed: Random seed for benchmark sampling.

        Returns:
            Aggregate EvaluationResult containing accuracy, tokens, and latency.
        """
        ...
