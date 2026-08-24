"""End-to-end orchestration for one development-search cycle."""

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from research_agent.agents.critic import CriticAgent
from research_agent.evaluation.protocol import ExperimentEvaluator
from research_agent.interfaces import (
    EvaluationResult,
    ExperienceRecord,
    MemoryCondition,
)
from research_agent.memory.library import ExperienceLibrary
from research_agent.search.controller import SearchController


FitnessFunction = Callable[[EvaluationResult], float]


class ResearchCycle:
    """Run one propose-evaluate-critique-store cycle.

    This is the central orchestrator that connects the search controller,
    benchmark evaluator, Critic agent, and experience library into a
    single development iteration.
    """

    def __init__(
        self,
        controller: SearchController,
        evaluator: ExperimentEvaluator,
        critic: CriticAgent,
        library: ExperienceLibrary,
        fitness_function: FitnessFunction,
    ) -> None:
        self.controller = controller
        self.evaluator = evaluator
        self.critic = critic
        self.library = library
        self.fitness_function = fitness_function

    async def run(
        self,
        cycle: int,
        total_cycles: int,
        seed: int,
        condition: MemoryCondition,
        benchmark: str,
    ) -> ExperienceRecord:
        """Execute and persist one candidate on the development split.

        The research loop proceeds as follows:
            1. Search Controller proposes a configuration.
            2. Evaluator runs it on a benchmark.
            3. Critic interprets the result.
            4. ExperienceLibrary saves the whole trajectory.
            5. Next cycle can learn from this new experience.

        Args:
            cycle: Zero-based index of the current cycle.
            total_cycles: Total number of cycles in this search run.
            seed: Shared seed for reproducibility.
            condition: Active memory-ablation condition.
            benchmark: Name of the target benchmark.

        Returns:
            The complete experience record for this cycle.
        """

        split = "development"
        config = await self.controller.propose_candidate(
            cycle=cycle,
            total_cycles=total_cycles,
            seed=seed,
            condition=condition,
            benchmark=benchmark,
        )
        result = await self.evaluator.evaluate(
            config=config,
            benchmark=benchmark,
            split=split,
            seed=seed,
        )
        review = await self.critic.review(
            config=config,
            result=result,
            baseline_context=self.controller.baseline_context,
        )
        fitness_score = self.fitness_function(result)

        retrieval_text = (
            f"The {config.reasoning_mode} strategy with "
            f"{config.agent_count} agents achieved "
            f"{result.accuracy:.2f} accuracy. "
            f"It used {result.input_tokens + result.output_tokens} total "
            f"tokens and {result.latency_seconds:.2f} seconds. "
            f"Critic recommendation: {review.future_recommendation}"
        )
        record = ExperienceRecord(
            experiment_id=str(uuid4()),
            cycle=cycle,
            config=config,
            result=result,
            fitness_score=fitness_score,
            critic_review=review,
            retrieval_text=retrieval_text, 
            created_at=datetime.now(timezone.utc),
        )
        self.library.append(record)
        return record
