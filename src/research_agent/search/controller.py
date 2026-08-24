"""Coordination of retrieval, search policy, generation, and validation."""

from research_agent.agents.ideator import IdeatorAgent
from research_agent.interfaces import ExperimentConfig, MemoryCondition
from research_agent.memory.library import ExperienceLibrary
from research_agent.memory.retrieval import (
    build_memory_context,
    retrieve_experiences,
)
from research_agent.search.policy import choose_search_decision
from research_agent.search.validation import (
    ConfigValidationError,
    validate_config,
)


class SearchController:
    """Produce valid, non-duplicate candidates under a fixed search budget.

    Attributes:
        ideator: IdeatorAgent instance for proposal generation.
        library: ExperienceLibrary instance for historical trajectories.
        model_id: Model deployment name.
        max_attempts: Maximum retry attempts for valid/unique generation.
        baseline_context: Shared baseline evidence string.
    """

    def __init__(
        self,
        ideator: IdeatorAgent,
        library: ExperienceLibrary,
        model_id: str,
        max_attempts: int = 10,
        baseline_context: str = "No fixed baseline measurements were supplied.",
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.ideator = ideator
        self.library = library
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.baseline_context = baseline_context

    async def propose_candidate(
        self,
        cycle: int,
        total_cycles: int,
        seed: int,
        condition: MemoryCondition,
        benchmark: str = "unspecified",
    ) -> ExperimentConfig:
        """Generate one executable candidate, retrying invalid proposals.

        Args:
            cycle: Current zero-based search cycle index.
            total_cycles: Total number of planned search cycles.
            seed: Shared random seed for search decision policy.
            condition: Active memory condition.
            benchmark: Target benchmark name.

        Returns:
            A validated, unique ExperimentConfig instance.

        Raises:
            RuntimeError: If max_attempts is exceeded without producing a valid proposal.
        """

        # Retrieve top 5 relevant prior experiences from the library
        experiences = retrieve_experiences(
            library=self.library,
            limit=5,
            split="development",
            condition=condition,
            seed=seed,
        )

        # Under no_memory condition, prior experiences are excluded from search policy
        policy_experiences = [] if condition == "no_memory" else experiences

        # Choose exploration or exploitation based on current cycle annealing schedule
        decision = choose_search_decision(
            cycle=cycle,
            total_cycles=total_cycles,
            seed=seed,
            experiences=policy_experiences,
        )

        # Format retrieved experiences into text context for the Ideator
        memory_context = build_memory_context(experiences, condition)

        # Track failure cause to feed back into retry prompt if invalid
        last_rejection = "no candidate was generated"
        last_duplicate: ExperimentConfig | None = None

        # Retry loop allowing up to max_attempts proposals
        for attempt in range(1, self.max_attempts + 1):
            config = await self.ideator.propose(
                memory_context=memory_context,
                model_id=self.model_id,
                seed=seed,
                condition=condition,
                search_mode=decision.mode,
                parent_ids=decision.parent_ids,
                benchmark=benchmark,
                baseline_context=self.baseline_context,
            )
            try:
                validate_config(config)
            except ConfigValidationError as error:
                last_rejection = f"invalid configuration: {error}"
                print(
                    f"Rejected candidate attempt {attempt}/"
                    f"{self.max_attempts}: {last_rejection}",
                    flush=True,
                )
                memory_context += (
                    "\n\nThe previous proposal was invalid: "
                    f"{error}. Return a corrected configuration."
                )
                continue

            if self.library.contains_config_hash(
                config_hash=config.config_hash,
                condition=condition,
                seed=seed,
            ):
                last_duplicate = config
                last_rejection = (
                    "duplicate executable configuration "
                    f"{config.config_hash}"
                )
                print(
                    f"Rejected candidate attempt {attempt}/"
                    f"{self.max_attempts}: {last_rejection}",
                    flush=True,
                )
                memory_context += (
                    "\n\nThe previous proposal duplicated an existing "
                    "strategy. Change at least one executable setting."
                )
                continue
            return config

        if last_duplicate is not None:
            print(
                "Candidate attempts exhausted with valid duplicates. "
                "Executing the last duplicate as an explicit replication: "
                f"{last_duplicate.config_hash}",
                flush=True,
            )
            return last_duplicate

        raise RuntimeError(
            "The Ideator could not produce a valid, unique configuration "
            f"after {self.max_attempts} attempts. "
            f"Last rejection: {last_rejection}.",
        )
