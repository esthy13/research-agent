"""Retrieval and prompt rendering for controlled memory conditions."""

from research_agent.interfaces import (
    BenchmarkSplit,
    ExperienceRecord,
    MemoryCondition,
)
from research_agent.memory.library import ExperienceLibrary


def retrieve_experiences(
    library: ExperienceLibrary,
    limit: int = 5,
    split: BenchmarkSplit = "development",
    condition: MemoryCondition | None = None,
    seed: int | None = None,
) -> list[ExperienceRecord]:
    """Return the highest-fitness recent records in the requested scope.

    Args:
        library: ExperienceLibrary containing trajectories.
        limit: Maximum number of experiences to retrieve.
        split: Benchmark split filter.
        condition: Optional memory condition filter.
        seed: Optional seed filter.

    Returns:
        List of up to `limit` deduplicated ExperienceRecord instances.
    """

    if limit <= 0:
        return []

    # Keep records matching requested split, memory condition, and seed
    records = [
        record
        for record in library.load_all()
        if record.result.split == split
        and (condition is None or record.config.condition == condition)
        and (seed is None or record.config.seed == seed)
    ]

    # Sort primarily by fitness score, secondarily by recency
    records.sort(
        key=lambda record: (record.fitness_score, record.created_at),
        reverse=True,
    )

    # Replications share an executable configuration hash. Keep the strongest
    # measurement for each strategy so repeated runs cannot crowd diverse
    # configurations out of the agent's limited memory context.
    unique_records: list[ExperienceRecord] = []
    seen_hashes: set[str] = set()
    for record in records:
        if record.config.config_hash in seen_hashes:
            continue
        unique_records.append(record)
        seen_hashes.add(record.config.config_hash)
        if len(unique_records) == limit:
            break
    return unique_records


def build_memory_context(
    experiences: list[ExperienceRecord],
    condition: MemoryCondition,
) -> str:
    """Render records according to the selected memory ablation.

    Args:
        experiences: Retrieved experience records to format.
        condition: Active memory condition determining depth of context.

    Returns:
        Formatted memory context string for LLM prompts.
    """

    # Return default string if no memory is available or allowed
    if condition == "no_memory" or not experiences:
        return "No previous experimental memory is available."

    sections: list[str] = []
    for record in experiences:
        lines = [
            f"Experiment ID: {record.experiment_id}",
            f"Hypothesis: {record.config.hypothesis}",
            f"Strategy: {record.config.reasoning_mode}",
            f"Roles: {', '.join(record.config.roles)}",
            f"Accuracy: {record.result.accuracy:.2f}",
            f"Fitness: {record.fitness_score:.2f}",
            "Total tokens: "
            f"{record.result.input_tokens + record.result.output_tokens}",
            f"Latency seconds: {record.result.latency_seconds:.2f}",
            f"Outcome: {record.retrieval_text}",
        ]
        # Include Critic diagnosis and recommendations only under full_memory condition
        if condition == "full_memory" and record.critic_review is not None:
            lines.extend(
                [
                    f"Critic diagnosis: {record.critic_review.diagnostic_note}",
                    "Recommendation: "
                    f"{record.critic_review.future_recommendation}",
                ],
            )
        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)
