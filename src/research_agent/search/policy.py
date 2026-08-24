"""Reproducible exploration/exploitation scheduling."""

import random

from pydantic import BaseModel, Field

from research_agent.interfaces import ExperienceRecord, SearchMode


class SearchDecision(BaseModel):
    """Controller decision for one candidate-generation cycle."""

    mode: SearchMode
    parent_ids: list[str] = Field(default_factory=list)


def exploration_probability(
    cycle: int,
    total_cycles: int,
    start: float = 0.8,
    end: float = 0.3,
) -> float:
    """Linearly anneal exploration from ``start`` to ``end``.

    Args:
        cycle: Current cycle index (0-based).
        total_cycles: Total number of cycles in search schedule.
        start: Initial exploration probability.
        end: Terminal exploration probability.

    Returns:
        The annealed exploration probability float in [end, start].
    """

    if total_cycles <= 1:
        return end
    progress = min(max(cycle / (total_cycles - 1), 0.0), 1.0)
    return start + (end - start) * progress


def choose_search_decision(
    cycle: int,
    total_cycles: int,
    seed: int,
    experiences: list[ExperienceRecord],
) -> SearchDecision:
    """Choose a seeded mode and the strongest parent when exploiting.

    Args:
        cycle: Current cycle index.
        total_cycles: Total cycles in run.
        seed: Random seed for decision reproducibility.
        experiences: Prior experience records available for exploitation.

    Returns:
        A SearchDecision specifying mode and parent IDs.
    """

    if not experiences:
        return SearchDecision(mode="exploration")

    probability = exploration_probability(cycle, total_cycles)
    rng = random.Random(f"{seed}:{cycle}")
    if rng.random() < probability:
        return SearchDecision(mode="exploration")

    return SearchDecision(
        mode="exploitation",
        parent_ids=[experiences[0].experiment_id],
    )
