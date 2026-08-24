"""Shared data contracts for candidate generation, evaluation, and memory."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryCondition = Literal[
    "full_memory",
    "memory_without_augmentation",
    "no_memory",
]
SearchMode = Literal["exploration", "exploitation"]
EvaluationStatus = Literal["success", "failed", "timeout"]
BenchmarkSplit = Literal["development", "test", "transfer"]


class StrictModel(BaseModel):
    """Base model that rejects fields outside the documented contract.

    All data contracts inherit from this class so that unknown fields
    are rejected at validation time rather than silently accepted.
    """

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictModel):
    """Executable experiment specification produced by the search system.

    Attributes:
        hypothesis: The Ideator's proposed idea.
        rationale: Motivation behind the proposed idea.
        system_prompt: Shared system prompt for all role calls.
        reasoning_mode: One of direct, chain_of_thought,
            critique_and_revision, or self_consistency.
        agent_count: Number of stateless model calls (1 to 5).
        roles: Unique role labels proposed by the Ideator.
        communication_order: Execution order of the roles.
        aggregation: How role outputs become one scored response.
        temperature: Sampling temperature for the experiment model.
        max_tokens: Output-token cap per call (100 to 4000).
        seed: Reproducibility and benchmark-sampling seed.
        condition: Active memory-ablation condition.
        parent_ids: Experiment IDs used as parents during exploitation.
        search_mode: Controller-selected exploration or exploitation.
        config_hash: Deterministic SHA-256 identity of executable settings.
        model_id: Fixed model deployment identifier.
    """

    hypothesis: str # ideator's idea
    rationale: str # motivation behind the proposed idea
    system_prompt: str
    reasoning_mode: str
    agent_count: int
    roles: list[str]
    communication_order: list[str]
    aggregation: str
    temperature: float
    max_tokens: int
    seed: int
    condition: MemoryCondition
    parent_ids: list[str] = Field(default_factory=list) # links to early experiments which influenced the proposal
    search_mode: SearchMode # exploration or exploitation
    config_hash: str = "" # unique identifier to recognize duplicate configs
    model_id: str


class EvaluationResult(StrictModel):
    """Aggregate measurements returned by an experiment evaluator.

    Attributes:
        config_hash: Hash linking this result to its configuration.
        benchmark: Name of the evaluated benchmark.
        split: Dataset partition used (development, test, or transfer).
        seed: Evaluation seed controlling item sampling order.
        accuracy: Fraction of correctly answered items in [0, 1].
        input_tokens: Total input tokens across all model calls.
        output_tokens: Total output tokens across all model calls.
        latency_seconds: Wall-clock evaluation time in seconds.
        status: Execution outcome (success, failed, or timeout).
        errors: Per-item error messages collected during evaluation.
    """

    config_hash: str
    benchmark: str
    split: BenchmarkSplit
    seed: int
    accuracy: float = Field(ge=0.0, le=1.0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0.0)
    status: EvaluationStatus
    errors: list[str] = Field(default_factory=list)


class CriticReview(StrictModel):
    """Evidence-grounded interpretation of one evaluation result.

    Attributes:
        evaluation_summary: Brief summary of the measured outcome.
        failure_category: Optional classification of the failure type.
        diagnostic_note: Detailed analysis of what happened and why.
        future_recommendation: One concrete follow-up change to try.
    """

    evaluation_summary: str
    failure_category: str | None = None
    diagnostic_note: str
    future_recommendation: str


class ExperienceRecord(StrictModel):
    """Persistent trajectory stored in the experience library.

    Attributes:
        cycle: Zero-based cycle index within the search trajectory.
        experiment_id: Unique identifier for this experiment run.
        config: The full configuration that was executed.
        result: Aggregate measurements from the evaluation.
        fitness_score: Numeric fitness derived from the result.
        critic_review: Optional structured diagnosis from the Critic.
        retrieval_text: Compact text summary used for memory retrieval.
        created_at: UTC timestamp when the record was created.
    """

    cycle: int = Field(default=0, ge=0)
    experiment_id: str
    config: ExperimentConfig
    result: EvaluationResult
    fitness_score: float
    critic_review: CriticReview | None = None
    retrieval_text: str
    created_at: datetime
