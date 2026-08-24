"""Semantic validation for configurations accepted by the evaluator."""

from research_agent.interfaces import ExperimentConfig
from research_agent.search.hashing import calculate_config_hash


ALLOWED_REASONING_MODES = {
    "direct",
    "chain_of_thought",
    "critique_and_revision",
    "self_consistency",
}

ALLOWED_AGGREGATIONS = {
    "single_answer",
    "final_revision",
    "majority_vote",
}


class ConfigValidationError(ValueError):
    """Raised when a configuration cannot be executed by the runner."""


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate cross-field constraints and hash integrity.

    Args:
        config: ExperimentConfig instance to validate.

    Returns:
        The validated ExperimentConfig instance.

    Raises:
        ConfigValidationError: If any executable or semantic constraint is violated.
    """

    errors: list[str] = []
    # Verify agent count is within allowed bounds
    if not 1 <= config.agent_count <= 5:
        errors.append("agent_count must be between 1 and 5")
    # Verify number of role definitions matches agent count
    if len(config.roles) != config.agent_count:
        errors.append("the number of roles must match agent_count")
    # Verify role names are distinct
    if len(set(config.roles)) != len(config.roles):
        errors.append("agent roles must be unique")
    if len(config.communication_order) != config.agent_count:
        errors.append("communication_order must include every agent once")
    if set(config.communication_order) != set(config.roles):
        errors.append(
            "communication_order must contain exactly the defined roles",
        )
    if config.reasoning_mode not in ALLOWED_REASONING_MODES:
        errors.append(f"unsupported reasoning mode: {config.reasoning_mode}")
    if config.aggregation not in ALLOWED_AGGREGATIONS:
        errors.append(f"unsupported aggregation: {config.aggregation}")
    if config.aggregation == "majority_vote" and config.agent_count < 3:
        errors.append("majority_vote requires at least three agents")
    if config.aggregation == "final_revision" and config.agent_count < 2:
        errors.append("final_revision requires at least two agents")
    if config.aggregation == "single_answer" and config.agent_count != 1:
        errors.append("single_answer requires exactly one agent")
    if (
        config.reasoning_mode == "direct"
        and config.aggregation != "single_answer"
    ):
        errors.append("direct reasoning requires single_answer aggregation")
    if (
        config.reasoning_mode == "self_consistency"
        and config.aggregation != "majority_vote"
    ):
        errors.append("self_consistency requires majority_vote aggregation")
    if config.reasoning_mode == "critique_and_revision":
        if config.aggregation != "final_revision":
            errors.append(
                "critique_and_revision requires final_revision aggregation",
            )
        if config.agent_count < 3:
            errors.append(
                "critique_and_revision requires at least three agents",
            )
    if not 0.0 <= config.temperature <= 2.0:
        errors.append("temperature must be between 0 and 2")
    if not 100 <= config.max_tokens <= 4000:
        errors.append("max_tokens must be between 100 and 4000")
    if not config.system_prompt.strip():
        errors.append("system_prompt cannot be empty")
    if not config.model_id.strip():
        errors.append("model_id cannot be empty")
    if config.config_hash != calculate_config_hash(config):
        errors.append("config_hash is missing or incorrect")
    if errors:
        raise ConfigValidationError("; ".join(errors))
    return config
