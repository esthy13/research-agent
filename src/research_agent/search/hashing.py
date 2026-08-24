"""Deterministic fingerprints for executable candidate configurations."""

import hashlib
import json

from research_agent.interfaces import ExperimentConfig


# Metadata is intentionally excluded so independent repetitions share a strategy ID.
EXECUTION_FIELDS = {
    "model_id",
    "system_prompt",
    "reasoning_mode",
    "agent_count",
    "roles",
    "communication_order",
    "aggregation",
    "temperature",
    "max_tokens",
}


def calculate_config_hash(config: ExperimentConfig) -> str:
    """Hash the canonical JSON representation of executable settings.

    Args:
        config: The experiment configuration to hash.

    Returns:
        A SHA-256 hex string identifying the executable configuration.
    """

    executable_config = config.model_dump(include=EXECUTION_FIELDS)
    canonical_json = json.dumps(
        executable_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def attach_config_hash(config: ExperimentConfig) -> ExperimentConfig:
    """Return a copy carrying its deterministic configuration hash.

    Args:
        config: The experiment configuration without a hash.

    Returns:
        A copy of the configuration with config_hash set.
    """

    return config.model_copy(
        update={"config_hash": calculate_config_hash(config)},
    )
