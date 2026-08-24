"""Model factories for the OpenAI-compatible course gateway."""

import os
from pathlib import Path

from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from dotenv import load_dotenv

from research_agent.interfaces import ExperimentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _create_model(
    temperature: float,
    max_tokens: int,
    model_name: str | None = None,
) -> tuple[OpenAIChatModel, OpenAIChatFormatter]:
    """Create a model and matching formatter from environment settings.

    Args:
        temperature: Sampling temperature for generation.
        max_tokens: Maximum output tokens per completion.
        model_name: Override for the MODEL_NAME env variable.
            Defaults to the value in the environment.

    Returns:
        A tuple of the configured chat model and its formatter.
    """

    model = OpenAIChatModel(
        model_name=model_name or os.environ["MODEL_NAME"],
        api_key=os.environ["MODEL_API_KEY"],
        stream=False,
        client_kwargs={"base_url": os.environ["BASE_URL"]},
        generate_kwargs={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    return model, OpenAIChatFormatter()


def create_critic_model() -> tuple[OpenAIChatModel, OpenAIChatFormatter]:
    """Create the low-temperature model used for result diagnosis.

    Returns:
        A tuple of the Critic model (temperature 0.1) and its formatter.
    """

    return _create_model(temperature=0.1, max_tokens=1000)


def create_ideator_model() -> tuple[OpenAIChatModel, OpenAIChatFormatter]:
    """Create the higher-temperature model used for candidate generation.

    Returns:
        A tuple of the Ideator model (temperature 0.7) and its formatter.
    """

    return _create_model(temperature=0.7, max_tokens=1500)


def create_experiment_model(
    config: ExperimentConfig,
) -> tuple[OpenAIChatModel, OpenAIChatFormatter]:
    """Create the model used to execute one experiment configuration.

    Args:
        config: The experiment specification providing temperature,
            max_tokens, and model_id.

    Returns:
        A tuple of the experiment model and its formatter.
    """

    return _create_model(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        model_name=config.model_id,
    )
