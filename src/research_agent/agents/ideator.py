"""AgentScope wrapper for memory-guided candidate generation."""

from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit

from research_agent.agents.prompts import (
    IDEATOR_SYSTEM_PROMPT,
    build_ideator_input,
)
from research_agent.interfaces import (
    ExperimentConfig,
    MemoryCondition,
    SearchMode,
)
from research_agent.search.hashing import attach_config_hash


class IdeatorAgent:
    """Generate one structured configuration under controller constraints.

    Wraps an AgentScope ReActAgent that produces a typed
    ExperimentConfig. Controller-owned fields (model_id, seed,
    condition, search_mode, parent_ids) are overwritten after
    generation to prevent the LLM from altering fixed controls.
    """

    def __init__(
        self,
        model: ChatModelBase,
        formatter: FormatterBase,
    ) -> None:
        self.model = model
        self.formatter = formatter

    async def propose(
        self,
        memory_context: str,
        model_id: str,
        seed: int,
        condition: MemoryCondition,
        search_mode: SearchMode,
        parent_ids: list[str],
        benchmark: str = "unspecified",
        baseline_context: str = "No fixed baseline measurements were supplied.",
    ) -> ExperimentConfig:
        """Return a normalized, hashed candidate configuration.

        Args:
            memory_context: Rendered text of prior experimental memory.
            model_id: Fixed model deployment for this experiment.
            seed: Reproducibility seed for the search run.
            condition: Memory-ablation condition in effect.
            search_mode: Whether to explore or exploit.
            parent_ids: Experiment IDs selected as parents.
            benchmark: Name of the target benchmark.
            baseline_context: Fixed baseline evidence string.

        Returns:
            A validated ExperimentConfig with the config_hash attached.

        Raises:
            RuntimeError: If the LLM does not return structured output.
        """

        agent = ReActAgent(
            name="Ideator",
            sys_prompt=IDEATOR_SYSTEM_PROMPT,
            model=self.model,
            formatter=self.formatter,
            memory=InMemoryMemory(),
            max_iters=5,
        )
        message = Msg(
            name="SearchController",
            content=build_ideator_input(
                memory_context=memory_context,
                model_id=model_id,
                seed=seed,
                condition=condition,
                search_mode=search_mode,
                parent_ids=parent_ids,
                benchmark=benchmark,
                baseline_context=baseline_context,
            ),
            role="user",
        )
        response = await agent(message, structured_model=ExperimentConfig)

        if response.metadata is None:
            raise RuntimeError(
                "The Ideator did not return a structured configuration.",
            )

        proposed_config = ExperimentConfig.model_validate(response.metadata)
        normalized_config = proposed_config.model_copy(
            update={
                "model_id": model_id,
                "seed": seed,
                "condition": condition,
                "search_mode": search_mode,
                "parent_ids": parent_ids,
                "config_hash": "",
            },
        )
        return attach_config_hash(normalized_config)
