"""AgentScope wrapper for structured experiment diagnosis."""

from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit

from research_agent.agents.prompts import (
    CRITIC_SYSTEM_PROMPT,
    build_critic_input,
)
from research_agent.interfaces import (
    CriticReview,
    EvaluationResult,
    ExperimentConfig,
)


class CriticAgent:
    """Produce a validated diagnosis for one completed experiment.

    Uses a low-temperature LLM call to interpret measured performance,
    identify failure modes without modifying scores, and propose a concrete
    executable follow-up recommendation.
    """

    def __init__(
        self,
        model: ChatModelBase,
        formatter: FormatterBase,
    ) -> None:
        self.model = model
        self.formatter = formatter

    async def review(
        self,
        config: ExperimentConfig,
        result: EvaluationResult,
        baseline_context: str = "No fixed baseline measurements were supplied.",
    ) -> CriticReview:
        """Review measured evidence without retaining cross-run chat state.

        Args:
            config: The executed experiment configuration.
            result: Measured performance result containing accuracy, tokens, etc.
            baseline_context: Fixed baseline evidence provided as context.

        Returns:
            A structured CriticReview instance with diagnosis and recommendation.

        Raises:
            RuntimeError: If the LLM does not return a valid structured review.
        """

        agent = ReActAgent(
            name="Critic",
            sys_prompt=CRITIC_SYSTEM_PROMPT,
            model=self.model,
            formatter=self.formatter,
            memory=InMemoryMemory(),
            max_iters=5,
        )
        message = Msg(
            name="Experiment",
            content=build_critic_input(
                config,
                result,
                baseline_context=baseline_context,
            ),
            role="user",
        )
        response = await agent(message, structured_model=CriticReview)

        if response.metadata is None:
            raise RuntimeError(
                "The Critic did not return a structured review.",
            )
        return CriticReview.model_validate(response.metadata)
