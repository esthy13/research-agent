"""Execute an experiment configuration as a concrete reasoning protocol."""

import inspect
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentscope.message import Msg

from research_agent.interfaces import ExperimentConfig


AnswerExtractor = Callable[[str], str | None]


@dataclass(frozen=True)
class ExecutionOutcome:
    """Final answer and aggregate usage from one configured execution."""

    text: str
    input_tokens: int
    output_tokens: int
    model_calls: int


@dataclass(frozen=True)
class _CallOutcome:
    """Normalized text and usage returned by one model call."""

    text: str
    input_tokens: int
    output_tokens: int


def _response_text(response: object) -> str:
    """Return plain text from an AgentScope model response."""

    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
        else:
            text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _usage_value(usage: object | None, *names: str) -> int:
    """Read the first available token counter from a usage object."""

    if usage is None:
        return 0
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


class ConfigurationExecutor:
    """Run direct, voting, critique, or collaborative configurations.

    Attributes:
        config: ExperimentConfig specifying roles, order, and aggregation.
        model: AgentScope model instance.
        formatter: Message formatter.
        extract_answer: Function to parse model output into benchmark labels.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: Any,
        formatter: Any,
        extract_answer: AnswerExtractor,
    ) -> None:
        self.config = config
        self.model = model
        self.formatter = formatter
        self.extract_answer = extract_answer

    async def execute(self, task_prompt: str) -> ExecutionOutcome:
        """Execute the protocol selected by the configuration.

        Args:
            task_prompt: Rendered question prompt for the benchmark item.

        Returns:
            An ExecutionOutcome with final text, token usage, and call count.
        """

        if self.config.reasoning_mode == "critique_and_revision":
            return await self._execute_critique_and_revision(task_prompt)
        if self.config.aggregation == "majority_vote":
            return await self._execute_self_consistency(task_prompt)
        if self.config.aggregation == "final_revision":
            return await self._execute_final_revision(task_prompt)
        return await self._execute_direct(task_prompt)

    async def _call_model(self, role: str, content: str) -> _CallOutcome:
        """Make one stateless role-specific model call.

        Args:
            role: Assigned agent role label.
            content: User prompt content sent to the model.

        Returns:
            A _CallOutcome containing raw text and token usage.
        """

        messages = [
            Msg(
                name="ExperimentConfig",
                content=(
                    f"{self.config.system_prompt}\n\n"
                    f"You are acting in the role: {role}."
                ),
                role="system",
            ),
            Msg(name=role, content=content, role="user"),
        ]
        formatted_messages = self.formatter.format(messages)
        if inspect.isawaitable(formatted_messages):
            formatted_messages = await formatted_messages

        response = await self.model(formatted_messages)
        usage = getattr(response, "usage", None)
        return _CallOutcome(
            text=_response_text(response),
            input_tokens=_usage_value(usage, "input_tokens", "prompt_tokens"),
            output_tokens=_usage_value(
                usage,
                "output_tokens",
                "completion_tokens",
            ),
        )

    def _reasoning_instruction(self) -> str:
        """Return the instruction associated with the reasoning mode.

        Returns:
            Reasoning instruction string appended to prompts.
        """

        if self.config.reasoning_mode == "chain_of_thought":
            return (
                "Reason through the problem step by step and verify the result "
                "before giving the requested final answer."
            )
        return "Solve the task directly and give the requested final answer."

    async def _execute_direct(self, task_prompt: str) -> ExecutionOutcome:
        role = self.config.communication_order[0]
        call = await self._call_model(
            role,
            f"{task_prompt}\n\n{self._reasoning_instruction()}",
        )
        return self._combine(call.text, [call])

    async def _execute_self_consistency(
        self,
        task_prompt: str,
    ) -> ExecutionOutcome:
        attempts: list[_CallOutcome] = []
        answers: list[str] = []

        for role in self.config.communication_order:
            call = await self._call_model(
                role,
                (
                    f"{task_prompt}\n\n"
                    "Produce an independent solution without relying on any "
                    "other agent's attempt. "
                    f"{self._reasoning_instruction()}"
                ),
            )
            attempts.append(call)
            answer = self.extract_answer(call.text)
            if answer is not None:
                answers.append(answer)

        if not answers:
            final_text = attempts[-1].text if attempts else ""
        else:
            counts = Counter(answers)
            largest_count = max(counts.values())
            # Resolve ties reproducibly in favor of the earliest proposed answer.
            winner = next(
                answer for answer in answers if counts[answer] == largest_count
            )
            final_text = f"Final answer: {winner}"

        return self._combine(final_text, attempts)

    async def _execute_critique_and_revision(
        self,
        task_prompt: str,
    ) -> ExecutionOutcome:
        roles = self.config.communication_order
        calls: list[_CallOutcome] = []

        solution = await self._call_model(
            roles[0],
            (
                f"{task_prompt}\n\n"
                "Produce the initial candidate solution. "
                f"{self._reasoning_instruction()}"
            ),
        )
        calls.append(solution)

        critiques: list[tuple[str, str]] = []
        for role in roles[1:-1]:
            critique = await self._call_model(
                role,
                (
                    f"{task_prompt}\n\n"
                    "Review the candidate solution below. Identify concrete "
                    "reasoning or calculation errors and explain how they "
                    "should be corrected. Do not merely agree with it.\n\n"
                    f"CANDIDATE SOLUTION:\n{solution.text}"
                ),
            )
            calls.append(critique)
            critiques.append((role, critique.text))

        critique_text = self._render_contributions(critiques)
        revision = await self._call_model(
            roles[-1],
            (
                f"{task_prompt}\n\n"
                "Create the final revised solution using the candidate and "
                "the critiques. Check the reasoning yourself and follow the "
                "task's requested final-answer format.\n\n"
                f"CANDIDATE SOLUTION:\n{solution.text}\n\n"
                f"CRITIQUES:\n{critique_text}"
            ),
        )
        calls.append(revision)
        return self._combine(revision.text, calls)

    async def _execute_final_revision(
        self,
        task_prompt: str,
    ) -> ExecutionOutcome:
        calls: list[_CallOutcome] = []
        contributions: list[tuple[str, str]] = []
        roles = self.config.communication_order

        for role in roles[:-1]:
            previous = self._render_contributions(contributions)
            previous_section = (
                f"\n\nPREVIOUS CONTRIBUTIONS:\n{previous}"
                if previous
                else ""
            )
            contribution = await self._call_model(
                role,
                (
                    f"{task_prompt}\n\n"
                    "Contribute a solution from your assigned role. Examine "
                    "any previous contributions, correct errors you notice, "
                    "and make your reasoning useful to the final synthesizer. "
                    f"{self._reasoning_instruction()}"
                    f"{previous_section}"
                ),
            )
            calls.append(contribution)
            contributions.append((role, contribution.text))

        synthesis = await self._call_model(
            roles[-1],
            (
                f"{task_prompt}\n\n"
                "Synthesize the role contributions into one final solution. "
                "Resolve disagreements, verify the answer independently, and "
                "follow the task's requested final-answer format.\n\n"
                "ROLE CONTRIBUTIONS:\n"
                f"{self._render_contributions(contributions)}"
            ),
        )
        calls.append(synthesis)
        return self._combine(synthesis.text, calls)

    @staticmethod
    def _render_contributions(contributions: list[tuple[str, str]]) -> str:
        """Render attributed agent outputs for a later model call."""

        return "\n\n".join(
            f"[{role}]\n{text}" for role, text in contributions
        )

    @staticmethod
    def _combine(
        final_text: str,
        calls: list[_CallOutcome],
    ) -> ExecutionOutcome:
        """Combine usage across all calls in an execution."""

        return ExecutionOutcome(
            text=final_text,
            input_tokens=sum(call.input_tokens for call in calls),
            output_tokens=sum(call.output_tokens for call in calls),
            model_calls=len(calls),
        )
