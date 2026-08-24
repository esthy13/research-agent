"""Prompt templates for the Ideator and Critic agents."""

from research_agent.interfaces import (
    EvaluationResult,
    ExperimentConfig,
    MemoryCondition,
    SearchMode,
)


CRITIC_SYSTEM_PROMPT = """
You are the Critic in an automated experimentation system.

Analyze an experiment using only its configuration and measured result.
Summarize the outcome, classify failures when appropriate, distinguish
evidence from hypotheses, and recommend one concrete follow-up change.

Do not invent measurements or claim that an untested change will work.
Recommend follow-ups only through executable ExperimentConfig fields. There is
no numerical role weighting, dynamic role rotation, tool use, model training,
or fine-tuning. Do not claim that prompt wording implements those mechanisms.
""".strip()


IDEATOR_SYSTEM_PROMPT = """
You are the Ideator in an automated experimentation system.

Propose one executable prompting or multi-agent configuration for a
reasoning benchmark. Use supplied experimental memory when available.
State a testable hypothesis and rationale, respect all fixed controls,
and change at least one executable setting when exploiting a parent.
Leave config_hash empty because the controller calculates it.

Do not invent results or claim that the proposal will certainly improve.
Propose only mechanisms implemented by the supplied execution contract. Do not
invent numerical role weighting, dynamic role rotation, tool use, model
training, fine-tuning, or fields that are absent from ExperimentConfig.
""".strip()


BENCHMARK_CONTEXTS = {
    "pubmedqa": (
        "PubMedQA biomedical question answering. Each item provides scientific "
        "abstract context and a research question. The solver must infer one "
        "final label: yes, no, or maybe. It must reason from the supplied "
        "evidence and handle uncertainty without adding outside facts."
    ),
    "mmlu_college_chemistry": (
        "MMLU college chemistry multiple-choice reasoning. Select exactly one "
        "answer label A, B, C, or D."
    ),
    "mmlu_college_physics": (
        "MMLU college physics multiple-choice reasoning. Select exactly one "
        "answer label A, B, C, or D."
    ),
    "gsm8k": (
        "GSM8K grade-school mathematical word problems. Return the final "
        "numeric answer in the required format."
    ),
    "bbh": (
        "BBH boolean-expression reasoning. Return exactly true or false in "
        "the required format."
    ),
}


EXECUTION_CONTRACT = """
- direct + single_answer: one role makes one model call.
- chain_of_thought: adds an explicit step-by-step verification instruction.
- self_consistency + majority_vote: at least three roles make independent
  calls; the executor parses their answers and returns the majority label.
- critique_and_revision + final_revision: the first role proposes a solution,
  intermediate roles critique it, and the last role produces the final answer.
- Other final_revision configurations: roles contribute in the fixed
  communication_order and the final role synthesizes their text.

The system_prompt is shared by every role. Role names only label calls, and
communication_order only fixes call order. There are no numerical role weights,
dynamic role rotation, learned parameters, tools, training, or fine-tuning.
Only propose hypotheses that the fields in ExperimentConfig actually execute.
""".strip()


NO_BASELINE_CONTEXT = "No fixed baseline measurements were supplied."


def benchmark_context(benchmark: str) -> str:
    """Return a concise task description for candidate generation.

    Args:
        benchmark: Name of the benchmark.

    Returns:
        Concise task context description string.
    """

    return BENCHMARK_CONTEXTS.get(
        benchmark,
        f"Reasoning benchmark named {benchmark}.",
    )


def build_critic_input(
    config: ExperimentConfig,
    result: EvaluationResult,
    baseline_context: str = NO_BASELINE_CONTEXT,
) -> str:
    """Render measured evidence for structured Critic review.

    Args:
        config: The executed experiment configuration.
        result: Measured outcome containing accuracy, tokens, and errors.
        baseline_context: Shared baseline reference text.

    Returns:
        Formatted prompt string for the Critic agent.
    """

    return f"""
Analyze the following experiment.

TARGET BENCHMARK:
{result.benchmark}

BENCHMARK TASK:
{benchmark_context(result.benchmark)}

FIXED BASELINE EVIDENCE:
{baseline_context}

EXECUTION CONTRACT:
{EXECUTION_CONTRACT}

EXPERIMENT CONFIGURATION:
{config.model_dump_json(indent=2)}

MEASURED RESULT:
{result.model_dump_json(indent=2)}

Return a structured CriticReview.
""".strip()


def build_ideator_input(
    memory_context: str,
    model_id: str,
    seed: int,
    condition: MemoryCondition,
    search_mode: SearchMode,
    parent_ids: list[str],
    benchmark: str = "unspecified",
    baseline_context: str = NO_BASELINE_CONTEXT,
) -> str:
    """Render fixed controls and retrieved evidence for the Ideator.

    Args:
        memory_context: Retrieved prior experimental trajectories.
        model_id: Model deployment identifier.
        seed: Random seed for search run.
        condition: Active memory ablation condition.
        search_mode: Exploration or exploitation.
        parent_ids: Parent experiment IDs for exploitation.
        benchmark: Target benchmark name.
        baseline_context: Baseline evidence text.

    Returns:
        Formatted prompt string for the Ideator agent.
    """

    parent_text = ", ".join(parent_ids) or "none"
    if search_mode == "exploration":
        mode_instruction = (
            "Propose a substantially new strategy. Do not copy a previous "
            "configuration, and return an empty parent_ids list."
        )
    else:
        mode_instruction = (
            f"Improve parent experiment(s) {parent_text}. Change at least "
            "one executable setting and preserve exactly these parent IDs."
        )

    return f"""
Propose one new experiment.

TARGET BENCHMARK:
{benchmark}

BENCHMARK TASK:
{benchmark_context(benchmark)}

OPTIMIZATION OBJECTIVE:
Maximize development accuracy. When accuracy is comparable, prefer fewer total
tokens and lower latency. The final answer must follow the benchmark format.

FIXED BASELINE EVIDENCE:
{baseline_context}

The baseline evidence is a shared experimental control, not retrieved memory.
It is identical under every memory-ablation condition.
Do not simply reproduce a fixed baseline; propose an executable change whose
hypothesis explains why it could improve this benchmark.

FIXED CONTROLS:
- Model: {model_id}
- Seed: {seed}
- Memory condition: {condition}
- Search mode: {search_mode}
- Parent IDs: {parent_text}

SEARCH INSTRUCTION:
{mode_instruction}

ALLOWED REASONING MODES:
- direct
- chain_of_thought
- critique_and_revision
- self_consistency

ALLOWED AGGREGATION METHODS:
- single_answer
- final_revision
- majority_vote

EXECUTION CONTRACT:
{EXECUTION_CONTRACT}

EXECUTION CONSTRAINTS:
- direct with single_answer requires exactly one agent
- single_answer configurations require exactly one agent
- self_consistency requires majority_vote and at least three agents
- critique_and_revision requires final_revision and at least three agents
- final_revision requires at least two agents
- majority_vote requires at least three agents
- agent_count must be between one and five
- max_tokens must be between 100 and 4000
- the hypothesis must describe a change that these fields truly execute

PREVIOUS EXPERIMENTAL MEMORY:
{memory_context}

Return one structured ExperimentConfig with an empty config_hash.
""".strip()
