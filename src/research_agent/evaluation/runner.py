"""Execute configurations on supported reasoning benchmarks."""

import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from research_agent.evaluation.executor import ConfigurationExecutor
from research_agent.interfaces import (
    BenchmarkSplit,
    EvaluationResult,
    ExperimentConfig,
)
from research_agent.model_factory import create_experiment_model
from research_agent.search.validation import validate_config

try:
    from datasets import Dataset, load_dataset
except ModuleNotFoundError as error:
    if error.name != "datasets":
        raise

    Dataset = object

    def load_dataset(*args: object, **kwargs: object) -> object:
        """Explain how to enable real benchmark loading."""

        raise ModuleNotFoundError(
            "The real evaluator requires the Hugging Face `datasets` package. "
            "Install the project dependencies with `uv pip install -e .` or "
            "install it directly with `uv pip install datasets`."
        ) from error


MAX_EXAMPLES = 100
GSM8K_DATASET = "openai/gsm8k"
BBH_DATASET = "Joschka/big_bench_hard"
BBH_TASK = "boolean_expressions"
PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
MMLU_DATASET = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
DATASET_SPLIT_SEED = 2026


def _extract_multiple_choice_answer(text: str) -> str | None:
    """Extract an A, B, C, or D answer from a multiple-choice response."""

    patterns = [
        r"final\s+answer\s*:?\s*[\(\[]?([A-D])[\)\]]?(?![A-Za-z])",
        r"(?:answer|option|choice)\s+(?:is\s+)?"
        r"[\(\[]?([A-D])[\)\]]?(?![A-Za-z])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    stripped = text.strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped

    parenthesized = re.findall(
        r"\(([A-D])\)",
        text,
        flags=re.IGNORECASE,
    )
    if parenthesized:
        return parenthesized[-1].upper()

    return None

def _extract_pubmedqa_answer(text: str) -> str | None:
    """Extract a yes, no, or maybe answer from a PubMedQA response."""

    final_answer_match = re.search(
        r"final\s+answer\s*:?\s*(yes|no|maybe)\b",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        return final_answer_match.group(1).lower()

    matches = re.findall(
        r"\b(yes|no|maybe)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None

    return matches[-1].lower()

def _pubmedqa_gold_answer(example: dict) -> str | None:
    """Return the normalized PubMedQA reference answer."""

    answer = str(example["final_decision"]).strip().lower()

    if answer not in {"yes", "no", "maybe"}:
        return None

    return answer

def _mmlu_gold_answer(example: dict) -> str | None:
    """Convert the MMLU answer index to its A-D label."""

    labels = "ABCD"
    answer = example["answer"]

    if isinstance(answer, int) and 0 <= answer < len(labels):
        return labels[answer]

    normalized = str(answer).strip().upper()

    if normalized in labels:
        return normalized

    if normalized.isdigit():
        index = int(normalized)
        if 0 <= index < len(labels):
            return labels[index]

    return None

def _mmlu_question_text(example: dict) -> str:
    """Render an MMLU question with its four answer choices."""

    labels = "ABCD"
    choices = list(example["choices"])

    if len(choices) != len(labels):
        raise ValueError("MMLU example must contain exactly four choices")

    rendered_choices = "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(labels, choices)
    )

    return (
        f"Question:\n{example['question']}\n\n"
        f"Answer choices:\n{rendered_choices}"
    )

def _load_mmlu_subject_split(
    subject: str,
    split: BenchmarkSplit,
) -> Dataset:
    """Create a deduplicated MMLU partition or full transfer collection."""

    # Use only the official test collection. The original dev and validation
    # splits are too small and contain overlaps with other splits.
    ds = load_dataset(
        MMLU_DATASET,
        subject,
        split="test",
        revision=MMLU_REVISION,
    )
    if "_source_index" not in ds.column_names:
        ds = ds.add_column("_source_index", list(range(len(ds))))

    seen_rows: set[tuple] = set()
    groups: dict[str, list[int]] = {}

    for index in range(len(ds)):
        example = ds[index]

        row_key = (
            str(example["question"]).strip(),
            tuple(str(choice).strip() for choice in example["choices"]),
            int(example["answer"]),
        )

        # Remove exact duplicate rows.
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)

        # Keep variants of the same question stem in the same partition.
        normalized_stem = " ".join(
            str(example["question"]).split(),
        ).casefold()

        groups.setdefault(normalized_stem, []).append(index)

    unique_indices = [
        index
        for group_indices in groups.values()
        for index in group_indices
    ]
    if split == "transfer":
        return ds.select(sorted(unique_indices))

    group_names = list(groups)
    random.Random(DATASET_SPLIT_SEED).shuffle(group_names)

    midpoint = len(group_names) // 2

    if split == "development":
        selected_groups = group_names[:midpoint]
    elif split == "test":
        selected_groups = group_names[midpoint:]
    else:
        raise ValueError(f"Unsupported split: {split}")

    selected_indices = [
        index
        for group_name in selected_groups
        for index in groups[group_name]
    ]

    return ds.select(selected_indices)


def _load_mmlu_physics_split(split: BenchmarkSplit) -> Dataset:
    return _load_mmlu_subject_split("college_physics", split)


def _load_mmlu_chemistry_split(split: BenchmarkSplit) -> Dataset:
    return _load_mmlu_subject_split("college_chemistry", split)

def _pubmedqa_question_text(example: dict) -> str:
    """Render the scientific context and question without leaking the answer."""

    context = example["context"]
    context_passages = context.get("contexts", [])
    rendered_context = "\n\n".join(
        str(passage) for passage in context_passages
    )

    return (
        f"Scientific context:\n{rendered_context}\n\n"
        f"Question:\n{example['question']}"
    )

def _normalize_numeric_answer(answer: str) -> str:
    """Normalize equivalent numeric strings for exact comparison."""

    try:
        normalized = format(Decimal(answer).normalize(), "f")
    except InvalidOperation:
        return answer

    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        return "0"
    return normalized


def _extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style response."""

    final_answer_match = re.search(
        r"final\s+answer\s*:?\s*(-?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        answer = final_answer_match.group(1).replace(",", "")
        return _normalize_numeric_answer(answer)

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    return _normalize_numeric_answer(numbers[-1].replace(",", ""))


def _extract_boolean_answer(text: str) -> str | None:
    """Extract a true/false answer from a BBH boolean response."""

    final_answer_match = re.search(
        r"final\s+answer\s*:?\s*(true|false)\b",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        return final_answer_match.group(1).lower()

    matches = re.findall(r"\b(true|false)\b", text, flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].lower()


def _gsm8k_gold_answer(example: dict) -> str | None:
    """Extract the gold final answer from a GSM8K example."""

    return _extract_numeric_answer(str(example["answer"]).split("####")[-1])


def _bbh_gold_answer(example: dict) -> str | None:
    """Extract the gold final answer from a BBH boolean example."""

    return _extract_boolean_answer(str(_example_value(example, ("target", "answer"))))


def _example_value(example: dict, keys: Sequence[str]) -> object:
    """Return the first present field from an example."""

    for key in keys:
        if key in example:
            return example[key]
    raise KeyError(f"Example is missing all expected fields: {', '.join(keys)}")


def _select_fraction(ds: Dataset, start: float, end: float) -> Dataset:
    """Select a deterministic contiguous fraction of a dataset."""

    start_index = int(len(ds) * start)
    end_index = int(len(ds) * end)
    if end_index <= start_index and len(ds) > 0:
        end_index = min(start_index + 1, len(ds))
    return ds.select(range(start_index, end_index))


@dataclass(frozen=True)
class BenchmarkSpec:
    """Benchmark-specific adapters used by the generic evaluation loop."""

    name: str
    prompt_name: str
    answer_instruction: str
    question_fields: tuple[str, ...]
    load_split: Callable[[BenchmarkSplit], Dataset]
    extract_model_answer: Callable[[str], str | None]
    extract_gold_answer: Callable[[dict], str | None]
    question_formatter: Callable[[dict], str] | None = None

    def question_text(self, example: dict) -> str:
        if self.question_formatter is not None:
            return self.question_formatter(example)

        return str(_example_value(example, self.question_fields))


def _load_gsm8k_split(split: BenchmarkSplit) -> Dataset:
    if split == "development":
        return load_dataset(GSM8K_DATASET, "main", split="train")
    if split == "test":
        return load_dataset(GSM8K_DATASET, "main", split="test")
    raise ValueError(f"Unsupported split: {split}")


def _load_bbh_boolean_split(split: BenchmarkSplit) -> Dataset:
    ds = load_dataset(BBH_DATASET, BBH_TASK, split=BBH_TASK)
    if split == "development":
        return _select_fraction(ds, 0.0, 0.8)
    if split == "test":
        return _select_fraction(ds, 0.8, 1.0)
    raise ValueError(f"Unsupported split: {split}")

def _load_pubmedqa_split(split: BenchmarkSplit) -> Dataset:
    """Load a fixed 500/500 development-test partition."""

    ds = load_dataset(
        PUBMEDQA_DATASET,
        PUBMEDQA_CONFIG,
        split="train",
    )
    ds = ds.shuffle(seed=DATASET_SPLIT_SEED)

    midpoint = len(ds) // 2

    if split == "development":
        return ds.select(range(0, midpoint))

    if split == "test":
        return ds.select(range(midpoint, len(ds)))

    raise ValueError(f"Unsupported split: {split}")

BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        name="gsm8k",
        prompt_name="GSM8K math problem",
        answer_instruction=(
            "Return the final numeric result on a line starting with "
            "'Final answer:'."
        ),
        question_fields=("question",),
        load_split=_load_gsm8k_split,
        extract_model_answer=_extract_numeric_answer,
        extract_gold_answer=_gsm8k_gold_answer,
    ),
    "bbh": BenchmarkSpec(
        name="bbh",
        prompt_name="BBH boolean expressions problem",
        answer_instruction=(
            "Return either true or false on a line starting with "
            "'Final answer:'."
        ),
        question_fields=("input", "question"),
        load_split=_load_bbh_boolean_split,
        extract_model_answer=_extract_boolean_answer,
        extract_gold_answer=_bbh_gold_answer,
    ),

    "pubmedqa": BenchmarkSpec(
    name="pubmedqa",
    prompt_name="PubMedQA biomedical reasoning problem",
    answer_instruction=(
        "Return yes, no, or maybe on a line starting with "
        "'Final answer:'."
    ),
    question_fields=(),
    load_split=_load_pubmedqa_split,
    extract_model_answer=_extract_pubmedqa_answer,
    extract_gold_answer=_pubmedqa_gold_answer,
    question_formatter=_pubmedqa_question_text,
    ),

    "mmlu_college_physics": BenchmarkSpec(
        name="mmlu_college_physics",
        prompt_name="MMLU college physics problem",
        answer_instruction=(
            "Reason concisely, using no more than 200 words. "
            "End the response with exactly one line in the form "
            "'Final answer: X', where X is A, B, C, or D."
        ),
        question_fields=(),
        load_split=_load_mmlu_physics_split,
        extract_model_answer=_extract_multiple_choice_answer,
        extract_gold_answer=_mmlu_gold_answer,
        question_formatter=_mmlu_question_text,
    ),
    "mmlu_college_chemistry": BenchmarkSpec(
        name="mmlu_college_chemistry",
        prompt_name="MMLU college chemistry problem",
        answer_instruction=(
            "Reason concisely, using no more than 200 words. "
            "End the response with exactly one line in the form "
            "'Final answer: X', where X is A, B, C, or D."
        ),
        question_fields=(),
        load_split=_load_mmlu_chemistry_split,
        extract_model_answer=_extract_multiple_choice_answer,
        extract_gold_answer=_mmlu_gold_answer,
        question_formatter=_mmlu_question_text,
    ),

}
SUPPORTED_BENCHMARKS: list[str] = list(BENCHMARKS)


def _benchmark_spec(benchmark: str) -> BenchmarkSpec:
    normalized = benchmark.lower()
    if normalized == "gs8mk":
        normalized = "gsm8k"
    if normalized == "bbh:boolean_expressions":
        normalized = "bbh"
    try:
        return BENCHMARKS[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(BENCHMARKS))
        raise ValueError(
            f"Unsupported benchmark: {benchmark}. Supported: {supported}",
        ) from error


def _build_solver_prompt(
    benchmark_spec: BenchmarkSpec,
    question: str,
) -> str:
    """Build the user prompt for one benchmark item."""

    instructions = [
        f"Solve this {benchmark_spec.prompt_name}.",
        benchmark_spec.answer_instruction,
        "",
        question,
    ]
    return "\n".join(instructions)


class Evaluator:
    """Evaluate benchmark items with a concrete configuration executor.

    Attributes:
        accuracy: Default accuracy threshold for mock evaluation.
        latency_seconds: Default latency per benchmark execution.
        max_examples: Subsampling cap on benchmark items (default 100).
        last_item_results: Detail records collected during last evaluation run.
    """

    def __init__(
        self,
        accuracy: float = 0.65,
        latency_seconds: float = 2.0,
        max_examples: int | None = MAX_EXAMPLES,
    ) -> None:
        if max_examples is not None and max_examples < 1:
            raise ValueError("max_examples must be positive or None")
        self.accuracy = accuracy
        self.latency_seconds = latency_seconds
        self.max_examples = max_examples
        self.last_item_results: list[dict[str, object]] = []

    async def evaluate(
        self,
        config: ExperimentConfig,
        benchmark: str,
        split: BenchmarkSplit,
        seed: int,
    ) -> EvaluationResult:
        """Run a benchmark sample and return aggregate metrics.

        Args:
            config: Executable experiment configuration.
            benchmark: Target benchmark identifier.
            split: Dataset split to evaluate against.
            seed: Random seed for benchmark item subsampling.

        Returns:
            Aggregate EvaluationResult containing accuracy, tokens, and errors.
        """

        validate_config(config)

        benchmark_spec = _benchmark_spec(benchmark)
        ds = benchmark_spec.load_split(split)

        model, formatter = create_experiment_model(config)
        executor = ConfigurationExecutor(
            config=config,
            model=model,
            formatter=formatter,
            extract_answer=benchmark_spec.extract_model_answer,
        )
        started = time.perf_counter()
        correct = 0
        evaluated = (
            len(ds)
            if self.max_examples is None
            else min(self.max_examples, len(ds))
        )
        indices = random.Random(seed).sample(range(len(ds)), evaluated)
        input_tokens = 0
        output_tokens = 0
        errors: list[str] = []
        runtime_error_count = 0
        self.last_item_results = []

        print("Running evaluation...")

        for i in indices:
            question = ds[i]
            item_id = str(
                question.get(
                    "id",
                    question.get("_source_index", i),
                ),
            )
            item_result: dict[str, object] = {
                "dataset_index": i,
                "item_id": item_id,
                "expected_answer": None,
                "predicted_answer": None,
                "correct": False,
                "error": None,
            }

            try:
                outcome = await executor.execute(
                    _build_solver_prompt(
                        benchmark_spec,
                        benchmark_spec.question_text(question),
                    ),
                )
                raw_output = outcome.text
                extracted_answer = benchmark_spec.extract_model_answer(raw_output)
                expected_answer = benchmark_spec.extract_gold_answer(question)
                item_result["expected_answer"] = expected_answer
                item_result["predicted_answer"] = extracted_answer

                input_tokens += outcome.input_tokens
                output_tokens += outcome.output_tokens

                if extracted_answer is None:
                    message = (
                        f"{item_id}: model did not produce an "
                        "extractable answer"
                    )
                    errors.append(message)
                    item_result["error"] = message
                    continue
                if expected_answer is None:
                    runtime_error_count += 1
                    message = f"{item_id}: could not extract gold answer"
                    errors.append(message)
                    item_result["error"] = message
                    continue
                if extracted_answer == expected_answer:
                    correct += 1
                    item_result["correct"] = True
            except Exception as error:
                runtime_error_count += 1
                message = f"{item_id}: {type(error).__name__}: {error}"
                errors.append(message)
                item_result["error"] = message
            finally:
                self.last_item_results.append(item_result)

        end = time.perf_counter()

        print(f"Evaluation time: {(end - started) / 60:.2f} minutes")

        return EvaluationResult(
            config_hash=config.config_hash,
            benchmark=benchmark,
            split=split,
            seed=seed,
            accuracy=correct / evaluated if evaluated else 0.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
            status="failed" if runtime_error_count else "success",
            errors=errors,
        )
