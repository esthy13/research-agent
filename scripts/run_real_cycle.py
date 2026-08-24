"""Run the full research loop with the real benchmark evaluator."""

import argparse
import asyncio
import os
from pathlib import Path

from research_agent.agents.critic import CriticAgent
from research_agent.agents.ideator import IdeatorAgent
from research_agent.cycle import ResearchCycle
from research_agent.evaluation.runner import Evaluator, SUPPORTED_BENCHMARKS
from research_agent.interfaces import (
    EvaluationResult,
    ExperimentConfig,
    MemoryCondition,
)
from research_agent.memory.library import ExperienceLibrary
from research_agent.model_factory import (
    create_critic_model,
    create_ideator_model,
)
from research_agent.search.controller import SearchController


MEMORY_CONDITIONS: tuple[MemoryCondition, ...] = (
    "full_memory",
    "memory_without_augmentation",
    "no_memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more real propose-evaluate-critique cycles.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of development-search cycles to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for candidate generation and benchmark sampling.",
    )
    parser.add_argument(
        "--condition",
        choices=MEMORY_CONDITIONS,
        default="full_memory",
        help="Memory ablation condition for the search loop.",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(SUPPORTED_BENCHMARKS),
        default="gsm8k",
        help="Benchmark used for development evaluation.",
    )
    parser.add_argument(
        "--library-path",
        default="data/experiences.jsonl",
        help=(
            "JSONL file used as the ExperienceLibrary. Use a dedicated path "
            "for final-report runs to avoid mixing pilot and production data."
        ),
    )
    parser.add_argument(
        "--max-candidate-attempts",
        type=int,
        default=10,
        help=(
            "Maximum Ideator proposals allowed for one cycle before the "
            "search run stops. Invalid and duplicate proposals count toward "
            "this limit."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        nargs=2,
        metavar=("CONFIG_JSON", "RESULT_JSON"),
        help=(
            "Fixed baseline configuration and development-result pair. May "
            "be supplied multiple times. Baselines are shared controls and "
            "are not part of retrieved experimental memory."
        ),
    )
    return parser.parse_args()


def load_baseline_context(
    baseline_pairs: list[list[str]],
    benchmark: str,
) -> str:
    """Load verified fixed-control measurements for the agent prompts."""

    if not baseline_pairs:
        return "No fixed baseline measurements were supplied."

    lines = []
    for config_name, result_name in baseline_pairs:
        config_path = Path(config_name)
        result_path = Path(result_name)
        config = ExperimentConfig.model_validate_json(
            config_path.read_text(encoding="utf-8"),
        )
        result = EvaluationResult.model_validate_json(
            result_path.read_text(encoding="utf-8"),
        )
        if result.config_hash != config.config_hash:
            raise ValueError(
                f"Baseline hash mismatch: {config_path} and {result_path}",
            )
        if result.benchmark.lower() != benchmark.lower():
            raise ValueError(
                f"Baseline {result_path} targets {result.benchmark}, "
                f"not {benchmark}",
            )

        label = config_path.stem.removesuffix("_1024")
        total_tokens = result.input_tokens + result.output_tokens
        lines.append(
            f"- {label}: accuracy={result.accuracy:.4f}, "
            f"total_tokens={total_tokens}, "
            f"latency_seconds={result.latency_seconds:.2f}, "
            f"reasoning_mode={config.reasoning_mode}, "
            f"agent_count={config.agent_count}, "
            f"aggregation={config.aggregation}, "
            f"temperature={config.temperature}, "
            f"max_tokens={config.max_tokens}, "
            f"system_prompt={config.system_prompt!r}"
        )
    return "\n".join(lines)


async def main() -> None:
    args = parse_args()
    library = ExperienceLibrary(file_path=args.library_path)
    baseline_context = load_baseline_context(
        args.baseline,
        benchmark=args.benchmark,
    )

    ideator_model, ideator_formatter = create_ideator_model()
    critic_model, critic_formatter = create_critic_model()

    controller = SearchController(
        ideator=IdeatorAgent(
            ideator_model,
            ideator_formatter,
        ),
        library=library,
        model_id=os.environ["MODEL_NAME"],
        max_attempts=args.max_candidate_attempts,
        baseline_context=baseline_context,
    )

    research_cycle = ResearchCycle(
        controller=controller,
        evaluator=Evaluator(),
        critic=CriticAgent(
            critic_model,
            critic_formatter,
        ),
        library=library,


        fitness_function=lambda result: result.accuracy,
    )

    existing_records = [
        record
        for record in library.load_all()
        if record.result.benchmark.lower() == args.benchmark.lower()
        and record.result.split == "development"
        and record.config.condition == args.condition
        and record.config.seed == args.seed
    ]
    completed_cycles = len(existing_records)
    if completed_cycles >= args.cycles:
        print(
            f"Search target already satisfied for condition={args.condition}: "
            f"{completed_cycles}/{args.cycles} saved experiences."
        )
        return

    next_cycle = (
        max(record.cycle for record in existing_records) + 1
        if existing_records
        else 0
    )
    remaining_cycles = args.cycles - completed_cycles
    print(
        f"Resuming condition={args.condition}: "
        f"{completed_cycles}/{args.cycles} experiences already saved; "
        f"running {remaining_cycles} remaining cycles."
    )

    last_record = None

    for offset in range(remaining_cycles):
        cycle_number = next_cycle + offset
        print(f"\nSTARTING REAL CYCLE {cycle_number}")

        last_record = await research_cycle.run(
            cycle=cycle_number,
            total_cycles=args.cycles,
            seed=args.seed,
            condition=args.condition,
            benchmark=args.benchmark,
        )

        print(
            f"Completed cycle {cycle_number}: "
            f"accuracy={last_record.result.accuracy:.2f}, "
            f"fitness={last_record.fitness_score:.2f}, "
            f"mode={last_record.config.search_mode}, "
            f"hash={last_record.config.config_hash}"
        )

    if last_record is not None:
        print("\nSAVED EXPERIENCE")
        print(last_record.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
