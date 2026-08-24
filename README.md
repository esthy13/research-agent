# Group C Research Agent

This repository implements a memory-guided search system for discovering
effective prompting and multi-agent reasoning configurations. The system does
not train or fine-tune a model. Instead, it repeatedly proposes an executable
configuration, evaluates it on a reasoning benchmark, diagnoses the measured
outcome, and stores the complete trajectory so later proposals can use prior
evidence.

The primary study uses PubMedQA to ask two questions:

1. Can automated configuration search outperform fixed single-agent baselines?
2. Does memory containing Critic diagnoses help more than raw result memory or
   no memory?
3. Does the selected final configuration transfer to new benchmarks?

The final frozen protocol, outputs, and report are available under
[`docs/`](docs/).

## Core idea

One development-search cycle is:

```text
Experience Library
        |
        v
retrieve relevant prior experiments
        |
        v
choose exploration or exploitation
        |
        v
Ideator proposes an ExperimentConfig
        |
        v
normalize -> hash -> validate -> reject duplicates
        |
        v
ConfigurationExecutor turns the config into model calls
        |
        v
Evaluator scores benchmark answers, tokens, and latency
        |
        v
Critic diagnoses the measured result
        |
        v
save one complete ExperienceRecord
        |
        +----> retrieved by a later cycle when memory is enabled
```

The separation between proposing and measuring is deliberate. The Ideator may
change the experimental treatment—prompt, reasoning mode, role topology,
temperature, or token budget—but it cannot change the model chosen for the run,
the benchmark data, the answer extractor, the split, the search decision, or
the scoring rule.

## What the system searches over

An `ExperimentConfig` contains a hypothesis and rationale plus the executable
settings below:

| Field | Meaning |
| --- | --- |
| `system_prompt` | Shared system prompt used by every role. |
| `reasoning_mode` | `direct`, `chain_of_thought`, `self_consistency`, or `critique_and_revision`. |
| `agent_count` | Number of stateless model calls/roles, from 1 to 5. |
| `roles` | Free-form, unique role labels proposed by the Ideator. |
| `communication_order` | Exact order in which those roles are executed. |
| `aggregation` | `single_answer`, `majority_vote`, or `final_revision`. |
| `temperature` | Sampling temperature used by the experiment model. |
| `max_tokens` | Output-token cap used by the experiment model. |
| `model_id` | Fixed model deployment used for execution. |
| `seed` | Reproducibility and benchmark-sampling seed. |
| `condition` | Active memory-ablation condition. |
| `search_mode` | Controller-selected exploration or exploitation mode. |
| `parent_ids` | Earlier experiment IDs used as parents during exploitation. |
| `config_hash` | Deterministic identity of the executable settings. |

Role names are labels, not different trained models. Every role uses the same
configured model and shared system prompt. A role's position in
`communication_order`, together with the reasoning/aggregation mode, determines
the instruction it receives.

### Implemented execution protocols

| Protocol | Concrete execution |
| --- | --- |
| Direct + single answer | One role makes one model call and its response is returned. |
| Chain of thought | Calls receive an additional step-by-step reasoning and verification instruction. It can be used with a single call or generic final revision. |
| Self-consistency + majority vote | At least three roles solve independently. Benchmark-specific answers are extracted, counted, and the majority label is returned. Ties go to the earliest tied answer. |
| Critique and revision + final revision | The first role proposes a solution, every middle role independently critiques that solution, and the final role receives the proposal and all critiques before revising it. |
| Generic final revision | Roles contribute sequentially; each role sees prior contributions and the last role synthesizes the final response. |

Configurations are semantically validated before execution. Examples of
enforced constraints are: direct reasoning requires one role and
`single_answer`; self-consistency requires at least three roles and
`majority_vote`; critique-and-revision requires at least three roles and
`final_revision`.

## Memory conditions

Every run stores its trajectory, but the retrieval layer controls what a later
Ideator is allowed to see:

| Condition | Information available to the next proposal |
| --- | --- |
| `full_memory` | Prior configurations, measurements, compact outcomes, Critic diagnoses, and recommendations. |
| `memory_without_augmentation` | Prior configurations and measurements, but no Critic diagnosis or recommendation. |
| `no_memory` | No prior trajectory and no parent-based exploitation. Records are still saved for audit and final analysis. |

Retrieval is scoped to the development split, condition, and seed. It ranks by
fitness and recency, keeps only the strongest record for each configuration
hash, and returns at most five distinct strategies. This prevents replications
of one strategy from filling the complete prompt context.

## Exploration and exploitation

The search policy starts with an exploration probability of 0.8 and decreases
it linearly to 0.3 over the configured cycle budget. Early cycles therefore
favor new strategies; later cycles more often refine the strongest retrieved
parent. The random decision is seeded with `seed:cycle`, so repeating the same
run produces the same search-mode schedule.

If the Ideator returns an invalid or duplicate configuration, the controller
adds the rejection reason to the next prompt and retries up to ten times by
default. If every valid attempt is a duplicate, the final duplicate is executed
as an explicit replication so each condition retains the same experiment
budget. If no valid proposal is produced, the cycle fails instead of silently
inventing a fallback.

## Shared data contracts

All main components exchange strict Pydantic models from `interfaces.py`:

| Contract | Producer | Consumers | Purpose |
| --- | --- | --- | --- |
| `ExperimentConfig` | Ideator, then normalized by Python | validator, executor, evaluator, Critic, memory | Exact experiment to execute. |
| `EvaluationResult` | Evaluator | Critic, fitness function, memory | Accuracy, tokens, latency, status, and errors. |
| `CriticReview` | Critic | memory and later full-memory prompts | Evidence-grounded diagnosis and next-step recommendation. |
| `ExperienceRecord` | `ResearchCycle` | JSONL library, retrieval, reporting scripts | Complete provenance for one cycle. |

Unknown schema fields are rejected. Python overwrites controller-owned fields
after structured LLM generation and calculates `config_hash` itself. The hash
uses canonical JSON over the actual executable settings, so identical
strategies share an identity even if their hypotheses, seeds, conditions, or
parent histories differ.

## End-to-end component interaction

`ResearchCycle.run()` is the central orchestrator:

1. `SearchController.propose_candidate()` retrieves memory and calls the search
   policy.
2. The controller asks `IdeatorAgent.propose()` for a structured candidate.
3. Python restores fixed controls, calculates the hash, validates cross-field
   semantics, and checks the library for duplicates.
4. `Evaluator.evaluate()` loads the requested benchmark split, samples items
   reproducibly, creates the configured experiment model, and delegates every
   item to `ConfigurationExecutor.execute()`.
5. The executor dispatches to direct, voting, critique/revision, or sequential
   synthesis behavior and aggregates model-call token usage.
6. The evaluator extracts benchmark-specific labels, compares them with gold
   answers, and returns aggregate accuracy, token use, latency, status, and
   item-level errors.
7. `CriticAgent.review()` receives the exact configuration, measured result,
   benchmark contract, and fixed baseline context. It returns a structured
   diagnosis without modifying the score.
8. The cycle calculates fitness (currently raw development accuracy), creates a
   compact retrieval summary, and appends an `ExperienceRecord` to JSONL.

The evaluator is also the boundary between generic reasoning protocols and
benchmark-specific behavior. The executor knows how agents interact; each
`BenchmarkSpec` knows how to load questions, format them, extract model answers,
and extract gold labels.

## Repository guide

### Runtime package

```text
src/research_agent/
├── interfaces.py
│   Strict Pydantic contracts shared by search, execution, Critic, and memory.
├── model_factory.py
│   Creates OpenAI-compatible AgentScope models and matching formatters.
│   Ideator uses temperature 0.7, Critic 0.1, and experiment calls use the
│   temperature/max_tokens proposed in ExperimentConfig.
├── cycle.py
│   Orchestrates one propose-evaluate-critique-store development cycle.
├── agents/
│   ├── ideator.py
│   │   Calls the LLM for a structured candidate, restores protected controls,
│   │   and attaches the authoritative configuration hash.
│   ├── critic.py
│   │   Produces a structured diagnosis from measured evidence.
│   └── prompts.py
│       Stable agent roles, benchmark context, execution contract, constraints,
│       baselines, search instruction, and retrieved-memory rendering.
├── search/
│   ├── controller.py
│   │   Coordinates retrieval, policy, generation, validation, retries, and
│   │   duplicate handling.
│   ├── policy.py
│   │   Reproducible exploration/exploitation schedule and parent selection.
│   ├── validation.py
│   │   Cross-field executable constraints and hash-integrity checks.
│   └── hashing.py
│       Canonical SHA-256 identity for executable configuration settings.
├── memory/
│   ├── library.py
│   │   Append-only JSONL persistence, provenance checks, and duplicate lookup.
│   └── retrieval.py
│       Scoped ranking, replication deduplication, and condition-dependent
│       prompt rendering.
└── evaluation/
    ├── protocol.py
    │   Async evaluator interface expected by ResearchCycle.
    ├── executor.py
    │   Concrete direct, voting, critique/revision, and synthesis model calls.
    └── runner.py
        Benchmark registry, dataset splits, question rendering, answer
        extraction, reproducible sampling, scoring, and result aggregation.
```

The `__init__.py` files mark importable Python packages; they contain no runtime
workflow of their own.

### Operational scripts

| File | Purpose |
| --- | --- |
| `scripts/create_baseline_config.py` | Write fixed naive or careful single-agent baseline configurations. |
| `scripts/evaluate_config.py` | Evaluate any saved configuration on a chosen benchmark/split and optionally save per-item labels. |
| `scripts/run_real_cycle.py` | Run or resume one condition's real development-search cycles. This is the main Python entry point for the autonomous loop. |
| `scripts/summarize_experiments.py` | Rank successful search records, deduplicate strategies, export top candidates, and summarize memory conditions. |
| `scripts/select_final_config.py` | Compare searched candidates and baselines over repeated full-development evaluations; freeze the winner by mean accuracy, then tokens and latency for exact ties. |
| `scripts/analyze_mmlu_transfer.py` | Validate frozen MMLU transfer artifacts and produce paired statistical summaries. |
| `scripts/slurm/run_pubmedqa_development_search.sh` | Complete five-phase PubMedQA protocol: baselines, search, ranking, confirmation/selection, and terminal held-out test. It supports resumption. |
| `scripts/slurm/run_mmlu_transfer_evaluation.sh` | Evaluate the frozen PubMedQA winner and both baselines on complete MMLU Chemistry and Physics transfer collections. |

### Documentation and frozen artifacts

| Path | Contents |
| --- | --- |
| `docs/FINAL_EXPERIMENT_PROTOCOL.md` | Frozen PubMedQA data split, search budget, baselines, selection rule, held-out policy, and launcher commands. |
| `docs/MMLU_TRANSFER_PROTOCOL.md` | Frozen cross-benchmark transfer protocol, leakage controls, metrics, and launcher command. |
| `docs/IMPLEMENTATION_AND_INTEGRATION.md` | Design rationale, paper mapping, and integration details. |
| `docs/pubmedqa_final/` | Complete frozen primary-study configurations, trajectories, aggregate evaluations, item labels, rankings, and selected winner. |
| `docs/calibration/`, `docs/final_results/`, `docs/diagnostics/`, `docs/smoke_results/` | Historical calibration and audit artifacts referenced by the report; they are not inputs to the frozen PubMedQA selection. |
| `data/.gitkeep` | Keeps the default local data directory in Git. Runtime `experiences.jsonl` files are generated and ignored. |

## Supported benchmarks

| Benchmark argument | Dataset/task | Extracted answer |
| --- | --- | --- |
| `pubmedqa` | Labeled PubMedQA | `yes`, `no`, or `maybe` |
| `gsm8k` | GSM8K main | normalized numeric answer |
| `bbh` | BIG-Bench Hard boolean expressions | `true` or `false` |
| `mmlu_college_physics` | MMLU college physics | `A`, `B`, `C`, or `D` |
| `mmlu_college_chemistry` | MMLU college chemistry | `A`, `B`, `C`, or `D` |

PubMedQA is deterministically shuffled with dataset seed 2026 and split 500/500
into development and test. Autonomous search uses the same seeded 100-example
development screen for every condition. Test results never enter the Ideator,
Critic, Experience Library, or configuration selection.

MMLU Chemistry and Physics also expose internal development/test partitions for
future search studies. The additional `transfer` split instead uses every unique
item in each subject's official test collection. It is reserved for evaluating
configurations frozen on another benchmark and must not feed subsequent search
or reselection.

## Installation

Requirements:

- Python 3.10 or newer;
- access to the course's OpenAI-compatible model gateway;
- Hugging Face dataset access and enough compute for the selected experiment
  budget;
- `uv` is recommended, although a normal editable `pip` installation works.

Create an isolated environment and install the locked project:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --active
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Copy the safe environment template and add the credentials supplied for the
course:

```bash
cp .env.example .env
```

```dotenv
MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct-FP8"
MODEL_API_KEY="..."
BASE_URL="..."
HF_TOKEN="hf_..."
```

`.env`, generated trajectories, result directories, model caches, and virtual
environments are ignored. Never commit credentials.

## Running one development search

The following command runs ten PubMedQA cycles under full memory and saves a
resumable trajectory:

```bash
python scripts/run_real_cycle.py \
  --cycles 10 \
  --seed 2026 \
  --condition full_memory \
  --benchmark pubmedqa \
  --library-path results/example_run/experiences.jsonl
```

`--cycles` is the target total, not an additional count. Running the same
command again reads matching records and executes only missing cycles.

For the primary comparison, use separate runs for all three conditions but the
same benchmark seed and one shared library path:

```bash
for condition in full_memory memory_without_augmentation no_memory; do
  python scripts/run_real_cycle.py \
    --cycles 10 \
    --seed 2026 \
    --condition "$condition" \
    --benchmark pubmedqa \
    --library-path results/example_run/experiences.jsonl
done
```

The frozen study also supplies naive and careful development baselines to every
Ideator/Critic prompt. The complete launcher below performs that setup
automatically; use it rather than the simplified commands above for exact
reproduction.

## Evaluating a fixed configuration

Create a baseline:

```bash
python scripts/create_baseline_config.py \
  --kind careful \
  --seed 2026 \
  --temperature 0.0 \
  --max-tokens 1024 \
  --output results/manual/careful.json
```

Evaluate it on 100 development examples and retain expected/predicted labels:

```bash
python scripts/evaluate_config.py \
  --config results/manual/careful.json \
  --benchmark pubmedqa \
  --split development \
  --seed 2026 \
  --max-examples 100 \
  --details-output results/manual/careful_items.json \
  --output results/manual/careful_result.json
```

Per-item artifacts store IDs, expected labels, predicted labels, correctness,
and errors. Raw model reasoning is deliberately not persisted.

## Exact PubMedQA reproduction

The authoritative protocol is
[`docs/FINAL_EXPERIMENT_PROTOCOL.md`](docs/FINAL_EXPERIMENT_PROTOCOL.md), and the
authoritative launcher is
[`scripts/slurm/run_pubmedqa_development_search.sh`](scripts/slurm/run_pubmedqa_development_search.sh).

On a SLURM cluster:

```bash
mkdir -p logs
sbatch scripts/slurm/run_pubmedqa_development_search.sh
```

The default job performs:

1. Create naive and careful direct baselines.
2. Evaluate both on the fixed 100-example development screen.
3. Run ten search cycles for each of the three memory conditions with seed 2026.
4. Rank successful trajectories and export the top three distinct strategies.
5. Evaluate those strategies and both baselines on all 500 development items
   with seeds 2026, 2027, and 2028.
6. Freeze the configuration with the highest mean development accuracy; exact
   ties prefer fewer tokens and then lower latency.
7. Evaluate the frozen winner and both baselines once on all 500 held-out test
   items, then terminate without further adaptation.

For a local or cluster pilot that never accesses the test split:

```bash
CYCLES_PER_RUN=2 \
CONDITIONS="full_memory" \
TOP_K=2 \
CONFIRMATION_SEEDS="2026" \
RUN_FINAL_TEST=0 \
bash scripts/slurm/run_pubmedqa_development_search.sh
```

The launcher is resumable. Point `EXPERIMENT_ROOT` at the existing output
directory; completed baselines, cycles, confirmation evaluations, and terminal
tests are reused:

```bash
EXPERIMENT_ROOT=/absolute/path/to/existing/run \
bash scripts/slurm/run_pubmedqa_development_search.sh
```

## Frozen MMLU transfer evaluation

The cross-benchmark study evaluates the already-frozen PubMedQA winner and both
predefined baselines on every unique item in the official MMLU College
Chemistry and College Physics test collections. It performs no MMLU search,
adaptation, or reselection. The complete design and interpretation constraints
are frozen in
[`docs/MMLU_TRANSFER_PROTOCOL.md`](docs/MMLU_TRANSFER_PROTOCOL.md).

Submit the resumable job from the repository root:

```bash
mkdir -p logs
sbatch scripts/slurm/run_mmlu_transfer_evaluation.sh
```

The job evaluates 100 Chemistry items and 91 deduplicated Physics items for
each of the naive, careful, and PubMedQA-selected configurations. Its
`analysis/` directory contains a validated Markdown report and machine-readable
JSON/CSV tables with Wilson intervals, paired bootstrap intervals, exact
McNemar tests, tokens, and latency.

## Generated PubMedQA experiment directory

The complete launcher creates:

```text
results/pubmedqa_search_<job>_<timestamp>/
├── experiences.jsonl
├── configs/
│   ├── naive_1024.json
│   ├── careful_1024.json
│   ├── confirmation/
│   └── final_selected_config.json
├── evaluations/
│   ├── search_sample/
│   ├── full_development/
│   └── final_test/
├── item_details/
│   ├── search_sample/
│   ├── full_development/
│   └── final_test/
├── top_candidate_configs/
├── ranked_development_experiments.json/.csv
├── condition_summary.json/.csv
├── top_candidates.json
└── final_selection.json/.csv
```

The committed primary-study snapshot follows the same organization under
`docs/pubmedqa_final/`.

## Selection and interpretation

Search fitness is currently raw development accuracy. Token use and latency are
shown to the agents and recorded, but they do not change the per-cycle fitness
ranking. Final confirmation is stricter: it compares searched candidates and
fixed baselines by mean full-development accuracy, using mean total tokens and
then mean latency only for exact ties.

This system performs configuration-level search and selection. It does not
update model weights, prove that later proposals become intrinsically better,
or implement numerical role weighting, dynamic role rotation, tools, training,
or fine-tuning. The final report discusses these limitations and the distinction
between search improvement and model self-improvement.

## Reproducibility checklist

Before treating a run as comparable to the frozen study, verify that:

- the same model deployment and `.env` gateway are used;
- PubMedQA uses dataset seed 2026 and search/evaluation seed 2026;
- every condition uses the same 100 development questions and cycle budget;
- the naive and careful baseline evidence is supplied identically to all
  conditions;
- only development results enter memory, Critic prompts, and selection;
- top candidates and both baselines receive the same full-development
  confirmation seeds;
- the final configuration is frozen before any test evaluation;
- no search, reselection, or adaptation follows the held-out test.

See the final protocol for the exact frozen values and the final report for the
scientific rationale, results, threats to validity, and discussion.
