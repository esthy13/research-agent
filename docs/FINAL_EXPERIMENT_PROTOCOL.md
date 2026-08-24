# Final experiment protocol

This document freezes the PubMedQA protocol used for the final report. Test
results are terminal outputs and must never enter the Ideator, Critic,
Experience Library, configuration selection, or another search run.

## Research question

The system automatically proposes and evaluates executable reasoning-agent
configurations. The final experiment asks:

1. Can autonomous search discover a PubMedQA configuration that improves over
   fixed naive and careful single-agent baselines?
2. Does full experimental memory outperform memory without Critic augmentation
   and no memory?
3. What accuracy, token, and latency trade-offs result from multi-agent
   protocols?

## Data partition

The 1,000 labeled PubMedQA examples are shuffled once with dataset seed 2026
and divided into fixed, disjoint halves:

- development: first 500 examples;
- test: remaining 500 examples.

Autonomous search evaluates every proposal on the same 100-example development
sample selected with evaluation seed 2026. The test partition is accessed only
after the final configuration is frozen.

## Fixed baselines

Both baselines use one direct solver, temperature 0.0, and a 1,024-token cap:

- naive: minimal answer instruction;
- careful: explicit reasoning and verification instruction.

Their 100-example development results are supplied to every Ideator and Critic
as fixed experimental controls. This evidence is identical across all memory
conditions and is not retrieved experimental memory.

## Executable search space

- direct: one model call;
- self-consistency: independent calls followed by parsed majority vote;
- critique and revision: initial solver, critics, and final reviser;
- final revision: ordered role contributions followed by a synthesizer.

Role names label calls, communication order fixes call order, and one shared
system prompt is used by all roles. Numerical role weighting, dynamic role
rotation, tools, training, and fine-tuning are not implemented and cannot be
claimed as experimental manipulations.

## Search budget and memory ablations

The default final job runs 10 cycles for each condition using fixed search and
evaluation seed 2026:

- `full_memory`;
- `memory_without_augmentation`;
- `no_memory`.

All conditions therefore evaluate proposals on the same 100 development
questions. Search fitness is accuracy. Token use and latency are included in
memory and Critic evidence so the agent can prefer efficient strategies when
accuracy is comparable.

The controller first requests up to 10 valid, unique proposals per cycle. If
every valid proposal duplicates an existing executable configuration, the last
duplicate is evaluated as an explicit replication instead of aborting the
search budget. Replications remain in the trajectory, but retrieved memory and
top-candidate export retain only the strongest measurement per configuration
hash so repetitions cannot crowd out strategy diversity.

## Full-development selection

After the 30 search cycles:

1. Successful proposals are ranked by development accuracy.
2. The top three distinct executable configurations are exported.
3. Those configurations and both baselines are evaluated on all 500
   development examples with evaluation orders 2026, 2027, and 2028.
4. The configuration with the highest mean accuracy is frozen. Mean total
   tokens and then mean latency break exact accuracy ties.

This prevents a searched configuration from being called the winner when a
baseline is stronger. Several searched candidates are confirmed so selection
does not depend only on the noisy 100-example screen.

## Held-out test

The frozen winner, naive baseline, and careful baseline are each evaluated once
on all 500 test examples. The job terminates after writing these results. No
test result is written to the Experience Library or passed to an agent.

## Reproducibility artifacts

The timestamped `results/pubmedqa_search_<job>_<timestamp>/` directory contains:

- `experiences.jsonl`: complete autonomous search trajectory;
- `ranked_development_experiments.*`: 100-example search ranking;
- `condition_summary.*`: memory-ablation summaries;
- `top_candidate_configs/`: searched finalists;
- `evaluations/full_development/`: repeated selection measurements;
- `final_selection.*`: baseline-aware final ranking;
- `configs/final_selected_config.json`: frozen winner;
- `evaluations/final_test/`: terminal held-out results;
- `item_details/`: expected and predicted labels without raw model reasoning.

## SLURM commands

Final run:

```bash
mkdir -p logs
sbatch scripts/slurm/run_pubmedqa_development_search.sh
```

Interrupted jobs can resume the same result directory. Existing baselines,
saved search experiences, confirmation results, and test results are reused:

```bash
sbatch --export=ALL,EXPERIMENT_ROOT=/absolute/path/to/existing/run \
  scripts/slurm/run_pubmedqa_development_search.sh
```

Non-final pilot that does not access test:

```bash
CYCLES_PER_RUN=2 \
CONDITIONS="full_memory" \
TOP_K=2 \
CONFIRMATION_SEEDS="2026" \
RUN_FINAL_TEST=0 \
sbatch scripts/slurm/run_pubmedqa_development_search.sh
```
