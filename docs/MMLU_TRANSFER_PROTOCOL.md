# Frozen MMLU transfer protocol

## Research question

Does the reasoning configuration selected exclusively on PubMedQA development
data transfer to unseen scientific multiple-choice tasks without additional
search or adaptation?

This is a cross-benchmark transfer study. College Chemistry and College Physics
are two subjects within the MMLU benchmark family, not two independent
datasets.

## Frozen configurations

The study compares three configurations already frozen by the PubMedQA study:

1. `naive`: the predefined naive single-agent baseline;
2. `careful`: the predefined careful single-agent baseline;
3. `selected`: the final configuration selected using PubMedQA development
   data.

The configuration JSON files are copied without modification from
`docs/pubmedqa_final/configs/`. Their system prompts, interaction protocols,
model IDs, temperatures, and token limits remain unchanged. MMLU supplies only
the task-specific question rendering and required A-D final-answer format.

## Evaluation data

The evaluator loads the official test collection for each subject:

- `cais/mmlu`, `college_chemistry`, official `test` collection;
- `cais/mmlu`, `college_physics`, official `test` collection.

The dataset is pinned to revision
`c30699e8356da336a370243923dbaf21066bb9fe`. Exact duplicate rows are removed,
leaving 100 Chemistry items and 91 Physics items. Every remaining item is
evaluated; there is no item subsampling. The result uses the explicit
`transfer` split label to distinguish this complete collection from the
internal MMLU development/held-out partitions available for future search
experiments.

Because all configurations were selected before this study and no MMLU result
is used to alter them, the complete subject collections can be used as
out-of-domain transfer evaluations.

## Reproducibility and leakage controls

- All three configurations receive exactly the same items.
- Evaluation seed 2026 fixes item execution order.
- Temperature remains frozen at 0.0 for all configurations.
- Raw model reasoning is not persisted.
- Item identifiers, expected labels, predicted labels, correctness, and errors
  are stored for paired analysis.
- MMLU results must not be added to the Experience Library or used to select,
  edit, or rerun a configuration.
- A failed transfer result is retained and reported; it must not trigger
  post-hoc adaptation.

## Reported outcomes

For each subject and for the pooled two-subject result, the analysis reports:

- accuracy and a 95% Wilson confidence interval;
- input, output, and total tokens;
- end-to-end latency;
- answer-extraction and runtime failures;
- selected-minus-baseline accuracy differences with deterministic paired
  bootstrap intervals;
- selected-only and baseline-only correct counts;
- two-sided exact McNemar p-values;
- token and latency ratios.

The subject-level tests are exploratory and are not corrected for multiple
comparisons. The pooled row aggregates item-level outcomes across the two
subjects and must not be described as evidence from two independent datasets.

## SLURM submission

From the repository root:

```bash
mkdir -p logs
sbatch scripts/slurm/run_mmlu_transfer_evaluation.sh
```

The launcher is resumable. To continue an interrupted run:

```bash
sbatch --export=ALL,EXPERIMENT_ROOT=/absolute/path/to/existing/run \
  scripts/slurm/run_mmlu_transfer_evaluation.sh
```

The completed experiment directory contains frozen configuration copies,
aggregate evaluations, item-level details, the protocol snapshot, and
validated JSON, CSV, and Markdown reports.
