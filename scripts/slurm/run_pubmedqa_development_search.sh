#!/bin/bash
#SBATCH --job-name=pubmedqa_search
#SBATCH --partition=study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/pubmedqa_search_%j.out
#SBATCH --error=logs/pubmedqa_search_%j.err

# =============================================================================
# PubMedQA search and final evaluation
#
# This job performs configuration development and then a terminal evaluation
# on the held-out PubMedQA test partition:
#
#   1. Create naive and careful single-agent baseline configurations.
#   2. Evaluate both baselines on the fixed 100-example development sample.
#   3. Run the autonomous propose-evaluate-critique-memory loop on the same
#      development sample under each requested memory condition.
#   4. Rank all successful configurations and export the top distinct
#      candidates.
#   5. Compare those candidates and both baselines on the complete 500-example
#      development partition using repeated runs. Freeze the winner by mean
#      accuracy, with tokens and latency used only for exact ties.
#   6. Evaluate the frozen winner and both predefined baselines once on all 500
#      held-out test examples.
#
# The test results are terminal outputs: they are not stored in the experience
# library, shown to the Ideator or Critic, or followed by additional search.
# Do not select another configuration or rerun the search based on test scores.
#
# Optional submission overrides:
#
#   CYCLES_PER_RUN=2 SEARCH_SEED=2026 CONDITIONS="full_memory" \
#     TOP_K=2 CONFIRMATION_SEEDS="2026" RUN_FINAL_TEST=0 \
#     sbatch scripts/slurm/run_pubmedqa_development_search.sh
#
# Full default protocol:
#
#   sbatch scripts/slurm/run_pubmedqa_development_search.sh
#
# Resume an interrupted run:
#
#   sbatch --export=ALL,EXPERIMENT_ROOT=/absolute/path/to/existing/run \
#     scripts/slurm/run_pubmedqa_development_search.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    REPO_DIR="${REPO_DIR:-$HOME/group-c-research-agent}"
fi

cd "$REPO_DIR"

if [ -d "$REPO_DIR/project" ]; then
    VENV_DIR="$REPO_DIR/project"
elif [ -d "$REPO_DIR/.venv" ]; then
    VENV_DIR="$REPO_DIR/.venv"
elif [ -d "$REPO_DIR/venv" ]; then
    VENV_DIR="$REPO_DIR/venv"
else
    echo "ERROR: Could not find project/, .venv/, or venv/."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# SLURM opens the output files before running this script. Create logs/ once
# in the repository before the first submission: mkdir -p logs
mkdir -p logs results

export PYTHONPATH="$REPO_DIR/src"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
mkdir -p "$HF_HOME"

if [ ! -f "$REPO_DIR/.env" ]; then
    echo "ERROR: $REPO_DIR/.env is missing."
    echo "Define MODEL_NAME, MODEL_API_KEY, and BASE_URL before submitting."
    exit 1
fi

python3 -c "import agentscope, datasets, dotenv, pydantic; print('Dependency check: OK')"
python3 -c "from research_agent.evaluation.runner import MAX_EXAMPLES; assert MAX_EXAMPLES == 100, f'Expected search MAX_EXAMPLES=100, found {MAX_EXAMPLES}'; print('Search sample size: 100')"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${SLURM_JOB_ID:-local}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_DIR/results/pubmedqa_search_${JOB_ID}_${TIMESTAMP}}"
LIBRARY_PATH="$EXPERIMENT_ROOT/experiences.jsonl"

CYCLES_PER_RUN="${CYCLES_PER_RUN:-10}"
SEARCH_SEED="${SEARCH_SEED:-2026}"
MAX_CANDIDATE_ATTEMPTS="${MAX_CANDIDATE_ATTEMPTS:-10}"
TOP_K="${TOP_K:-3}"
CONDITIONS="${CONDITIONS:-full_memory memory_without_augmentation no_memory}"
BASELINE_SEED="${BASELINE_SEED:-2026}"
CONFIRMATION_SEEDS="${CONFIRMATION_SEEDS:-2026 2027 2028}"
RUN_FINAL_TEST="${RUN_FINAL_TEST:-1}"

mkdir -p "$EXPERIMENT_ROOT/configs"
mkdir -p "$EXPERIMENT_ROOT/configs/confirmation"
mkdir -p "$EXPERIMENT_ROOT/evaluations/search_sample"
mkdir -p "$EXPERIMENT_ROOT/evaluations/full_development"
mkdir -p "$EXPERIMENT_ROOT/evaluations/final_test"
mkdir -p "$EXPERIMENT_ROOT/item_details/search_sample"
mkdir -p "$EXPERIMENT_ROOT/item_details/full_development"
mkdir -p "$EXPERIMENT_ROOT/item_details/final_test"

echo "============================================================"
echo "  PubMedQA Search and Final Evaluation"
echo "============================================================"
echo "Node:                $HOSTNAME"
echo "Date:                $(date)"
echo "Job ID:              $JOB_ID"
echo "Repository:          $REPO_DIR"
echo "Python:              $(python3 --version)"
echo "Experiment root:     $EXPERIMENT_ROOT"
echo "Cycles per run:      $CYCLES_PER_RUN"
echo "Fixed search seed:   $SEARCH_SEED"
echo "Candidate attempts:  $MAX_CANDIDATE_ATTEMPTS"
echo "Top candidates:      $TOP_K"
echo "Memory conditions:   $CONDITIONS"
echo "Confirmation seeds:  $CONFIRMATION_SEEDS"
echo "Run final test:       $RUN_FINAL_TEST"
echo "============================================================"

echo "Phase 1/5: Creating and evaluating fixed baselines"

if [ ! -f "$EXPERIMENT_ROOT/configs/naive_1024.json" ]; then
    python3 scripts/create_baseline_config.py \
        --kind naive \
        --output "$EXPERIMENT_ROOT/configs/naive_1024.json" \
        --seed "$BASELINE_SEED" \
        --temperature 0.0 \
        --max-tokens 1024
else
    echo "Reusing existing naive baseline configuration."
fi

if [ ! -f "$EXPERIMENT_ROOT/configs/careful_1024.json" ]; then
    python3 scripts/create_baseline_config.py \
        --kind careful \
        --output "$EXPERIMENT_ROOT/configs/careful_1024.json" \
        --seed "$BASELINE_SEED" \
        --temperature 0.0 \
        --max-tokens 1024
else
    echo "Reusing existing careful baseline configuration."
fi

for baseline in naive careful; do
    baseline_result="$EXPERIMENT_ROOT/evaluations/search_sample/${baseline}_pubmedqa_100.json"
    if [ ! -f "$baseline_result" ]; then
        python3 scripts/evaluate_config.py \
            --config "$EXPERIMENT_ROOT/configs/${baseline}_1024.json" \
            --benchmark pubmedqa \
            --split development \
            --seed "$BASELINE_SEED" \
            --max-examples 100 \
            --details-output "$EXPERIMENT_ROOT/item_details/search_sample/${baseline}_pubmedqa_100.json" \
            --output "$baseline_result"
    else
        echo "Reusing existing ${baseline} baseline development result."
    fi
done

echo "Phase 2/5: Running autonomous configuration search"

for condition in $CONDITIONS; do
    echo "------------------------------------------------------------"
    echo "condition=$condition seed=$SEARCH_SEED cycles=$CYCLES_PER_RUN"
    echo "Started: $(date)"
    echo "------------------------------------------------------------"

    python3 scripts/run_real_cycle.py \
        --cycles "$CYCLES_PER_RUN" \
        --seed "$SEARCH_SEED" \
        --condition "$condition" \
        --benchmark pubmedqa \
        --max-candidate-attempts "$MAX_CANDIDATE_ATTEMPTS" \
        --baseline "$EXPERIMENT_ROOT/configs/naive_1024.json" \
            "$EXPERIMENT_ROOT/evaluations/search_sample/naive_pubmedqa_100.json" \
        --baseline "$EXPERIMENT_ROOT/configs/careful_1024.json" \
            "$EXPERIMENT_ROOT/evaluations/search_sample/careful_pubmedqa_100.json" \
        --library-path "$LIBRARY_PATH"

    echo "Finished: $(date)"
done

echo "Phase 3/5: Ranking development configurations"

python3 scripts/summarize_experiments.py \
    --library-path "$LIBRARY_PATH" \
    --output-dir "$EXPERIMENT_ROOT" \
    --benchmark pubmedqa \
    --split development \
    --top-k "$TOP_K"

cp "$EXPERIMENT_ROOT/configs/naive_1024.json" \
    "$EXPERIMENT_ROOT/configs/confirmation/naive.json"
cp "$EXPERIMENT_ROOT/configs/careful_1024.json" \
    "$EXPERIMENT_ROOT/configs/confirmation/careful.json"
find "$EXPERIMENT_ROOT/configs/confirmation" \
    -type f -name 'candidate_*.json' -delete
cp "$EXPERIMENT_ROOT"/top_candidate_configs/candidate_*.json \
    "$EXPERIMENT_ROOT/configs/confirmation/"

echo "Phase 4/5: Full-development confirmation and final selection"

for config_path in "$EXPERIMENT_ROOT"/configs/confirmation/*.json; do
    label="$(basename "$config_path" .json)"
    for confirmation_seed in $CONFIRMATION_SEEDS; do
        confirmation_result="$EXPERIMENT_ROOT/evaluations/full_development/${label}_seed_${confirmation_seed}.json"
        if [ ! -f "$confirmation_result" ]; then
            python3 scripts/evaluate_config.py \
                --config "$config_path" \
                --benchmark pubmedqa \
                --split development \
                --seed "$confirmation_seed" \
                --max-examples 500 \
                --details-output "$EXPERIMENT_ROOT/item_details/full_development/${label}_seed_${confirmation_seed}.json" \
                --output "$confirmation_result"
        else
            echo "Reusing confirmation result: ${label}, seed ${confirmation_seed}."
        fi
    done
done

python3 scripts/select_final_config.py \
    --config-dir "$EXPERIMENT_ROOT/configs/confirmation" \
    --results-dir "$EXPERIMENT_ROOT/evaluations/full_development" \
    --benchmark pubmedqa \
    --output-config "$EXPERIMENT_ROOT/configs/final_selected_config.json" \
    --output-dir "$EXPERIMENT_ROOT"

echo "Phase 5/5: Running the terminal held-out test evaluation"

if [ "$RUN_FINAL_TEST" = "1" ]; then
    for baseline in naive careful; do
        baseline_test="$EXPERIMENT_ROOT/evaluations/final_test/${baseline}_pubmedqa_test.json"
        if [ ! -f "$baseline_test" ]; then
            python3 scripts/evaluate_config.py \
                --config "$EXPERIMENT_ROOT/configs/${baseline}_1024.json" \
                --benchmark pubmedqa \
                --split test \
                --seed "$BASELINE_SEED" \
                --max-examples 500 \
                --details-output "$EXPERIMENT_ROOT/item_details/final_test/${baseline}_pubmedqa_test.json" \
                --output "$baseline_test"
        else
            echo "Reusing final-test result for ${baseline}."
        fi
    done

    final_test="$EXPERIMENT_ROOT/evaluations/final_test/final_selected_pubmedqa_test.json"
    if [ ! -f "$final_test" ]; then
        python3 scripts/evaluate_config.py \
            --config "$EXPERIMENT_ROOT/configs/final_selected_config.json" \
            --benchmark pubmedqa \
            --split test \
            --seed "$BASELINE_SEED" \
            --max-examples 500 \
            --details-output "$EXPERIMENT_ROOT/item_details/final_test/final_selected_pubmedqa_test.json" \
            --output "$final_test"
    else
        echo "Reusing final-test result for the selected configuration."
    fi
else
    echo "Final test skipped because RUN_FINAL_TEST=$RUN_FINAL_TEST."
fi

echo ""
echo "============================================================"
echo "PubMedQA search and final evaluation completed."
echo "Results:  $EXPERIMENT_ROOT"
echo "Finished: $(date)"
echo ""
echo "No search or adaptation was performed after the test phase."
echo "============================================================"
