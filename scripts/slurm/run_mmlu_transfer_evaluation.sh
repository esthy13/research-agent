#!/bin/bash
#SBATCH --job-name=mmlu_transfer
#SBATCH --partition=study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mmlu_transfer_%j.out
#SBATCH --error=logs/mmlu_transfer_%j.err

# =============================================================================
# Frozen cross-benchmark transfer evaluation
#
# This job evaluates three configurations that were frozen before MMLU access:
#
#   1. the naive single-agent baseline;
#   2. the careful single-agent baseline;
#   3. the final configuration selected on PubMedQA development data.
#
# Each configuration is evaluated once on every unique item in the official
# MMLU College Chemistry and College Physics test collections. The job does not
# perform search, adaptation, or configuration selection on MMLU. It writes
# aggregate results, item-level correctness, Wilson confidence intervals,
# paired bootstrap intervals, exact McNemar tests, and cost/latency summaries.
#
# Submit:
#
#   mkdir -p logs
#   sbatch scripts/slurm/run_mmlu_transfer_evaluation.sh
#
# Resume an interrupted job:
#
#   sbatch --export=ALL,EXPERIMENT_ROOT=/absolute/path/to/existing/run \
#     scripts/slurm/run_mmlu_transfer_evaluation.sh
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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${SLURM_JOB_ID:-local}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_DIR/results/mmlu_transfer_${JOB_ID}_${TIMESTAMP}}"
TRANSFER_SEED="${TRANSFER_SEED:-2026}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: DRY_RUN must be 0 or 1."
    exit 1
fi

SOURCE_CONFIG_DIR="$REPO_DIR/docs/pubmedqa_final/configs"
CONFIG_DIR="$EXPERIMENT_ROOT/configs"
EVALUATION_DIR="$EXPERIMENT_ROOT/evaluations/transfer"
DETAILS_DIR="$EXPERIMENT_ROOT/item_details/transfer"
ANALYSIS_DIR="$EXPERIMENT_ROOT/analysis"

mkdir -p "$CONFIG_DIR"
mkdir -p "$EVALUATION_DIR"
mkdir -p "$DETAILS_DIR"
mkdir -p "$ANALYSIS_DIR"

copy_frozen_config() {
    local source_path="$1"
    local destination_path="$2"
    if [ ! -f "$source_path" ]; then
        echo "ERROR: Missing source configuration: $source_path"
        exit 1
    fi
    if [ -f "$destination_path" ]; then
        if ! cmp -s "$source_path" "$destination_path"; then
            echo "ERROR: Existing frozen configuration differs: $destination_path"
            exit 1
        fi
        echo "Verified existing frozen configuration: $destination_path"
    else
        cp "$source_path" "$destination_path"
    fi
}

transfer_artifacts_are_reusable() {
    local result_path="$1"
    local details_path="$2"
    local config_path="$3"
    local benchmark="$4"
    python3 - \
        "$result_path" \
        "$details_path" \
        "$config_path" \
        "$benchmark" <<'PY'
import json
import sys
from pathlib import Path

from research_agent.interfaces import EvaluationResult, ExperimentConfig

result = EvaluationResult.model_validate_json(Path(sys.argv[1]).read_text())
details = json.loads(Path(sys.argv[2]).read_text())
config = ExperimentConfig.model_validate_json(Path(sys.argv[3]).read_text())
benchmark = sys.argv[4]
valid = (
    result.status == "success"
    and result.split == "transfer"
    and result.benchmark == benchmark
    and result.config_hash == config.config_hash
    and details.get("config_hash") == config.config_hash
    and details.get("benchmark") == benchmark
    and details.get("split") == "transfer"
    and isinstance(details.get("items"), list)
    and bool(details["items"])
)
raise SystemExit(0 if valid else 1)
PY
}

copy_frozen_config \
    "$SOURCE_CONFIG_DIR/naive_1024.json" \
    "$CONFIG_DIR/naive.json"
copy_frozen_config \
    "$SOURCE_CONFIG_DIR/careful_1024.json" \
    "$CONFIG_DIR/careful.json"
copy_frozen_config \
    "$SOURCE_CONFIG_DIR/final_selected_config.json" \
    "$CONFIG_DIR/selected.json"

cp "$REPO_DIR/docs/MMLU_TRANSFER_PROTOCOL.md" \
    "$EXPERIMENT_ROOT/MMLU_TRANSFER_PROTOCOL.md"

python3 - "$CONFIG_DIR" <<'PY'
import sys
from pathlib import Path

from research_agent.interfaces import ExperimentConfig

paths = sorted(Path(sys.argv[1]).glob("*.json"))
configs = [
    ExperimentConfig.model_validate_json(path.read_text())
    for path in paths
]
assert len(configs) == 3
assert len({config.model_id for config in configs}) == 1
print("Frozen configuration check: OK")
print("Model:", configs[0].model_id)
PY

python3 - <<'PY'
from research_agent.evaluation.runner import _benchmark_spec

names = ("mmlu_college_chemistry", "mmlu_college_physics")
counts = {
    name: len(_benchmark_spec(name).load_split("transfer"))
    for name in names
}
print("Transfer item counts:", counts)
assert counts == {
    "mmlu_college_chemistry": 100,
    "mmlu_college_physics": 91,
}
PY

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run completed: dependencies, configurations, and datasets verified."
    exit 0
fi

echo "============================================================"
echo "  Frozen MMLU Transfer Evaluation"
echo "============================================================"
echo "Node:                $HOSTNAME"
echo "Date:                $(date)"
echo "Job ID:              $JOB_ID"
echo "Repository:          $REPO_DIR"
echo "Python:              $(python3 --version)"
echo "Experiment root:     $EXPERIMENT_ROOT"
echo "Evaluation seed:     $TRANSFER_SEED"
echo "Configurations:      naive careful selected"
echo "Benchmarks:          mmlu_college_chemistry mmlu_college_physics"
echo "Split:               transfer (complete deduplicated official test)"
echo "Search/adaptation:   disabled"
echo "============================================================"

for benchmark in mmlu_college_chemistry mmlu_college_physics; do
    for label in naive careful selected; do
        result_path="$EVALUATION_DIR/${label}_${benchmark}.json"
        details_path="$DETAILS_DIR/${label}_${benchmark}.json"
        config_path="$CONFIG_DIR/${label}.json"

        if [ -s "$result_path" ] && [ -s "$details_path" ]; then
            if transfer_artifacts_are_reusable \
                "$result_path" \
                "$details_path" \
                "$config_path" \
                "$benchmark"; then
                echo "Reusing existing transfer result: $label on $benchmark"
                continue
            fi
            echo "Existing artifacts are incomplete, failed, or mismatched."
            echo "Rerunning $label on $benchmark and overwriting that pair."
        fi

        echo "------------------------------------------------------------"
        echo "Evaluating $label on $benchmark"
        echo "Started: $(date)"
        echo "------------------------------------------------------------"

        python3 scripts/evaluate_config.py \
            --config "$config_path" \
            --benchmark "$benchmark" \
            --split transfer \
            --seed "$TRANSFER_SEED" \
            --all-examples \
            --details-output "$details_path" \
            --output "$result_path"

        echo "Finished: $(date)"
    done
done

echo "Creating validated statistical summaries"

python3 scripts/analyze_mmlu_transfer.py \
    --experiment-root "$EXPERIMENT_ROOT" \
    --output-dir "$ANALYSIS_DIR"

echo ""
echo "============================================================"
echo "Frozen MMLU transfer evaluation completed."
echo "Results:  $EXPERIMENT_ROOT"
echo "Report:   $ANALYSIS_DIR/TRANSFER_REPORT.md"
echo "Finished: $(date)"
echo ""
echo "No MMLU search, adaptation, or reselection was performed."
echo "============================================================"
