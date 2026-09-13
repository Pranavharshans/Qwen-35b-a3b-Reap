#!/bin/bash -l
#SBATCH --job-name=reverse-reap-q38
#SBATCH --partition=rtxpro6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:rtxpro6k:8
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
unset SLURM_EXPORT_ENV

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: reverse_reap.slurm REPO CONFIG SOURCE_PLAN MODEL_DIR STATE_DIR [THINKING_CONFIG] [THROUGH_TASK] [RUN_ROOT]" >&2
  exit 64
fi

repo_dir=$1
config_path=$2
source_plan=$3
model_dir=$4
state_dir=$5
thinking_config=${6:-}
through_task=${7:-}
run_root=${8:-}

module purge
module load cuda/12.8
module load python

cd "$repo_dir"
export UV_CACHE_DIR="${TMPDIR:?FAU must provide TMPDIR}/reverse-reap-uv-cache"
export HF_HOME="${TMPDIR}/reverse-reap-hf-cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$state_dir"

materialized_plan="$state_dir/fau-materialized-plan.yaml"
materialize=(uv run --frozen --no-sync python scripts/fau/materialize_plan.py
  "$source_plan" "$materialized_plan" --config "$config_path" --model-dir "$model_dir"
  --execution-mode fau-slurm)
if [[ -n "$thinking_config" ]]; then
  materialize+=(--thinking-config "$thinking_config")
fi
if [[ -n "$through_task" ]]; then
  materialize+=(--through-task "$through_task")
fi
if [[ -n "$run_root" ]]; then
  materialize+=(--run-root "$run_root")
fi
"${materialize[@]}"

srun uv run --frozen --no-sync reverse-reap run-all \
  "$config_path" "$materialized_plan" "$state_dir" \
  --heartbeat-seconds 30 --stale-after-seconds 180
