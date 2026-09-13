#!/bin/bash
set -euo pipefail

submit=false
thinking_config=""
through_task=""
job_name=""
run_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) submit=true; shift ;;
    --thinking-config) thinking_config=$(realpath "$2"); shift 2 ;;
    --through-task) through_task=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --run-root) run_root=$(realpath "$2"); shift 2 ;;
    *) break ;;
  esac
done
if [[ $# -ne 4 ]]; then
  echo "usage: $0 [--submit] [--job-name NAME] [--run-root PATH] [--thinking-config PATH] [--through-task ID] CONFIG SOURCE_PLAN MODEL_DIR STATE_DIR" >&2
  exit 64
fi
if [[ -n "$job_name" && ! "$job_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]]; then
  echo "invalid job name: $job_name" >&2
  exit 64
fi

repo_dir=$(git rev-parse --show-toplevel)
config_path=$(realpath "$1")
plan_path=$(realpath "$2")
model_dir=$(realpath "$3")
state_input=$4
if [[ -e "$state_input" ]]; then
  [[ -d "$state_input" ]] || { echo "state path is not a directory: $state_input" >&2; exit 66; }
  state_dir=$(realpath "$state_input")
else
  state_parent=$(dirname "$state_input")
  state_base=$(basename "$state_input")
  [[ -d "$state_parent" ]] || { echo "state directory parent is missing: $state_parent" >&2; exit 66; }
  state_dir="$(cd "$state_parent" && pwd -P)/$state_base"
  if [[ "$submit" == true ]]; then
    mkdir -p "$state_dir"
  fi
fi
job_script="$repo_dir/scripts/fau/reverse_reap.slurm"

for path in "$config_path" "$plan_path" "$model_dir" "$job_script"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 66; }
done
if [[ -n "$thinking_config" && ! -e "$thinking_config" ]]; then
  echo "missing thinking config: $thinking_config" >&2
  exit 66
fi

command=(sbatch)
if [[ -n "$job_name" ]]; then
  command+=(--job-name="$job_name")
fi
command+=("$job_script" "$repo_dir" "$config_path" "$plan_path" "$model_dir" "$state_dir")
if [[ -n "$thinking_config" ]]; then
  command+=("$thinking_config")
else
  command+=("")
fi
if [[ -n "$through_task" ]]; then
  command+=("$through_task")
elif [[ -n "$run_root" ]]; then
  command+=("")
fi
if [[ -n "$run_root" ]]; then
  command+=("$run_root")
fi
printf 'FAU submission command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$submit" == true ]]; then
  command -v sbatch >/dev/null || { echo "sbatch is unavailable; run this on an FAU login node" >&2; exit 69; }
  "${command[@]}"
else
  echo "dry run only; add --submit to submit"
fi
