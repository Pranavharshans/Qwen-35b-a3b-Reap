#!/bin/bash
set -euo pipefail

submit=false
if [[ ${1:-} == "--submit" ]]; then
  submit=true
  shift
fi
if [[ $# -ne 4 ]]; then
  echo "usage: $0 [--submit] CONFIG SOURCE_PLAN MODEL_DIR STATE_DIR" >&2
  exit 64
fi

repo_dir=$(git rev-parse --show-toplevel)
config_path=$(realpath "$1")
plan_path=$(realpath "$2")
model_dir=$(realpath "$3")
mkdir -p "$4"
state_dir=$(realpath "$4")
job_script="$repo_dir/scripts/fau/reverse_reap.slurm"

for path in "$config_path" "$plan_path" "$model_dir" "$job_script"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 66; }
done

command=(sbatch "$job_script" "$repo_dir" "$config_path" "$plan_path" "$model_dir" "$state_dir")
printf 'FAU submission command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$submit" == true ]]; then
  command -v sbatch >/dev/null || { echo "sbatch is unavailable; run this on an FAU login node" >&2; exit 69; }
  "${command[@]}"
else
  echo "dry run only; add --submit to submit"
fi
