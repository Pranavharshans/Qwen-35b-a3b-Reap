# GLM-5.3-Flash-BF16 execution guide

This is a separately governed infrastructure track for
`zai-org/GLM-5.3-Flash-BF16` at immutable revision
`a5b45eb41df6402735dedc900be14a42e8d5e538`. It does not reuse Qwen run IDs,
telemetry, candidates, controls, evidence labels, or outputs. The exact GLM checkpoint has
not passed Gate A in this repository, so the only validated claim today is hardware-free
configuration, plan, and launcher support.

## Safety boundary

The launcher has exactly two modes and previews by default:

- `direct` runs the normal `reverse-reap run-all` controller on a cloud or other CUDA host;
  it does not require or call Slurm.
- `fau-slurm` uses the existing FAU Alex batch contract: one `rtxpro6k` node with eight RTX
  PRO 6000 GPUs. Preview does not call `sbatch`.

`--execute` is the state-change gate. An execution also requires `--run-id` (or a config whose
`run_id` is already pinned), so the command reviewed in preview cannot silently move to a new
run directory. No launcher command provisions hardware, buys credits, downloads weights, or
publishes artifacts.

## Prerequisites

Use Python 3.12, `uv`, the frozen lockfile, and a local clone of this repository:

```bash
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv sync --frozen --extra gpu
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run --frozen --no-sync \
  reverse-reap validate-config configs/smoke-glm53-flash-bf16.yaml
```

For direct execution, the `glm53-direct` hardware preflight is vendor-neutral but fail-closed:
CUDA 12.8+, PyTorch 2.11+, at least 700 GiB aggregate GPU memory, and at least 120 GiB free on
the run filesystem. This is a conservative feasibility floor for the roughly 640 GB of raw
BF16 parameters plus runtime overhead, not proof that a particular topology will load. The
model directory may live on separately provisioned storage. Alex uses the stricter checked-in
eight-RTX-PRO-6000 profile.

## Metadata first, weights second

Download only the small official metadata and weight index into a metadata directory, then
pin a new config from the template and inspect the report:

```bash
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run --frozen --no-sync \
  reverse-reap preflight-model \
  configs/glm53-flash-bf16.template.yaml \
  /absolute/path/to/glm53-pinned.yaml \
  /absolute/path/to/glm53-metadata \
  /absolute/path/to/glm53-model-preflight.json
```

Only after the report passes, storage is checked, and the exact download is approved, stage
the verified weights into an empty external model directory:

```bash
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run --frozen --no-sync \
  reverse-reap download-weights \
  /absolute/path/to/glm53-model-preflight.json \
  /absolute/path/to/GLM-5.3-Flash-BF16
```

The weight download is a separate network/storage action. It is not performed by preview,
tests, or the launcher. Keep model shards and generated run artifacts outside Git.

## Mode 1: direct CUDA host

Preview with an already staged model directory and an isolated GLM state root:

```bash
bash scripts/launch_glm53.sh --direct \
  --config /absolute/path/to/glm53-pinned.yaml \
  --plan configs/execution-plan-smoke.yaml \
  --model-dir /absolute/path/to/GLM-5.3-Flash-BF16 \
  --state-root /absolute/path/to/glm53-runs
```

The preview validates the exact GLM model identity, BF16 precision, paths, execution plan,
budget fields, and prospective run state. It renders a run-specific config and donor-aware
plan path without creating either. Copy the printed run ID only after reviewing all paths and
the budget. To start the governed direct controller:

```bash
bash scripts/launch_glm53.sh --direct --execute \
  --run-id '<RUN_ID_FROM_PREVIEW>' \
  --config /absolute/path/to/glm53-pinned.yaml \
  --plan configs/execution-plan-smoke.yaml \
  --model-dir /absolute/path/to/GLM-5.3-Flash-BF16 \
  --state-root /absolute/path/to/glm53-runs
```

Execution writes the pinned config, materialized plan, and launch record under that GLM run
root, then invokes the existing controller. It does not invoke `sbatch`.

## Mode 2: FAU Alex Slurm

Run the preview on an Alex login node with absolute cluster paths:

```bash
bash scripts/launch_glm53.sh --fau-slurm \
  --config /absolute/cluster/path/glm53-pinned.yaml \
  --plan configs/execution-plan-smoke.yaml \
  --model-dir /absolute/cluster/path/GLM-5.3-Flash-BF16 \
  --state-root /absolute/cluster/path/glm53-runs
```

The output is a future submission command with job name `reverse-reap-glm53`; preview never
looks up or invokes `sbatch`. After reviewing the exact run ID, configuration, allocation,
paths, budget, and rendered command, an explicitly authorized submission is:

```bash
bash scripts/launch_glm53.sh --fau-slurm --execute \
  --run-id '<RUN_ID_FROM_PREVIEW>' \
  --config /absolute/cluster/path/glm53-pinned.yaml \
  --plan configs/execution-plan-smoke.yaml \
  --model-dir /absolute/cluster/path/GLM-5.3-Flash-BF16 \
  --state-root /absolute/cluster/path/glm53-runs
```

That command alone maps to `sbatch`. It does not prove that the allocation can load or
instrument the model. Never submit, cancel, requeue, or otherwise change an Alex job without
fresh authorization for that specific action.

## Run state, budgets, and recovery

Each mode resolves a GLM-specific run ID and stores state beneath the supplied root. Source
plans are never mutated. Refuse an existing non-empty run unless intentionally resuming with
`--resume`; a resume must reuse the same pinned run ID and artifacts.

Before execution, replace the checked-in placeholder accounting values with the actual GPU
hours, provider/allocation rate, storage ceiling, and deadline. A changed model revision,
dataset manifest, prompt template, decoding setting, intervention, or budget creates a new
run. Keep the 20% reserve enforced by the controller.

During a run, inspect controller state and Slurm logs (for FAU) at least every 15 minutes.
Stop safely on failed architecture or GPU preflight, revision/hash drift, no forward progress
for 15 minutes, repeated OOM, more than two identical failure signatures, disk pressure,
telemetry validation failure, or projected budget/deadline overrun. Do not continue from a
partially written or mismatched artifact.

## Evidence and publication limits

A launcher preview or successful process exit is not scientific validation. The next
permitted scientific action is a separately authorized, bounded exact-checkpoint Gate A
probe. Until it passes, do not start calibration, freeze candidates, make causal claims, or
transfer Qwen results to GLM.

“Push to hub” for this implementation means pushing source code and documentation to the
configured GitHub `origin`. It does not mean uploading model weights, extracted experts,
datasets, telemetry, or run bundles to Hugging Face. No publication is automatic.
