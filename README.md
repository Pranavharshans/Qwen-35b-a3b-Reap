# Reverse-REAP for Qwen3.5-35B-A3B

This repository localizes, causally tests, and losslessly extracts coding-critical routed
experts from the official `Qwen/Qwen3.5-35B-A3B` checkpoint. REAP and routing statistics
produce candidates; only frozen ablations plus untouched replication can support the
`coding-critical-v0` label.

The v0 scope ends at expert extraction. Extracted tensors are not a standalone model and
cannot be inserted directly into a smaller host model without later representation-bridge
research.

## Fixed v0 contract

- Donor revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- Source and primary execution precision: BF16
- Architecture: 40 MoE layers, 256 routed experts per layer, top-8 routing
- Primary condition: thinking disabled, deterministic greedy decoding, batch size 1
- Intervention: zero selected weighted expert contributions without rerouting or renormalizing
- Hardware acceptance target: exactly four RTX 3090 GPUs with at least 24 GiB each

Read `AGENTS.md`, `prd.md`, and `roadmap.md` before changing or executing the experiment.

## Local validation

Python 3.12 and `uv` are required. These checks do not download or load the donor:

```bash
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv sync --frozen --extra dev
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run ruff check src tests scripts
```

The Torch instrumentation module is skipped when the local environment has no Torch. A
hardware-free pass therefore does not prove exact-checkpoint compatibility.

## Cheapest execution sequence

Use a provisioned four-RTX-3090 host with at least 100 GB free for model files and more space
for run artifacts. Update only the budget, provider rate, storage limit, and deadline in the
pinned configuration before resolving a run ID. A configuration change requires a new run.

1. Verify GPUs and storage before downloading weights:

   ```bash
   python scripts/gpu_preflight.py --output runs/smoke/preflight.json
   python scripts/validate_artifact.py preflight runs/smoke/preflight.json
   ```

2. Resolve and verify donor metadata. This also regenerates the pinned config from the
   template and refuses architecture drift:

   ```bash
   reverse-reap preflight-model \
     configs/smoke-3090-bf16.yaml \
     configs/pinned-3090-bf16.yaml \
     /models/qwen-metadata \
     runs/smoke/model-preflight.json
   ```

3. Download the 14 verified BF16 weight shards only after both preflights pass:

   ```bash
   reverse-reap download-weights runs/smoke/model-preflight.json /models/qwen
   ```

4. Freeze the source dataset and nested cost tiers:

   ```bash
   reverse-reap fetch-datasets configs/dataset-sources.yaml datasets/manifests/source-full.jsonl
   reverse-reap freeze-dataset-tiers \
     datasets/manifests/source-full.jsonl datasets/manifests
   python scripts/validate_dataset.py datasets/manifests/full.jsonl --full
   ```

5. Build the isolated code evaluator and main runtime. Pin the main image by digest in
   `REVERSE_REAP_BASE_IMAGE`; do not use a floating CUDA tag for a scientific run:

   ```bash
   docker build -t reverse-reap-evaluator:local evaluator
   export REVERSE_REAP_EVALUATOR_IMAGE=reverse-reap-evaluator:local
   export REVERSE_REAP_BASE_IMAGE='REPLACE_WITH_APPROVED_IMAGE@sha256:REPLACE'
   export REVERSE_REAP_MODEL_DIR=/absolute/path/to/model
   docker compose build reverse-reap
   ```

6. Run only the smoke graph first:

   ```bash
   docker compose run --rm reverse-reap \
     run-all configs/pinned-3090-bf16.yaml configs/execution-plan-smoke.yaml \
     runs/smoke/state --heartbeat-seconds 30 --stale-after-seconds 180
   ```

Stop on a failed GPU preflight, instrumentation Gate A, telemetry invariant, determinism
check, dataset gate, budget gate, or repeated failure signature. Do not proceed to the full
graph merely because the process completed; inspect the evidence label and gate report.

## Full governed graph

Generate the complete plan only after the smoke run passes:

```bash
reverse-reap make-full-plan configs/execution-plan-full.yaml
reverse-reap run-all configs/pinned-3090-bf16.yaml configs/execution-plan-full.yaml \
  runs/v0/state --heartbeat-seconds 30 --stale-after-seconds 180
```

The controller writes atomic task state, hashes inputs and outputs, reserves 20% of the
declared budget, retries an identical failure at most twice, and records periodic heartbeats.
No model weights or extracted tensors are committed or uploaded automatically.

## Evidence outputs

- Routing records: one row per token, layer, and top-k route
- Candidate manifest: hashed and immutable before causal validation
- Controls: 20 layer-matched, 20 frequency-matched, highest-frequency, and
  lowest-differential sets
- Causal report: coding/control paired drops, uncertainty, random percentile, and replication
- Extraction bundle: safetensors, source-key map, tensor hashes, and independent byte check
- Run bundle: configurations, state, artifact hashes, gate outcomes, and limitations

Checked-in JSON Schemas under `schemas/` define the configuration, state, routing,
candidate, and extraction contracts.
