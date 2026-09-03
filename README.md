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
   # Optional override; compose.yaml already pins PyTorch 2.7.1/CUDA 12.8 by digest.
   export REVERSE_REAP_BASE_IMAGE='pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2'
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

## SWE-bench scoring boundary

Repository-repair responses are not treated as scoreable until the official SWE-bench
Docker harness completes them. The harness is pinned to repository revision
`02e7a74ffd0b707aab73d203fe87bdc7c76afc8e`.

Export one generated condition:

```bash
reverse-reap export-swebench runs/v0/baseline-validation-a.jsonl \
  runs/v0/swebench/c0-a-predictions.jsonl --model-name qwen35a3b-c0-a
```

Run the official harness in a separate CPU/Docker environment. Generated patches are
untrusted; do not run them directly on the GPU host filesystem:

```bash
git clone https://github.com/SWE-bench/SWE-bench.git /opt/SWE-bench
git -C /opt/SWE-bench checkout 02e7a74ffd0b707aab73d203fe87bdc7c76afc8e
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --split test \
  --predictions_path runs/v0/swebench/c0-a-predictions.jsonl \
  --max_workers 4 --cache_level env --run_id qwen35a3b-c0-a
```

Merge the official report back into the generated condition:

```bash
reverse-reap merge-swebench \
  runs/v0/baseline-validation-a.jsonl \
  qwen35a3b-c0-a.json \
  runs/v0/baseline-validation-a-scored.jsonl
```

The merge rejects foreign instance IDs, refuses overwrite, retains incomplete/error items,
and passes the scoreability gate only at 95% or above. Use the scored files for determinism
and causal comparison. The official harness requires substantial CPU storage; keep its image
cache outside the model volume.

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
