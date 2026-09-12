# Reverse-REAP for supported Qwen MoE donors

This repository localizes, causally tests, and losslessly extracts coding-critical routed
experts from the official `Qwen/Qwen3.5-35B-A3B` checkpoint. REAP and routing statistics
produce candidates; only frozen ablations plus untouched replication can support the
`coding-critical-v0` label.

The original Qwen3.5 experiment and evidence remain immutable. A separately approved
compatibility path supports the official BF16 `Qwen/Qwen3.8-Flash-Next` checkpoint at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. Qwen3.5 candidates and conclusions do not transfer
to Qwen3.8: it requires a new run ID, telemetry, candidates, controls, causal validation,
replication, and extraction.

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

## Qwen3.8-Flash-Next compatibility

The Qwen3.8 path validates the official `qwen4_exp` text tower: 48 MoE layers, 512 routed
experts per layer, top-10 routing, hidden size 2,560, routed/shared expert width 640, and the
fused `gate_up_proj`/`down_proj` tensor layout. The metadata-first workflow remains the same:

```bash
uv run reverse-reap validate-config configs/smoke-qwen38-flash-next-bf16.yaml
uv run reverse-reap preflight-model \
  configs/qwen38-flash-next-bf16.template.yaml \
  configs/smoke-qwen38-flash-next-bf16.yaml \
  /path/to/qwen38-metadata runs/qwen38/model-preflight.json
uv run reverse-reap download-weights \
  runs/qwen38/model-preflight.json /path/to/Qwen3.8-Flash-Next
```

The pinned config makes `preflight-model` refuse revision drift rather than silently rewrite
it. Support is currently hardware-free and metadata/index validated; an exact-checkpoint Gate
A probe is still required before calibration or an expert claim.

### Official Qwen3.8 FP8 checkpoint

`Qwen/Qwen3.8-Flash-Next-FP8` is a separate donor at immutable revision
`236dfdf285828023ca3bcd3f37366c58a3469b13`. It uses dynamic activations and 128x128
blockwise FP8 weights. Its per-expert gate/up/down weights and all three inverse-scale tensors
are validated and extracted byte-for-byte; they are never cast, fused, or treated as BF16
source tensors.

Metadata-first preflight remains hardware-free and does not download the 131 weight shards:

```bash
uv run reverse-reap preflight-model \
  configs/qwen38-flash-next-fp8.template.yaml \
  configs/smoke-qwen38-flash-next-fp8.yaml \
  /cluster/metadata/qwen38-fp8 runs/qwen38-fp8/model-preflight.json
```

Use `configs/qwen38-flash-next-fp8-full.yaml` for direct FAU execution,
`configs/qwen38-flash-next-fp8-full-thinking.yaml` for the separate thinking condition, and
`configs/qwen38-flash-next-fp8-bridge-capture.yaml` for pass-2 target capture. FP8 requires
its own telemetry, candidates, causal validation, and replication; BF16 results do not transfer.

## FAU Alex Slurm execution

The FAU path uses the same single-writer `run-all` controller and source plan as the regular
path. It requests one complete `rtxpro6k` node (8 × RTX PRO 6000, 96 GiB each), loads CUDA
12.8 and Python through environment modules, keeps caches on node-local `$TMPDIR`, and uses a
clean exported environment. Prepare the environment and weights on Alex, then preview and
submit:

```bash
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv sync --frozen --extra gpu

scripts/fau/submit_reverse_reap.sh \
  configs/smoke-qwen38-flash-next-bf16.yaml configs/execution-plan-smoke.yaml \
  /absolute/cluster/path/Qwen3.8-Flash-Next runs/qwen38/state

scripts/fau/submit_reverse_reap.sh --submit \
  configs/smoke-qwen38-flash-next-bf16.yaml configs/execution-plan-smoke.yaml \
  /absolute/cluster/path/Qwen3.8-Flash-Next runs/qwen38/state
```

The first command is a dry run. The second calls `sbatch` and must run on an FAU login node.
No job is submitted by setup or tests. The job materializes a run-specific plan with absolute
cluster paths and the eight-GPU FAU preflight; it never mutates the source plan.

### Qwen3.8 two-pass FAU workflow

Scoring may remain on a separate VM. FAU produces immutable generation/capture artifacts and
hash-bound handoffs; it does not execute generated code in this workflow.

After reviewing the dataset catalog, dry-run the first pass through frozen expert-candidate
analysis. The cutoff prevents FAU from entering the downstream scoring/causal tasks:

```bash
scripts/fau/submit_reverse_reap.sh \
  --through-task candidate-analysis \
  configs/qwen38-flash-next-bf16-full.yaml configs/execution-plan-v0.yaml \
  /absolute/cluster/path/Qwen3.8-Flash-Next runs/qwen38/pass1-state
```

Add `--submit` to that command only after reviewing the rendered command and run budget. When
pass 1 has produced a passed, frozen Qwen3.8 Gate C artifact, transfer generation artifacts
to the scoring VM as needed and copy the candidate artifact without modification to
`runs/qwen38/inputs/candidate-manifest.json`. Then dry-run the second, independently identified
teacher-forced capture pass:

```bash
scripts/fau/submit_reverse_reap.sh \
  configs/qwen38-flash-next-bf16-bridge-capture.yaml \
  configs/execution-plan-qwen38-bridge-capture.yaml \
  /absolute/cluster/path/Qwen3.8-Flash-Next runs/qwen38/pass2-state
```

Again, add `--submit` only for the authorized launch. Pass 2 records selected-expert input,
replayed output, weighted output, router identity/weight, token identity, and full provenance
in resumable BF16 shards. It ends with a hash-verified handoff manifest. It does not score,
train a bridge, publish weights, or reuse the pass-1 run ID.

The checked-in Qwen3.8 full configurations use allocation-accounting placeholder rates and a
future deadline. Review and pin the actual allocation budget, storage ceiling, deadline, and
dataset hash before any real job; changing any of them creates a new run ID.

## CPU analysis engines

The `analyze` stage has two engines with identical scientific semantics
(Gate C rule, permutation guard, fail-closed behavior, frozen artifacts):

- `--engine fast` (default): one-pass streaming aggregation into a compact
  per-(sample, layer, expert) table, a telemetry-SHA-256-keyed aggregate cache
  (`.cache/analysis/<sha>.npz`, reused only on an exact hash and metadata
  match), and vectorized NumPy bootstrap/permutation inference with one shared
  replicate loop across the whole cardinality grid.
- `--engine reference`: the original dict-based implementation, preserved as a
  scientific oracle.

```bash
uv run reverse-reap analyze telemetry.jsonl out/ --engine fast
uv run reverse-reap analyze telemetry.jsonl out/ --engine reference
uv run python scripts/benchmark_analysis.py --telemetry telemetry.jsonl \
    --output /tmp/analysis-bench --top-n 8 --grid 4 8 16
```

Equivalence is enforced by `tests/test_analysis_optimized.py`: expert
identities and ordering, bootstrap Jaccards (exactly equal) and intervals,
permutation null multisets and p-values, cardinality-grid decisions, candidate
and control memberships, determinism under a fixed seed, and fail-closed
parity. Known float-level limitations, documented there and in
`src/reverse_reap/analysis_fast.py`: floating-point summation order (NumPy
`bincount` versus per-key `np.mean`) can move boundary-tie comparisons by
~1e-16 on exactly symmetric fixtures, and the reference's
`unique_null_statistics` diagnostic can split mathematical ties into last-bit
variants that the vectorized engine collapses; real telemetry shows neither
effect.

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

## Opt-in bridge training v1

Bridge training is a separately governed follow-on for the frozen,
post-trained `Qwen/Qwen3.5-2B` host. It consumes only a verified donor handoff,
an explicitly mapped host-layer sidecar, and host hidden-state records. The
bridge trains input/output adapters and a capped residual gate around frozen
extracted SwiGLU experts; it never saves host weights. v1 is vector
supervision over precomputed host states, not host end-to-end language-model
training or an LM-loss claim. See
[`docs/bridge-training.md`](docs/bridge-training.md) and the intentionally
unlaunchable template [`configs/bridge-qwen35-2b.yaml`](configs/bridge-qwen35-2b.yaml).

The CPU-only preflight and split-repair commands are available as
`bridge-preflight`, `freeze-host-states`, and `repair-bridge-manifest`; the
GPU-facing host-state capture is explicit as `capture-host-states`.
`train-bridge` requires a fully hash-bound config and the optional GPU
dependencies. Random-expert control is reported unavailable until a separately
verified random extraction exists.

The paired coding evaluation is documented in
[`docs/bridge-benchmark.md`](docs/bridge-benchmark.md). One
`run-bridge-benchmark` command executes two deterministic base runs and two
deterministic trained-bridge runs on both a 25-item pilot and the frozen full
HumanEval+ set. Pilot score never suppresses the full tier; integrity and budget
failures still stop it. Raw generations can be scored later on the qualified
Docker/KVM boundary with `score-bridge-benchmark`.

## Official MBPP+ four-condition bridge benchmark

The MBPP+ follow-on uses the official EvalPlus v0.3.1 evaluator instead of the
repository's generic Python scorer. It loads the host once, strictly verifies
the trained bridge checkpoint and all four extracted frozen experts, then emits
four separate conditions: base and bridged inference with thinking disabled and
enabled. See [`docs/mbppplus-bridge-benchmark.md`](docs/mbppplus-bridge-benchmark.md)
and [`configs/mbppplus-bridge-qwen35-2b.yaml`](configs/mbppplus-bridge-qwen35-2b.yaml).

```bash
reverse-reap run-mbpp-bridge-benchmark /path/to/pinned-mbpp-benchmark.yaml
reverse-reap validate-mbpp-bridge-benchmark /path/to/pinned-mbpp-benchmark.yaml
```

Generated code is untrusted, so official scoring remains on a Docker-capable
scorer. Build the revision-labelled, digest-pinned image with
`scripts/prepare_evalplus_docker.py`, transfer the small run directory, and run:

```bash
reverse-reap score-mbpp-bridge-benchmark /path/to/pinned-mbpp-benchmark.yaml \
  --evalplus-image 'localhost:5000/reverse-reap-evalplus@sha256:<digest>'
```

The report keeps MBPP base tests, MBPP+ extended tests, and thinking modes
separate. It is a capability comparison, not causal evidence.

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
