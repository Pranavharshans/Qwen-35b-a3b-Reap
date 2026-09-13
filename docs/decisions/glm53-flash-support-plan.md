# GLM-5.3-Flash-BF16 support plan

Status: infrastructure-only; no GLM weights, GPU run, Slurm job, or publication is
authorized by this plan. The exact checkpoint Gate A remains unverified.

## Scope and gates

This plan adds a GLM-specific dispatch boundary around the existing governed
`reverse-reap run-all` controller. It does not fork the science pipeline, transfer
Qwen evidence, alter Qwen run state, train or merge weights, or publish artifacts.
The two supported modes are:

- `direct`: a cloud or other CUDA host, with the generic `glm53-direct` aggregate-VRAM
  preflight; it never requires or invokes `sbatch`.
- `fau-slurm`: the existing FAU Alex contract, one `rtxpro6k` node with six RTX PRO
  6000 GPUs, routed through the existing submission helper; preview never invokes
  `sbatch`.

Preview is the default. `--execute` is the explicit launch gate. Before any real
execution, the operator must approve the exact checkpoint and budget, pass metadata
preflight, stage weights in the supplied model directory, and review the rendered
command. Stop on model/revision drift, failed hardware or budget gates, stale state,
repeated failures, or any request to publish weights. The exact checkpoint/revision
for this draft is `zai-org/GLM-5.3-Flash-BF16` at
`a5b45eb41df6402735dedc900be14a42e8d5e538`.

## Tasks

### RR-GLM53-01 — donor contract and metadata guards

status: `COMPLETE`

task_id: `RR-GLM53-01`

objective: Register the exact GLM BF16 donor and validate its metadata/index shape
without downloading full weights.

input files and hashes:

- `configs/glm53-flash-bf16.template.yaml` — SHA-256 `28e8486da347ab573941e1e09f61ac9a0fd4323334a75393bb592d3c6244dccd`
- `configs/smoke-glm53-flash-bf16.yaml` — SHA-256 `698a241f1fe24a9d0714e619de21427617df8c8b69f13896053c22bdf6a27641`
- `src/reverse_reap/donors.py` — SHA-256 `d7bbd248d185621f5b919daa2b1227cedd023f3d526021752dc2ea577d5bed7f`
- `src/reverse_reap/config.py` — SHA-256 `52858a506abc751f0508a95c2de2aa95b6cfda3be2cf34ed2e0bfa3d463ab3d6`
- `src/reverse_reap/model_preflight.py` — SHA-256 `9e2ea2b303bd0633a07af43ffa7936c31ddf66e6063092332ba90fbcb1fbecd4`

expected outputs: A registered donor contract, exact revision/config identity
checks, sparse-layer-aware metadata validation, and per-expert BF16 index checks.

definition of done: Hardware-free tests enforce model ID, pinned revision,
45 decoder layers with dense layers 0–2, 42 sparse MoE layers at absolute layers
3–44, 288 routed experts, top-8 routing, and the per-expert tensor layout.

validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run --frozen --no-sync pytest -q tests/test_config.py tests/test_model_preflight.py tests/test_extraction.py tests/test_qwen35.py`

estimated GPU hours: `0`

estimated storage: `<1 MiB` committed; metadata-only fixtures are temporary.

dependencies: Official metadata at the exact revision; no full weight download.

failure behavior: Reject identity, revision, architecture, sparse-layer, or tensor
layout drift before weight download and leave existing Qwen state untouched.

### RR-GLM53-02 — exact-checkpoint runtime adapter

status: `DEFERRED`

task_id: `RR-GLM53-02`

objective: Prove that the existing runtime/instrumentation/controller can execute
the GLM model and preserve absolute sparse layer identities.

input files and hashes:

- `src/reverse_reap/qwen35.py` — SHA-256 `e1aeecc2301dc190804257124dace0248905d61c545037be74e4a16b05395aa5`
- `src/reverse_reap/runtime.py` — SHA-256 `6ed76c1443ee3ff7506d7e94cc238cd64f7bda8ab4d508ca5b24bdf0fb72c091`
- `src/reverse_reap/instrumentation.py` — SHA-256 `2ffc8a6c9633c182239d60690a03c007417604b3eb4d3c1ee94107073565a833`
- `src/reverse_reap/extraction.py` — SHA-256 `db28e854e780a82d9442990a6d2e50a65101c130f99b4e87c8010b8f678934a8`

expected outputs: An exact-checkpoint Gate A probe, validated runtime hooks, and
only then a separately approved GLM calibration/evaluation task.

definition of done: A real pinned checkpoint proves logits/no-op behavior, module
mapping, top-k route counts, and safe extraction on absolute sparse layers 3–44.

validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run --frozen --no-sync pytest -q tests/test_runtime.py tests/test_instrumentation.py tests/test_extraction.py` plus the exact-checkpoint Gate A command recorded by the operator.

estimated GPU hours: `0` until separately approved; Gate A budget must be estimated
from a bounded pilot before launch.

estimated storage: `0` committed; model and run artifacts remain external and ignored.

dependencies: RR-GLM53-01, approved exact checkpoint, compatible Transformers/PyTorch,
and fresh human authorization for GLM GPU work.

failure behavior: Remain `DEFERRED`/`WAITING_FOR_HUMAN` on missing exact metadata,
unsupported module mapping, OOM, or failed Gate A; do not weaken gates or reuse Qwen
evidence.

### RR-GLM53-03 — two-mode execution infrastructure

status: `ACTIVE`

task_id: `RR-GLM53-03`

objective: Provide one fail-closed GLM launcher with preview-by-default direct and
FAU-Slurm modes, shared plan materialization, isolated run IDs, and operator docs.

input files and hashes:

- `configs/execution-plan-smoke.yaml` — SHA-256 `32ada0f1c610fe8af9618ea66aa6a751d0988416e1ae80f2f3b25caa074b5b62`
- `scripts/launch_glm53.py` — SHA-256 `b7b3e80f7ffe722cc4ce418d0c7fd213e205287afa81e5d382c43bff2f19c7a1`
- `src/reverse_reap/plan_materializer.py` — SHA-256 `284bbeb5c9fe1cd8875b5e262d38538e3d413f182b6e88d30d4fec75b77358d9`
- `scripts/fau/materialize_plan.py` — SHA-256 `5021476185dd8d60f8633462271ab1c358ee57d745c3eb6fb730edee3521ece2`
- `scripts/fau/submit_reverse_reap.sh` — SHA-256 `f5ca01537e81d9222c7da1bb8b63dfd02e1d41cd7d8bedf8772205a2643adc04`
- `scripts/fau/reverse_reap.slurm` — SHA-256 `afe396058e0879ac79a42ec143f4e218f84b0ab921b830cdb985453a88bb109d`
- `scripts/gpu_preflight.py` — SHA-256 `386ef010a63dfb11cd8a8a3f4e5c01b0bc0bab95d254b02dbcb603e16238cf32`
- `tests/test_glm53_launcher.py` — SHA-256 `e9d5f6ffc30a6d647ee64d049bf54dd0e97ac75837a77a226c62d2e89495e6fe`
- `tests/test_fau_slurm.py` — SHA-256 `fce5e9b74a79583e60ed03a8bb2ece63af5a886ff21ebedc50e8480815a642d8`
- `tests/test_gpu_preflight.py` — SHA-256 `75853af8811157a6ef4ab3b576dc454023f074ed7e08d6abae812fdf384d7bcc`

expected outputs: A render-only launcher boundary, run-specific GLM config/plan
materialization on explicit execution, unchanged Qwen default materialization, FAU
job naming/run-root plumbing, generic GLM preflight, hardware-free tests, and this
operator documentation.

definition of done: Both modes reject the wrong model or missing paths before render;
preview creates no state and never calls `sbatch`; direct execution maps to one fake
`run-all` command; FAU execution maps to one fake `--submit` command; materialized
plans contain the GLM path/identity, top-8, 42 sparse layers at 3–44, and the correct
mode profile; shell syntax, lint, tests, and diff checks pass.

validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run --frozen --no-sync pytest -q tests/test_glm53_launcher.py tests/test_gpu_preflight.py tests/test_fau_slurm.py && bash -n scripts/launch_glm53.sh scripts/fau/submit_reverse_reap.sh scripts/fau/reverse_reap.slurm && UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run --frozen --no-sync ruff check scripts/launch_glm53.py src/reverse_reap/plan_materializer.py scripts/fau/materialize_plan.py scripts/gpu_preflight.py tests/test_glm53_launcher.py tests/test_gpu_preflight.py tests/test_fau_slurm.py && git diff --check`

estimated GPU hours: `0` for implementation and hardware-free validation.

estimated storage: `<1 MiB` committed; no model weights or run artifacts.

dependencies: RR-GLM53-01; existing governed controller/FAU contract; explicit human
approval before any `--execute` or Slurm submission.

failure behavior: Fail closed before rendering on invalid identity, paths, plan, run
ID, profile, or state; preserve source plans and Qwen run/output paths; never fall
back to `/models/qwen`, a Qwen config, an implicit `sbatch`, or an alternate resource
contract.

## Current limitations

The implementation is infrastructure-only and does not verify the exact GLM
checkpoint's Gate A, module mapping, throughput, memory fit, causal gates, or output
quality. The `glm53-direct` profile requires CUDA, torch 2.11+, CUDA runtime 12.8+,
at least 700 GiB aggregate GPU memory, and 120 GiB free run-filesystem space; this is
a conservative BF16 feasibility floor, not a proof that the checkpoint loads. The
model directory may be pre-staged on separate storage. The checked-in budget values
are placeholders and require review before a real run. No weights are published by
the launcher, controller, tests, or documentation.
