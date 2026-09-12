# Qwen3.8 FAU two-pass generation plan

Status: implementation plan (2026-09-12)

## RR-Q38-TP-01

- task_id: `RR-Q38-TP-01`
- objective: Make targeted bridge-data capture donor-neutral across the approved Qwen3.5 and Qwen3.8 checkpoints.
- input files and hashes: current bridge-capture implementation, approved donor registry, frozen dataset manifest, and a passed Gate C candidate manifest.
- expected outputs: Qwen3.8-valid capture manifests and BF16 target-vector shards with exact donor/config/dataset/candidate provenance.
- definition of done: no Qwen3.5 expert IDs or hidden size are assumed; candidates are range-checked against the selected donor contract; Qwen3.5 behavior remains supported.
- validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q tests/test_bridge_capture.py tests/test_fau_slurm.py tests/test_plans.py`
- estimated GPU hours: `0` for implementation and tests.
- estimated storage: `< 1 MB` repository changes.
- dependencies: Qwen3.8 donor contract and fused-MoE adapter.
- failure behavior: fail closed before GPU execution on donor, revision, candidate, dataset, or vector-shape mismatch.

## RR-Q38-TP-02

- task_id: `RR-Q38-TP-02`
- objective: Add an FAU-ready second-pass plan and concise two-pass submission workflow while keeping scoring external.
- input files and hashes: Qwen3.8 config, frozen full dataset, first-pass passed Gate C candidate manifest, exact BF16 model directory.
- expected outputs: Qwen3.8 bridge-capture config, plan, Slurm materialization, hash-bound handoff bundle, and operator documentation.
- definition of done: the first generation plan and second capture plan can each be dry-run or submitted in one command; each gets a separate immutable run namespace; no scorer, bridge trainer, or job submission runs during tests.
- validation command: `bash -n scripts/fau/submit_reverse_reap.sh && UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q tests/test_fau_slurm.py tests/test_plans.py`
- estimated GPU hours: `0` for implementation; the checked-in second-pass estimate is bounded by its configuration and controller reserve.
- estimated storage: `< 1 MB` repository changes; runtime capture allowance is declared in configuration.
- dependencies: `RR-Q38-TP-01` and a human-frozen dataset/candidate manifest before real execution.
- failure behavior: dry-run by default; refuse missing paths; no automatic scoring, training, or submission.

## RR-Q38-TP-03

- task_id: `RR-Q38-TP-03`
- objective: Validate, commit only scoped files, and push the current feature branch.
- input files and hashes: outputs of `RR-Q38-TP-01` and `RR-Q38-TP-02`.
- expected outputs: test evidence and matching local/remote commit.
- definition of done: narrow and full hardware-free tests pass, unrelated user changes remain unstaged, and the scoped commit is pushed.
- validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q && UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run ruff check src tests scripts`
- estimated GPU hours: `0`.
- estimated storage: test cache only under `/tmp`.
- dependencies: `RR-Q38-TP-01`, `RR-Q38-TP-02`.
- failure behavior: report failures and do not claim exact-checkpoint GPU validation.
