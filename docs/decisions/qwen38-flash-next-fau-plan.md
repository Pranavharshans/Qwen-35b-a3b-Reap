# Qwen3.8-Flash-Next and FAU execution plan

Status: implementation plan (2026-09-12)

## RR-Q38-01

- task_id: `RR-Q38-01`
- objective: Add a fail-closed donor registry and fused-MoE adapter for the official BF16 `Qwen/Qwen3.8-Flash-Next` checkpoint while preserving the existing Qwen3.5 path.
- input files and hashes: `AGENTS.md` `df645b54...`; `prd.md` `86f6d085...`; `roadmap.md` `a296c081...`; `src/reverse_reap/config.py` `84c41658...`; `src/reverse_reap/model_preflight.py` `f399c054...`; `src/reverse_reap/qwen35.py` `68ceabfb...`; official model revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- expected outputs: model-family contract, generic fused-expert adapter, Qwen3.8 config template, schema update, hardware-free tests.
- definition of done: both donors validate by exact model ID/revision metadata; incompatible metadata fails; extraction derives dimensions from the approved contract; no existing Qwen3.5 behavior regresses.
- validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q tests/test_config.py tests/test_qwen35.py tests/test_model_preflight.py tests/test_extraction.py`
- estimated GPU hours: `0`
- estimated storage: `< 1 MB` repository changes; metadata-only preflight later requires negligible storage.
- dependencies: none.
- failure behavior: stop with unsupported/architecture-mismatch evidence; do not download weights or claim runtime validation.

## RR-Q38-02

- task_id: `RR-Q38-02`
- objective: Add a native FAU Alex Slurm submission wrapper for the same governed CLI plan used locally.
- input files and hashes: `README.md` `f93662e5...`; FAU Alex documentation checked 2026-09-12; `configs/execution-plan-smoke.yaml`; `compose.yaml`; `Dockerfile`.
- expected outputs: checked-in Slurm profile, render/submit command, generated-script validation, concise operator documentation.
- definition of done: a user can validate, render, and submit in a few commands; generated jobs use `#!/bin/bash -l`, `--export=NONE`, `unset SLURM_EXPORT_ENV`, single-node `rtxpro6k`, CUDA 12.8+, node-local `$TMPDIR`, explicit repository/model/run paths, and the normal `reverse-reap run-all` controller.
- validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q tests/test_fau_slurm.py && bash -n scripts/fau/submit_reverse_reap.sh`
- estimated GPU hours: `0` (no job submission in this implementation task)
- estimated storage: `< 1 MB`.
- dependencies: `RR-Q38-01`.
- failure behavior: refuse invalid/missing paths or unsupported resource profiles before calling `sbatch`; never submit from tests.

## RR-Q38-03

- task_id: `RR-Q38-03`
- objective: Run hardware-free regression validation, inspect the isolated diff, commit only feature files, and push the current branch.
- input files and hashes: outputs of `RR-Q38-01` and `RR-Q38-02`.
- expected outputs: test evidence, local commit, matching remote commit.
- definition of done: narrow tests and full hardware-free suite pass; unrelated dirty files remain unstaged; push succeeds.
- validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q && UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run ruff check src tests scripts`
- estimated GPU hours: `0`
- estimated storage: test/cache only in `/tmp`.
- dependencies: `RR-Q38-01`, `RR-Q38-02`.
- failure behavior: report exact failed command and do not represent unsupported exact-checkpoint/GPU behavior as validated.
