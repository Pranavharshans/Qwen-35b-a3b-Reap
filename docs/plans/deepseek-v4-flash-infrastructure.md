# DeepSeek V4 Flash infrastructure implementation plan

Status: approved by fresh human instruction; infrastructure only; no GPU launch or Slurm
submission is authorized by this plan.

Scope boundary: this is a separate DeepSeek track. It must not change the Qwen v0 donor,
schemas, configurations, scientific claims, run state, datasets, or extraction artifacts.

## DSV4-INFRA-001 — Model-neutral launch renderer

- Status: COMPLETE
- Objective: implement a small, testable launcher that validates a dedicated DeepSeek V4
  Flash configuration and renders the same pinned vLLM workload for either `direct` or
  `fau_slurm` transport.
- Input files and hashes:
  - `AGENTS.md`: `df645b54b12212f40f1b03b9e8bbee33f6f4c97b872082e5054adf5b9b6f12e6`
  - `prd.md`: `86f6d085bb79fe1db2d45c400e6fec6e1ee90cac8e880578c58049dd6278cdec`
  - `roadmap.md`: `a296c08188dc67db13a6f66090b99b32bcbe191070a26269e580d3171cc009b5`
  - `src/reverse_reap/cli.py`: `f34606477ee09d4edee6fe9edc23a063db3feae39b00507de4fa27d28a68e3d0`
- Expected outputs: isolated DeepSeek configuration model, command renderer, CLI entry point,
  and unit tests; no changes to the Qwen `ExperimentConfig`.
- Definition of done: a pinned non-placeholder 40-character model revision is mandatory;
  direct mode prints or executes an argv-safe vLLM command; FAU mode writes or prints an
  `sbatch` script containing that same workload; execution and submission require distinct
  explicit opt-in flags and fail closed otherwise.
- Validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q
  tests/test_deepseek_infra.py`
- Estimated GPU hours: `0`
- Estimated storage: `<0.01 GB`
- Dependencies: none
- Failure behavior: terminal; preserve generated evidence and do not fall back to Qwen or a
  different DeepSeek checkpoint.

## DSV4-INFRA-002 — Two immutable example modes

- Status: COMPLETE
- Status history: PENDING -> ACTIVE -> COMPLETE
- Objective: add separate, reviewable example configurations for direct/cloud GPU access and
  FAU Slurm without embedding credentials, account identifiers, or provider metadata.
- Input files and hashes:
  - `README.md`: `10a08cb9d5d4b36e455bba366c8fdf6b06d5193a7b23cdc1140e0a27baf427ec`
  - output of `DSV4-INFRA-001`
- Expected outputs: `configs/deepseek-v4-flash-direct.yaml` and
  `configs/deepseek-v4-flash-fau-slurm.yaml`.
- Definition of done: both configs point to the exact same official 0731 model and immutable
  revision; only the launch transport differs; hardware/account/site values that require FAU
  confirmation are explicit placeholders or user-overridable fields.
- Validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run reverse-reap
  deepseek-launch <config> --check`
- Estimated GPU hours: `0`
- Estimated storage: `<0.01 GB`
- Dependencies: `DSV4-INFRA-001`
- Failure behavior: terminal; never silently supply an FAU account, partition, constraint, or
  paid-provider budget.

## DSV4-INFRA-003 — Operator documentation

- Status: COMPLETE
- Status history: PENDING -> ACTIVE -> COMPLETE
- Objective: document installation, immutable pinning, direct preview/launch, FAU script
  preview/submission, health checks, outputs, troubleshooting, authorization boundaries, and
  the absence of exact-checkpoint runtime validation.
- Input files and hashes:
  - `README.md`: `10a08cb9d5d4b36e455bba366c8fdf6b06d5193a7b23cdc1140e0a27baf427ec`
  - outputs of `DSV4-INFRA-001` and `DSV4-INFRA-002`
- Expected outputs: `docs/deepseek-v4-flash-infrastructure.md` plus a short root README link.
- Definition of done: a new operator can distinguish direct execution from FAU scheduling,
  preview every state-changing command, locate logs/run directories, and see that no job,
  instance, model download, or weight publication was performed during implementation.
- Validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q
  tests/test_deepseek_infra.py tests/test_reporting.py`
- Estimated GPU hours: `0`
- Estimated storage: `<0.01 GB`
- Dependencies: `DSV4-INFRA-001`, `DSV4-INFRA-002`
- Failure behavior: retry once for documentation/test mismatch, then terminal.

## DSV4-INFRA-004 — Repository validation and delivery

- Status: COMPLETE
- Objective: validate the isolated implementation, inspect the diff, and prepare delivery to
  the configured GitHub remote.
- Input files and hashes:
  - `pyproject.toml`: `f8de0969cf38023f4715242b23ac58d1ab278b0e7b5b7dfb04d43c222b7cc2b3`
  - outputs of `DSV4-INFRA-001` through `DSV4-INFRA-003`
- Expected outputs: passing targeted/full hardware-free tests, lint result, reviewed diff, and
  a commit suitable for `origin` (`Pranavharshans/Qwen-35b-a3b-Reap`).
- Definition of done: no unrelated or Qwen contract changes are present; exact commands and
  exit codes are recorded; commit and push are reported separately.
- Validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q &&
  UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run ruff check src tests scripts`
- Estimated GPU hours: `0`
- Estimated storage: `<0.01 GB`
- Dependencies: `DSV4-INFRA-001`, `DSV4-INFRA-002`, `DSV4-INFRA-003`
- Failure behavior: do not commit or push on validation failure; report the narrow blocker.
