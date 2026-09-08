# Causal Pilot — Final Report (Generation + Scoring + Gate D)

- Run ID: `20260907T190651Z-qwen35a3b-direct-1657b169-3ea2699f`
- Model: Qwen3.5-35B-A3B rev `59d61f3…`, BF16, greedy, batch 8, seed 20260903
- Governed plans: `configs/execution-plan-causal-pro6000.yaml` (gen) +
  `configs/execution-plan-causal-score.yaml` (score), same run ID throughout
- Outcome: **generation COMPLETE, scoring COMPLETE, Gate B FAIL (fail-closed
  halt), Gate D label `observational-candidates`, passed False**
- Classification: pipeline halted at `causal-gates` (pre-committed terminal);
  `run-bundle` never ran, so no recorded classification — maps to
  **INCOMPLETE** with a directionally null Gate D. No replication/extraction
  run (standing STOP-after-Gate-D order).

## Phase 1 — Generation (1× RTX PRO 6000, instance 50178611, DESTROYED)

- Host commit `1657b16`; config `configs/pinned-pro6000-bf16-gen.yaml`
- Frozen inputs 23/23 hash-matched; pre-gate PASS (0/0 mismatches)
- **26/26 files, 1,300/1,300 rows**, bundle self-verify zero mismatches
- Bundle SHA-256:
  `ca6f32c2f2cefbe3252518ec76289ca23e29de8d6c40dbeaa4c8a2ea35354d81`
- Cost: 6.39 h × $1.3139/h ≈ **$8.40** (within $10 ceiling)

## Phase 2 — Scoring (KVM VM on RTX 4070S host, instance 50216764, DESTROYED)

GPU never touched (CPU-only docker evals; KVM template requires a GPU host).
Instance $0.1356/h; ~2.1 h wall ≈ **$0.28** (within $2.50 cap).

- `verify-generation-inputs`, `docker-evaluator-prep`: COMPLETE
- `swebench-harness-prep`: COMPLETE (harness `02e7a74`, tasks `3d07b464`)
- `score-conditions`: COMPLETE — 26/26 pre-scored files
- `swebench-harness-score`: COMPLETE — **26/26 final scored files,
  1,300/1,300 rows** (local copy verified: 26 finals, 1,300 rows)
- `causal-gates`: **FAILED_TERMINAL** (pre-committed: Gate B fail halts run)
- `run-bundle`: never ran (correct per fail-closed design)

Local artifacts: `runs/causal-pilot/20260907T190651Z-qwen35a3b-direct-1657b169-3ea2699f/`
(`scored/`, `gates/`, `state/`, `causal-report.json`, `generation-bundle.json`).

## Gate results

- **Gate B (determinism/coverage): FAIL** — 50 samples, 0 mismatches
  (determinism held), but scoreable fraction **0.84 < frozen 0.95**.
- **Gate D: `observational-candidates`, passed False.** coding_drop 0.0,
  95% CI [0.0, 0.0]; criteria 1/5 true (`no_broad_output_collapse` only);
  `replication: null`. Random-control drops cluster at 0.0–0.0625
  (median 0.03125, p95 0.0625) — selected-expert masking shows no effect
  above the random floor.

## Root cause of Gate B failure (science finding, not infra)

All 208 SWE-bench rows errored identically: the official harness could not
apply any model patch (`patch: Only garbage was found in the patch input`,
0 completed / 208 errors across 26/26 condition reports). The generation
prompt for swebench rows (`src/reverse_reap/sources.py:100`) is
`Repository: {repo}\nIssue:\n{problem_statement}` — it never requests diff
format, so the model wrote prose explanations (e.g. a 3.7 KB Django `Q`
essay for `django__django-14017`), and `export_predictions`
(`src/reverse_reap/swebench.py:38-40`) submits raw prose as `model_patch`
when no fenced block exists. 0% patch applicability ⇒ coverage can never
reach 0.95 ⇒ Gate B structurally unpassable under this prompt. The
fail-closed halt worked as designed: no causal inference was run on
unevaluable data. Fixing this means re-prompting for diffs + regeneration
(new GPU run) — NOT done; needs approval as a design change.

## Deviations & fixes (all committed, pushed to `origin/codex/pilot-plan`)

- Scoring VM: $0.1356/h (over $0.10/h target), 24 GB RAM / 117 GB disk
  (under ≥32 GB / ≥120 GB floors) — functionally fine except where noted.
- `fcc907d`: controller skipped directory outputs when hashing
  (`IsADirectoryError` on the pinned `swebench-tasks` clone) + regression test.
- `7d6b0f0`: `run_swebench_harness` resolves harness/task-repo/work paths to
  absolute before `cwd=condition_work` (relative paths broke the harness child).
- `25908e0`: harness eval capped at `--max-workers 4` (plan change,
  parallelism only) + 16 GB swap on host, after 8 workers blew past 24 GB
  RAM and the kernel OOM-killed the harness mid-`c0` and the controller
  with it (stale RUNNING state auto-recovered on relaunch).

## Spend (total ≈ $8.68)

- Generation VM: ≈ $8.40. Scoring VM: ≈ $0.28. No further cloud spend
  pending; both VMs destroyed, billing stopped.
