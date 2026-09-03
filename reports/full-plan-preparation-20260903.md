# Full-Plan Preparation — 2026-09-03

**Status: prepared only. This plan must not be launched without explicit human approval. No run has been started.**

Artifact: `configs/execution-plan-v0.yaml`, generated with `src/reverse_reap/plans.py::write_full_plan` (defaults: `pinned_config=configs/pinned-3090-bf16.yaml`, `thinking_config=configs/pinned-thinking-3090-bf16.yaml`, `run_dir=runs/v0/${RUN_ID}`, `dataset_manifest=datasets/manifests/full.jsonl`) and verified to load with `reverse_reap.controller.load_plan`. 42 tasks; 30 carry the Gate C `run_if` on `candidate-manifest.json` field `gate_passed == true`; there are zero ungated manifest consumers.

## 1. Length-matched dataset tiers (exact counts, read-only from the GPU host)

Collected over `datasets/manifests/<tier>-lengthmatched.jsonl` on the GPU host (`/workspace/reverse-reap`), read-only.

### pilot-lengthmatched.jsonl — total 224

- Per domain: coding 160, control 64
- Per split: selection 53, calibration 69, replication 52, validation 50
- Domain x split: coding selection 40, coding calibration 40, coding replication 40, coding validation 40; control selection 13, control calibration 29, control replication 12, control validation 10

### medium-lengthmatched.jsonl — total 889

- Per domain: coding 633, control 256
- Per split: selection 210, calibration 266, replication 195, validation 218
- Domain x split: coding selection 160, coding calibration 160, coding replication 153, coding validation 160; control selection 50, control calibration 106, control replication 42, control validation 58

### full-lengthmatched.jsonl — total 3684

- Per domain: coding 1684, control 2000
- Per split: selection 766, calibration 1450, replication 713, validation 755
- Domain x split: coding selection 354, coding calibration 678, coding replication 332, coding validation 320; control selection 412, control calibration 772, control replication 381, control validation 435

## 2. GPU-hour estimates (plan-declared, summed)

Evaluation stage — 29 tasks, 218.0 GPU-hours:

| Stage | Tasks | GPU-hours each | Subtotal |
|---|---|---|---|
| C0 baseline validation (a, b) | 2 | 8 | 16 |
| C2 selected-set validation | 1 | 8 | 8 |
| C3 layer-random controls | 20 | 8 | 160 |
| C4 frequency-matched control | 1 | 8 | 8 |
| C5 lowest-differential control | 1 | 8 | 8 |
| Replications (baseline + selected) | 2 | 8 | 16 |
| Thinking pilots (C1 baseline + C6 selected, limit 20) | 2 | 1 | 2 |
| **Evaluation subtotal** | **29** | | **218.0** |

Non-evaluation stage — 25.21 GPU-hours: gpu-preflight 0.01, instrumentation-probe 0.5, telemetry-calibration 12, telemetry-selection 12, single-expert-intervention-probe 0.5, extract-candidates 0.2. The remaining 7 tasks (dataset-freeze, dataset-token-length-audit, merge-telemetry, candidate-analysis, baseline-determinism, causal-report, final-bundle) declare 0 GPU-hours.

**Plan total: 243.21 GPU-hours across 42 tasks.**

## 3. Storage estimate

Sum of per-task `estimated_storage_gb` = **235 GB**. This already includes the ~60 GB merged telemetry that the `merge-telemetry` task declares; the two per-split capture tasks add 30 GB each, extraction 20 GB, dataset freeze 3 GB, and each of the 29 evaluation tasks 3 GB (or 1 GB for the thinking pilots), with 1 GB defaults on the small artifact tasks. Largest single task output is the 60 GB merged telemetry file, which is below `storage_limit_gb` on its own. Against the configured `storage_limit_gb` of 250, the plan leaves about 15 GB of headroom.

## 4. Cost at the configured provider rate

The provider rate is taken from the actual pinned configs — `provider_rate_usd_per_hour: 1` appears in both `configs/pinned-3090-bf16.yaml` and `configs/pinned-thinking-3090-bf16.yaml` (it is not fixture-only). At $1/GPU-hour, the plan total is **$243.21** (243.21 GPU-hours x $1/hr). The controller books cost as `consumed_gpu_hours x provider_rate_usd_per_hour` per completed task.

Configured budget parameters (both pinned configs): direct `max_gpu_hours 4`, `max_cost_usd 5`; thinking `max_gpu_hours 8`, `max_cost_usd 10`; both `storage_limit_gb 250`, `deadline_utc 2027-01-01T00:00:00Z`.

Important consequence, reported factually and unchanged: `evaluate_budget` applies a 0.8 reserve factor, so the effective limits are 3.2 GPU-hours / $4 (direct) and 6.4 GPU-hours / $8 (thinking). The budget check is cumulative consumed plus the next stage's estimate, enforced before each task. With the pinned smoke-scale budgets the run will therefore reach `WAITING_FOR_HUMAN` before `telemetry-calibration` (0.51 consumed + 12 projected = 12.51 > 3.2 hours). Proceeding past instrumentation requires a human to deliberately raise the budget; the controller cannot overspend silently.

## 5. Stopping conditions (all enforced by existing code, unchanged)

1. **Budget gates.** `evaluate_budget` (max_gpu_hours, max_cost_usd via the 0.8 reserve, deadline_utc) is evaluated before every task against cumulative consumed hours plus the projected stage; a denial, or a task whose `estimated_storage_gb` exceeds `storage_limit_gb`, transitions the task to `WAITING_FOR_HUMAN` with a failure signature and `run_all` returns.
2. **Gate C failure -> NULL, not a crash.** If `candidate-analysis` produces `gate_passed: false`, all 30 gated stages (intervention probe, C2/C3/C4/C5 validation, both replications, causal report, extraction, thinking pilots) transition to COMPLETE with `SKIPPED_GATE` signatures and the run terminates in a NULL-flavored report with no causal claim.
3. **Gate B determinism failure -> terminal stop.** `baseline-determinism` (failure_behavior `terminal`) requires >= 0.95 scoreable fraction and zero repeated response/score mismatches across the two C0 baseline passes; any mismatch or a scoreable fraction below 0.95 marks the comparison failed and the task terminates the run.
4. **Retry exhaustion -> FAILED_TERMINAL stops run_all.** A `failure_behavior: terminal` task fails terminally on its first failure; a `retry` task fails terminally after the same failure signature recurs beyond two attempts; `run_all` returns on any status other than COMPLETE/FAILED_RETRYABLE, so a `FAILED_TERMINAL` halts the whole run.

## 6. Launch authorization

This plan is prepared only. It must not be launched — no `run_all`, no task dispatch, no GPU work — without explicit human approval.
