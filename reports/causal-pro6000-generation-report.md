# PRO 6000 Causal-Generation Run — Completion Report

- Run ID: `20260907T190651Z-qwen35a3b-direct-1657b169-3ea2699f`
- Host commit: `1657b16` (B8 production path + pre-staged-weights disk gate)
- Config: `configs/pinned-pro6000-bf16-gen.yaml` (batch 8, seed 20260903,
  Qwen3.5-35B-A3B rev `59d61f3…`, BF16, greedy, use_cache)
- Plan: `configs/execution-plan-causal-pro6000.yaml` (7 tasks)
- Instance: vast.ai `50178611` (`causal-pro6000-gen`), 1× RTX PRO 6000 96GB, $1.3139/h

## Results (all local artifacts verified)

- `verify-frozen-inputs`: 23/23 manifests hash-matched, 0 mismatches
- `gpu-preflight` (pro6000 profile): PASS — 1 GPU, PRO 6000, 96 GiB, sm_120,
  torch 2.11.0+cu128 / CUDA 12.8, disk gate OK
- `gen-validation-baselines`: 3/3 files (150 generations)
- `response-determinism-pre-gate`: **PASS** — baseline_pair 0 mismatches,
  noop_equivalence 0 mismatches (fail-closed gate, terminal)
- `gen-validation-interventions`: 3/3 files (150 generations)
- `gen-random-controls`: 20/20 files (1,000 generations)
- `generation-bundle`: **26/26 files, 1,300/1,300 rows, self-verify passed,
  zero mismatches**
- Bundle SHA-256: `ca6f32c2f2cefbe3252518ec76289ca23e29de8d6c40dbeaa4c8a2ea35354d81`
- Local copy: `runs/causal-pilot/20260907T190651Z-qwen35a3b-direct-1657b169-3ea2699f/`
  (bundle hash identical to on-host value)

## Cost / ceilings

- VM uptime at wind-down: 383.59 min (6.39 h) × $1.3139/h ≈ **$8.40**
- Within gen-config ceilings (8 h / $10) and plan reserve (6.4 h / $8.00).
- Superseded failed run `20260907T185944Z-…-d2cb0628-…` preserved on host only
  (preflight env + unsatisfiable 120 GiB disk gate, fixed by 1657b16); not copied.

## Next (NOT started)

- Phase 2: provision low-cost KVM scoring VM, verify bundle pins, score under
  frozen scorer, Gate D report. STOP after Gate D — no replication/extraction
  without approval.
