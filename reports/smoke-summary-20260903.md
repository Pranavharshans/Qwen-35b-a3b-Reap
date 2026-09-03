# Smoke Run Summary — 2026-09-03

**This file is a frozen, write-once record of the v0 smoke run. It must never be edited in place; any correction or follow-up goes into a new report.**

## Outcome

**observational-candidates / smoke-scale NULL.**

No causal or extraction claim is made from this run. The ranked candidates are observational only.

## Run identity and provenance

- Run ID: `20260903T123424Z-qwen35a3b-direct-04cb47a5-03cacc64`
- Code commit: `04cb47a502ae787572a8676c6c9178d9c9b664b1`
- Telemetry SHA-256: `91f948f83a96d018869461b3374d20b4266184885322e79f00471ebf1f1fe2e6` (1,695,360 rows)
- Analysis commit: `9cc07aca661bb9819abca613b9ac8f03f3d1ee29`

## Candidate manifest

- Path: `runs/smoke/analysis-check2/candidate-manifest.json`
- File SHA-256: `657881972a84e20bd80f98222450ee1f4f2884b6211fe3e5132025d155d3ff47`
- Recorded internal `manifest_sha256`: `b10796c95a741aa13dbea0ac4cf31944cc36cee5ab965c28c20e775b03091ab2`
- Provenance chain closed: the manifest's `source_hashes.telemetry` equals the telemetry SHA-256 above.
- `top_n`: 8
- 8,843 experts ranked in the shared coding∩control universe; 1,113 single-domain experts recorded unranked.

## Gate A: PASS

`preflight.json` recorded `passed=true`. Host configuration:

- 4x NVIDIA GeForce RTX 3090 (24 GiB each, compute capability (8,6))
- CUDA runtime 12.6, driver 590.48.01
- torch 2.7.1+cu126
- ~189.6 GiB disk free
- Zero errors.

## Telemetry validation: valid

`valid=true` — 20 chunks/samples, 1,695,360 routing rows, 211,920 token-layer groups, top-8 of 256 experts across 40 layers, 5,298 analysed tokens.

## Gate C: FAIL at smoke scale

- Bootstrap median Jaccard **0.7778** (200 iterations, seed 20260903) — passes the >=0.60 threshold.
- Label-permutation p = **0.4643** — fails the <=0.05 threshold.

The permutation design itself is valid:

- Method: `global-count-preserving-exact-enumeration`
- 56 attainable assignments, 56 evaluated, 56 unique null statistics, labels changed, `design_valid=true`
- Fallback reason: all 3 strata are single-domain (function-synthesis: 2 coding; repository-bug-repair: 1 coding; general-knowledge: 5 control)
- Observed top-8 differential sum: 46.1776

This is a **valid low-power failure** (3 coding / 5 control analysis-window samples), not an invalid design. No GPU work may be restarted merely to obtain a passing p-value.

## Downstream gating

The intervention, ablation, and extraction stages in both the smoke and full plans are now hard-gated on `candidate-manifest.json` field `gate_passed == true` (commit `8b79530` — "fix: gate every manifest-consuming stage on candidate-manifest gate_passed"). A failed Gate C therefore skips those stages by construction.
