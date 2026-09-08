# Agentic SWE probe plan (12 sessions, CPU-validated, awaiting approval)

Status: proposal only. No GPU provisioned, no model run, no 208 repair, no
scoring. The `swe-edit-v1` scaffold is CPU-validated (report `775af44c…`);
the Qwen multi-turn adapter (`swe_agentic.py`) is implemented with 20
model-free tests passing. This document is the human review gate before any
paid 12-session probe.

## Why this exists

The v2/v4 one-shot approach produced diff-shaped answers but 0/12 patches
applied: the model cannot reliably calculate hunk positions. The scaffold lets
the model inspect the exact frozen base and request structured exact-match
edits; accepted edits are serialized mechanically with
`git diff --binary --no-ext-diff --no-textconv`. The model still decides every
code change. Prior evidence is preserved unchanged (v2 failure, v4/B8 OOM,
v4/B6 0/12 applicability).

## Context strategy (deterministic, condition-independent)

Full transcript, no compaction: every turn appends the raw assistant response
plus the canonical-JSON tool result, identically for all four conditions.
Nothing is summarized, dropped, or reordered, so baseline-repeat and no-op
comparisons remain exact. Alternatives (bounded tool-result windows,
transcript compaction) were rejected: any reduction risks diverging across
conditions when action sequences differ, which would invalidate the identity
gates. Frozen policy caps (`configs/swe-edit-policy-v1.json`: 8 turns,
24,000 input / 8,192 output tokens, 900 s) are enforced fail-closed by the
scaffold; the adapter additionally refuses any turn whose prompt exceeds the
model config input limit before calling the donor.

## Measured context growth (model-free benchmark, exact tokenizer)

`scripts/bench_swe_agentic_context.py` replays a fixed realistic 6-turn
script (list → search → read → read → edit → finish) per frozen v3 task
against real exact-base trees, counting every turn with the exact donor
tokenizer (`59d61f3…` bundle, metadata-only). No model is called, so live
sessions may run more turns; the 24k/8-turn caps bound that fail-closed.

| Sample | Turns | Max per-turn prompt | Cumulative in | Cumulative out | Largest tool response |
|---|---|---|---|---|---|
| django__django-14017 | 6 | 3,933 | 13,628 | 840 | 5,230 B / 1,403 tok (search) |
| matplotlib__matplotlib-23563 | 6 | 5,798 | 22,155 | 1,167 | 5,426 B / 1,967 tok (search) |
| sympy__sympy-14396 | 6 | 4,857 | 17,990 | 737 | 4,731 B / 1,623 tok (search) |

Worst cumulative input: **22,155 / 24,000** (1,845 headroom). A live 8-turn
session can exhaust the cap; that stops fail-closed and is recorded, never
truncated. Largest single prompt: **5,798** tokens.

Consequence: the v4 one-shot config limit (`max_input_tokens: 3072`) cannot
hold agentic turns. The probe needs a new config with `max_input_tokens:
24576` (24k policy + template margin), `max_new_tokens: 1024` per turn,
BF16, greedy, thinking disabled — new fingerprint and immutable run ID.

## Batch comparison (70 GiB BF16 weights + 327,680 B KV per token, +10%)

| Batch | Expected at 22,155 tok | Ceiling (92% of 96) | Verdict |
|---|---|---|---|
| B1 | 77.44 GiB | 88.32 | FITS (10.9 margin) |
| B2 | 84.87 GiB | 88.32 | fits (3.4 margin, rejected: thin + batching complexity) |
| B4 | 99.75 GiB | 88.32 | EXCEEDS |
| B6 | 114.62 GiB | 88.32 | EXCEEDS (B6 qualified only for ~2k single prompts) |

Selected: **B1 only**. One session at a time in frozen sample order makes
identical batching across conditions trivial and keeps 10.9 GiB headroom for
live sessions that run longer than the 6-turn script. No B2/B4/B6.

## Fixed probe configuration (12 sessions)

- Sessions: 3 frozen samples × 4 conditions
  (`c0-baseline-a`, `c0-baseline-b`, `c0-noop-masked`, `c2-selected`).
- Donor loaded once and reused; per-forward `intervene_qwen35` wrapping with
  the existing semantics (selected masked, no-op empty mask, baselines
  untouched). Greedy, thinking disabled, exact-tokenizer accounting every turn.
- Scheduling: sequential B1, frozen sample order, identical tool-result
  rendering and budgets across conditions. Model `RUN_ID` resolved once from
  the new config fingerprint; per-session run ids encode the condition.
- Preflight (same VM, before any session): B1 prefill of a synthetic 24k
  context plus the longest measured turn (5,798 tok). Require peak ≤92%,
  no OOM/NaN, exact donor/runtime/config verification. Stop on failure.
- Expected loads: 1 model load. Runtime planning range 30–60 inference
  minutes for 72 model turns after load (12 × ~6), plus ~25 min setup
  (70 GB weights, env). At $1.0–$1.6/hr: **≈$1.50–$2.50 all-in**.
  Proposed ceiling: 3 host-hours, **$4.80** (consistent with prior runs).
  These are ranges; the pilot measures and reports actuals.
- Checkpoint/resume: scaffold atomic `state.json` + binding per session;
  run manifest binds config/task/policy/transcript/patch hashes; resume via
  expected-binding; heartbeat + 15-minute watchdog as before.

## Gates (frozen)

Hard (stop, no 208 regardless unless human approves):
baseline-repeat transcript+patch identity, no-op identity, no OOM/NaN,
no truncation, budget compliance, provenance/contract match.
Measured (scorecard, not pass/fail for the method): per-session
applicability via `git apply --cached --check` at exact frozen bases (12/12
results reported with reasons), turns/tokens/VRAM/throughput/cost.
The 208-output repair and any scoring require a separate human approval
after this probe report, whatever the applicability outcome.
