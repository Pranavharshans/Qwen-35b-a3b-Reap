# MBPP+ four-condition bridge benchmark

This follow-on benchmark compares the frozen `Qwen/Qwen3.5-2B` host with the
trained four-expert bridge on official MBPP/MBPP+ tests. It does not upgrade the
experts beyond `observational-candidates` and does not establish causality.

## Governed execution plan

| Task | Objective | Inputs and hashes | Expected outputs | Definition of done | Validation | GPU hours | Storage | Dependencies | Failure behavior |
|---|---|---|---|---|---|---:|---:|---|---|
| `mbpp-freeze` | Freeze the official MBPP+ v0.2.0 task order | Official release archive, SHA-256 in config | `task-freeze.json`, `tasks.jsonl` | Exactly 378 unique official tasks and deterministic pilot membership | Built-in archive, schema, count, ID and content-hash checks | 0 | <10 MB | none | terminal |
| `artifact-preflight` | Prove host, checkpoint and extracted experts are the pinned artifacts | Host manifest, bridge config/checkpoint, extraction manifest/tensors | `load-verification.json` | Every hash matches; four mappings and eight expert tensors load; experts and host are frozen | Built-in strict-key, shape, dtype, revision and hash checks | <0.05 | <50 MB | `mbpp-freeze` | terminal |
| `four-condition-generation` | Generate base/bridge with thinking disabled/enabled | Frozen tasks and verified runtime | Four condition JSONL files plus tier reports and telemetry | Complete all four 50-task pilot prefixes first, then all 378 tasks regardless of pilot scores; thinking prompts differ; every bridge sidecar records calls, gate and residual activity | `validate-mbpp-bridge-benchmark` | configurable, measured by heartbeat | <1 GB | `artifact-preflight` | resumable by validated prefix; terminal on integrity/budget failure |
| `official-evalplus-score` | Score all four files with the upstream harness | Four JSONL files, MBPP archive, digest-pinned EvalPlus image at pinned revision | Four official result JSON files, hashes, state and paired reports | Official base and plus statuses exist for all 378 tasks | image-label check and exact official task-universe reconciliation | 0 | <1 GB | generation | terminal; never fall back to custom scorer |

Only the lead execution agent writes run state. The pilot is the first 50
members of the frozen hash order. All four pilot prefixes are completed and
validated before generation continues to the full 378-task set. Pilot scores
never suppress full generation.

## Conditions

The GPU command writes exactly these primary files:

```text
conditions/base-thinking-off.jsonl
conditions/base-thinking-on.jsonl
conditions/bridge-thinking-off.jsonl
conditions/bridge-thinking-on.jsonl
```

All conditions use greedy decoding and the same task order. Thinking modes are
never pooled. The bridge hooks are absent for both base conditions. For each
bridged condition, generation fails unless telemetry proves that every mapped
sidecar executed, its learned gate opened, and it emitted a nonzero residual.

## Pre-pilot policy amendment

Two preserved one-sample preflights are the rationale for treating truncated
thinking output as an item-level benchmark failure rather than a pipeline
integrity failure:

- `20260910T075838Z-qwen35-2b-mbpp-bridge-0ff6beb` (2,048-token thinking cap):
  `base-thinking-on` hit the cap with an unclosed reasoning block. Report
  SHA-256 `b13fa426b13067b16bcdbb1d3dfafc998929c41c84f581cca466650ef5429d66`;
  outcome SHA-256
  `44b45648653686e453fb3ec3669f915e20c1d8bd03d4f4301bf7e428018d844a`.
- `20260910T090943Z-qwen35-2b-mbpp-bridge-0ff6beb` (4,096-token thinking cap):
  the same condition again consumed the full cap with an unclosed block.
  Report SHA-256
  `176bd4eb0f1f728ab3b58bc883497803c7a2f6b8f69e5ec5f79535ce0ca92afc`;
  outcome SHA-256
  `66c42d5c9249080d91cbf9aca600c4b8fb2c56492344d5c412122020a36c1f4c`.

Both runs remain classified `FAIL / WAITING_FOR_HUMAN`; this policy does not
reclassify them. Generation records `cap_hit`, `reasoning_opened`,
`reasoning_closed`, `final_answer_present`, `sanitizable`, `score_eligible` and
`failure_reason` on every row. Such rows stay in the denominator and are never
excluded, replaced, retried or regenerated. Official EvalPlus sanitization
still receives every row: recoverable code is scored normally; a row without a
recoverable executable remains in the scored task universe with a deterministic
failed result. Per-condition rates are reported for cap hits, unclosed
reasoning, missing final answers and sanitization failures.

The pre-registered pilot safety gate stops after the 50-task pilot with a
feasibility failure when any condition exceeds a 5% cap-hit rate or a 5%
structurally-invalid rate, or has missing/duplicate rows, non-finite telemetry,
or incomplete bridge engagement. When every integrity and rate gate passes,
generation proceeds to all 378 tasks regardless of capability score.

## Commands

Download the immutable official release artifact and verify the SHA-256 shown
in the template configuration. Then create a new, fully pinned configuration
from `configs/mbppplus-bridge-qwen35-2b.yaml`.

Generate all four conditions on the preserved GPU VM:

```bash
reverse-reap run-mbpp-bridge-benchmark /path/to/pinned-mbpp-benchmark.yaml
```

Build a digest-pinned official EvalPlus v0.3.1 image on a Docker-capable scorer:

```bash
python scripts/prepare_evalplus_docker.py --output-dir /path/to/evalplus-image
```

The image contains the official HumanEval+ v0.1.10 archive at
`/opt/evalplus-data/HumanEvalPlus.jsonl.gz`, pinned by SHA-256
`e62f4130146963d969da64553f407a66e52d095adbfed4ee6733b4d59e14a3ed`. Image
preparation and scoring both verify the OCI labels and the archive bytes before
accepting the image. The locked-down scorer sets `HUMANEVAL_OVERRIDE_PATH` to
that image path because `evalplus.sanitize` loads HumanEval+ and MBPP+ together.

Copy the completed run directory and the official MBPP+ archive to that scorer,
then score all conditions with one command:

```bash
reverse-reap score-mbpp-bridge-benchmark /path/to/pinned-mbpp-benchmark.yaml \
  --evalplus-image "$(cat /path/to/evalplus-image/evalplus-image.txt)"
```

The scorer runs `evalplus.sanitize --mbpp_version v0.2.0` and
`evalplus.evaluate --dataset mbpp --version v0.2.0` inside a network-disabled,
read-only, capability-dropped Docker container. EvalPlus e5d0ed0 does not
support `--output-file`; it deterministically writes
`<samples path without .jsonl>_eval_results.json`, which the scorer requires
and validates before parsing. It reports official MBPP base-test and MBPP+
base-plus-extra pass rates separately.
