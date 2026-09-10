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

Copy the completed run directory and the official MBPP+ archive to that scorer,
then score all conditions with one command:

```bash
reverse-reap score-mbpp-bridge-benchmark /path/to/pinned-mbpp-benchmark.yaml \
  --evalplus-image "$(cat /path/to/evalplus-image/evalplus-image.txt)"
```

The scorer runs `evalplus.sanitize` and `evalplus.evaluate --dataset mbpp`
inside a network-disabled, read-only, capability-dropped Docker container. It
reports official MBPP base-test and MBPP+ base-plus-extra pass rates separately.
