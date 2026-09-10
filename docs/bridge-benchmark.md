# Base versus trained-bridge benchmark

This benchmark answers one narrow question: does the frozen
`Qwen/Qwen3.5-2B` host score differently when the trained four-expert bridge
is attached? It does not reinterpret the original observational expert label
or prove causality.

## Frozen design

The primary task source is a revision-pinned `evalplus/humanevalplus` manifest.
Freeze it before the paid run:

```bash
reverse-reap freeze-humanevalplus <DATASET_COMMIT_SHA> \
  datasets/manifests/humanevalplus-bridge.jsonl
```

Only Python HumanEval+ rows with unit tests are eligible. Original
`openai/openai_humaneval` remains an explicitly named compatibility option and
must never be reported as HumanEval+. Selection
is deterministic by `selection_seed` and content hash:

- pilot: first 25 selected tasks;
- full: every eligible HumanEval task (or the predeclared `full_items` cap);
- the full tier always runs after a score-valid or score-poor pilot;
- an integrity failure still stops the pipeline.

Each tier runs exactly four generation conditions:

1. `base-a`
2. `base-b`
3. `bridge-a`
4. `bridge-b`

The repeated conditions must have identical generated token IDs. Base and
bridge use the same prompts, tokenizer, greedy decoding, thinking-disabled
chat template, input/output limits and sample order.

## Single generation command

Copy `configs/bridge-benchmark-qwen35-2b.yaml`, replace every placeholder,
pin the host-file manifest and hashes, then run:

```bash
reverse-reap run-bridge-benchmark /path/to/pinned-benchmark.yaml
```

The command loads the host once, verifies and loads the bridge checkpoint,
runs all four pilot generations, enforces both repeat gates, and then runs all
four full-tier generations regardless of pilot score. It stores task freezes,
prompts, input and generated token IDs, raw and cleaned completions, hashes,
timings, per-condition JSONL and tier reports. Atomic `state.json` and versioned
scoring-state records distinguish running, complete, and terminal-failure outcomes
so a stopped run is never mistaken for a finished comparison.
Bridged rows also contain streaming per-expert gate means and residual L2
contribution norms; full hidden activations are never retained.

Scientific scores never control progression from pilot to full. The following
integrity failures do stop progression: source/host/bridge hash drift, wrong
checkpoint keys, repeat nondeterminism, context overflow, unavailable model,
OOM, invalid output, budget/deadline, or evaluator failure.

## Scoring boundary

Generated Python is untrusted. On a Docker-capable host, set
`scoring_mode: docker-local` and pin the evaluator image by digest to generate
and score in the same command. A normal Vast GPU container may not provide
nested Docker; use `scoring_mode: deferred` there. The generation command
still performs all four runs and preserves every output.

Transfer the small run directory to the previously qualified KVM scorer, then
run one scoring command:

```bash
reverse-reap score-bridge-benchmark /path/to/pinned-benchmark.yaml \
  --evaluator-image 'localhost:5000/reverse-reap-evaluator@sha256:<digest>'
```

Scoring executes every completion with no network, a read-only root,
capabilities dropped, bounded processes, CPU, memory and time, and a disposable
filesystem. It writes scored copies rather than modifying raw generations.
The continuation-boundary v3 scorer always reconstructs executable text from
the immutable `raw_completion`, preserves leading indentation, and records both
the normalizer version and reconstructed completion. Its `*-v3` scored files,
reports, states, reference preflight and artifact manifest never overwrite the
original or v2 scoring evidence. Because normalized task prompts may omit their
trailing newline, v3 joins each prompt and reference/generated continuation with
exactly one line boundary while preserving continuation indentation.
Before accepting model scores, every canonical reference solution must pass
its own frozen tests inside the exact pinned evaluator image. Any reference
failure invalidates the scorer and stops the benchmark.

The scoring command may run after the generation deadline: that deadline
prevents further paid generation, while already-frozen outputs remain valid
inputs to deferred CPU scoring.

The final paired report includes base and bridge pass rates, bridge fixes,
bridge regressions, both-pass/both-fail cases, an exact paired p-value, and a
2,000-resample paired bootstrap interval.

## Required interpretation

A completed paired score is a capability comparison, not causal proof. A
positive bridge delta is promising only if it exceeds uncertainty and survives
later shuffled-pair and equal-parameter controls. This four-run command tests
the essential base-versus-trained-bridge comparison; the broader controls are
a separate approved evaluation generation.
