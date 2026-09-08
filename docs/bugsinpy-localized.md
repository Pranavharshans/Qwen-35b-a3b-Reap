# Localized BugsInPy validation replacement

## Status and claim boundary

The implementation is ready for CPU-only task selection and environment validation. No eight-task
projection, replacement manifest, Docker image, GPU run, or causal result is frozen by this commit.
The failed SWE-bench runs remain `INCOMPLETE` and unchanged.

This condition is deliberately labelled **oracle-localized repository repair**. The task projection
may use the gold patch metadata to identify one changed production filename, but neither the gold
patch body nor the fixed repository content may enter a prompt or runtime filesystem. Consequently,
the condition measures repair after file localization; it does not measure repository navigation.

Upstream is pinned to `soarsmu/BugsInPy` revision
`11c5f1eea954a42132cfd06bf257766a7963e0fd`. A human must approve the exact eight-row task
projection hash before a replacement manifest is built.

## Task selection contract

Choose exactly eight Python bugs that satisfy all of the following before inspecting model output:

- one production `.py` file is changed by the upstream fix;
- the buggy commit and target blob resolve by full SHA;
- the failing test reproduces in a disposable container;
- the full target file fits the predeclared tokenizer budget;
- the fixed version makes the same test pass during dataset preflight;
- no duplicate project/bug or cross-split content is introduced.

The strict task JSON schema is represented by `LocalizedTask` in
`src/reverse_reap/localized_repair.py`. `failure_output` must be captured from the buggy revision.
The task projection contains the oracle filename but never patch text or fixed source. Freeze and
publish its SHA-256 for human approval before preparation.

## Build the replacement manifest

```bash
uv run python scripts/prepare_bugsinpy_localized.py \
  --base-manifest datasets/manifests/pilot-lengthmatched.jsonl \
  --tasks runs/bugsinpy-localized/tasks.json \
  --approved-tasks-sha256 "$APPROVED_TASKS_SHA256" \
  --output datasets/manifests/pilot-lengthmatched-bugsinpy-localized.jsonl \
  --provenance runs/bugsinpy-localized/manifest-provenance.json
```

The command refuses overwrite and requires exactly eight validation SWE-bench rows and eight
approved replacements. Every other manifest byte is preserved.

## Generation output boundary

The prompt requests the complete replacement contents of one file. No diff syntax is requested.
For each condition, convert its eight generated rows into patches:

```bash
uv run python scripts/export_bugsinpy_localized.py \
  --evaluation runs/causal-repair/generations/c2-selected.jsonl \
  --tasks runs/bugsinpy-localized/tasks.json \
  --output runs/causal-repair/patches/c2-selected.bugsinpy-patches.jsonl
```

The exporter rejects fences, empty/no-op output, invalid Python, foreign tasks, duplicate coverage,
and unsafe paths. It replaces only the approved base file and asks Git to emit the patch. Mechanical
applicability proves serialization, not task correctness.

## Pinned task-image contract

Each approved task needs an independently built image containing its exact buggy checkout and
dependencies at a fixed absolute path. Freeze an eight-row JSON array with:

```json
{
  "task_id": "project-1",
  "image": "registry/task@sha256:<64 hex>",
  "repository_path": "/opt/repo",
  "test_command": ["python", "-m", "pytest", "-q", "tests/test_target.py"],
  "image_source_sha256": "<64 hex>"
}
```

The build recipe and image must first prove buggy FAIL and fixed PASS. The runtime image must contain
only the buggy tree; never bake the fixed tree or gold patch into it. The harness runs with no
network, a read-only root, dropped capabilities, bounded CPU/RAM/PIDs/time, and an executable tmpfs.
It copies the buggy tree to tmpfs, applies the model-derived patch, and executes only the pinned test
argv. Patch/application errors remain unscoreable; completed failing tests are scoreable failures.

```bash
uv run python scripts/run_bugsinpy_localized_harness.py \
  --conditions configs/causal-pilot-conditions.json \
  --prescored-dir runs/causal-repair/prescored \
  --patches-dir runs/causal-repair/patches \
  --tasks runs/bugsinpy-localized/tasks.json \
  --images runs/bugsinpy-localized/task-images.json \
  --output-dir runs/causal-repair/scored
```

## Required next gate

Before regenerating 208 rows, run a three-task/four-condition exact-checkpoint probe. Require
baseline-repeat and no-op identity, 12/12 syntactically valid full files, 12/12 mechanically
applicable patches, and 12/12 completed test executions. Task resolution may be zero and remains an
honest measured outcome; infrastructure/applicability coverage may not fall below the frozen gate.
Do not combine the prior 1,092 outputs with replacements until a separate composite-provenance plan
is reviewed and approved.
