# Repository-context repair: source v3

Status: implemented offline; exact-checkpoint applicability remains unverified. No paid
run is authorized by this document. Preserve the failed source-v1/v2 runs unchanged.
The human requested this implementation on 2026-09-08. Candidates, scoring, dataset
membership, output limit, input limit and Gate B/D criteria remain unchanged.

The source-v2 prompt lacked base source context. Diff-shaped outputs do not establish
applicability. Wrong hunk counts and premature EOS remain possible even with context;
this repair supplies exact context but does not repair model outputs after generation.
Generation's `truncated` flag only detects the output limit, not why EOS occurred.

## Frozen retrieval policy

Only six input fields are accepted: `sample_id`, `source_id`, `repo`, `base_commit`,
`problem_statement`, `repo_dir`. Prepare this projection from the pinned task metadata;
never feed complete SWE records containing `patch`, `test_patch`, `FAIL_TO_PASS`,
`PASS_TO_PASS`, hints or solutions. Record and independently verify its SHA-256 and
full 40-character base commits against the frozen task metadata before execution.
This projection is a trusted provenance boundary: code cannot prove that an operator
did not put a solution into `problem_statement` or choose a wrong base SHA.
`prepare_swe_context.py` requires `--tasks-sha256` from that independent review and
rejects any different projection bytes. Do not compute this argument inline from an
unreviewed projection: it is the independently approved binding to the frozen metadata.

Retrieval reads only `git ls-tree` and `git cat-file` objects at that exact commit,
with replace objects disabled. It never reads working-tree files or later commits.
Only regular Python source files are eligible; tests, fixtures, hidden paths, symlinks,
submodules and non-Python assets are excluded. Issue word overlap plus path matches
rank fixed 40-line chunks, ties broken by path and starting line. Limits: 200KB/file,
50MB total scanned source, 100,000 tree entries, 2,400 context bytes. Each selected
chunk records path, line range, blob OID, full-file SHA-256 and chunk SHA-256.
Retrieval depends on task inputs only, never intervention condition or outcomes.
This deliberately simple retrieval can miss relevant source. A source-context probe
failure must be reported without tuning retrieval using gold patches.

## Bounded task plan

| task_id | objective | inputs and hashes | expected outputs | done/validation | GPU h / storage | dependencies | failure behavior |
|---|---|---|---|---|---|---|---|
| prepare | Freeze v3 validation prompts | frozen manifest SHA + approved allowlisted task projection SHA + base SHAs | new manifest + context JSON | unchanged non-target bytes; exact-base provenance; `pytest tests/test_swe_context.py` | 0 / repository mirrors plus <10MB outputs | none | stop on missing source/budget/hash mismatch |
| audit | Exact tokenizer input audit and immutable run contract | new config fingerprint + manifest/context SHA | context-contract.json | every prompt <=1024 tokens; no truncation; matching run ID | 0 / tokenizer files | prepare | stop; larger limits require separate amendment |
| probe | Three preselected tasks x four conditions | frozen conditions SHA + candidate hashes + exact donor SHA + context contract | 12 rows/checkpoints | baseline pair/noop identical; exact expected IDs | proposed <=1 GPU host-hour generation, setup budget separately approved /120GB disk | audit + explicit paid authorization | checkpoint, preserve, stop |
| apply | Check all 12 raw diffs at frozen bases | probe file hashes + same task projection SHA | applicability report | `validate_swe_context_probe.py`, exit 0 only all apply + invariants | 0 / mirrors + small temporary indices | probe | record failed applicability; no 208-row run |

## Preparation commands (CPU only)

Use the same three previously frozen probe sample IDs. `tasks.json` contains exactly
those three allowlisted records. `ids.txt` contains their sample IDs, one per line.
Mirror paths are operator-provided local repositories containing the exact base objects.
No checkout or generated-code execution is necessary.

```bash
uv run python scripts/prepare_swe_context.py \
  --manifest datasets/manifests/pilot-lengthmatched.jsonl \
  --tasks runs/swe-context-v3/tasks.json \
  --tasks-sha256 "$APPROVED_TASK_PROJECTION_SHA256" \
  --output datasets/manifests/pilot-lengthmatched-swe-v3-probe.jsonl \
  --provenance runs/swe-context-v3/context.json
```

The checked-in `configs/pinned-pro6000-bf16-gen-swefix-v3.yaml` is a NON-LAUNCHABLE
template with an intentionally expired deadline; the runner rejects it before loading
weights. Resolve a fresh approved deadline and remaining budget in a new config before
execution. Its changed fingerprint must appear
in a newly resolved run ID. Pin the benchmark-validated Torch 2.11.0+cu128 environment;
do not reinstall the repository's older GPU extra over it. Preserve 1024 input/output
tokens, BF16, greedy B8, thinking disabled. Prepare a fresh deadline and approved
remaining budget before any launch. The existing old budget does not authorize rent.

## Next GPU command, prepared only

`RUN_ID` must be resolved once with the new config fingerprint, and exported. Use
fresh output/checkpoint paths; the v3 runner rejects old unbound checkpoints. The
context audit runs before donor loading and rejects oversized prompts rather than
silently truncating source. The external launcher must enforce approved elapsed time,
cost reserve and 15-minute progress watchdog; this script is not a billing controller.

```bash
python scripts/run_swe_repair_gen.py \
  --config configs/pinned-pro6000-bf16-gen-swefix-v3.yaml \
  --model-path /models/qwen \
  --dataset-manifest datasets/manifests/pilot-lengthmatched-swe-v3-probe.jsonl \
  --context-provenance runs/swe-context-v3/context.json \
  --conditions configs/causal-pilot-conditions.json \
  --condition-ids c0-baseline-a c0-baseline-b c0-noop-masked c2-selected \
  --sample-ids-file runs/swe-context-v3/ids.txt \
  --output-dir "runs/swe-context-v3/${RUN_ID}/probe" \
  --checkpoint-dir "runs/swe-context-v3/${RUN_ID}/checkpoints" \
  --heartbeat-path "runs/swe-context-v3/${RUN_ID}/heartbeat.json" \
  --fingerprint-path "runs/swe-context-v3/${RUN_ID}/environment.json" \
  --run-id "$RUN_ID" --mode probe
```

Copy/hash outputs before ending GPU rental. Applicability runs CPU-only with a
temporary Git index; it does not execute or apply the model patch to the worktree:

```bash
uv run python scripts/validate_swe_context_probe.py \
  --tasks runs/swe-context-v3/tasks.json \
  --probe-dir "runs/swe-context-v3/${RUN_ID}/probe" \
  --report "runs/swe-context-v3/${RUN_ID}/applicability.json"
```

Stop after the probe. Applicability is not task correctness and is not Gate D.
The old 1,092-output reuse proposal also needs review: changing batch membership can
change BF16 generation, so unchanged per-sample prompt bytes alone are insufficient
to prove batched reuse equivalence. Do not automatically composite or score a full run.
