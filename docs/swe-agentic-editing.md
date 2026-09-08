# Deterministic SWE repository-editing scaffold

## Outcome and boundary

`swe-edit-v1` replaces model-written diff line numbers with model-directed exact-match edits.
The model still decides what to inspect and what code to change. The scaffold only provides a
bounded read/edit interface and mechanically serializes accepted changes with:

```text
git diff --binary --no-ext-diff --no-textconv
```

This is CPU-only infrastructure. It does not call a model, provision hardware, alter the scorer,
or authorize the 12-output probe or 208-output repair. The original v2/v4 failed patches remain
evidence and are not rewritten.

## Trust boundary

The input task contains exactly six fields: `sample_id`, `source_id`, `repo`, full
`base_commit`, `problem_statement`, and local `repo_dir`. Gold `patch`, `test_patch`,
`FAIL_TO_PASS`, `PASS_TO_PASS`, hints, outcome fields, and post-fix content are rejected by the
allowlist. At session creation, only regular Git blobs at the exact base commit are read with
replacement objects disabled. They are copied into a new inert Git repository; the source
checkout, hooks, filters, symlinks, submodules, and later commits are never materialized.

The model protocol has five single-action JSON messages:

```json
{"action":"list","path":"pkg"}
{"action":"search","query":"literal text","path":"pkg"}
{"action":"read","path":"pkg/file.py","start_line":1,"end_line":120}
{"action":"edit","path":"pkg/file.py","old_text":"exact unique source","new_text":"replacement"}
{"action":"finish"}
```

There is no shell, command execution, arbitrary file creation/deletion, network operation, glob,
regex, Python evaluation, or access outside the inert session repository. Paths containing
traversal, control characters, backslashes, `.git`, symlinks, or non-regular files fail closed.
An edit is accepted only when `old_text` occurs exactly once in an existing UTF-8 regular file.

Run the model process and this driver inside an OS sandbox with networking disabled (for example,
a disposable container with `--network=none`, read-only input mounts, and only the session output
directory writable). The protocol has no network action, but application code cannot prove host
firewall state by itself.

## Frozen default budgets

The policy in `configs/swe-edit-policy-v1.json` allows at most 8 turns, 16 tool calls, 24,000
input tokens, 8,192 output tokens, and 900 seconds per sample-condition session. File count,
materialization, list/search/read/edit, and final-patch byte limits are also fixed. Token counts
must come from the exact donor tokenizer; the scaffold does not estimate tokens from characters.

Every atomic checkpoint binds the run ID, task projection, configuration hash, model revision,
tokenizer hash, policy hash, exact base commit, and materialized blob manifest. It also hashes the
current Git diff. A changed binding or edited workspace cannot be resumed silently. The completed
artifacts include a mechanically generated patch and complete JSONL transcript with SHA-256.

## CPU-only use

Create a session:

```bash
uv run python scripts/run_swe_edit_session.py start \
  --session-dir /tmp/swe-edit/session-001 \
  --task /tmp/swe-edit/task.json \
  --policy configs/swe-edit-policy-v1.json \
  --run-id DRYRUN-sample-001 \
  --config-sha256 <64-hex-config-hash> \
  --model-revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --tokenizer-sha256 <64-hex-tokenizer-bundle-hash>
```

The command writes `initial-prompt.txt` and `binding.json`. After the generation loop measures the
turn with the exact tokenizer, advance it using a file containing exactly one JSON response:

```bash
uv run python scripts/run_swe_edit_session.py step \
  --session-dir /tmp/swe-edit/session-001 \
  --policy configs/swe-edit-policy-v1.json \
  --expected-binding /tmp/swe-edit/session-001/binding.json \
  --response /tmp/swe-edit/response.json \
  --input-tokens 1800 --output-tokens 90
```

Repeat until `finish` or a frozen budget is exhausted. A production generation adapter must keep
condition-independent repository tools, identical batch grouping within each sample-condition
comparison, greedy decoding, thinking disabled, and the existing expert intervention semantics.

## Validation

These checks download neither model weights nor datasets:

```bash
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv sync --frozen --extra dev
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest tests/test_swe_edit.py -q
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run ruff check src tests scripts
```

The synthetic suite covers exact-base materialization, post-fix exclusion, answer-field rejection,
symlink/path traversal denial, deterministic list/search/read behavior, unique exact-match edits,
ambiguous edits, mechanical patch applicability, checkpoint binding/workspace drift, atomic
artifacts, replay determinism, and all session budgets.

## Cost model and limitations

The protocol permits 8 turns; the expected useful path is 4–6 model turns per sample-condition:
two to four list/search/read actions, one or two edits, then finish. A turn may consume up to the
overall 24k/8k session caps, so a GPU probe must preflight the actual transcript-length growth.

Using the prior one-shot PRO 6000 B6 probe only as a rough lower bound, 12 agentic sessions are
expected to take about 20–45 inference minutes after model load, while 208 sessions could take
roughly 2–5 inference hours. At $1.0–$1.6/hour this suggests approximately $1–$3 for a 12-session
probe including setup and $3–$9 for the 208 repair on a prepared host. These are planning ranges,
not benchmark results; no paid run should use them as evidence. A bounded exact-checkpoint pilot
must measure turns, tokens, prefill/decode throughput, VRAM, wall time, and provider cost first.

Known limitation: this commit implements and validates the safe CPU state machine, not the Qwen
multi-turn generation adapter. It proves that valid model-selected exact-match edits become
applicable patches; it does not prove Qwen will choose correct edits or that SWE-bench tests pass.
The official unchanged harness remains the only capability scorer.
