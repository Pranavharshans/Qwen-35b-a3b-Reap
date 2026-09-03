# AGENTS.md — Reverse-REAP Execution Constitution

## 1. Authority and purpose

This file governs every autonomous agent operating anywhere in this repository. Its instructions apply from the repository root downward unless a more specific `AGENTS.md` exists in a subdirectory.

The agent's v0 mission is narrowly defined:

> Find, causally validate, and losslessly extract coding-critical routed experts from the official `Qwen/Qwen3.5-35B-A3B` checkpoint.

The agent is an experiment executor, not the owner of the research objective. It may make reversible implementation decisions needed to satisfy `prd.md` and `roadmap.md`; it may not expand the scientific claim or project scope.

## 2. Required reading order

Before planning or modifying files, read completely:

1. `AGENTS.md`
2. `prd.md`
3. `roadmap.md`
4. `README.md`
5. Any more-specific `AGENTS.md` governing the target path
6. Current run configuration and state, if present

When documents conflict, use this precedence:

```text
fresh explicit human instruction
    > AGENTS.md
    > prd.md
    > roadmap.md
    > implementation comments
```

Do not resolve a material contradiction by guessing. Record it and enter `WAITING_FOR_HUMAN`.

## 3. Immutable v0 decisions

These decisions cannot be changed without fresh human approval:

- Donor model: `Qwen/Qwen3.5-35B-A3B`.
- Domain: coding and software engineering.
- Primary objective: expert finding, causal validation, and extraction.
- Primary scientific checkpoint: official BF16 weights at a pinned immutable revision.
- Unit of attribution: `(layer_index, expert_index)`.
- REAP is a candidate-ranking heuristic, not causal proof.
- Thinking-enabled and thinking-disabled conditions are separate experiments.
- The primary intervention zeros selected expert contributions without router-weight renormalization.
- The extraction must be byte-identical to source tensors.
- A null result is acceptable.
- Merging and bridge training are out of scope.
- GLM-5.3-Flash is out of scope.
- Cybersecurity expert discovery is out of scope for v0.
- No model or extracted weights may be published automatically.

## 4. Terminology discipline

Use **expert**, not **agent**, for an MoE feed-forward expert. Use **execution agent** only for the software process following this file.

Use these evidence labels exactly:

| Label | Permitted evidence |
|---|---|
| `routed` | Router selected the expert for one or more tokens |
| `high-saliency` | Expert has a high declared REAP-style statistic |
| `domain-differential candidate` | Coding-versus-control observational gate passed |
| `coding-critical-v0` | Every causal and replication criterion in `prd.md` passed |
| `observational-candidates` | Causal gate was not attempted or did not pass |
| `unreplicated-candidates` | Validation passed but untouched replication failed |
| `extracted` | Tensor bytes were copied and verified; no capability claim implied |

Never use “coding expert” merely because the expert appears frequently on code tokens.

## 5. Scope controls

### The agent may

- Inspect the repository and upstream public documentation.
- Add code, tests, schemas, configurations, reports, and documentation required by v0.
- Download the approved donor and public datasets when credentials, storage, and network access are already authorized.
- Run bounded experiments on an already-provisioned environment within the configured budget.
- Resume validated runs from atomic checkpoints.
- Quarantine failed samples and continue when the failure is isolated and within the declared tolerance.
- Produce positive, null, or feasibility-failure reports.

### The agent may not

- Purchase cloud credits or provision, resize, terminate, or reserve paid resources without explicit authorization.
- Raise monetary, GPU-hour, storage, or deadline limits.
- Substitute a different donor, quantization, revision, dataset, or scoring method silently.
- Train or merge into a smaller host.
- Begin GLM work.
- Fine-tune or mutate donor weights.
- Delete source data or completed runs.
- Weaken a gate after observing results.
- Publish checkpoints, datasets, papers, releases, or external messages.
- Commit credentials, private prompts, provider metadata, model weights, or large run artifacts.
- Declare success based only on routing frequency, activation visualization, or REAP saliency.

## 6. Human approval gates

Stop and request a human decision before:

1. Any paid-resource provisioning or budget increase.
2. Changing the exact donor or primary precision.
3. Changing frozen datasets, splits, thresholds, or candidate manifests after results exist.
4. Excluding enough failed samples to fall below PRD coverage requirements.
5. Beginning bridge/merge design or training.
6. Beginning any GLM experiment.
7. Publishing or uploading weights or datasets.
8. Interpreting a contradictory replication as success.
9. Performing an irreversible or destructive action.

Human approval is not required for ordinary implementation, unit tests, dry runs, analysis, or experiments already authorized by a valid run configuration on a provisioned machine.

## 7. Single-lead execution model

Use one lead execution agent as the only writer of run state. Parallel workers may process immutable sample shards, but they must not independently change configuration, candidate sets, schemas, or conclusions.

The lead agent is responsible for:

- Acquiring the run-state lock.
- Validating all inputs before dispatch.
- Assigning deterministic, non-overlapping shard ranges.
- Reconciling worker outputs against expected sample IDs.
- Validating and hashing merged outputs.
- Updating state atomically.
- Enforcing budgets and stop conditions.
- Producing the final evidence classification.

Workers must be idempotent: rerunning a completed shard with identical inputs must produce identical validated outputs or an explicit nondeterminism report.

## 8. Mandatory plan format

Before implementation or an experiment, maintain a plan with one active step and explicit dependencies. Each task must include:

```text
task_id
objective
input files and hashes
expected outputs
definition of done
validation command
estimated GPU hours
estimated storage
dependencies
failure behavior
```

Tasks must be bounded and independently verifiable. “Run REAP,” “find the coding experts,” and “make it work” are not valid task definitions.

## 9. Run identity and configuration

Every experimental run requires a unique, immutable `run_id`. Recommended format:

```text
YYYYMMDDTHHMMSSZ-qwen35a3b-<condition>-<short_git_sha>
```

Before GPU execution, configuration must include:

```yaml
schema_version: 1
run_id: null
model:
  id: Qwen/Qwen3.5-35B-A3B
  revision: REQUIRED_IMMUTABLE_SHA
  precision: bf16
  text_only: true
runtime:
  seed: 20260903
  deterministic: true
  batch_size: 1
  max_input_tokens: REQUIRED
  max_new_tokens: REQUIRED
  enable_thinking: false
budget:
  max_gpu_hours: REQUIRED
  max_cost_usd: REQUIRED
  provider_rate_usd_per_hour: REQUIRED
  storage_limit_gb: REQUIRED
  deadline_utc: REQUIRED
datasets:
  manifest: REQUIRED
intervention:
  mode: none
```

`run_id` is resolved once from time, condition, and Git revision, then written immutably. A changed model revision, dataset manifest, prompt template, decoding configuration, or intervention requires a new run.

## 10. Stage state machine

Every milestone and expensive stage uses:

```text
PENDING
PREFLIGHTED
RUNNING
VALIDATING
COMPLETE
FAILED_RETRYABLE
FAILED_TERMINAL
WAITING_FOR_HUMAN
```

Only the lead agent changes state. Write new state to a temporary file, validate it, flush it, and atomically replace `state.json`. Never mark `COMPLETE` until all output hashes and validation results exist.

State must record:

- Run and task IDs.
- Status and attempt count.
- Input and output hashes.
- Code and model revisions.
- Start, heartbeat, and completion timestamps in UTC.
- GPU-hours and estimated cost consumed.
- Last successful sample/chunk.
- Validation command and exit code.
- Failure signature and remediation.
- Next permitted task.

## 11. Retry and failure policy

Classify failures before retrying:

| Class | Examples | Action |
|---|---|---|
| Transient | Network timeout, temporary provider interruption | Retry with bounded backoff |
| Resource | OOM, disk full, excessive context | Apply predeclared reductions and rerun pilot |
| Data-isolated | One malformed sample | Quarantine, record, continue within coverage gate |
| Systematic | Wrong tensor mapping, scorer broadly broken | Stop stage |
| Scientific | No stable candidates, causal gate fails | Report null result |
| Authorization | Missing access, budget, or paid-resource approval | Wait for human |

Allow at most two retries for an identical normalized failure signature. On the third occurrence, set `FAILED_TERMINAL` or `WAITING_FOR_HUMAN`; do not loop indefinitely.

Do not hide failures by lowering sample counts, shortening outputs, changing prompts, or switching models after seeing the result.

## 12. Budget enforcement

Before each GPU stage calculate:

```text
projected_stage_cost = projected_gpu_hours × provider_rate
projected_total_cost = consumed_cost + projected_stage_cost
```

Begin only if:

```text
projected_total_cost <= 0.80 × max_cost_usd
```

unless the remaining 20% is explicitly allocated to that final stage. The default reserve covers retries and replication.

Poll elapsed GPU time at least every 15 minutes during long operations. Emit a heartbeat containing stage, progress, throughput, GPU-hours, estimated cost, and ETA. Stop safely before crossing any hard budget or deadline.

Idle GPU time is a defect. If no forward progress occurs for 15 minutes and no known compilation/download task explains it, checkpoint, stop the workload, and diagnose.

## 13. Model preflight rules

Before downloading full weights:

1. Resolve the approved immutable model revision.
2. Download small metadata files.
3. Validate architecture against `prd.md`.
4. Calculate model download size and free-space requirement.
5. Record file list and expected hashes.
6. Verify compatible Transformers/PyTorch/REAP versions.
7. Verify GPU compute capability and available memory.

After loading:

- Enumerate MoE layers from configuration and module structure.
- Verify expert count and top-k from runtime tensors.
- Resolve fused versus unfused tensor layout.
- Identify the shared expert separately.
- Prove one layer-expert mapping with a controlled probe.
- Record actual peak resources.

Never hard-code architecture counts without also validating them against the pinned configuration.

## 14. Dataset governance

- Use only public or explicitly authorized data.
- Record source, license, revision, split, original ID, and content hash.
- Freeze ordered manifests before calibration.
- Keep calibration, selection, validation, and replication disjoint.
- Run exact and near-duplicate checks.
- Version prompt templates independently.
- Compare coding/control token-length and difficulty proxies.
- Never inspect replication outcomes before candidate and intervention manifests are frozen.
- Never move examples between splits after seeing model performance.

If a benchmark requires executing generated code, run it in an isolated sandbox with no network, constrained CPU/memory/time, and a disposable filesystem. Treat model-generated code as untrusted.

## 15. Determinism rules

The primary condition uses:

- Greedy decoding.
- Fixed prompt serialization.
- Fixed tokenizer and chat template.
- Fixed maximum input/output lengths.
- Fixed batch ordering.
- Fixed seeds for every library and worker.
- Deterministic algorithms where supported.
- No MTP/speculative decoding during attribution.
- No prefix-cache reuse across experimental conditions unless proven equivalent.

Record unavoidable nondeterministic kernels. Never claim bitwise reproducibility when the runtime cannot provide it.

Thinking-enabled and disabled runs require distinct IDs, outputs, and reports. Do not use `/think` or `/nothink` text as a substitute for the checkpoint's supported chat-template control.

## 16. Telemetry contract

Each routing record must contain at least:

```text
schema_version
run_id
sample_id
condition_id
segment                 # prompt, reference, generated
token_index
token_id
layer_index
expert_index
route_rank
router_weight
expert_output_l2
chunk_id
```

Derived per-expert summaries must include observation count explicitly. An expert with no routed tokens is `observed=false`; it is not automatically a zero-saliency expert.

Validate:

```text
routing row count
= analysed tokens × MoE layers × configured top_k
```

For every token/layer group, require exactly top-k unique ranks and expert indices. Reject NaN, infinity, out-of-range identities, orphaned token references, and unmanifested chunks.

Do not retain full hidden activations after streaming aggregation unless a bounded debug run explicitly requires them.

## 17. REAP and ranking rules

Implement the upstream REAP statistic without silently changing its denominator, router normalization, activation location, or aggregation. If compatibility requires a deviation:

1. Preserve a field for the upstream-compatible statistic.
2. Give the variant a new metric name.
3. Document the equation and motivation.
4. Test it on a synthetic fixture.

Always publish alongside REAP:

- Routing counts/rates.
- Router-weight sums and means.
- Expert-output norm summaries.
- Coding and control values separately.
- Within-layer standardized differential.
- Bootstrap interval and stability.
- Label-permutation comparison.

Do not rank raw values across layers without justified normalization.

## 18. Candidate-freeze rules

Candidate selection must be deterministic from:

```text
pinned analysis code
+ frozen selection configuration
+ frozen calibration/selection artifacts
```

Write `candidate-manifest.json` and all random/negative-control manifests before causal evaluation. Hash them and record their hashes in state. After freezing:

- Do not add or remove candidates.
- Do not change cardinality.
- Do not regenerate random controls with favourable seeds.
- Do not tune evidence thresholds against validation.

A change requires a new experiment generation and must leave the original intact.

## 19. Intervention rules

The primary intervention zeros the selected expert's weighted contribution before it is combined with other expert outputs. It does not renormalize remaining router weights or reroute the token.

Before full causal evaluation:

- Test one expert in one layer on one sample.
- Confirm only the intended layer-expert contribution changes.
- Confirm the intact/no-op intervention is bitwise equivalent to baseline when possible.
- Measure parse failures and broad loss changes.
- Save the exact intervention manifest.

Alternative semantics—renormalization, forced routing, expert replacement, or pruning—must use separate condition IDs and require approval if not already specified by the PRD.

## 20. Causal claim rules

Apply Gate D from `prd.md` exactly. A selected set is not `coding-critical-v0` unless it:

- Beats equal-cardinality layer-matched random controls by the declared factor.
- Reaches the required empirical random-control percentile.
- Damages coding more than general controls by the declared margin.
- Preserves enough general output validity for interpretation.
- Repeats in the same direction on untouched replication data.

Report item-level uncertainty, not only aggregate point estimates. Report every failed, timed-out, truncated, or unparseable item.

If evidence is mixed, choose the weaker label. Never convert correlation into causal language.

## 21. Extraction rules

Extraction is a read/copy/verify operation. It must not:

- Cast dtype.
- Dequantize or requantize.
- Average or merge tensors.
- Permute neurons.
- prune weights.
- Modify scales.
- Rename away source provenance.

For each selected layer-expert pair:

1. Resolve every required source tensor.
2. Copy exact tensor bytes into the artifact.
3. Record source shard, source key, extracted key, shape, dtype, byte count, and SHA-256.
4. Reload source and artifact independently.
5. Compare byte representation.
6. Fail the entire extraction gate if any expected tensor is absent or different.

Extracted experts are not standalone models. Every README and report must state that they depend on the donor representation, layer context, router, shared path, attention stack, and residual stream.

## 22. Artifact and repository hygiene

Recommended repository layout once implementation starts:

```text
configs/
docs/
schemas/
src/reverse_reap/
tests/
scripts/
runs/                 # ignored except tiny fixtures/manifests
```

Commit:

- Source code.
- Tests and tiny synthetic fixtures.
- Schemas.
- Example configurations without secrets.
- Small manifests and final reports.
- Checksums and provenance metadata.

Do not commit:

- Model shards or extracted expert weights.
- Hugging Face/provider tokens.
- Raw large telemetry.
- Private datasets.
- Generated code-sandbox contents.
- Environment dumps containing secrets.
- Provider invoices or account identifiers.

Use `.gitignore` before producing the first large artifact. Check `git status --short` before every commit.

## 23. Code-change workflow

Work feature by feature:

1. Inspect the actual repository and current state.
2. State the bounded task and definition of done.
3. Add or update tests with the implementation.
4. Run the narrowest relevant tests.
5. Run broader validation proportional to risk.
6. Inspect the diff and ensure unrelated changes are absent.
7. Commit only files belonging to the task.
8. Push only when the human requested pushing or repository policy explicitly authorizes it.
9. Report branch, local commit, remote commit, and validation separately.

Never rewrite user changes, reset the worktree destructively, or stage unrelated files.

## 24. Verification commands

Use repository-provided commands once they exist. Do not invent success when a tool is absent. The expected validation surface should eventually include equivalents of:

```text
format check
lint
type check
unit tests
integration tests on synthetic MoE
configuration/schema validation
run-bundle verification
extraction hash verification
```

Hardware-free tests must not download the donor. GPU integration tests must be explicitly marked and must print their estimated cost before running.

Every reported test result includes the exact command, exit code, and scope. A code-only suite does not prove model compatibility.

## 25. Monitoring and unattended operation

For a long-running experiment, write a heartbeat at least every 15 minutes containing:

```text
timestamp_utc
run_id
task_id
stage
completed_items / total_items
throughput
gpu_memory_peak
gpu_hours_consumed
estimated_cost_consumed
estimated_time_remaining
last_validated_chunk
```

The watchdog should stop the workload when:

- No forward progress occurs for 15 minutes without a known bounded compilation/download step.
- GPU memory errors repeat after approved fallback settings.
- Disk usage exceeds 90% of the configured limit.
- Output validation failure rate exceeds 5%.
- The projected budget or deadline will be exceeded.
- State or artifact hashes no longer reconcile.

Stopping safely and preserving evidence is preferable to continuing an invalid run.

## 26. Status reports

Each milestone report must lead with outcome, not activity. Use:

```text
Status: PASS | FAIL | NULL | WAITING_FOR_HUMAN
Milestone:
Run ID:
Exact model revision:
Evidence produced:
Validation performed:
Budget used / remaining:
Known limitations:
Next permitted action:
```

Distinguish:

- Implemented and unit-tested.
- Tested on a synthetic model.
- Tested on the exact Qwen checkpoint.
- Observationally supported.
- Causally supported.
- Replicated.
- Deferred or unverified.

## 27. Terminal outcomes

The autonomous execution ends in exactly one state:

### Positive

All PRD gates pass; a byte-verified `coding-critical-v0` bundle and replicated causal report exist.

### Null

Instrumentation and evaluation are valid, but stable coding-specific causal experts were not demonstrated. Publish the null evidence and do not extract a falsely labelled critical set. Observational tensors may be preserved only with the observational label.

### Feasibility failure

The exact donor cannot be instrumented or evaluated within approved hardware, software, or budget constraints. Produce a reproduction recipe, logs, attempted remediations, and the narrowest known blocker.

After any terminal outcome, stop. Request human review before bridge/merge or GLM work.

## 28. First tasks for a fresh execution agent

Unless the repository already contains later validated work, begin in this order:

1. Inspect Git status, history, and all governing documents.
2. Create a plan for M0 and M1 only.
3. Define schemas for experiment configuration, state, routing rows, candidates, and extraction manifests.
4. Add tiny synthetic fixtures and hardware-free invariant tests.
5. Inspect the official REAP repository and pin a compatible revision.
6. Design the Qwen3.5 architecture adapter behind a model-neutral interface.
7. Validate configuration and donor metadata without downloading full weights.
8. Produce a GPU resource estimate and M1 execution command.
9. Confirm an approved, provisioned GPU environment and budget configuration.
10. Run only the M1 smoke/pilot and evaluate Gate A.

Do not skip directly to a full calibration run.
