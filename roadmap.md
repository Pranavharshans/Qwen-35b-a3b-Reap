# Reverse-REAP v0 Roadmap

## 1. Mission and boundary

The v0 mission is to locate, causally validate, and losslessly extract coding-critical routed experts from `Qwen/Qwen3.5-35B-A3B`.

The roadmap ends at a verified expert bundle and evidence report. It does not implement a bridge, merge experts into a smaller host, train a recipient, or run GLM-5.3-Flash. Those are gated follow-on projects.

```text
Qwen architecture validation
        ↓
dataset freeze
        ↓
baseline + telemetry validation
        ↓
Reverse-REAP observational ranking
        ↓
candidate freeze
        ↓
causal ablation against controls
        ↓
lossless expert extraction
        ↓
independent replication and v0 report
        ↓
human go/no-go for bridge research
        ↓
human go/no-go for GLM replication
```

## 2. Execution principles

1. Start with the smallest test capable of disproving an assumption.
2. Never spend full-dataset GPU time before the preceding gate passes.
3. Pin every changing dependency and dataset revision.
4. Keep thinking-enabled and thinking-disabled evidence separate.
5. Use observational signals to nominate candidates, not to prove causality.
6. Freeze hypotheses, thresholds, and candidate sets before validation.
7. Treat null results and feasibility failures as valid deliverables.
8. Never overwrite completed run artifacts.
9. Extract source tensors without transformation.
10. Require human approval before expanding scope to merging or GLM.

## 3. Recommended compute strategy

### Primary prototype

Use one RTX PRO 6000 Blackwell 96 GB or another CUDA GPU with at least 80 GB VRAM for the official BF16 Qwen checkpoint. Begin with batch size 1, short contexts, text-only loading, and no MTP. Increase throughput only after telemetry correctness is established.

### Why not begin with an optimized serving runtime?

vLLM and SGLang are appropriate for throughput baselines, but fused MoE kernels may hide individual expert outputs or make interventions difficult. The primary calibration path should be an instrumentable PyTorch/Transformers implementation. Serving-runtime support is a later optimization and must reproduce the reference telemetry on a fixed micro-corpus.

### Budget configuration

Before any paid run, define:

```text
MAX_GPU_HOURS
MAX_COST_USD
PROVIDER_RATE_USD_PER_HOUR
STORAGE_LIMIT_GB
RUN_DEADLINE_UTC
```

The agent may use an already-provisioned machine within these bounds. It may not purchase credits, create paid instances, or increase limits without explicit human authorization.

## 4. Milestone overview

| Milestone | Outcome | Main gate |
|---|---|---|
| M0 | Repository and experiment contract ready | Documentation reviewed |
| M1 | Exact donor and runtime proven instrumentable | Gate A pilot |
| M2 | Coding and control corpora frozen | Dataset audit |
| M3 | Reproducible unmodified baseline | Gate B |
| M4 | Valid full calibration telemetry | Telemetry invariants |
| M5 | Candidate layer-expert manifest frozen | Gate C |
| M6 | Causal criticality tested | Gate D |
| M7 | Selected tensors extracted and verified | Gate E |
| M8 | Untouched replication completed | Same-direction replication |
| M9 | v0 report and decision package published | Human review |

## 5. Detailed milestones

## M0 — Repository foundation

### Objective

Provide enough specification for a fresh agent to execute without inventing the research question.

### Tasks

- Adopt `prd.md` as the requirements authority.
- Adopt `AGENTS.md` as the execution constitution.
- Create a conventional Python project layout when implementation begins.
- Add configuration schemas for experiments, datasets, interventions, and run state.
- Add a command that validates configuration without loading the model.
- Add CI checks for unit tests, formatting, type checks, and manifest schemas.
- Record all unresolved decisions in `docs/decisions/` rather than hiding them in code comments.

### Deliverables

- `prd.md`
- `roadmap.md`
- `AGENTS.md`
- Later implementation: validated example configuration and dry-run command

### Exit criteria

- Scope, donor, terminology, gates, and output contract agree across all three documents.
- No merge or GLM task appears in the v0 execution queue.

## M1 — Donor preflight and instrumentation spike

### Objective

Prove that the exact pinned Qwen checkpoint exposes correct routing and activation signals before building the full pipeline.

### Tasks

1. Resolve the latest approved immutable revision for `Qwen/Qwen3.5-35B-A3B` and write it into the experiment configuration.
2. Download configuration/tokenizer metadata first; compare all architecture fields with the PRD donor contract.
3. Record model shard sizes and storage requirements before downloading weights.
4. Load text-only BF16 weights with batch size 1 and a conservative context limit.
5. Locate all MoE layers, router modules, routed expert tensors, and shared-expert tensors.
6. Verify expected counts from the pinned config rather than hard-coding 40/256/8.
7. Run one short deterministic prompt with instrumentation disabled.
8. Run the same prompt with router capture enabled.
9. Add streaming capture of expert output norms required by REAP.
10. Confirm instrumented and uninstrumented generated token IDs are identical.
11. Mask one known routed expert for one layer and confirm the layer output changes without crashing.
12. Measure peak VRAM, RAM, disk, and tokens/second.

### Tests

- Unit test maps synthetic router tensors to `(token, layer, rank, expert, weight)` correctly.
- Expert indices are inside `[0, num_experts)`.
- Each analysed token has exactly `top_k` unique routed expert indices per MoE layer.
- Captured expert output norm is finite and non-negative.
- Shared experts are not mislabelled as routed experts.
- Instrumentation changes neither logits nor token IDs when interventions are disabled.

### Deliverables

- `architecture-report.json`
- `environment.json`
- `probe-routing.parquet` or equivalent typed columnar file
- `probe-reap-summary.parquet`
- `instrumentation-validation.md`
- Peak-resource measurements

### Stop conditions

- Architecture differs from the approved donor.
- Any signal cannot be mapped unambiguously to a layer-expert identity.
- Capture changes the baseline output.
- OOM persists after batch 1, text-only loading, shorter context, and documented memory controls.

### Exit gate

Gate A from `prd.md` passes on at least 10 samples.

## M2 — Dataset construction and freeze

### Objective

Create disjoint coding and control corpora that support differential and causal claims.

### Candidate sources

Final sources must be reviewed for licensing and reproducibility. Candidate categories include:

- HumanEval+/EvalPlus-style function synthesis.
- MBPP+/EvalPlus-style function synthesis.
- A pinned subset of LiveCodeBench or another contamination-aware code benchmark.
- Executable bug-repair or debugging tasks with deterministic tests.
- Code-understanding tasks with exact or unit-test scoring.
- Matched reasoning and technical controls with comparable token lengths.

Dataset choice is not delegated to runtime improvisation. The agent proposes a manifest; the manifest must pass automated audits before the freeze marker is written.

### Tasks

- Normalize all samples to a versioned schema.
- Preserve original sample IDs and source revisions.
- Record prompt template version separately from source content.
- Remove exact and near duplicates across all splits.
- Detect train/selection/validation leakage introduced by derived variants.
- Measure target/control prompt-token distributions.
- Assign immutable calibration, selection, validation, and replication splits.
- Build small, medium, and full manifests without changing membership order.
- Store expected scoring method and timeout for every sample.
- Hash every normalized record and the ordered dataset manifest.

### Scale ladder

| Tier | Coding items | Control items | Purpose |
|---|---:|---:|---|
| Smoke | 10 | 10 | Schema, scorer, and telemetry |
| Pilot | 50 | 50 | Resource estimate and preliminary stability |
| Medium | 200 | 200 | Candidate-ranking rehearsal |
| Full v0 | ≥500 | ≥500 | Frozen analysis and validation |

### Exit criteria

- Licenses and citations are recorded.
- No cross-split duplicate survives the audit.
- At least 95% of items pass scorer preflight.
- Target/control length comparison is published.
- Full dataset manifest is immutable and hashed.

## M3 — Baseline evaluation

### Objective

Establish what the intact donor can actually do under the exact experimental settings.

### Tasks

- Run the smoke suite under C0: thinking disabled and greedy decoding.
- Verify scoring, timeouts, truncation, and response parsing.
- Repeat C0 and test token-level determinism.
- Run the pilot suite and estimate total GPU time and storage.
- Run C1 separately with thinking enabled and its own output budget.
- Record per-stratum accuracy/pass rate, generated tokens, latency, and failures.
- Freeze the generation configurations used by later interventions.

### Required comparisons

- C0 repeated run versus C0 original.
- C0 versus C1 performance and token-count distributions.
- Capture disabled versus capture enabled on the same micro-corpus.

### Exit gate

Gate B passes. If baseline coding performance is too low to measure degradation reliably, stop and report that the chosen tasks are unsuitable.

## M4 — Calibration telemetry

### Objective

Collect validated router and expert-output statistics over the frozen calibration split.

### Tasks

1. Estimate storage and runtime from the pilot.
2. Start the full calibration only if it fits the remaining budget with 20% reserve.
3. Stream aggregate statistics per layer-expert pair.
4. Persist token-level routing rows in bounded chunks.
5. Keep prompt, reference/teacher-forced completion, and free-generation segments distinguishable.
6. Validate each chunk before marking it complete.
7. Resume from the last validated chunk after interruption.
8. Produce coverage, load-balance, routing-entropy, and REAP-saliency summaries.

### Validation invariants

```text
routing_rows = analysed_tokens × moe_layers × top_k
```

Also validate:

- Unique ranks per token/layer.
- Unique selected experts per token/layer.
- Finite router weights and activation norms.
- Sequential sample and chunk references.
- Exact manifest byte counts and SHA-256 hashes.
- Every aggregate can be reconciled to its token-level source rows.

### Exit criteria

- All chunks pass validation.
- No missing layer silently becomes a zero-activity layer.
- Observed/unobserved state is explicit.
- Calibration report contains no causal language.

## M5 — Reverse-REAP candidate ranking

### Objective

Produce and freeze an observational shortlist of potentially coding-specific experts.

### Analysis sequence

1. Calculate upstream-compatible REAP saliency independently for coding and controls.
2. Calculate routing frequency and router-mass baselines.
3. Standardize within layer to avoid comparing incompatible raw layer scales.
4. Calculate coding-minus-matched-control effects.
5. Bootstrap samples within each stratum.
6. Run label permutations to estimate the null distribution.
7. Test ranking sensitivity to prompt length and programming language.
8. Compare C0 and C1 without pooling them.
9. Construct candidate sets across a predeclared cardinality grid.
10. Choose the smallest stable set that satisfies Gate C.

### Candidate-set controls

Precompute and freeze:

- At least 20 layer-matched random sets.
- At least 20 frequency-matched random sets.
- A highest-frequency set.
- A lowest-differential set.
- If feasible, a standard task-agnostic REAP-retained set of equal size.

### Deliverables

- `expert-ranking.parquet`
- `bootstrap-stability.json`
- `label-permutation.json`
- `candidate-manifest.json`
- Random and negative-control manifests
- Observational report with uncertainty

### Exit gate

Gate C passes, or the project terminates with a valid observational/null report. The agent must not tune the threshold using validation outcomes.

## M6 — Causal validation

### Objective

Determine whether the frozen candidate set is necessary for coding performance beyond chance and general model degradation.

### Intervention order

1. One-sample, one-expert masking sanity check.
2. One-sample selected-set check for output validity.
3. Smoke validation across selected, random, and negative controls.
4. Pilot validation to estimate effect sizes and runtime.
5. Full frozen validation only if earlier checks are interpretable.
6. Individual-expert leave-one-out tests within the selected set, budget permitting.
7. C1 reasoning-enabled replication after C0 primary analysis.

### Primary masking semantics

Set the selected expert contribution to zero after normal expert computation and before combination, without renormalizing surviving router weights. This estimates the necessity of the removed contribution. Any renormalized or rerouted experiment receives a separate condition ID.

### Measurements

- Coding pass-rate/accuracy delta from intact baseline.
- Matched-control and general-control deltas.
- Item-level paired confidence intervals.
- Selected-set percentile within random-set null distributions.
- Output parse/error/truncation rates.
- Token loss or perplexity where meaningful.
- Latency and peak memory overhead.
- Per-stratum effects and heterogeneity.

### Interpretation checks

- A global language collapse is not coding specificity.
- A large loss caused by one malformed layer is not a distributed expert finding.
- A selected set that behaves like high-frequency controls has not passed causal localization.
- A result present only in the selection data has not replicated.

### Exit gate

Apply Gate D exactly. Label the set `coding-critical-v0` only if every criterion passes. Otherwise retain the honest label `observational-candidates`.

## M7 — Lossless expert extraction

### Objective

Package the frozen, validated layer-expert tensors for later bridge research without altering them.

### Tasks

- Resolve each selected layer-expert pair to source tensor names.
- Copy gate, up, and down projections and associated scale tensors.
- Preserve dtype, shape, byte order, and tensor naming provenance.
- Record whether tensors came from fused or per-expert source layouts.
- Include no unrelated experts.
- Store causal and observational metadata alongside, not inside, tensor values.
- Reload source and extraction through independent code paths.
- Compare tensor bytes and SHA-256 hashes.
- Calculate total extracted parameter count and storage.
- Produce a future-bridge compatibility note without implementing the bridge.

### Deliverables

```text
extraction/
  experts.safetensors
  extraction-manifest.json
  source-to-extracted-map.json
  checksums.sha256
  verification-report.json
  README.md
```

### Exit gate

Gate E passes for every tensor. One missing or transformed tensor fails the extraction stage.

## M8 — Untouched replication

### Objective

Test the frozen claim once on data that influenced neither scoring nor selection.

### Tasks

- Lock code and configuration before revealing replication outcomes.
- Evaluate baseline, selected-set ablation, and frozen random controls.
- Do not replace failed items unless the dataset itself is corrupt and the exclusion rule was predeclared.
- Report the same metrics and confidence intervals as validation.
- Compare effect direction, magnitude, and rank stability.

### Exit criteria

- Same-direction coding-specific degradation is observed.
- Output validity remains interpretable.
- Any attenuation or contradiction is stated prominently.

If replication fails, downgrade `coding-critical-v0` to `unreplicated-candidates` and preserve the extracted tensors only as research artifacts.

## M9 — v0 report and handoff

### Objective

Produce a decision package that a human can audit without rerunning the experiment.

### Required report sections

- Exact donor revision and environment.
- Dataset manifests and licenses.
- Baseline performance.
- Telemetry integrity results.
- Observational ranking and uncertainty.
- Causal interventions and controls.
- Replication outcome.
- Extracted artifact checksums.
- Known limitations and failed attempts.
- Actual GPU hours and cost.
- Positive, null, or feasibility-failure terminal classification.

### Human decisions after v0

Decision 1: Is the causal evidence strong enough to design a bridge into a small host?

Decision 2: Is the procedure stable and affordable enough to port to GLM-5.3-Flash?

Neither decision may be made automatically by the execution agent.

## 6. Indicative schedule

These are planning ranges, not promises. Actual duration must be recalculated from the M1/M3 pilots on the rented hardware.

| Phase | Expected human/agent elapsed time | GPU exposure |
|---|---|---|
| M0 | 0.5–1 day | None |
| M1 | 1–3 days | 2–8 hours |
| M2 | 1–3 days | None or minimal |
| M3 | 0.5–2 days | 2–10 hours |
| M4 | 1–3 days | 8–30 hours |
| M5 | 0.5–2 days | Primarily CPU |
| M6 | 2–7 days | 20–80 hours |
| M7 | 0.5–1 day | 1–4 hours plus storage I/O |
| M8 | 1–3 days | 8–30 hours |
| M9 | 0.5–1 day | None |

The causal random-control sweep is likely the dominant cost. Reduce sample count only through a documented amendment made before observing final results; never quietly reduce the number of controls to fit a preferred conclusion.

## 7. Agent checkpoint protocol

Every milestone follows this state machine:

```text
PENDING
  → PREFLIGHTED
  → RUNNING
  → VALIDATING
  → COMPLETE

Any state may transition to:
  → FAILED_RETRYABLE
  → FAILED_TERMINAL
  → WAITING_FOR_HUMAN
```

The state file must include:

- Milestone and task ID.
- Attempt number.
- Input artifact hashes.
- Output artifact hashes.
- Start/update/end timestamps in UTC.
- GPU-hours and estimated cost consumed.
- Validation command and exit status.
- Failure reason or gate decision.
- Next permitted action.

No more than two retries are allowed for the same failure signature. A third occurrence is terminal and requires human review.

## 8. GLM progression gate

GLM-5.3-Flash becomes eligible for a new planning phase only if:

- Qwen Gate A instrumentation is reliable.
- A candidate set can be frozen without leakage.
- Causal intervention semantics are validated.
- Extraction is byte-exact.
- Actual Qwen cost and duration are known.
- At least one positive or scientifically useful null result is documented.
- A separate GLM hardware/runtime feasibility study is approved.

The GLM plan must account for its different architecture, 288 routed experts across 43 MoE layers, FP8 tensor layout, much larger storage footprint, and multi-GPU requirements. Qwen code must not assume its adapter generalizes to GLM.

## 9. Future bridge handoff—not implementation

The v0 extraction bundle should make a later transfer study possible by preserving:

- Donor hidden size and expert intermediate size.
- Original layer and expert identity.
- Complete expert tensor set.
- Source checkpoint hashes.
- Input/output activation statistics useful for bridge initialization.
- Causal effect sizes and domain specificity.
- Candidate cardinality Pareto curves.

The future project must compare a selected-expert bridge against random-expert, zero/disabled-bridge, equal-parameter LoRA, and random frozen-MLP controls. This roadmap authorizes none of those training runs.

## 10. Completion checklist

- [ ] Exact official donor revision pinned.
- [ ] Architecture adapter validated.
- [ ] Instrumentation is behavior-preserving when disabled.
- [ ] Coding and control datasets frozen and hashed.
- [ ] Baseline reproducible.
- [ ] Telemetry row counts and values validated.
- [ ] REAP and differential rankings generated.
- [ ] Candidate and control manifests frozen.
- [ ] Causal validation completed without changing thresholds.
- [ ] Replication split evaluated once.
- [ ] Expert tensors extracted byte-for-byte.
- [ ] Run bundle and checksums verified.
- [ ] Actual cost and limitations reported.
- [ ] Terminal outcome classified honestly.
- [ ] Human review requested before bridge or GLM work.
