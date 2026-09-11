# Four-Expert Bridge Rescue Experiments

## Status and evidence boundary

This document defines five **separate, sequential, exploratory experiments** for
the existing Qwen3.5-2B bridge and its four frozen extracted donor experts. It
does not authorize execution, paid infrastructure, retraining, publication, or
changes to the completed MBPP+ result.

The completed MBPP+ comparison is the motivation, not a tuning set. The bridge
was approximately neutral with thinking disabled and worse on official coding
accuracy with thinking enabled, despite reducing some cap hits and unfinished
reasoning. Those 378 tasks have been inspected and scored, so no configuration
may be selected on them and then described as independently validated.

The four donor experts remain `observational-candidates`; their tensors are
`extracted`. These experiments can optimize how the existing bridge is used,
but they cannot upgrade the experts to `coding-critical-v0`.

## Shared dataset design

Freeze one new ordered pool of **164 coding problems** that has not been used in
the bridge's MBPP+ or earlier HumanEval+ analyses:

- **64 development problems**: the only problems available to Experiments 1-5
  for configuration comparison and selection.
- **100 untouched confirmation problems**: opened once, after one final policy
  and all thresholds are frozen.

The 64-problem development subset is large enough for economical screening:
one changed result equals 1.5625 percentage points. It is not large enough for
a strong final claim, and repeated reuse across five experiments increases
selection bias. The 100-problem confirmation subset therefore remains sealed;
one changed result equals one percentage point.

Candidate datasets include a reproducibly executable subset of BigCodeBench or
a time-pinned LiveCodeBench release. Before choosing one, verify its official
revision, license, task-level sandbox requirements, reference pass rate, and
absence from every existing Reverse-REAP run. Freeze source IDs, order, prompt
serialization, entry points, tests, hashes, difficulty strata, token limits,
and the 64/100 split before GPU generation. Do not substitute MBPP+ or the
previously inspected HumanEval+ outputs as untouched confirmation evidence.

Every experiment uses thinking-enabled inference as its primary condition.
Include the unmodified base and the current always-on bridge as paired controls
on the same task IDs. Use greedy decoding, identical prompts, identical token
limits, deterministic execution, and official sandboxed scoring. Invalid,
truncated, unclosed, and unsanitizable rows remain failures in the denominator.

## Common decision rule

Development experiments are ranking exercises, not confirmation. For every
variant report official pass rate, paired fixes and breaks versus base, exact
paired p-value, paired bootstrap interval, cap hits, structural failures,
generated tokens, runtime, and bridge telemetry.

Select a variant only when:

1. it has more fixes than breaks on the 64 development problems;
2. its official accuracy is not below the unmodified base;
3. its structural-invalid rate does not increase;
4. it has zero missing, duplicate, reordered, OOM, NaN, or Inf rows; and
5. its effect is mechanistically consistent with the experiment's hypothesis.

After Experiment 5, freeze at most one final policy. Run that policy and the
unmodified base once on the 100 untouched confirmation problems. A useful
follow-on signal requires at least five net additional solves (+5 percentage
points), a positive paired interval or a clearly labelled underpowered trend,
and a relative cap-hit reduction of at least 20% without increased structural
failures. Report anything weaker as exploratory or null.

## Experiment 1 — Fixed bridge-strength sweep

**Hypothesis:** the trained bridge is directionally useful but its current
residual contribution oversteers the host.

Compare the base, current always-on bridge, and fixed runtime multipliers
`0.025`, `0.05`, `0.10`, and `0.15` on all 64 development problems. A multiplier
scales the already trained gated residual; it does not edit checkpoint weights
or silently change the trained `gate_cap`.

```text
task_id: bridge-rescue-01-strength
objective: select at most one fixed residual multiplier
input files and hashes: frozen 64-task development manifest, host manifest,
  bridge checkpoint, extraction manifest, expert artifact, prompt/config hashes
expected outputs: six condition files, telemetry, official scores, paired report
definition of done: all expected rows reconcile and one result is selected or NULL
validation command: repository strength-sweep validator plus official scorer
estimated GPU hours: measure with an 8-task preflight before authorizing the 64
estimated storage: less than 5 GB excluding the already present host/artifacts
dependencies: shared dataset freeze and runtime-policy implementation
failure behavior: fail closed; do not add strengths after viewing scores
```

## Experiment 2 — Adaptive late activation

**Hypothesis:** the bridge helps terminate unusually long reasoning but harms
productive early reasoning when active from token zero.

Compare base, always-on bridge, activation after 1,024 generated reasoning
tokens, activation after 1,536 tokens, and a predeclared ramp: zero through
1,024, linearly increasing to the selected Experiment 1 multiplier by token
2,048. Disable the bridge immediately after the tokenizer-observed `</think>`
boundary. Run all variants on the same 64 development problems.

```text
task_id: bridge-rescue-02-late-activation
objective: determine whether late activation preserves accuracy while reducing loops
input files and hashes: Experiment 1 decision, shared manifests and artifact hashes
expected outputs: five condition files, activation traces, telemetry, paired report
definition of done: every bridge row records activation token and applied multiplier
validation command: policy-boundary tests, generation validator, official scorer
estimated GPU hours: measured 8-task preflight projection before launch
estimated storage: less than 5 GB incremental
dependencies: Experiment 1 complete or NULL; token-aware runtime policy
failure behavior: fail closed on missing think boundary or policy telemetry
```

## Experiment 3 — Per-expert and leave-one-out ablation

**Hypothesis:** one or more of the four observational experts causes most of the
accuracy damage, while another may carry the termination benefit.

Run ten predeclared conditions on the same 64 problems: base, all four experts,
each of four experts alone, and each of four leave-one-out combinations. Use the
best non-adaptive strength from Experiment 1; do not combine this experiment
with threshold tuning. Report every expert by `(donor_layer, donor_expert)` and
mapped host layer.

```text
task_id: bridge-rescue-03-expert-ablation
objective: identify helpful, harmful, redundant, and interaction-dependent mappings
input files and hashes: selected strength, four frozen mappings and shared manifests
expected outputs: ten condition files, per-mapping telemetry and interaction report
definition of done: exact expert masks verified and every condition officially scored
validation command: mask isolation tests, hook-leak test, official paired scorer
estimated GPU hours: measured 8-task preflight projection before launch
estimated storage: less than 8 GB incremental
dependencies: Experiment 1 result; immutable per-expert runtime mask
failure behavior: report NULL if no subset beats base; do not search arbitrary subsets
```

## Experiment 4 — Layer and reasoning-phase timing

**Hypothesis:** expert residuals are useful only at particular host depths or
during a particular reasoning phase.

Using the best predeclared expert subset from Experiment 3, compare five
conditions on the same 64 problems: base, all mapped layers, early mapped layers
only, late mapped layers only, and thinking-phase-only injection. Layer groups
must be declared from the existing mappings before results are viewed; do not
move experts to new host layers in this experiment.

```text
task_id: bridge-rescue-04-layer-timing
objective: localize any useful bridge effect by existing host depth and phase
input files and hashes: selected strength/subset and frozen mapping definitions
expected outputs: five condition files, layer/phase telemetry, paired report
definition of done: hooks execute only at authorized layers and phases
validation command: layer-isolation, think-boundary, no-hook-leak, official scorer
estimated GPU hours: measured 8-task preflight projection before launch
estimated storage: less than 5 GB incremental
dependencies: Experiments 1 and 3; layer and phase policy support
failure behavior: fail closed on ambiguous phase state; never remap layers post hoc
```

## Experiment 5 — Learned conditional gate

**Hypothesis:** a small controller can predict when the frozen bridge should be
off, weak, or active better than a fixed length threshold.

Train only a compact controller from separately frozen training trajectories.
Its inputs may include generated-token position, host-hidden summaries, current
learned gate statistics, and predeclared repetition features. It must include a
null route and must not receive MBPP+, the 64 development labels, or the 100
confirmation labels as training targets. Compare base, the best deterministic
policy from Experiments 1-4, and the learned controller on the 64 development
problems before freezing one final policy.

```text
task_id: bridge-rescue-05-learned-gate
objective: learn conditional bridge use without modifying host or expert tensors
input files and hashes: separate training trajectories, frozen bridge and prior decision
expected outputs: controller-only checkpoint, training report, three scored conditions
definition of done: controller provenance, null-route use, deterministic replay,
  controls, scores and hashes all validate
validation command: controller checkpoint validator, replay tests, official scorer
estimated GPU hours: pilot and budget projection required before training authorization
estimated storage: less than 10 GB excluding reusable host/artifacts
dependencies: Experiments 1-4 and new controller trainer/runtime
failure behavior: select deterministic policy or base when learned gate does not win
```

## Required implementation before execution

The repository can already load the frozen host, bridge checkpoint, four expert
tensors, attach one sidecar per mapped host layer, collect aggregate bridge
telemetry, generate the four fixed MBPP+ conditions, and score official EvalPlus
outputs. It **cannot run these five experiments safely without code changes**.

Implement and test the following as separate commits before any experiment:

1. A strict, hash-bound runtime-policy schema containing residual multiplier,
   expert allowlist, existing-layer allowlist, activation threshold/ramp, and
   thinking-phase policy. Defaults must reproduce the existing always-on path.
2. A token/phase controller updated by the generation loop and read by sidecar
   hooks without global cross-run state. Record the exact applied policy for
   every generated token or an auditable bounded summary.
3. Runtime residual scaling that does not mutate the bridge checkpoint or its
   trained `gate_cap`.
4. Exact per-expert and per-layer masks with isolation tests proving excluded
   sidecars emit zero residual and included sidecars retain their original output.
5. New condition/config generation, immutable fingerprints, unique run IDs,
   resumable files, expected-row reconciliation, policy telemetry, and paired
   official-score reports for arbitrary predeclared variants.
6. A controller-only trainer and checkpoint schema for Experiment 5, including
   null-route, determinism, frozen host/expert assertions, split isolation,
   equal-parameter control, and no host-weight serialization.
7. Hardware-free fake-model tests, exact-checkpoint one-sample preflights,
   deterministic repeats, hook-removal tests, budget projections, and archive
   verification for every new runtime path.

Do not encode all five experiments as one execution command. Each experiment
gets its own config, fingerprint, run ID, state lock, budget, report, and human
checkpoint. Shared orchestration may prepare immutable inputs, but it must stop
after each experiment. This containment prevents a later agent from silently
turning an exploratory sweep into an unrestricted search for a positive score.

