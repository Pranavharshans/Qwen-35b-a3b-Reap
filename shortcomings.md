# Reverse-REAP Bridge: Shortcomings and Recommended v2

## Status and purpose

This document records the limitations of the current Qwen3.5-35B-A3B to
Qwen3.5-2B expert bridge, the evidence boundaries, how it differs from a larger
fusion architecture such as Mini-Whale-1-12B, and a recommended next design.

The current expert set remains `observational-candidates`. The bridge remains a
`trained-observational-bridge-unvalidated` artifact. Neither label establishes
that the extracted experts are coding-critical or that donor coding capability
has been transferred.

## What v1 implements

The v1 bridge attaches four frozen, byte-verified donor experts to four fixed
Qwen3.5-2B host layers:

| Donor expert | Host layer |
|---|---:|
| layer 3, expert 26 | 2 |
| layer 7, expert 18 | 4 |
| layer 35, expert 239 | 18 |
| layer 37, expert 5 | 20 |

For each mapping, the host hidden state passes through a trainable nonlinear
input adapter, the frozen donor SwiGLU expert, a trainable output adapter, and
an input-dependent capped gate. The delta is added to the host residual stream.
The host checkpoint and extracted expert tensors remain frozen.

```text
host hidden state
    -> input adapter (2048 -> 256 -> 2048)
    -> frozen donor expert
    -> output adapter (2048 -> 256 -> 2048)
    -> input-dependent gate (maximum 0.25)
    -> residual addition at one fixed host layer
```

Training used frozen vector supervision over captured host and donor states. It
did not use end-to-end language-model loss, coding correctness, or execution
feedback. No modified host weights were saved.

## What the evidence supports

The project has demonstrated that:

- the four extracted expert artifacts can execute inside the smaller host;
- source expert tensors remain frozen and byte-verified;
- the bridge is deterministic under the pinned runtime;
- disabled/no-op operation preserves the original host path;
- trained sidecars execute, open finite gates, and emit finite residuals;
- bridge training reduced its vector loss on train, validation, and test data;
- the MBPP+ pilot produced small positive point estimates but no statistically
  demonstrated capability improvement;
- qualitative examples suggest a possible reasoning-termination effect when
  the bridge prevents a long thinking loop from reaching the token ceiling.

The current full benchmark is unfinished until all four conditions are
generated, officially scored, paired, and reported. Early throughput, cap-hit,
or anecdotal response differences are not capability results.

## Shortcomings of v1

### 1. Only four experts were transplanted

The donor contains 40 routed layers with 256 experts per layer and top-8
routing. Four experts are an extremely sparse sample of that capacity. Coding
behaviour may be distributed across simultaneously routed experts, the shared
expert path, attention, router weights, and their composition over many layers.

Four experts may affect response style or reasoning termination without
carrying enough information to supply broad coding competence.

### 2. Candidate discovery was observational and underpowered

REAP-style saliency and coding-versus-control routing produced candidates, not
causal proof. One candidate, `(3,26)`, later failed its frozen minimum coding
coverage requirement. The threshold was not weakened, so the set must not be
described as the best experts or as coding-critical.

### 3. Discovery did not adequately cover thinking-enabled coding

Primary discovery used thinking-disabled behaviour. The benchmark indicates
that the most visible bridge effect may occur during thinking-enabled
generation. Experts involved in planning, verification, and reasoning
termination may differ from those selected from shorter disabled traces.

### 4. The donor's total parameter count overstates what was transferred

Qwen3.5-35B-A3B has roughly 35B total parameters but activates only a fraction
per token. Four expert tensors do not transfer a dense 35B model or the donor's
complete active path. Attention, routers, shared paths, and the other active
experts remain absent.

### 5. There is no learned router

Each v1 expert is permanently attached to one host layer. The bridge cannot
learn:

- which expert a token needs;
- whether no expert is appropriate;
- whether several expert behaviours should cooperate;
- how use should differ across code, prose, planning, and final answers;
- how to balance utilization and prevent expert collapse.

This is likely the largest functional difference from a genuine fused MoE.

### 6. Layer alignment is heuristic

Donor layers were mapped to host layers by approximate relative depth. Matching
depth does not establish representational compatibility. Layer insertion should
be learned or selected using held-out alignment evidence.

### 7. The training objective is disconnected from task success

Vector reconstruction does not directly optimize next-token probabilities,
correct algorithms, compilable code, test passage, coding specificity, or
general-capability retention. The first qualitative MBPP+ pairs show the bridge
can stop a runaway trace while shared logic and interface mistakes survive.

### 8. The host is completely frozen

This makes the experiment clean and reversible, but prevents the host from
learning how to interpret donor-derived residuals. A small, separately stored
host LoRA may be necessary for meaningful transfer.

### 9. Residual safety is incomplete

The input-dependent gate caps scalar contribution, but v1 lacks a complete
stability system including:

- normalization of each mapped expert delta before injection;
- an explicit delta-to-host norm-ratio clamp;
- a learned residual-repair path;
- router entropy and load-balancing objectives;
- a null/no-expert route.

### 10. Improvement is not yet attributable to donor knowledge

A complete controlled comparison has not established whether any gain comes
from donor knowledge, ordinary added parameters, bridge training data, changed
residual dynamics, shortened reasoning, or benchmark variance. Required
controls include disabled, untrained, shuffled-pair, equal-parameter adapter,
random frozen MLP, and random extracted-expert conditions.

### 11. The 2B host may be too weak

The inexpensive 2B host may lack the capacity to exploit transplanted features.
A 4B host may integrate them better. Its relative gain might be smaller because
its baseline is stronger, but a gain would be more meaningful. The existing 2B
bridge cannot be reused: new host states, mappings, adapters, training, and
evaluation would be required.

### 12. Benchmark coverage remains narrow

MBPP+ covers bounded function synthesis, not repository repair, multi-file
reasoning, long-context coding, tool use, or cybersecurity. A single benchmark
can also confuse formatting and termination improvements with algorithmic gain.

### 13. Efficiency can be mistaken for capability

Fewer tokens, fewer cap hits, and faster generation matter only if correctness
is preserved. Premature termination can also be fast. Reports must keep
efficiency, structural validity, sanitized-code accuracy, and official test
accuracy separate.

## Comparison with Mini-Whale-1-12B

The supplied Mini-Whale description occupies a very different design point:

| Dimension | Mini-Whale description | Reverse-REAP v1 |
|---|---|---|
| Host | Qwen3-4B | Qwen3.5-2B |
| Donor modules | 260 unique experts, 303 placements | 4 experts, 4 placements |
| Augmentation | all 36 host layers | 4 host layers |
| Routing | learned top-2 router per token | fixed expert per mapped layer |
| Host adaptation | attention LoRA, later merged | host fully frozen |
| Residual controls | norm, sigmoid gate, norm clamp, repair | capped input-dependent gate |
| Objective | coding instruction SFT | frozen vector alignment |
| Deployment | custom monolithic fused model | detachable sidecar |
| Scientific isolation | many changes combined | narrow and reversible |

Mini-Whale has much greater capacity and can learn which donor expert to use,
so it has a greater chance of changing coding capability. However, the supplied
description alone does not substantiate claims such as "codes like DeepSeek."
A falling training loss is not a benchmark. Reproducible baselines, official
coding evaluations, paired controls, artifact provenance, and independent
verification are still required.

Its source should also be checked for the exact rank-7 factorization, expert
reuse, expert provenance, router training, parameter accounting, quantized
deployment, and immutable DeepSeek revision. Reverse-REAP is easier to audit
but much less expressive; Mini-Whale is more expressive but combines too many
changes to attribute a gain automatically to donor experts.

## Recommended v2: routed expert-bank distillation

The recommended architecture lies between v1 and a 260-expert monolith. It
should transfer composed donor behaviour into compact modules that operate
natively in the host representation space.

```text
Qwen3.5-4B host hidden state
            |
      learned sparse router
       /      |       \
      v       v        v
 compact   compact   compact       8-16 student experts
 student   student   student
       \      |       /
          top-1/top-2 mixture
                 |
      normalized and norm-clamped delta
                 |
        gated residual addition
```

### Stage 1: stronger donor discovery

- Run thinking-disabled and thinking-enabled discovery separately.
- Use harder verified coding tasks and difficulty-matched controls.
- Pre-register sufficient context/output allowance for thinking traces.
- Evaluate candidate cardinalities such as 4, 8, 12, and 16.
- Combine routing, weight, output magnitude, domain differential, stability,
  and causal intervention evidence.
- Preserve untouched replication data.

### Stage 2: capture composed donor targets

For relevant tokens, record bounded and streamed targets:

- host hidden states at candidate insertion layers;
- selected expert inputs and outputs;
- the aggregate routed MoE residual after top-k weighting;
- router probabilities and selected identities;
- shared-expert contribution where accessible;
- donor top-k logits or log-probabilities;
- task, domain, segment, and thinking-mode labels.

The aggregate residual is more informative than one expert output because it
preserves the composition the donor actually used.

### Stage 3: train compact host-native student experts

- Use 8-16 bottleneck SwiGLU students in the host hidden dimension.
- Initialize them as near-zero residual modules.
- Distil aggregate donor residual direction and norm.
- Retain extracted tensors as teachers and research artifacts rather than
  mandatory inference dependencies.
- Train an equal-parameter donor-free bank as a control.

### Stage 4: train a sparse router

- Route top-1 or top-2 students per token.
- Include an explicit null/no-expert route.
- Use entropy and load-balancing regularization.
- Monitor collapse, utilization, domain specificity, and layer usage.
- Freeze routing before final evaluation.

### Stage 5: permit bounded host adaptation

After representation training stabilizes, add a small LoRA to selected host
attention/output projections. Preserve the original host and store LoRA
separately. This lets the host consume the residual without becoming
unrestricted fine-tuning.

### Stage 6: connect training to capability

Train progressively:

1. representation distillation;
2. donor-logit distillation on coding trajectories;
3. verified coding supervised fine-tuning;
4. optional sandboxed execution-feedback optimization.

Each stage needs a new immutable identity and evaluation before proceeding.

### Stage 7: run discriminating controls

Compare equal trainable-parameter budgets for:

- untouched host;
- host plus ordinary adapters;
- host plus shuffled donor targets;
- host plus random frozen experts;
- host plus selected raw experts;
- host plus distilled expert bank;
- each applicable condition with and without bounded host LoRA.

This separates donor knowledge from ordinary adaptation, added parameters,
routing, and optimization effects.

## Recommended prototype scale

- Host: Qwen3.5-4B.
- Candidate donor experts: 12-16, initially used as teachers.
- Student bank: 8 compact host-native experts.
- Routing: top-2 plus a null route.
- Student bottleneck: 256-512.
- Host adaptation: approximately 5-20M LoRA parameters.
- Initial data: 2-10M verified coding/control tokens.
- Training hardware: 24-48GB VRAM depending on context, optimizer,
  checkpointing, and quantization; validate with a B1/B2/B4 preflight.
- Evaluation: paired MBPP+/HumanEval+, a harder repository or program-repair
  benchmark, and matched general controls.

## Decision criteria

Do not move to a larger donor or GLM merely because v2 trains. Require:

- reproducible improvement on held-out coding tasks;
- more paired fixes than paired breaks;
- confidence intervals and paired tests consistent with a real effect;
- acceptable general-capability retention;
- improvement beyond equal-parameter and shuffled/random controls;
- stable sparse-router usage without collapse;
- acceptable latency, memory, and deployment cost;
- replication on untouched data.

If these conditions fail, preserve the result as a null or feasibility finding.
Do not relabel observational experts as transferred coding capability.

## Bottom line

The v1 experiment was useful: it validated extraction, cross-model expert
execution, deterministic bridge training, reversible residual injection, and a
possible reasoning-termination effect. It also exposed why four frozen experts
plus vector alignment are unlikely to transfer broad capability by themselves.

The best next step is neither simply more v1 nor an immediate 260-expert
monolith. It is controlled routed expert-bank distillation: stronger discovery,
composed donor targets, compact host-native students, a learned sparse router,
limited host adaptation, capability-aware training, and equal-budget controls.
