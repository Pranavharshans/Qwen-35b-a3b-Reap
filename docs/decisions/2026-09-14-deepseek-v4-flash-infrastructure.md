# Decision: DeepSeek V4 Flash launch infrastructure

Date: 2026-09-14
Status: accepted for the infrastructure slice
Authorization: fresh human authorization for this repository change

## Decision

Add an isolated, infrastructure-only launcher for the exact checkpoint
`deepseek-ai/DeepSeek-V4-Flash-0731` at revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. The launcher exposes two explicit
transport modes for one canonical vLLM workload:

- `direct` for an already-provisioned local or cloud GPU environment;
- `fau_slurm` for one FAU Slurm node and one task, with explicit `--submit`.

Preview and hardware-free checks are the default. State-changing execution and
submission remain opt-in, fail closed on placeholders or unsafe overrides, and
must receive fresh operator authorization at the time of the action.

## Scope boundary

This decision does not amend the Qwen v0 donor, scientific objective, schemas,
datasets, experiment state, attribution gates, or extraction contract in
`AGENTS.md`, `prd.md`, or `roadmap.md`. It does not begin a DeepSeek experiment,
GPU run, model download, cloud provisioning, FAU scheduler submission, or model
publication. Hardware sizing, runtime compatibility, and throughput remain
unverified until a separately authorized bounded preflight.

“Push to Hub” for this change means pushing repository source, configuration,
tests, and documentation to the configured GitHub `origin`; it never means
uploading model weights, datasets, credentials, or extracted tensors to the
Hugging Face Hub.

## Future approval boundaries

Before `--execute`, an operator must replace private paths and the example run
ID, verify the target environment, inspect the canonical argv, and authorize
the direct process. Before `--submit`, an operator must additionally confirm
FAU account/partition/constraint values, cluster modules and activation, the
pre-existing Slurm log parent directories, the generated script, and a fresh
submission authorization. Any change to model ID, revision, precision,
transport semantics, or distributed orchestration requires a new reviewed
decision and implementation slice.

See [the DeepSeek V4 Flash infrastructure guide](../deepseek-v4-flash-infrastructure.md)
for the operational contract and exact preview/check commands.
