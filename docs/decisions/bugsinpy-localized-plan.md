# Localized BugsInPy replacement plan

Status: IMPLEMENTATION ACTIVE

This is a new validation experiment, not a repair or relabelling of the failed
SWE-bench runs. It replaces exactly eight validation SWE-bench rows with eight
human-approved, single-file BugsInPy tasks. The task condition is labelled
`oracle-localized repository repair`: the target filename may be derived from
benchmark metadata, but the gold patch and fixed tree are never model inputs.

| task_id | objective | inputs and hashes | expected outputs | definition of done | validation | GPU hours | storage | dependencies | failure behavior |
|---|---|---|---|---|---|---:|---:|---|---|
| BIP-01 | Freeze task and image contracts | approved task projection, BugsInPy revision, base commits | schemas and provenance | strict allowlists reject answer-bearing fields and unpinned images | focused tests | 0 | <1 MB | none | stop on trust-boundary drift |
| BIP-02 | Build replacement manifest | frozen pilot manifest plus eight approved tasks | new immutable manifest | exactly eight validation SWE rows replaced; all other bytes unchanged | manifest validator | 0 | <5 MB | BIP-01 | refuse overwrite/membership drift |
| BIP-03 | Convert full-file responses to patches | generation rows plus exact base Git objects | applicable patches | model output replaces one approved file; Git creates patch mechanically | temporary-index `git apply --check` | 0 | <100 MB | BIP-02 | mark empty/unsafe output unscoreable |
| BIP-04 | Score in pinned task images | digest-pinned images and commands | completed/resolved/error report | no-network constrained containers execute the unchanged task tests | synthetic mocked harness tests | 0 | <5 GB/task image | BIP-03 | retain errors in denominator |
| BIP-05 | Prepare bounded exact-checkpoint probe | committed implementation and frozen artifacts | probe plan only | costs, prompts, images and coverage preflighted | configuration tests | 0 | TBD | BIP-04 | wait for paid-resource approval |

Only implementation and hardware-free validation are authorized now. No task
selection, dataset freeze, image build, GPU generation, causal scoring,
replication, or extraction is launched automatically.
