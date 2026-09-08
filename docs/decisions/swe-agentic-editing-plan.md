# SWE repository-editing scaffold implementation plan

Status: IMPLEMENTED — `SWE-EDIT-04` validation active

This bounded CPU-only task replaces one-shot handwritten unified diffs with a deterministic
repository-editing protocol. It does not change datasets, scorers, candidates, gates, or model
outputs, and it authorizes no paid inference.

| task_id | objective | input files and hashes | expected outputs | definition of done | validation command | estimated GPU hours | estimated storage | dependencies | failure behavior |
|---|---|---|---|---|---|---:|---:|---|---|
| SWE-EDIT-01 | Freeze the protocol, policy, and threat model | `AGENTS.md`, `prd.md`, `roadmap.md`, `README.md`; current Git SHA | policy JSON, design/runbook | Protocol is bounded, deterministic, exact-base-only, and explicitly excludes answer-bearing data | `uv run ruff check src tests scripts` | 0 | <1 MB | none | Stop before implementation on a governing conflict |
| SWE-EDIT-02 | Implement isolated repository tools and exact-match edits | SWE-EDIT-01 outputs; six-field task projection | `src/reverse_reap/swe_edit.py`, CLI driver | Safe list/search/read/edit/finish works against regular blobs from the exact commit; final patch is emitted by Git | `uv run pytest tests/test_swe_edit.py -q` | 0 | <100 MB/session | SWE-EDIT-01 | Reject unsafe paths, symlinks, ambiguity, budget drift, or checkpoint drift |
| SWE-EDIT-03 | Prove security, determinism, and resumption on synthetic repositories | implementation and policy hashes | integration/security tests | Replays are deterministic; hashes bind task/config/model; no source checkout, hook, network, or answer-bearing input is required | `uv run pytest tests/test_swe_edit.py -q` | 0 | <100 MB | SWE-EDIT-02 | Fail closed and leave no scientific claim |
| SWE-EDIT-04 | Validate on a disposable CPU host and freeze handoff | committed SHA | validation report and proposed GPU-probe recipe | Focused/full hardware-free tests and lint pass at the exact commit | commands in `docs/swe-agentic-editing.md` | 0 | <1 GB | SWE-EDIT-03 | No GPU; report feasibility limitation and stop |

The active step is `SWE-EDIT-04`; `SWE-EDIT-01` through `SWE-EDIT-03` are complete. Only the lead
execution agent writes code or run state. A
separate VM worker may validate an immutable committed revision and return evidence, but may not
change configuration, code, or conclusions.
