"""Compute the causal-pilot validation gates from scored condition files (CPU only).

Writes three reports into the output directory (never overwriting existing
report files), then exits:

* determinism-report.json — official Gate B on the scored baseline pair with
  its FROZEN semantics: all rows of the condition files are the denominator
  (no scoreable restriction), requiring scoreable_fraction >= 0.95 AND zero
  response/score mismatches. SWE-bench rows reach full coverage through the
  pinned official harness in scripts/run_swebench_harness.py, never through
  exclusion. A failure exits 3 so the plan task stops the run.
* gate-d-report.json — causal_gate_report (Gate D) over the validation-split
  baseline, selected, and the 20 layer-matched random controls. Replication
  is NOT part of this stage (a separate governed plan after Gate D passes).
  Gate D's scientific outcome (pass or fail) is never a task failure.
* supplementary-controls.json — coding/control drops for the frequency-matched
  and lowest-differential specificity controls against the same baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reverse_reap.causal import (
    CausalError,
    _domain_drop,
    causal_gate_report,
    compare_deterministic_evaluations,
)


def write_if_absent(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite report: {path}")
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def scoreable_map(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["sample_id"]: row for row in rows if row.get("scoreable")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--causal-report",
        type=Path,
        default=None,
        help="additional copy of the Gate D payload consumed by build_run_bundle",
    )
    parser.add_argument("--random-count", type=int, default=20)
    args = parser.parse_args()
    scored = args.scored_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gate_b = compare_deterministic_evaluations(
        scored / "c0-baseline-a.jsonl", scored / "c0-baseline-b.jsonl"
    )
    determinism_report = {"gate": "B", "passed": gate_b["passed"], **gate_b}
    write_if_absent(args.output_dir / "determinism-report.json", determinism_report)

    random_paths = [
        scored / f"c3-layer-random-{index:03d}.jsonl" for index in range(args.random_count)
    ]
    missing = [str(path) for path in random_paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing scored random controls: {missing}")
    gate_d = causal_gate_report(
        scored / "c0-baseline-a.jsonl",
        scored / "c2-selected.jsonl",
        random_paths,
    )
    write_if_absent(args.output_dir / "gate-d-report.json", gate_d)
    if args.causal_report is not None:
        args.causal_report.parent.mkdir(parents=True, exist_ok=True)
        write_if_absent(args.causal_report, gate_d)

    baseline = scoreable_map(scored / "c0-baseline-a.jsonl")
    supplementary = {}
    for condition_id in ("c4-frequency-matched", "c5-lowest-differential"):
        intervention = scoreable_map(scored / f"{condition_id}.jsonl")
        supplementary[condition_id] = {
            "coding_drop": _domain_drop(baseline, intervention, "coding"),
            "control_drop": _domain_drop(baseline, intervention, "control"),
        }
    write_if_absent(args.output_dir / "supplementary-controls.json", supplementary)

    print(json.dumps({"gate_b_passed": determinism_report["passed"],
                      "gate_d_passed": gate_d["passed"],
                      "gate_d_label": gate_d["label"]}, sort_keys=True), flush=True)
    return 0 if determinism_report["passed"] else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CausalError as error:
        print(f"CausalError: {error}", file=sys.stderr)
        sys.exit(2)
