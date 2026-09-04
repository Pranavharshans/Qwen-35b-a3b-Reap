"""Compute the causal-pilot gates from scored condition files (CPU only).

Writes three reports into the output directory (never overwriting existing
report files), then exits:

* determinism-report.json — official Gate B on the scored baseline pair,
  reported BOTH unrestricted and restricted to scoreable rows. The frozen
  0.95 scoreable threshold is applied to the scoreable-restricted variant
  because swebench rows are structurally scoreable=False on every host
  without the SWEBench harness; the unrestricted variant is recorded for
  transparency. A failure exits 3 so the plan task can stop the run.
* gate-d-report.json — causal_gate_report (Gate D) over baseline, selected,
  the 20 layer-matched random controls, and the replication pair. Gate D's
  scientific outcome (pass or fail) is never a task failure.
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

    gate_b_unrestricted = compare_deterministic_evaluations(
        scored / "c0-baseline-a.jsonl", scored / "c0-baseline-b.jsonl"
    )
    gate_b_scoreable = compare_deterministic_evaluations(
        scored / "c0-baseline-a.jsonl",
        scored / "c0-baseline-b.jsonl",
        restrict_scoreable=True,
    )
    determinism_report = {
        "gate": "B",
        "unrestricted": gate_b_unrestricted,
        "scoreable_restricted": gate_b_scoreable,
        "passed": gate_b_scoreable["passed"],
        "population_note": (
            "swebench rows are structurally scoreable=False without the SWEBench "
            "harness; the frozen 0.95 threshold is evaluated on scoreable rows"
        ),
    }
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
        replication_baseline_path=scored / "c0-replication-baseline.jsonl",
        replication_selected_path=scored / "c2-replication-selected.jsonl",
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
