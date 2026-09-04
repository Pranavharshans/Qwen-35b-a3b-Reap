"""Response-determinism pre-gate between GPU phases.

Checks that one or more pairs of raw generation files are response-identical
(no scoring required). Each pair is ``NAME=FIRST_PATH:SECOND_PATH``; repeat
``--pair`` for every required equivalence, e.g. the deterministic baseline
pair and the no-op intervention equivalence pair. On any mismatch the report
is still written and the process exits 3 so the plan task stops the run
before further GPU spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reverse_reap.causal import compare_generation_determinism


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="NAME=FIRST.jsonl:SECOND.jsonl (repeatable)",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.report.exists():
        raise SystemExit(f"refusing to overwrite report: {args.report}")
    checks = {}
    for spec in args.pair:
        name, _, paths = spec.partition("=")
        first, sep, second = paths.partition(":")
        if not name or not sep:
            raise SystemExit(f"malformed --pair {spec!r}; expected NAME=FIRST:SECOND")
        checks[name] = compare_generation_determinism(Path(first), Path(second))
    report = {
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
