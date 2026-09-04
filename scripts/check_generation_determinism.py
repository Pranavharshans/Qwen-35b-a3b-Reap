"""Response-determinism pre-gate between GPU phases.

Compares the two identical baseline conditions' raw generation files (no
scoring required). On any response mismatch the report is still written and
the process exits 3 so the plan task stops the run before further GPU spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reverse_reap.causal import compare_generation_determinism


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = compare_generation_determinism(args.first, args.second)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite report: {args.report}")
    args.report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
