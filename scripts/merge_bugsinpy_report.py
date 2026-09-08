"""Merge a localized BugsInPy harness report without changing the denominator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reverse_reap.localized_repair import merge_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_report(args.evaluation, args.report, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
