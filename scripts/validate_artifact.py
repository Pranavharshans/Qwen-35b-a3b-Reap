#!/usr/bin/env python3
"""Small deterministic validation commands used by the autonomous plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=["preflight", "probe", "dataset", "candidate", "causal", "extraction"],
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-gate-pass", action="store_true")
    args = parser.parse_args()
    if args.kind == "dataset":
        rows = [line for line in args.path.read_text(encoding="utf-8").splitlines() if line]
        return 0 if rows else 2
    value = read_json(args.path)
    passed = True
    if args.kind in {"preflight", "probe"}:
        passed = value.get("passed") is True
    elif args.kind == "candidate":
        passed = bool(value.get("experts")) and (
            not args.require_gate_pass or value.get("gate_passed")
        )
    elif args.kind == "causal":
        passed = value.get("label") in {
            "coding-critical-v0",
            "observational-candidates",
            "unreplicated-candidates",
        } and set(value.get("criteria", {})) == {
            "twice_random_median",
            "at_or_above_random_p95",
            "coding_specificity_2pp",
            "replication_direction",
            "no_broad_output_collapse",
        }
    elif args.kind == "extraction":
        passed = value.get("label") == "extracted" and all(
            tensor.get("verified") for tensor in value.get("tensors", [])
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
