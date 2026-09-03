#!/usr/bin/env python3
"""Validate frozen dataset coverage independently of the acquisition process."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from reverse_reap.datasets import load_manifest


def validate(path: Path, require_full: bool) -> list[str]:
    samples = load_manifest(path)
    errors = []
    domains = Counter(sample.domain for sample in samples)
    coding_strata = {sample.stratum for sample in samples if sample.domain == "coding"}
    control_strata = {sample.stratum for sample in samples if sample.domain == "control"}
    languages = {sample.language for sample in samples if sample.domain == "coding"}
    splits = Counter(sample.split for sample in samples)
    expected_coding = {
        "function-synthesis",
        "code-understanding",
        "repository-bug-repair",
    }
    if require_full and (domains["coding"] < 500 or domains["control"] < 500):
        errors.append(f"full tier requires 500 per domain, got {dict(domains)}")
    if not expected_coding <= coding_strata:
        errors.append(f"coding strata are incomplete: {sorted(coding_strata)}")
    if len(languages - {None}) < 2 or "python" not in languages:
        errors.append(f"at least two coding languages including Python are required: {languages}")
    if not {"matched-reasoning", "general-knowledge"} <= control_strata:
        errors.append(f"matched and general controls are required: {sorted(control_strata)}")
    if set(splits) != {"calibration", "selection", "validation", "replication"}:
        errors.append(f"all four frozen splits must be non-empty: {dict(splits)}")
    unit_test_rows = [sample for sample in samples if sample.scorer == "unit_tests"]
    if any(not sample.tests for sample in unit_test_rows):
        errors.append("one or more unit-test rows have no tests")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    errors = validate(args.manifest, args.full)
    for error in errors:
        print(error)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
