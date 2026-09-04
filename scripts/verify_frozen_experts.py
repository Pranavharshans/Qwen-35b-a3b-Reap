"""Verify every frozen expert manifest referenced by the conditions spec.

Hash-matches each condition's expert manifest on disk against the SHA-256
recorded in the spec when the candidate analysis froze it. Any mismatch or
missing file is fatal (exit 1) — the causal run must consume exactly the
frozen top-4 candidates and frozen control sets, byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.conditions.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise SystemExit(f"unsupported conditions spec schema: {spec.get('schema_version')}")

    failures = []
    seen = set()
    for condition in spec["conditions"]:
        reference = condition["expert_manifest"]
        if reference is None:
            continue
        expected = condition["expert_manifest_sha256"]
        if (reference, expected) in seen:
            continue
        seen.add((reference, expected))
        path = Path(reference)
        if not path.exists():
            failures.append(f"{reference}: missing")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"{reference}: expected {expected}, found {actual}")
    if failures:
        for failure in failures:
            print(f"FROZEN-INPUT MISMATCH {failure}", file=sys.stderr)
        return 1
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite report: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"verified_manifests": len(seen), "mismatches": []}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"frozen inputs verified: {len(seen)} expert manifests hash-match the spec", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
