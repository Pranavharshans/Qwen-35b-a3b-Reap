"""Build and verify the hash manifest for causal generation response artifacts.

The causal run is split into a GPU generation phase and a CPU scoring phase on
separate hosts (user directive 2026-09-04). Only hashed manifests and response
artifacts cross the host boundary, so the generation phase ends by freezing a
bundle that pins the SHA-256 of every condition's response file, and the
scoring phase begins by re-verifying every pin. Fail-closed throughout.

Build (generation host, after all 26 conditions are generated):
    python scripts/generation_bundle.py build \
        --conditions configs/causal-pilot-conditions.json \
        --generations-dir runs/causal-pilot/${RUN_ID}/generations \
        --dataset-manifest datasets/manifests/pilot-lengthmatched.jsonl \
        --output runs/causal-pilot/${RUN_ID}/generation-bundle.json

Verify (scoring host, before any scoring work):
    python scripts/generation_bundle.py verify \
        --bundle runs/causal-pilot/${RUN_ID}/generation-bundle.json \
        --generations-dir runs/causal-pilot/${RUN_ID}/generations \
        --report runs/causal-pilot/${RUN_ID}/verify-generation-inputs.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


class BundleError(RuntimeError):
    """Raised when generation artifacts fail a structural or hash check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _validation_sample_ids(manifest_path: Path) -> list[str]:
    """Validation-split sample IDs in manifest order (the generation order)."""
    ids = [
        row["sample_id"] for row in _load_jsonl(manifest_path) if row.get("split") == "validation"
    ]
    if not ids:
        raise BundleError(f"manifest {manifest_path} has no validation-split samples")
    return ids


def _validate_condition_file(
    path: Path,
    condition: dict,
    expected_ids: list[str],
    condition_id: str,
) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise BundleError(f"missing or empty generation file for {condition_id}: {path}")
    rows = _load_jsonl(path)
    if len(rows) != len(expected_ids):
        raise BundleError(f"{condition_id}: expected {len(expected_ids)} rows, found {len(rows)}")
    for row, expected_id in zip(rows, expected_ids, strict=True):
        if row.get("sample_id") != expected_id:
            raise BundleError(
                f"{condition_id}: sample_id order drift at position "
                f"{expected_ids.index(expected_id)}: {row.get('sample_id')!r} != {expected_id!r}"
            )
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise BundleError(f"{condition_id}/{expected_id}: empty or missing response")
        if row.get("condition_id") != condition_id:
            raise BundleError(
                f"{condition_id}/{expected_id}: row condition_id is {row.get('condition_id')!r}"
            )
        if row.get("split") != "validation":
            raise BundleError(f"{condition_id}/{expected_id}: row split is {row.get('split')!r}")
        expected_masked = 0 if condition.get("expert_manifest") is None else 1
        masked = row.get("masked_experts")
        if not isinstance(masked, int) or (masked > 0) != (expected_masked > 0):
            raise BundleError(
                f"{condition_id}/{expected_id}: masked_experts {masked!r} contradicts "
                f"the frozen condition spec (expert_manifest="
                f"{condition.get('expert_manifest')!r})"
            )
    return len(rows)


def build_bundle(
    conditions_path: Path,
    generations_dir: Path,
    manifest_path: Path,
    output_path: Path,
    run_id: str,
) -> dict:
    if output_path.exists():
        raise BundleError(f"refusing to overwrite existing bundle: {output_path}")
    spec = json.loads(conditions_path.read_text(encoding="utf-8"))
    conditions = spec.get("conditions") or []
    if not conditions:
        raise BundleError(f"conditions spec {conditions_path} has no conditions")
    condition_ids = [c["condition_id"] for c in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise BundleError("duplicate condition_id in conditions spec")
    declared = (spec.get("splits") or {}).get("validation", {}).get("samples")
    expected_ids = _validation_sample_ids(manifest_path)
    if declared is not None and len(expected_ids) != declared:
        raise BundleError(
            f"manifest validation split has {len(expected_ids)} samples but the "
            f"spec pins {declared}"
        )
    files = {}
    for condition in conditions:
        condition_id = condition["condition_id"]
        path = generations_dir / f"{condition_id}.jsonl"
        rows = _validate_condition_file(path, condition, expected_ids, condition_id)
        files[condition_id] = {"path": str(path), "sha256": _sha256(path), "rows": rows}
    bundle = {
        "schema_version": 1,
        "run_id": run_id,
        "conditions_spec_sha256": _sha256(conditions_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "files": files,
        "total_rows": sum(entry["rows"] for entry in files.values()),
    }
    _atomic_write_json(output_path, bundle)
    return bundle


def verify_bundle(bundle_path: Path, generations_dir: Path, report_path: Path) -> dict:
    if report_path.exists():
        raise BundleError(f"refusing to overwrite existing report: {report_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    mismatches = []
    verified = 0
    for condition_id, entry in sorted(bundle.get("files", {}).items()):
        path = generations_dir / f"{condition_id}.jsonl"
        if not path.is_file():
            mismatches.append({"condition_id": condition_id, "reason": "missing file"})
            continue
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            mismatches.append(
                {
                    "condition_id": condition_id,
                    "reason": "sha256 mismatch",
                    "expected": entry.get("sha256"),
                    "actual": digest,
                }
            )
            continue
        rows = _load_jsonl(path)
        if rows and len(rows) != entry.get("rows"):
            mismatches.append(
                {
                    "condition_id": condition_id,
                    "reason": "row count mismatch",
                    "expected": entry.get("rows"),
                    "actual": len(rows),
                }
            )
            continue
        verified += 1
    report = {
        "bundle": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "verified_files": verified,
        "expected_files": len(bundle.get("files", {})),
        "mismatches": mismatches,
        "passed": not mismatches and verified == len(bundle.get("files", {})) and verified > 0,
    }
    _atomic_write_json(report_path, report)
    if mismatches or not report["passed"]:
        raise BundleError(
            f"generation bundle verification failed: {len(mismatches)} mismatch(es); "
            f"report at {report_path}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    build = subparsers.add_parser("build", help="hash and validate all generation files")
    build.add_argument("--conditions", type=Path, required=True)
    build.add_argument("--generations-dir", type=Path, required=True)
    build.add_argument("--dataset-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--run-id", default=os.environ.get("RUN_ID", "unscoped"))
    verify = subparsers.add_parser("verify", help="re-verify a bundle against copied files")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--generations-dir", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "build":
            bundle = build_bundle(
                args.conditions,
                args.generations_dir,
                args.dataset_manifest,
                args.output,
                run_id=args.run_id,
            )
            print(
                f"generation bundle written: {args.output} "
                f"({len(bundle['files'])} files, {bundle['total_rows']} rows)"
            )
        else:
            report = verify_bundle(args.bundle, args.generations_dir, args.report)
            print(
                f"generation bundle verified: {report['verified_files']}/"
                f"{report['expected_files']} files match"
            )
    except BundleError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
