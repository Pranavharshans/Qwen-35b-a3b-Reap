"""Build the composite 1300-row generation bundle (1092 reused + 208 regen).

Inputs:
  --original-generations-dir  original run's generations/ (26 x 50, source-v1 SWE prose)
  --swefix-dir                new 208 outputs (26 x 8, source-v2 diffs, {cid}.swefix.jsonl)
  --dataset-manifest          NEW manifest (pilot-lengthmatched-swe-v2.jsonl, validation order)
  --conditions                frozen conditions spec (unchanged)
  --reuse-manifest            hash-verified reuse manifest (1092 entries)
  --output-dir                destination generations/ (26 x 50 composite)
  --bundle-output             generation-bundle.json for the NEW run_id
  --run-id                    new immutable run ID
  --provenance-output         provenance JSON (separate reused vs regenerated)

Rules (fail-closed):
  * every reused row must appear in the reuse manifest with matching
    prompt/output/score hashes; any mismatch aborts (never silently reuse);
  * composite per condition = 42 reused (non-SWE, byte-identical prompts) + 8
    regen (SWE, source-v2), sorted in NEW manifest validation order;
  * exactly 1300 unique (sample_id, condition_id) rows, no duplicates/missing;
  * separate provenance: reused rows list original_run_id; regen rows list
    new run_id + swefix source; never implies one generation execution;
  * bundle pins SHA-256 per file + total_rows 1300; verify-mode must pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-generations-dir", type=Path, required=True)
    parser.add_argument("--swefix-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if args.bundle_output.exists():
        print(f"refusing to overwrite bundle: {args.bundle_output}", file=sys.stderr)
        return 2
    if args.provenance_output.exists():
        print(f"refusing to overwrite provenance: {args.provenance_output}", file=sys.stderr)
        return 2

    spec = json.loads(args.conditions.read_text(encoding="utf-8"))
    manifest_rows = _load_jsonl(args.dataset_manifest)
    expected_ids = [r["sample_id"] for r in manifest_rows if r.get("split") == "validation"]
    if len(expected_ids) != 50:
        print(f"manifest validation split has {len(expected_ids)} rows, expected 50", file=sys.stderr)
        return 2
    manifest_by = {r["sample_id"]: r for r in manifest_rows if r.get("split") == "validation"}

    reuse = json.loads(args.reuse_manifest.read_text(encoding="utf-8"))
    reuse_index = {(e["condition_id"], e["sample_id"]): e for e in reuse["entries"]}
    if len(reuse_index) != 1092:
        print(f"reuse manifest has {len(reuse_index)} entries, expected 1092", file=sys.stderr)
        return 2
    original_run_id = reuse["original_run_id"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict = {"new_run_id": args.run_id, "original_run_id": original_run_id,
                        "reused": [], "regenerated": []}
    seen: set[tuple[str, str]] = set()
    files: dict = {}

    for cond in spec["conditions"]:
        cid = cond["condition_id"]
        orig_path = args.original_generations_dir / f"{cid}.jsonl"
        fix_path = args.swefix_dir / f"{cid}.swefix.jsonl"
        if not orig_path.is_file():
            print(f"missing original file: {orig_path}", file=sys.stderr)
            return 2
        if not fix_path.is_file():
            print(f"missing swefix file: {fix_path}", file=sys.stderr)
            return 2
        orig_rows = {r["sample_id"]: r for r in _load_jsonl(orig_path)}
        fix_rows = {r["sample_id"]: r for r in _load_jsonl(fix_path)}
        if len(fix_rows) != 8 or any(v.get("scorer") != "swebench" for v in fix_rows.values()):
            print(f"{cid}: swefix must have exactly 8 swebench rows", file=sys.stderr)
            return 2

        composite: list[dict] = []
        for sid in expected_ids:
            m = manifest_by[sid]
            key = (cid, sid)
            if key in seen:
                print(f"duplicate row: {key}", file=sys.stderr)
                return 2
            seen.add(key)
            if m["scorer"] != "swebench":
                # Reused row: must be in reuse manifest with matching hashes.
                entry = reuse_index.get(key)
                if entry is None:
                    print(f"reused row missing from manifest (fail-closed): {key}", file=sys.stderr)
                    return 2
                g = orig_rows.get(sid)
                if g is None:
                    print(f"original row missing: {key}", file=sys.stderr)
                    return 2
                if _sha_text(g["response"]) != entry["output_sha256"]:
                    print(f"output hash mismatch (fail-closed): {key}", file=sys.stderr)
                    return 2
                if g.get("condition_id") != cid or g.get("split") != "validation":
                    print(f"metadata mismatch: {key}", file=sys.stderr)
                    return 2
                composite.append(g)
                provenance["reused"].append({"condition_id": cid, "sample_id": sid,
                                            "original_run_id": original_run_id})
            else:
                g = fix_rows.get(sid)
                if g is None:
                    print(f"regen row missing: {key}", file=sys.stderr)
                    return 2
                if g.get("condition_id") != cid or g.get("split") != "validation":
                    print(f"regen metadata mismatch: {key}", file=sys.stderr)
                    return 2
                if not g.get("response", "").strip():
                    print(f"empty regen response: {key}", file=sys.stderr)
                    return 2
                composite.append(g)
                provenance["regenerated"].append({"condition_id": cid, "sample_id": sid,
                                                 "run_id": args.run_id, "source": "swefix-v2"})

        dest = args.output_dir / f"{cid}.jsonl"
        if dest.exists():
            print(f"refusing to overwrite composite file: {dest}", file=sys.stderr)
            return 2
        dest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in composite),
                        encoding="utf-8")
        files[cid] = {"path": str(dest), "sha256": _sha_file(dest), "rows": len(composite)}

    if len(seen) != 1300:
        print(f"composite has {len(seen)} unique rows, expected 1300", file=sys.stderr)
        return 2
    if len(provenance["reused"]) != 1092 or len(provenance["regenerated"]) != 208:
        print("provenance counts wrong", file=sys.stderr)
        return 2

    bundle = {
        "schema_version": 1,
        "run_id": args.run_id,
        "conditions_spec_sha256": _sha_file(args.conditions),
        "dataset_manifest_sha256": _sha_file(args.dataset_manifest),
        "files": files,
        "total_rows": sum(v["rows"] for v in files.values()),
        "provenance": "composite: 1092 reused (original run {}) + 208 regenerated ({} source-v2); never one execution".format(
            original_run_id, args.run_id),
    }
    if bundle["total_rows"] != 1300:
        print(f"bundle total {bundle['total_rows']} != 1300", file=sys.stderr)
        return 2
    args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"total_rows": bundle["total_rows"], "reused": 1092, "regenerated": 208,
                      "bundle": str(args.bundle_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
