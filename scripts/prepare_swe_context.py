"""Freeze source-v3 validation prompts from an allowlisted task projection (CPU only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reverse_reap.datasets import canonical_json
from reverse_reap.swe_context import VERSION, ContextError, build_context, digest, render_prompt


def prepare(manifest: Path, tasks_file: Path, output: Path, provenance: Path,
            tasks_sha256: str) -> None:
    if output.exists() or provenance.exists():
        raise ContextError("refusing to overwrite a manifest or provenance")
    if digest(tasks_file.read_bytes()) != tasks_sha256:
        raise ContextError("task projection differs from independently approved SHA-256")
    tasks = json.loads(tasks_file.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ContextError("expected a nonempty list of allowlisted task projections")
    by_id = {t["sample_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ContextError("duplicate task identities")
    original = manifest.read_bytes().splitlines(keepends=True)
    lines, records, found = [], [], set()
    for line in original:
        row = json.loads(line)
        sid = row["sample_id"]
        if sid not in by_id:
            lines.append(line)  # Preserve every unaffected byte, including non-SWE serialization.
            continue
        if row["scorer"] != "swebench" or row["split"] != "validation":
            raise ContextError("context changes are restricted to validation SWE rows")
        if sid in found or by_id[sid]["source_id"] != row["source_id"]:
            raise ContextError("duplicate/mismatched sample identity")
        task = by_id[sid]
        context = build_context(task)
        row["prompt"] = render_prompt(task, context)
        row["prompt_template_version"] = VERSION
        row["content_sha256"] = digest(canonical_json({
            "prompt": row["prompt"], "reference": row.get("reference"), "tests": row.get("tests")
        }))
        context["prompt_sha256"] = digest(row["prompt"].encode())
        records.append(context)
        lines.append((json.dumps(row, sort_keys=True) + "\n").encode())
        found.add(sid)
    if found != set(by_id):
        raise ContextError("task projection contains IDs absent from manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"".join(lines))
    provenance.write_text(json.dumps({
        "version": VERSION, "source_manifest_sha256": digest(manifest.read_bytes()),
        "manifest_sha256": digest(output.read_bytes()),
        "task_projection_sha256": digest(tasks_file.read_bytes()), "contexts": records,
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "tasks", "output", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tasks-sha256", required=True,
                        help="Independently approved projection hash from frozen task metadata")
    args = parser.parse_args()
    prepare(args.manifest, args.tasks, args.output, args.provenance, args.tasks_sha256)


if __name__ == "__main__":
    main()
