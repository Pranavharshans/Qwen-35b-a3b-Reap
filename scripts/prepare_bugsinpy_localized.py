"""Replace exactly eight validation SWE rows with approved localized BugsInPy tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reverse_reap.localized_repair import (
    BUGSINPY_REVISION,
    LocalizedRepairError,
    atomic_write,
    load_tasks,
    sha256,
    task_to_sample,
)


def prepare(base: Path, tasks_path: Path, approved_sha256: str, output: Path,
            provenance: Path) -> dict:
    if output.exists() or provenance.exists():
        raise LocalizedRepairError("refusing to overwrite manifest or provenance")
    if sha256(tasks_path.read_bytes()) != approved_sha256:
        raise LocalizedRepairError("task projection does not match approved SHA-256")
    tasks = load_tasks(tasks_path)
    if len(tasks) != 8:
        raise LocalizedRepairError("replacement requires exactly eight approved tasks")
    replacements = [task_to_sample(task) for task in tasks]
    old_lines = base.read_bytes().splitlines(keepends=True)
    targets = []
    for index, line in enumerate(old_lines):
        row = json.loads(line)
        if row.get("split") == "validation" and row.get("scorer") == "swebench":
            targets.append(index)
    if len(targets) != 8:
        raise LocalizedRepairError(
            f"expected exactly eight validation SWE rows, got {len(targets)}"
        )
    if len({sample.sample_id for sample in replacements}) != 8:
        raise LocalizedRepairError("replacement sample IDs are not unique")
    output_lines = list(old_lines)
    for index, sample in zip(targets, replacements, strict=True):
        output_lines[index] = canonical(sample.model_dump(mode="json"))
    atomic_write(output, b"".join(output_lines))
    unchanged = all(
        output_lines[index] == old_lines[index]
        for index in range(len(old_lines))
        if index not in targets
    )
    report = {
        "schema_version": 1,
        "label": "oracle-localized repository repair",
        "source_repository": "https://github.com/soarsmu/BugsInPy.git",
        "source_revision": BUGSINPY_REVISION,
        "base_manifest_sha256": sha256(base.read_bytes()),
        "task_projection_sha256": approved_sha256,
        "replacement_manifest_sha256": sha256(output.read_bytes()),
        "replaced_rows": 8,
        "unaffected_bytes_identical": unchanged,
        "tasks": [{"task_id": task.task_id, "buggy_commit": task.buggy_commit,
                   "target_path": task.target_path} for task in tasks],
    }
    atomic_write(
        provenance, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    )
    return report


def canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode() + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--approved-tasks-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base_manifest, args.tasks, args.approved_tasks_sha256,
            args.output, args.provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
