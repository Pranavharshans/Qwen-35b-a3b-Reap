"""Convert localized full-file generation rows into mechanically applicable patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reverse_reap.localized_repair import load_tasks, response_to_patch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    tasks = {task.task_id: task for task in load_tasks(args.tasks)}
    patches = []
    for line in args.evaluation.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("scorer") != "bugsinpy":
            continue
        task_id = str(row["source_id"])
        if task_id not in tasks:
            raise SystemExit(f"unmanifested BugsInPy task: {task_id}")
        patches.append({**response_to_patch(tasks[task_id], str(row.get("response", ""))),
                        "sample_id": row["sample_id"],
                        "condition_id": row["condition_id"]})
    if len(patches) != len(tasks) or len({item["task_id"] for item in patches}) != len(patches):
        raise SystemExit("patch coverage must equal the exact task projection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in patches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
