"""Run localized BugsInPy patches in digest-pinned, networkless task images."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from reverse_reap.localized_repair import (
    LocalizedRepairError,
    TaskImage,
    load_tasks,
    merge_report,
)

PATCH_APPLIED = "__REVERSE_REAP_PATCH_APPLIED__"


def run_task(image: TaskImage, patch: str, *, timeout_seconds: int = 900) -> dict:
    """Apply one patch and run its pre-pinned test command in a disposable container."""
    with tempfile.TemporaryDirectory(prefix="bugsinpy-harness-") as directory:
        patch_path = Path(directory) / "patch.diff"
        patch_path.write_text(patch, encoding="utf-8")
        script = (
            'cp -a "$1"/. /tmp/repo; cd /tmp/repo; '
            "git apply --check /input/patch.diff; git apply /input/patch.diff; "
            f"echo {PATCH_APPLIED}; shift; exec \"$@\""
        )
        command = [
            "docker", "run", "--rm", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=256",
            "--memory=4096m", "--cpus=2", "--tmpfs=/tmp:rw,exec,nosuid,size=2g",
            "--mount", f"type=bind,src={patch_path},dst=/input/patch.diff,readonly",
            image.image, "sh", "-eu", "-c", script, "sh", image.repository_path,
            *image.test_command,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired as error:
            return {"completed": False, "resolved": False, "timed_out": True,
                    "stdout": (error.stdout or "")[-4000:],
                    "stderr": (error.stderr or "")[-4000:]}
    applied = PATCH_APPLIED in result.stdout
    return {"completed": applied, "resolved": applied and result.returncode == 0,
            "timed_out": False, "return_code": result.returncode,
            "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--prescored-dir", type=Path, required=True)
    parser.add_argument("--patches-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tasks = {task.task_id: task for task in load_tasks(args.tasks)}
    image_rows = json.loads(args.images.read_text(encoding="utf-8"))
    images = {row["task_id"]: TaskImage.model_validate(row) for row in image_rows}
    if set(images) != set(tasks) or len(images) != len(image_rows):
        raise LocalizedRepairError("task-image manifest must exactly cover approved tasks")
    for image in images.values():
        image.validate_boundary()
        if image.test_command != tasks[image.task_id].failing_test_command:
            raise LocalizedRepairError(
                f"{image.task_id}: task-image test command differs from frozen task"
            )
    spec = json.loads(args.conditions.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition in spec["conditions"]:
        condition_id = condition["condition_id"]
        evaluation = args.prescored_dir / f"{condition_id}.preswebench.jsonl"
        patches_path = args.patches_dir / f"{condition_id}.bugsinpy-patches.jsonl"
        destination = args.output_dir / f"{condition_id}.jsonl"
        if destination.exists():
            raise LocalizedRepairError(f"refusing to overwrite {destination}")
        patches = [json.loads(line) for line in patches_path.read_text().splitlines()
                   if line.strip()]
        by_id = {str(row["task_id"]): row for row in patches}
        if set(by_id) != set(tasks) or len(by_id) != len(patches):
            raise LocalizedRepairError(f"{condition_id}: patch coverage mismatch")
        completed, resolved, errors, details = [], [], [], []
        for task_id in sorted(tasks):
            result = run_task(images[task_id], str(by_id[task_id]["patch"]))
            details.append({"task_id": task_id, **result})
            if result["completed"]:
                completed.append(task_id)
                if result["resolved"]:
                    resolved.append(task_id)
            else:
                errors.append(task_id)
        report = {"completed_ids": completed, "resolved_ids": resolved,
                  "error_ids": errors, "details": details}
        report_path = args.output_dir / f"{condition_id}.bugsinpy-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        merge_report(evaluation, report_path, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
