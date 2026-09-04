"""Pinned boundary between generated patches and the official SWE-bench harness."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class SwebenchError(RuntimeError):
    """Raised when SWE-bench evidence is missing, ambiguous, or incomplete."""


SWE_BENCH_REPOSITORY = "https://github.com/SWE-bench/SWE-bench.git"
SWE_BENCH_REVISION = "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e"
SWE_BENCH_DATASET = "princeton-nlp/SWE-bench_Lite"
SWE_BENCH_SPLIT = "test"
# At the pinned harness revision the hosted princeton-nlp/SWE-bench_Lite rows
# lack the image/eval_script columns make_test_spec requires (measured:
# KeyError 'image'), so the harness MUST be driven with a task repo
# (per-instance eval.sh/Dockerfile/gold.patch). The task repo is therefore
# part of the pinned evaluation contract, cloned at this revision.
SWE_BENCH_TASKS_REPOSITORY = "https://github.com/SWE-bench/swe-bench-tasks.git"
SWE_BENCH_TASKS_REVISION = "3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _patch(text: str) -> str:
    fenced = re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced[-1].strip() if fenced else text.strip()


def export_predictions(
    evaluation_path: Path, destination: Path, *, model_name: str
) -> dict[str, Any]:
    if destination.exists():
        raise SwebenchError(f"refusing to overwrite predictions: {destination}")
    predictions = []
    for row in _rows(evaluation_path):
        if row.get("scorer") != "swebench":
            continue
        instance_id = str(row.get("source_id", ""))
        if not instance_id:
            raise SwebenchError("SWE-bench evaluation row is missing source_id")
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": _patch(str(row.get("response", ""))),
            }
        )
    if not predictions:
        raise SwebenchError("evaluation contains no SWE-bench predictions")
    if len({item["instance_id"] for item in predictions}) != len(predictions):
        raise SwebenchError("duplicate SWE-bench instance IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions),
        encoding="utf-8",
    )
    return {
        "valid": True,
        "predictions": len(predictions),
        "dataset": SWE_BENCH_DATASET,
        "split": SWE_BENCH_SPLIT,
        "harness_repository": SWE_BENCH_REPOSITORY,
        "harness_revision": SWE_BENCH_REVISION,
        "path": str(destination),
    }


def merge_report(
    evaluation_path: Path, report_path: Path, destination: Path
) -> dict[str, Any]:
    if destination.exists():
        raise SwebenchError(f"refusing to overwrite merged evaluation: {destination}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    completed = set(report.get("completed_ids", []))
    resolved = set(report.get("resolved_ids", []))
    errors = set(report.get("error_ids", []))
    if resolved - completed:
        raise SwebenchError("resolved IDs are not a subset of completed IDs")
    rows = _rows(evaluation_path)
    expected = {str(row["source_id"]) for row in rows if row.get("scorer") == "swebench"}
    if completed - expected or errors - expected:
        raise SwebenchError("harness report contains unrequested instance IDs")
    merged = []
    for row in rows:
        if row.get("scorer") != "swebench":
            merged.append(row)
            continue
        instance_id = str(row["source_id"])
        if instance_id in completed:
            row = {
                **row,
                "scoreable": True,
                "passed": instance_id in resolved,
                "swebench_completed": True,
                "error": None,
            }
        else:
            row = {
                **row,
                "scoreable": False,
                "passed": False,
                "swebench_completed": False,
                "error": "official SWE-bench harness did not complete this instance",
            }
        merged.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    scoreable = sum(bool(row.get("scoreable")) for row in merged)
    return {
        "valid": True,
        "rows": len(merged),
        "swebench_expected": len(expected),
        "swebench_completed": len(completed),
        "swebench_errors": len(errors),
        "scoreable_fraction": scoreable / len(merged) if merged else 0.0,
        "passed_gate_b_scoreability": bool(merged) and scoreable / len(merged) >= 0.95,
        "path": str(destination),
    }
