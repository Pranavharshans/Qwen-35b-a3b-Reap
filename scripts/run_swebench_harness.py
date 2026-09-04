"""Score SWE-bench rows via the PINNED official harness — never silently exclude.

For every condition with swebench rows in its pre-scored file:

1. ``export_predictions`` writes the official prediction contract.
2. The official harness at the pinned revision evaluates the predictions in
   its own containers (``--dataset_name princeton-nlp/SWE-bench_Lite --split
   test``). Instance images are pulled once and cached by Docker.
3. ``merge_report`` merges completed/resolved verdicts into the final scored
   file, marking any instance the harness did not complete as scoreable=False
   with an explicit error — Gate B's frozen 0.95 denominator then judges the
   honest coverage.

Conditions without swebench rows are passed through unchanged. All outputs
are fail-closed (never overwritten).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from reverse_reap.swebench import (
    SWE_BENCH_DATASET,
    SWE_BENCH_REVISION,
    SWE_BENCH_SPLIT,
    SwebenchError,
    export_predictions,
    merge_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--prescored-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--harness-repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    spec = json.loads(args.conditions.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise SystemExit(f"unsupported conditions spec schema: {spec.get('schema_version')}")
    harness_python = args.harness_repo / "venv" / "bin" / "python"
    if not harness_python.exists():
        raise SystemExit(f"harness venv missing: {harness_python}")
    revision = subprocess.run(
        ["git", "-C", str(args.harness_repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != SWE_BENCH_REVISION:
        raise SystemExit(f"harness revision {revision!r} != pinned {SWE_BENCH_REVISION!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for condition in spec["conditions"]:
        condition_id = condition["condition_id"]
        prescored = args.prescored_dir / f"{condition_id}.preswebench.jsonl"
        final = args.output_dir / f"{condition_id}.jsonl"
        if final.exists():
            print(f"[skip] {condition_id}: final scored file already exists", flush=True)
            continue
        if not prescored.exists():
            raise SystemExit(f"pre-scored file missing: {prescored}")
        rows = [
            json.loads(line)
            for line in prescored.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not any(row.get("scorer") == "swebench" for row in rows):
            shutil.copyfile(prescored, final)
            print(f"[passthrough] {condition_id}: no swebench rows", flush=True)
            continue

        condition_work = args.work_dir / condition_id
        predictions_dir = condition_work / "predictions"
        reports_dir = condition_work / "reports"
        logs_dir = condition_work / "logs"
        for path in (predictions_dir, reports_dir, logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        predictions = predictions_dir / "predictions.jsonl"
        if predictions.exists():
            predictions.unlink()  # derived staging file, regenerated each attempt
        export_predictions(prescored, predictions, model_name=f"reverse-reap/{condition_id}")

        started = time.monotonic()
        result = subprocess.run(
            [
                str(harness_python), "-m", "swebench.harness.run_evaluation",
                "--predictions_path", str(predictions),
                "--dataset_name", SWE_BENCH_DATASET,
                "--split", SWE_BENCH_SPLIT,
                "--run_id", condition_id,
                "--report_dir", str(reports_dir),
                "--log_dir", str(logs_dir),
                "--max_workers", str(args.max_workers),
            ],
            capture_output=True, text=True,
        )
        elapsed = time.monotonic() - started
        harness_log = condition_work / "harness-stderr.log"
        harness_log.write_text(result.stderr[-100_000:], encoding="utf-8")
        if result.returncode != 0:
            raise SystemExit(
                f"official harness failed for {condition_id} (rc={result.returncode}, "
                f"{elapsed:.0f}s); stderr in {harness_log}"
            )

        reports = [path for path in reports_dir.rglob("*.json") if path.is_file()]
        if len(reports) != 1:
            raise SystemExit(f"expected exactly one harness report for {condition_id}, "
                             f"found {len(reports)}")
        summary = merge_report(prescored, reports[0], final)
        summary["harness_seconds"] = elapsed
        summary["harness_revision"] = revision
        (args.output_dir / f"{condition_id}.summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[swebench] {condition_id}: completed={summary['swebench_completed']}/"
            f"{summary['swebench_expected']} errors={summary['swebench_errors']} "
            f"scoreable_fraction={summary['scoreable_fraction']:.3f} ({elapsed:.0f}s)",
            flush=True,
        )
    print("swebench harness scoring complete", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SwebenchError as error:
        print(f"SwebenchError: {error}", file=sys.stderr)
        sys.exit(2)
