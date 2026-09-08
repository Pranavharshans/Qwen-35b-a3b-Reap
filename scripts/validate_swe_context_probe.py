"""Check source-v3 probe against each exact base Git tree without applying code."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from reverse_reap.swe_context import ContextError, _git, validate_task

CONDITIONS = ("c0-baseline-a", "c0-baseline-b", "c0-noop-masked", "c2-selected")


def check_patch(task: dict, patch: str) -> tuple[bool, str]:
    validate_task(task)
    if "```" in patch or not patch.lstrip().startswith(("diff --git ", "--- ")):
        return False, "response is not a raw diff"
    repo = Path(task["repo_dir"])
    base = task["base_commit"]
    if _git(repo, "rev-parse", f"{base}^{{commit}}").decode().strip() != base:
        raise ContextError("base commit mismatch")
    with tempfile.TemporaryDirectory(prefix="swe-apply-") as temp:
        # Alternate index checks committed content, never an untrusted working tree.
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"), GIT_NO_REPLACE_OBJECTS="1")
        subprocess.run(["git", "-C", str(repo), "read-tree", base], env=env,
                       check=True, capture_output=True, timeout=30)
        result = subprocess.run(
            ["git", "-C", str(repo), "apply", "--cached", "--check", "-"],
            input=patch.rstrip("\n") + "\n", env=env, text=True,
            capture_output=True, timeout=60,
        )
        return result.returncode == 0, result.stderr[:1000]


def validate(tasks: list[dict], probe: Path) -> dict:
    by_id = {t["sample_id"]: t for t in tasks}
    if not tasks or len(by_id) != len(tasks):
        raise ContextError("empty/duplicate probe tasks")
    loaded, checks = {}, []
    for cid in CONDITIONS:
        rows = [json.loads(line) for line in
                (probe / f"{cid}.sweprobe.jsonl").read_text().splitlines()]
        by_sample = {r["sample_id"]: r for r in rows}
        if len(by_sample) != len(rows) or set(by_sample) != set(by_id):
            raise ContextError("probe must have exact identical sample coverage in all four arms")
        loaded[cid] = by_sample
        for sid, row in by_sample.items():
            if row["condition_id"] != cid:
                raise ContextError("condition mismatch")
            passed, reason = check_patch(by_id[sid], row["response"])
            checks.append({"condition": cid, "sample_id": sid,
                           "base_commit": by_id[sid]["base_commit"],
                           "passed": passed, "reason": reason,
                           "generated_tokens": row.get("generated_tokens"),
                           "truncated": row.get("truncated")})
    deterministic = all(
        loaded[CONDITIONS[0]][sid]["response"] == loaded[cid][sid]["response"]
        for sid in by_id for cid in CONDITIONS[1:3]
    )
    return {"passed": deterministic and all(r["passed"] for r in checks),
            "deterministic": deterministic, "checks": checks,
            "method": "git apply --cached --check with temporary index at exact base SHA"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tasks", "probe-dir", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ContextError("refusing to overwrite report")
    report = validate(json.loads(args.tasks.read_text()), args.probe_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
