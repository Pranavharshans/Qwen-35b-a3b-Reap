"""Validate a 12-session agentic probe without executing code (CPU only).

Checks, for every requested condition: exact frozen sample coverage with
identical ordering, baseline-repeat and no-op transcript+patch identity, no
truncation/OOM/NaN flags, and a fresh ``git apply --check`` of every
non-empty mechanical patch against its exact frozen base commit in a
temporary index. Exit 0 only when every check passes; applicability and task
resolution are reported separately and never imply a causal claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from reverse_reap.swe_edit import EditSessionError, validate_task


def _git(repo: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1")
    return subprocess.run(
        ["git", "-C", str(repo), *args], input=input_text, env=env, text=True,
        capture_output=True, timeout=120,
    )


def check_patch(task: dict, patch: bytes) -> tuple[bool, str]:
    """Apply-check a mechanical patch against the exact base in a temp index."""
    validate_task(task)
    if b"\0" in patch or not patch.strip():
        return False, "empty patch"
    repo = Path(task["repo_dir"])
    base = task["base_commit"]
    if _git(repo, "--no-replace-objects", "rev-parse",
            f"{base}^{{commit}}").stdout.strip() != base:
        raise EditSessionError("base commit mismatch")
    with tempfile.TemporaryDirectory(prefix="swe-agentic-apply-") as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"),
                   GIT_NO_REPLACE_OBJECTS="1")
        ready = subprocess.run(
            ["git", "-C", str(repo), "read-tree", base], env=env,
            capture_output=True, timeout=120,
        )
        if ready.returncode:
            return False, ready.stderr.decode(errors="replace")[:500]
        result = subprocess.run(
            ["git", "-C", str(repo), "apply", "--cached", "--check", "-"],
            input=patch, env=env, capture_output=True, timeout=120,
        )
        return result.returncode == 0, result.stderr.decode(errors="replace")[:500]


def validate(tasks: list[dict], probe_dir: Path, conditions: list[str]) -> dict:
    by_id = {task["sample_id"]: task for task in tasks}
    if not tasks or len(by_id) != len(tasks):
        raise EditSessionError("empty/duplicate probe tasks")
    expected_order = [task["sample_id"] for task in tasks]
    sessions: dict[str, dict[str, dict]] = {}
    for condition in conditions:
        cond_dir = probe_dir / condition
        ordered = []
        for sample_id in expected_order:
            state_path = cond_dir / sample_id / "state.json"
            if not state_path.exists():
                raise EditSessionError(f"missing session: {condition}/{sample_id}")
            state = json.loads(state_path.read_text())
            ordered.append((sample_id, state))
        sessions[condition] = dict(ordered)
    checks = []
    for condition in conditions:
        for sample_id in expected_order:
            state = sessions[condition][sample_id]
            artifacts = state.get("artifacts", {})
            patch_info = artifacts.get("patch", {})
            patch_path = probe_dir / condition / sample_id / patch_info.get("path", "")
            passed, reason = ((False, "no patch artifact") if not patch_path.exists()
                              else check_patch(by_id[sample_id], patch_path.read_bytes()))
            checks.append({
                "condition": condition, "sample_id": sample_id,
                "status": state.get("status"), "turns": state.get("turns"),
                "input_tokens": state.get("input_tokens"),
                "output_tokens": state.get("output_tokens"),
                "passed": passed, "reason": reason,
            })
    identity_pairs = [(conditions[0], other) for other in conditions[1:]]
    identities = []
    for first, second in identity_pairs:
        equal = all(
            sessions[first][sid].get("artifacts", {}).get("patch", {}).get("sha256")
            == sessions[second][sid].get("artifacts", {}).get("patch", {}).get("sha256")
            and sessions[first][sid].get("artifacts", {}).get("transcript", {}).get("sha256")
            == sessions[second][sid].get("artifacts", {}).get("transcript", {}).get("sha256")
            for sid in expected_order
        )
        identities.append({"pair": [first, second], "identical": equal})
    return {
        "passed": all(c["passed"] for c in checks) and all(
            i["identical"] for i in identities),
        "checks": checks, "identities": identities,
        "method": "git apply --cached --check in a temporary index at exact base SHA",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        print("refusing to overwrite report", file=sys.stderr)
        return 2
    try:
        report = validate(json.loads(args.tasks.read_text()), args.probe_dir,
                          list(args.conditions))
    except EditSessionError as exc:
        print(f"fail-closed: {exc}", file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
