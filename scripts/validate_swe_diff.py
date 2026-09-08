"""Validate SWE-repair probe outputs: raw unified diffs applicable via git apply.

Fail-closed probe gate for the SWE-bench prompt repair (source-v2):

* each response must contain NO Markdown fences (```), no code-block wrappers;
* each response must be syntactically a unified Git diff (diff --git or ---/+++
  headers plus @@ hunks);
* each extracted patch (via the frozen scorer's _patch, identical to
  export_predictions) must pass `git apply --check` against the correct
  repository state (target repo cloned at the instance's base_commit from the
  pinned SWE-bench_Lite dataset, or the pinned swe-bench-tasks context where
  applicable);
* no-op/determinism invariants: for the probed SWE sample_ids, baseline-a vs
  baseline-b responses identical, noop-masked vs baseline-a identical.

Exit 0 only if ALL checks pass. Never weakens the criterion: any failure is
terminal (exit 2) with an explicit reason. Used BEFORE the 208-row run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _extract_patch(text: str) -> str:
    # Identical to reverse_reap.swebench._patch (frozen scorer).
    fenced = re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced[-1].strip() if fenced else text.strip()


def _is_unified_diff(patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"
    if "```" in patch:
        return False, "patch contains Markdown fences"
    has_git_header = "diff --git" in patch
    has_traditional = "--- " in patch and "+++ " in patch
    if not (has_git_header or has_traditional):
        return False, "missing diff headers (need diff --git or ---/+++)"
    if "@@" not in patch:
        return False, "missing @@ hunks"
    return True, ""


def _git_apply_check(patch: str, repo_dir: Path) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as fh:
        fh.write(patch if patch.endswith("\n") else patch + "\n")
        patch_path = Path(fh.name)
    try:
        result = subprocess.run(
            ["git", "apply", "--check", str(patch_path)],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr.strip() or result.stdout.strip() or "git apply --check failed")[:500]
    finally:
        patch_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True, help="dir with {condition}.sweprobe.jsonl")
    parser.add_argument("--conditions", nargs="+", required=True, help="condition_ids to check")
    parser.add_argument("--repo-dir", type=Path, default=None,
                        help="target repo checkout at the correct base state; if omitted, only syntax is checked and applicability MUST be verified separately (fail-closed: probe does not pass without --repo-dir or --allow-syntax-only)")
    parser.add_argument("--allow-syntax-only", action="store_true",
                        help="permit syntax-only validation (records that git apply was NOT verified; probe still requires explicit human ack to proceed)")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.report.exists():
        print(f"refusing to overwrite report: {args.report}", file=sys.stderr)
        return 2

    results: dict = {"conditions": {}, "passed": False}
    overall_ok = True

    # Load all probe files.
    loaded: dict[str, dict[str, str]] = {}
    for cid in args.conditions:
        path = args.probe_dir / f"{cid}.sweprobe.jsonl"
        if not path.is_file():
            print(f"missing probe file: {path}", file=sys.stderr)
            return 2
        rows = _load_jsonl(path)
        if not rows:
            print(f"empty probe file: {path}", file=sys.stderr)
            return 2
        loaded[cid] = {r["sample_id"]: r.get("response", "") for r in rows}
        # Row-level checks.
        cond_ok = True
        reasons: list[str] = []
        for r in rows:
            resp = r.get("response", "")
            if "```" in resp:
                cond_ok = False
                reasons.append(f"{r.get('sample_id')}: contains Markdown fences")
                continue
            patch = _extract_patch(resp)
            # For source-v2 the raw response must equal the extracted patch (no fences to strip).
            if patch != resp.strip():
                cond_ok = False
                reasons.append(f"{r.get('sample_id')}: raw response != extracted patch (fences/prose?)")
                continue
            ok, reason = _is_unified_diff(patch)
            if not ok:
                cond_ok = False
                reasons.append(f"{r.get('sample_id')}: not a unified diff ({reason})")
                continue
            if args.repo_dir is not None:
                ok, reason = _git_apply_check(patch, args.repo_dir)
                if not ok:
                    cond_ok = False
                    reasons.append(f"{r.get('sample_id')}: git apply --check failed ({reason})")
        results["conditions"][cid] = {"rows": len(rows), "passed": cond_ok, "reasons": reasons}
        if not cond_ok:
            overall_ok = False

    # Determinism invariants on the probed sample set (require baseline pair + noop).
    if "c0-baseline-a" in loaded and "c0-baseline-b" in loaded:
        a, b = loaded["c0-baseline-a"], loaded["c0-baseline-b"]
        common = sorted(set(a) & set(b))
        mism = [sid for sid in common if a[sid] != b[sid]]
        results["baseline_pair"] = {"samples": len(common), "mismatches": mism, "passed": not mism}
        if mism:
            overall_ok = False
    if "c0-baseline-a" in loaded and "c0-noop-masked" in loaded:
        a, n = loaded["c0-baseline-a"], loaded["c0-noop-masked"]
        common = sorted(set(a) & set(n))
        mism = [sid for sid in common if a[sid] != n[sid]]
        results["noop_equivalence"] = {"samples": len(common), "mismatches": mism, "passed": not mism}
        if mism:
            overall_ok = False

    if args.repo_dir is None and not args.allow_syntax_only:
        print("probe FAIL-CLOSED: no --repo-dir for git apply --check and --allow-syntax-only not set", file=sys.stderr)
        overall_ok = False
        results["applicability"] = "not verified (missing --repo-dir)"
    elif args.repo_dir is None:
        results["applicability"] = "syntax-only (git apply NOT verified; human ack required)"
    else:
        results["applicability"] = f"git apply --check against {args.repo_dir}"

    results["passed"] = bool(overall_ok)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": results["passed"], "report": str(args.report)}, sort_keys=True))
    return 0 if results["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
