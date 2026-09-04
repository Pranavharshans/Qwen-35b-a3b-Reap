"""Install the PINNED official SWE-bench harness (CPU-only, idempotent).

Measured at the pinned revision (2026-09-04 scoring-host probe):

* The harness venv needs Python <= 3.12 (the reverse-reap venv may be 3.14);
  pass ``--python`` to select the interpreter used for the venv.
* At this revision the hosted princeton-nlp/SWE-bench_Lite rows lack the
  image/eval_script columns the harness requires, so the SWE-bench task repo
  (swe-bench-tasks) is cloned as well and pinned; every harness invocation
  must pass ``--task-repo`` pointing at it.

Clones SWE-bench/SWE-bench at SWE_BENCH_REVISION and SWE-bench/swe-bench-tasks
at SWE_BENCH_TASKS_REVISION, creates a venv, installs the harness in editable
mode, verifies the import, and writes harness-report.json. Refuses to run if
any installed revision drifts from its pin.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reverse_reap.swebench import (
    SWE_BENCH_DATASET,
    SWE_BENCH_REPOSITORY,
    SWE_BENCH_REVISION,
    SWE_BENCH_TASKS_REPOSITORY,
    SWE_BENCH_TASKS_REVISION,
)


def run(command: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, capture_output=True, text=True)


def clone_pinned(repository: str, revision: str, destination: Path) -> None:
    if destination.exists():
        head = run(["git", "-C", str(destination), "rev-parse", "HEAD"]).stdout.strip()
        if head == revision:
            return
        raise SystemExit(f"{destination} revision {head!r} != pinned {revision!r}")
    result = run(["git", "clone", repository, str(destination)])
    if result.returncode != 0:
        raise SystemExit(f"clone failed: {result.stderr[-800:]}")
    result = run(["git", "-C", str(destination), "checkout", revision])
    if result.returncode != 0:
        raise SystemExit(f"checkout failed: {result.stderr[-800:]}")
    result = run(["git", "-C", str(destination), "rev-parse", "HEAD"])
    if result.stdout.strip() != revision:
        raise SystemExit(f"installed revision drift: {result.stdout.strip()!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True,
                        help="destination for the pinned swe-bench-tasks clone")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the harness venv (needs <=3.12)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "harness-report.json"

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("harness_revision") == SWE_BENCH_REVISION:
            clone_pinned(SWE_BENCH_TASKS_REPOSITORY, SWE_BENCH_TASKS_REVISION,
                         args.tasks_dir)
            print(f"harness already installed at pinned revision {SWE_BENCH_REVISION}",
                  flush=True)
            return 0
        raise SystemExit(
            f"installed harness revision {report.get('harness_revision')} != pinned "
            f"{SWE_BENCH_REVISION}; refusing to proceed"
        )

    install = args.install_dir
    if install.exists():
        raise SystemExit(f"install dir exists without a valid report; remove it first: {install}")
    clone_pinned(SWE_BENCH_REPOSITORY, SWE_BENCH_REVISION, install)
    clone_pinned(SWE_BENCH_TASKS_REPOSITORY, SWE_BENCH_TASKS_REVISION, args.tasks_dir)

    result = run([args.python, "-m", "venv", str(install / "venv")])
    if result.returncode != 0:
        raise SystemExit(f"venv failed: {result.stderr[-800:]}")
    venv_python = install / "venv" / "bin" / "python"
    for command in (
        [str(venv_python), "-m", "pip", "install", "-q", "--upgrade", "pip"],
        [str(venv_python), "-m", "pip", "install", "-q", "-e", str(install)],
    ):
        result = run(command)
        if result.returncode != 0:
            raise SystemExit(f"harness pip install failed: {result.stderr[-1500:]}")
    result = run([str(venv_python), "-c", "import swebench; print('import-ok')"])
    if result.returncode != 0 or "import-ok" not in result.stdout:
        raise SystemExit(f"harness import failed: {result.stderr[-800:]}")

    report = {
        "harness_repository": SWE_BENCH_REPOSITORY,
        "harness_revision": SWE_BENCH_REVISION,
        "tasks_repository": SWE_BENCH_TASKS_REPOSITORY,
        "tasks_revision": SWE_BENCH_TASKS_REVISION,
        "dataset": SWE_BENCH_DATASET,
        "python": str(venv_python),
        "installed": True,
    }
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"harness installed at {SWE_BENCH_REVISION}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
