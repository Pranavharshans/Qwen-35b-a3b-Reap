"""Install the PINNED official SWE-bench harness (CPU-only, idempotent).

Clones SWE-bench/SWE-bench at the revision pinned in reverse_reap.swebench,
creates a venv, installs the harness in editable mode, and verifies the
import. Writes harness-report.json. Refuses to run if the installed revision
ever drifts from the pin.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reverse_reap.swebench import SWE_BENCH_DATASET, SWE_BENCH_REPOSITORY, SWE_BENCH_REVISION


def run(command: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "harness-report.json"

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("harness_revision") == SWE_BENCH_REVISION:
            print(f"harness already installed at pinned revision {SWE_BENCH_REVISION}", flush=True)
            return 0
        raise SystemExit(
            f"installed harness revision {report.get('harness_revision')} != pinned "
            f"{SWE_BENCH_REVISION}; refusing to proceed"
        )

    install = args.install_dir
    if install.exists():
        raise SystemExit(f"install dir exists without a valid report; remove it first: {install}")
    result = run(["git", "clone", SWE_BENCH_REPOSITORY, str(install)])
    if result.returncode != 0:
        raise SystemExit(f"clone failed: {result.stderr[-800:]}")
    result = run(["git", "-C", str(install), "checkout", SWE_BENCH_REVISION])
    if result.returncode != 0:
        raise SystemExit(f"checkout failed: {result.stderr[-800:]}")
    result = run(["git", "-C", str(install), "rev-parse", "HEAD"])
    if result.stdout.strip() != SWE_BENCH_REVISION:
        raise SystemExit(f"installed revision drift: {result.stdout.strip()!r}")

    result = run([sys.executable, "-m", "venv", str(install / "venv")])
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
        "dataset": SWE_BENCH_DATASET,
        "python": str(venv_python),
        "installed": True,
    }
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"harness installed at {SWE_BENCH_REVISION}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
