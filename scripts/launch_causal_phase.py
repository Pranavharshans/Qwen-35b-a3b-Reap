"""Launch a run-ID-scoped causal phase under the governing controller.

The CLI (``reverse-reap run-all``) resolves the run id from the config at
every invocation, so a relaunch against the same state dir would derive a
NEW timestamped run id and write unfinished-task artifacts into a second
run directory. This launcher resolves the run id exactly once and keeps
the state dir, plan-path expansion, task env, and any later resume on
that single id:

    fresh:  python scripts/launch_causal_phase.py \
                --config configs/pinned-3090-bf16-gen.yaml \
                --plan configs/execution-plan-causal-gen.yaml \
                --state-root runs/causal-pilot/state
    resume: python scripts/launch_causal_phase.py \
                --config configs/pinned-3090-bf16-gen.yaml \
                --plan configs/execution-plan-causal-gen.yaml \
                --resume runs/causal-pilot/state/<run_id>

Fresh launches fail closed if the derived state dir already exists. The
launch record (run id, state dir, plan, config fingerprint) is written to
``<state_dir>/launch-record.json`` before any task runs. ``--dry-run``
prints the record and exits without launching anything (used by tests and
pre-launch checks).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reverse_reap.config import ExperimentConfig, load_config
from reverse_reap.controller import run_all

RUN_ID_PARTS = 5  # <stamp>-qwen35a3b-<mode>-<sha8>-<fp8>


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def resolve_launch(
    config: ExperimentConfig,
    state_root: Path,
    resume_state: Path | None,
    *,
    git_sha_value: str,
) -> tuple[str, Path]:
    """Resolve (run_id, state_dir) exactly once.

    Fresh launches derive the run id from the config and HEAD and refuse to
    reuse an existing state dir. Resume launches take the run id from the
    state dir name so a relaunch keeps writing into the original run
    namespace.
    """
    if resume_state is not None:
        run_id = resume_state.name
        if len(run_id.split("-")) != RUN_ID_PARTS or not run_id[0].isdigit():
            raise SystemExit(f"--resume state dir is not run-ID-scoped: {resume_state}")
        if not resume_state.is_dir():
            raise SystemExit(f"--resume state dir does not exist: {resume_state}")
        return run_id, resume_state
    run_id = config.run_id or config.resolve_run_id(git_sha_value)
    state_dir = state_root / run_id
    if state_dir.exists():
        raise SystemExit(
            f"state dir already exists: {state_dir} — a fresh launch must never "
            "reuse it; pass --resume to continue that run"
        )
    return run_id, state_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.state_root) == bool(args.resume):
        parser.error("exactly one of --state-root or --resume is required")

    config = load_config(args.config)
    run_id, state_dir = resolve_launch(
        config,
        args.state_root or args.resume.parent,
        args.resume,
        git_sha_value=git_sha(),
    )
    record = {
        "run_id": run_id,
        "state_dir": str(state_dir),
        "plan": str(args.plan),
        "config": str(args.config),
        "config_sha256": config.fingerprint(),
    }
    print(json.dumps(record, indent=2))
    if args.dry_run:
        return 0
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "launch-record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    result = run_all(args.plan, config, state_dir, run_id=run_id)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
