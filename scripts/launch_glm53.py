#!/usr/bin/env python3
"""Preview or execute a governed GLM-5.3-Flash-BF16 run.

The launcher is deliberately a small dispatch boundary.  It validates the
same experiment configuration and execution plan used by ``reverse-reap``
before rendering one of two commands:

* ``direct`` invokes the normal ``reverse-reap run-all`` controller.
* ``fau-slurm`` delegates to ``scripts/fau/submit_reverse_reap.sh``.

Preview is the default and never creates a state directory or invokes either
runner.  Execution writes a run-ID-pinned config snapshot before dispatch so a
later controller resume cannot silently derive a different run ID.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from reverse_reap.config import ExperimentConfig, load_config
from reverse_reap.donors import GLM53_BF16_MODEL_ID
from reverse_reap.plan_materializer import render_plan

Mode = Literal["direct", "fau-slurm"]
_MODES = frozenset(("direct", "fau-slurm"))
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class LauncherError(RuntimeError):
    """Raised when a launch cannot be rendered safely."""


@dataclass(frozen=True)
class LaunchContext:
    """Fully validated, immutable dispatch inputs."""

    mode: Mode
    config_path: Path
    plan_path: Path
    model_dir: Path
    state_dir: Path
    run_root: Path
    run_id: str
    repo_dir: Path
    config: ExperimentConfig
    execute: bool
    resume: bool
    reverse_reap_command: tuple[str, ...]
    fau_submitter: Path | None = None
    thinking_config: Path | None = None
    through_task: str | None = None
    job_name: str = "reverse-reap-glm53"
    heartbeat_seconds: float = 30.0
    stale_after_seconds: float = 180.0
    materialized_plan: Path | None = None


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _require_file(path: Path, label: str) -> Path:
    resolved = _absolute(path)
    if not resolved.is_file():
        raise LauncherError(f"{label} does not exist or is not a file: {path}")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    resolved = _absolute(path)
    if not resolved.is_dir():
        raise LauncherError(f"{label} does not exist or is not a directory: {path}")
    return resolved


def _validate_model_config(path: Path) -> ExperimentConfig:
    try:
        config = load_config(path)
    except Exception as error:  # pydantic/yaml errors should be operator-facing
        raise LauncherError(f"invalid experiment config {path}: {error}") from error
    if config.model.id != GLM53_BF16_MODEL_ID:
        raise LauncherError(
            f"config model.id must be exactly {GLM53_BF16_MODEL_ID}; got {config.model.id}"
        )
    if config.model.source_precision != "bf16" or config.model.execution_precision != "bf16":
        raise LauncherError("GLM-5.3-Flash-BF16 requires source and execution precision bf16")
    if config.model.revision == "0" * 40:
        raise LauncherError(
            "config uses the all-zero revision placeholder; run metadata preflight "
            "and pin the exact GLM revision before execution"
        )
    return config


def _validate_thinking_config(path: Path) -> Path:
    resolved = _require_file(path, "thinking config")
    _validate_model_config(resolved)
    return resolved


def _git_sha(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LauncherError(f"could not resolve the repository revision: {error}") from error
    value = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{8,64}", value):
        raise LauncherError(f"git returned an invalid revision: {value!r}")
    return value


def _resolve_run_id(
    config: ExperimentConfig,
    *,
    requested: str | None,
    repo_dir: Path,
) -> str:
    if requested is not None:
        if not _RUN_ID.fullmatch(requested):
            raise LauncherError("--run-id must be a path-safe identifier")
        if config.run_id is not None and config.run_id != requested:
            raise LauncherError("--run-id differs from the run_id already pinned in the config")
        return requested
    run_id = config.run_id or config.resolve_run_id(_git_sha(repo_dir))
    if not _RUN_ID.fullmatch(run_id):
        raise LauncherError(f"resolved run_id is not path-safe: {run_id!r}")
    return run_id


def _resolve_state_paths(
    *,
    state_root: Path | None,
    state_dir: Path | None,
    run_id: str,
    resume: bool,
) -> tuple[Path, Path]:
    if (state_root is None) == (state_dir is None):
        raise LauncherError("provide exactly one of --state-root or --state-dir")
    if state_root is not None:
        root = _absolute(state_root)
        if root.exists() and not root.is_dir():
            raise LauncherError(f"state root is not a directory: {state_root}")
        if not root.exists() and not root.parent.is_dir():
            raise LauncherError(f"state root parent does not exist: {root.parent}")
        run_root = root / run_id
        target = run_root / "state"
    else:
        assert state_dir is not None
        target = _absolute(state_dir)
        if target.exists() and not target.is_dir():
            raise LauncherError(f"state directory is not a directory: {state_dir}")
        if not target.exists() and not target.parent.is_dir():
            raise LauncherError(f"state directory parent does not exist: {target.parent}")
        run_root = target.parent

    if resume:
        if not target.is_dir():
            raise LauncherError(f"--resume requires an existing state directory: {target}")
    elif target.exists() and any(target.iterdir()):
        raise LauncherError(
            f"state directory is non-empty: {target}; pass --resume for an intentional continuation"
        )
    if state_root is not None and run_root.exists() and any(run_root.iterdir()) and not resume:
        raise LauncherError(
            f"run directory already contains artifacts: {run_root}; choose a new run ID "
            "or pass --resume"
        )
    return run_root, target


def _split_command(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        value = os.environ.get("REVERSE_REAP_COMMAND", "reverse-reap")
    parts = tuple(shlex.split(value) if isinstance(value, str) else value)
    if not parts or any(not part for part in parts):
        raise LauncherError("the reverse-reap command must contain at least one executable token")
    return parts


def prepare_launch(
    *,
    mode: Mode,
    config_path: Path,
    plan_path: Path,
    model_dir: Path,
    state_root: Path | None = None,
    state_dir: Path | None = None,
    repo_dir: Path | None = None,
    execute: bool = False,
    resume: bool = False,
    run_id: str | None = None,
    reverse_reap_command: str | Sequence[str] | None = None,
    fau_submitter: Path | None = None,
    thinking_config: Path | None = None,
    through_task: str | None = None,
    job_name: str = "reverse-reap-glm53",
    heartbeat_seconds: float = 30.0,
    stale_after_seconds: float = 180.0,
) -> LaunchContext:
    """Validate all launch inputs before any command is rendered."""
    if mode not in _MODES:
        raise LauncherError(f"mode must be one of {sorted(_MODES)}")
    if heartbeat_seconds <= 0 or stale_after_seconds <= 0:
        raise LauncherError("heartbeat and stale-after intervals must be positive")
    if not _JOB_NAME.fullmatch(job_name):
        raise LauncherError("job name must contain only letters, digits, '_' or '-'")

    config_path = _require_file(config_path, "experiment config")
    plan_path = _require_file(plan_path, "execution plan")
    model_dir = _require_directory(model_dir, "model directory")
    repo_dir = _require_directory(repo_dir or Path(__file__).parents[1], "repository")
    config = _validate_model_config(config_path)
    if execute and run_id is None and config.run_id is None:
        raise LauncherError(
            "--execute requires --run-id or a config with run_id pinned; "
            "reuse the run_id printed by a reviewed preview"
        )
    if through_task is not None and mode != "fau-slurm":
        raise LauncherError("--through-task is only supported in fau-slurm mode")
    if thinking_config is not None:
        if mode != "fau-slurm":
            raise LauncherError("--thinking-config is only supported in fau-slurm mode")
        thinking_config = _validate_thinking_config(thinking_config)

    submitter: Path | None = None
    if mode == "fau-slurm":
        submitter = _require_file(
            fau_submitter or repo_dir / "scripts" / "fau" / "submit_reverse_reap.sh",
            "FAU submission helper",
        )

    resolved_run_id = _resolve_run_id(config, requested=run_id, repo_dir=repo_dir)
    run_root, resolved_state_dir = _resolve_state_paths(
        state_root=state_root,
        state_dir=state_dir,
        run_id=resolved_run_id,
        resume=resume,
    )
    try:
        render_plan(
            plan_path,
            config=config_path,
            model_dir=model_dir,
            thinking_config=thinking_config,
            through_task=through_task,
            execution_mode=mode,
            run_root=run_root,
        )
    except Exception as error:
        raise LauncherError(
            f"invalid or incompatible execution plan {plan_path}: {error}"
        ) from error
    return LaunchContext(
        mode=mode,
        config_path=config_path,
        plan_path=plan_path,
        model_dir=model_dir,
        state_dir=resolved_state_dir,
        run_root=run_root,
        run_id=resolved_run_id,
        repo_dir=repo_dir,
        config=config,
        execute=execute,
        resume=resume,
        reverse_reap_command=_split_command(reverse_reap_command),
        fau_submitter=submitter,
        thinking_config=thinking_config,
        through_task=through_task,
        job_name=job_name,
        heartbeat_seconds=heartbeat_seconds,
        stale_after_seconds=stale_after_seconds,
        materialized_plan=(
            run_root / "plan" / "materialized-plan.yaml" if mode == "direct" else None
        ),
    )


def render_command(context: LaunchContext, *, config_path: Path | None = None) -> list[str]:
    """Return the argv that will be executed; never invoke a shell."""
    config = str(config_path or context.config_path)
    if context.mode == "direct":
        plan = str(context.materialized_plan or context.plan_path)
        return [
            *context.reverse_reap_command,
            "run-all",
            config,
            plan,
            str(context.state_dir),
            "--heartbeat-seconds",
            str(context.heartbeat_seconds),
            "--stale-after-seconds",
            str(context.stale_after_seconds),
        ]
    if context.fau_submitter is None:  # defensive; prepare_launch always supplies it
        raise LauncherError("FAU submission helper was not resolved")
    command = [str(context.fau_submitter)]
    if context.execute:
        command.append("--submit")
    command.extend(("--job-name", context.job_name))
    command.extend(("--run-root", str(context.run_root)))
    if context.thinking_config is not None:
        command.extend(("--thinking-config", str(context.thinking_config)))
    if context.through_task is not None:
        command.extend(("--through-task", context.through_task))
    command.extend(
        (
            config,
            str(context.plan_path),
            str(context.model_dir),
            str(context.state_dir),
        )
    )
    return command


def _write_pinned_config(context: LaunchContext) -> Path:
    destination = context.run_root / "config" / "glm53-config.yaml"
    payload = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LauncherError(f"config is not a YAML object: {context.config_path}")
    payload["run_id"] = context.run_id
    rendered = yaml.safe_dump(payload, sort_keys=False)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise LauncherError(f"pinned config snapshot already differs: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def _write_materialized_plan(context: LaunchContext, config_path: Path) -> Path:
    """Write the validated direct-mode plan without touching its source."""
    if context.materialized_plan is None:
        raise LauncherError("a materialized plan is only required for direct mode")
    try:
        rendered = render_plan(
            context.plan_path,
            config=config_path,
            model_dir=context.model_dir,
            thinking_config=context.thinking_config,
            through_task=context.through_task,
            execution_mode=context.mode,
            run_root=context.run_root,
        )
    except Exception as error:
        raise LauncherError(f"could not materialize execution plan: {error}") from error
    destination = context.materialized_plan
    serialized = yaml.safe_dump(rendered, sort_keys=False)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise LauncherError(f"materialized plan already differs: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    return destination


def _write_launch_record(context: LaunchContext, command: Sequence[str], config_path: Path) -> Path:
    destination = context.run_root / "launch-record.json"
    record = {
        "schema_version": 1,
        "run_id": context.run_id,
        "mode": context.mode,
        "config": str(config_path),
        "source_config": str(context.config_path),
        "plan": str(context.plan_path),
        "model_dir": str(context.model_dir),
        "state_dir": str(context.state_dir),
        "config_sha256": context.config.fingerprint(),
        "command": list(command),
        "execution_requested": True,
    }
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise LauncherError(f"launch record already differs: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def _describe(context: LaunchContext, command: Sequence[str], *, config_path: Path) -> None:
    action = "execute" if context.execute else "preview"
    print("GLM-5.3-Flash-BF16 launcher")
    print(f"mode: {context.mode}")
    print(f"action: {action}")
    print(f"run_id: {context.run_id}")
    print(f"config: {config_path}")
    print(f"source_plan: {context.plan_path}")
    if context.materialized_plan is not None:
        print(f"plan: {context.materialized_plan}")
    else:
        print(f"plan: {context.plan_path} (materialized by the FAU job)")
    print(f"model_dir: {context.model_dir}")
    print(f"state_dir: {context.state_dir}")
    print(f"command: {shlex.join(command)}")
    if context.execute:
        print(
            "dispatch: explicit execution requested; the governed runner will "
            "enforce its plan gates"
        )
    else:
        print("dispatch: preview only; no GPU, filesystem, or scheduler state change is requested")


def execute_launch(context: LaunchContext) -> int:
    """Pin launch metadata and dispatch exactly one validated command."""
    config_path = _write_pinned_config(context)
    if context.mode == "direct":
        _write_materialized_plan(context, config_path)
    command = render_command(context, config_path=config_path)
    _write_launch_record(context, command, config_path)
    _describe(context, command, config_path=config_path)
    env = os.environ.copy()
    env["RUN_ID"] = context.run_id
    try:
        result = subprocess.run(command, cwd=context.repo_dir, env=env, check=False)
    except OSError as error:
        print(f"error: could not execute launcher command: {error}", file=sys.stderr)
        return 69
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or explicitly execute the governed GLM-5.3-Flash-BF16 "
            "direct or FAU-Slurm dispatch."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mode", choices=sorted(_MODES), dest="mode_flag")
    mode_group.add_argument("--direct", action="store_const", const="direct", dest="mode_flag")
    mode_group.add_argument(
        "--fau-slurm", action="store_const", const="fau-slurm", dest="mode_flag"
    )
    parser.add_argument("mode_positional", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("positional_paths", nargs="*", metavar="PATH")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--repo-dir", type=Path)
    parser.add_argument("--fau-submitter", type=Path)
    parser.add_argument("--thinking-config", type=Path)
    parser.add_argument("--through-task")
    parser.add_argument("--job-name", default="reverse-reap-glm53")
    parser.add_argument("--run-id")
    parser.add_argument("--reverse-reap-command")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--stale-after-seconds", type=float, default=180.0)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--execute", action="store_true")
    action_group.add_argument("--preview", "--dry-run", action="store_true", dest="preview")
    parser.add_argument("--resume", action="store_true")
    return parser


def _mode_and_paths(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Mode, list[Path]]:
    positional = []
    if args.mode_positional is not None:
        positional.append(args.mode_positional)
    positional.extend(args.positional_paths)
    if args.mode_flag is not None:
        if positional and positional[0] in _MODES:
            parser.error("mode was supplied both as a flag and a positional argument")
        mode = args.mode_flag
    else:
        if not positional or positional[0] not in _MODES:
            parser.error(
                "choose exactly one mode: --direct, --fau-slurm, "
                "or a direct/fau-slurm positional"
            )
        mode = positional.pop(0)
    return mode, [Path(item) for item in positional]


def _path_values(
    args: argparse.Namespace, positional: list[Path], parser: argparse.ArgumentParser
) -> dict[str, Path | None]:
    """Merge option and positional paths while allowing ``--state-root``."""
    names = ("config", "plan", "model_dir")
    values: dict[str, Path | None] = {name: getattr(args, name) for name in names}
    missing_names = [name for name in names if values[name] is None]
    if len(positional) > len(missing_names) + 1:
        parser.error("too many positional paths; expected CONFIG PLAN MODEL_DIR [STATE_DIR]")
    for name, value in zip(missing_names, positional, strict=False):
        values[name] = value
    remaining = positional[len(missing_names) :]
    values["state_dir"] = args.state_dir
    if remaining:
        if values["state_dir"] is not None:
            parser.error("state directory was supplied both positionally and with --state-dir")
        values["state_dir"] = remaining[0]
    if args.state_root is None and values["state_dir"] is None:
        parser.error("provide --state-root or a STATE_DIR path")
    missing = [
        name.replace("_", "-")
        for name, value in values.items()
        if value is None and not (name == "state_dir" and args.state_root is not None)
    ]
    if missing:
        parser.error(f"missing required launch paths: {', '.join(missing)}")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode, positional = _mode_and_paths(args, parser)
    values = _path_values(args, positional, parser)
    if args.state_root is not None and args.state_dir is not None:
        parser.error("choose only one of --state-root or --state-dir")
    if args.state_root is None:
        args.state_dir = values["state_dir"]
    elif values["state_dir"] is not None:
        parser.error("the fourth positional path cannot be combined with --state-root")
    if args.preview and args.execute:
        parser.error("--preview/--dry-run and --execute are mutually exclusive")
    try:
        context = prepare_launch(
            mode=mode,
            config_path=values["config"],
            plan_path=values["plan"],
            model_dir=values["model_dir"],
            state_root=args.state_root,
            state_dir=args.state_dir,
            repo_dir=args.repo_dir,
            execute=args.execute,
            resume=args.resume,
            run_id=args.run_id,
            reverse_reap_command=args.reverse_reap_command,
            fau_submitter=args.fau_submitter,
            thinking_config=args.thinking_config,
            through_task=args.through_task,
            job_name=args.job_name,
            heartbeat_seconds=args.heartbeat_seconds,
            stale_after_seconds=args.stale_after_seconds,
        )
        if context.execute:
            return execute_launch(context)
        command = render_command(context)
        _describe(context, command, config_path=context.config_path)
        return 0
    except LauncherError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
