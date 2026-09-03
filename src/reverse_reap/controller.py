"""Single-writer, resumable autonomous task controller with live heartbeat."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from reverse_reap.budget import evaluate_budget
from reverse_reap.config import ExperimentConfig, StrictModel
from reverse_reap.state import RunState, Status, atomic_write_state, load_state


class ControllerError(RuntimeError):
    """Raised when an autonomous task cannot be safely dispatched."""


class GateCondition(StrictModel):
    path: Path
    field: str
    equals: bool | str | int | float


class TaskDefinition(StrictModel):
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    objective: str = Field(min_length=1)
    definition_of_done: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    validation_command: list[str] = Field(min_length=1)
    inputs: list[Path] = Field(default_factory=list)
    outputs: list[Path] = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    estimated_gpu_hours: float = Field(ge=0)
    estimated_storage_gb: float = Field(ge=0)
    failure_behavior: Literal["retry", "terminal", "wait_for_human"] = "retry"
    run_if: GateCondition | None = None


class ExecutionPlan(StrictModel):
    schema_version: Literal[1]
    tasks: list[TaskDefinition]


def load_plan(path: Path) -> ExecutionPlan:
    plan = ExecutionPlan.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    ids = [task.task_id for task in plan.tasks]
    if len(ids) != len(set(ids)):
        raise ControllerError("task IDs must be unique")
    known = set(ids)
    for task in plan.tasks:
        unknown = set(task.dependencies) - known
        if unknown:
            raise ControllerError(
                f"task {task.task_id} has unknown dependencies: {sorted(unknown)}"
            )
    return plan


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive_run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControllerError("another lead controller holds the run lock") from error
        yield


def _state_path(state_dir: Path, task_id: str) -> Path:
    return state_dir / f"{task_id}.json"


def _completed(state_dir: Path, task_id: str) -> bool:
    path = _state_path(state_dir, task_id)
    return path.exists() and load_state(path).status == Status.COMPLETE


def consumed_gpu_hours(state_dir: Path) -> float:
    total = 0.0
    for path in state_dir.glob("*.json"):
        state = load_state(path)
        if state.status == Status.COMPLETE:
            total += state.consumed_gpu_hours
    return total


_ENV_VARIABLE = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<name2>[A-Za-z_][A-Za-z0-9_]*)"
)


def expand_command(command: list[str], env: dict[str, str] | None = None) -> list[str]:
    source = os.environ if env is None else env

    def substitute(part: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group("name") or match.group("name2")
            return str(source.get(name, match.group(0)))

        return _ENV_VARIABLE.sub(replace, part)

    expanded = [substitute(part) for part in command]
    unresolved = [part for part in expanded if "$" in part]
    if unresolved:
        raise ControllerError(f"command has unresolved environment variables: {unresolved}")
    return expanded


RUN_ID_VARIABLE = "RUN_ID"


def expand_scoped_path(path: Path, run_id: str) -> Path:
    """Resolve the run-scoped ``${RUN_ID}`` placeholder in declared task paths.

    Plan outputs and inputs may embed ``${RUN_ID}`` so each governed run writes
    to its own namespace instead of colliding with a previous run's frozen
    artifacts. Any other unresolved variable fails closed.
    """
    text = str(path).replace("${RUN_ID}", run_id).replace("$RUN_ID", run_id)
    if "$" in text:
        raise ControllerError(f"path has unresolved variables: {path}")
    return Path(text)


def _subprocess_env(run_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env[RUN_ID_VARIABLE] = run_id
    return env


def _run_with_heartbeat(
    command: list[str],
    state: RunState,
    state_path: Path,
    log_path: Path,
    interval: float,
    env: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            expand_command(command, env), stdout=log, stderr=subprocess.STDOUT, env=env
        )
        state.process_id = process.pid
        state.heartbeat_at_utc = datetime.now(UTC)
        atomic_write_state(state_path, state)
        while True:
            try:
                exit_code = process.wait(timeout=interval)
                state.process_id = None
                return exit_code
            except subprocess.TimeoutExpired:
                state.heartbeat_at_utc = datetime.now(UTC)
                state.updated_at_utc = state.heartbeat_at_utc
                atomic_write_state(state_path, state)


def _pid_alive(process_id: int | None) -> bool:
    if process_id is None:
        return False
    try:
        os.kill(process_id, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def recover_stale_tasks(state_dir: Path, *, stale_after_seconds: float) -> list[str]:
    recovered = []
    cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
    for path in state_dir.glob("*.json"):
        state = load_state(path)
        heartbeat = state.heartbeat_at_utc or state.updated_at_utc
        if state.status != Status.RUNNING or heartbeat >= cutoff:
            continue
        if _pid_alive(state.process_id):
            raise ControllerError(
                f"task {state.task_id} has a stale heartbeat but process "
                f"{state.process_id} is alive"
            )
        state.process_id = None
        try:
            state.register_retry("stale-dead-process")
        except ValueError:
            state.transition(Status.FAILED_TERMINAL)
        else:
            state.transition(Status.FAILED_RETRYABLE)
            recovered.append(state.task_id)
        atomic_write_state(path, state)
    return recovered


def run_status(state_dir: Path) -> dict[str, Any]:
    states = [load_state(path) for path in sorted(state_dir.glob("*.json"))]
    return {
        "tasks": [
            {
                "task_id": state.task_id,
                "status": state.status,
                "attempt": state.attempt,
                "heartbeat_at_utc": state.heartbeat_at_utc,
                "process_id": state.process_id,
                "failure_signature": state.failure_signature,
            }
            for state in states
        ],
        "consumed_gpu_hours": consumed_gpu_hours(state_dir),
    }


def run_all(
    plan_path: Path,
    config: ExperimentConfig,
    state_dir: Path,
    *,
    run_id: str,
    heartbeat_seconds: float = 30,
    stale_after_seconds: float = 180,
) -> dict[str, Any]:
    """Recover safely and run eligible tasks until completion or a terminal state."""
    recover_stale_tasks(state_dir, stale_after_seconds=stale_after_seconds)
    completed = []
    while True:
        result = run_next(
            plan_path,
            config,
            state_dir,
            run_id=run_id,
            heartbeat_seconds=heartbeat_seconds,
        )
        if result.get("task_id") and result["status"] == Status.COMPLETE:
            completed.append(result["task_id"])
            continue
        if result["status"] == Status.FAILED_RETRYABLE:
            continue
        return {"status": result["status"], "completed_this_run": completed, "last": result}


def run_next(
    plan_path: Path,
    config: ExperimentConfig,
    state_dir: Path,
    *,
    run_id: str,
    heartbeat_seconds: float = 30,
) -> dict[str, Any]:
    """Run at most one eligible task; repeat invocation resumes the plan."""
    plan = load_plan(plan_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(state_dir / ".lead.lock"):
        eligible = next(
            (
                task
                for task in plan.tasks
                if not _completed(state_dir, task.task_id)
                and all(_completed(state_dir, dependency) for dependency in task.dependencies)
            ),
            None,
        )
        if eligible is None:
            return {"status": "COMPLETE", "message": "no eligible unfinished tasks"}
        path = _state_path(state_dir, eligible.task_id)
        if eligible.run_if is not None:
            condition_path = expand_scoped_path(eligible.run_if.path, run_id)
            if not condition_path.exists():
                raise ControllerError(f"task gate artifact is missing: {condition_path}")
            condition_payload = json.loads(condition_path.read_text(encoding="utf-8"))
            actual: Any = condition_payload
            for part in eligible.run_if.field.split("."):
                if not isinstance(actual, dict) or part not in actual:
                    raise ControllerError(
                        f"task gate field {eligible.run_if.field} is missing from {condition_path}"
                    )
                actual = actual[part]
            if actual != eligible.run_if.equals:
                if not path.exists():
                    state = RunState(
                        run_id=run_id,
                        task_id=eligible.task_id,
                        config_sha256=config.fingerprint(),
                        input_hashes={str(condition_path): file_sha256(condition_path)},
                        failure_signature=(f"SKIPPED_GATE:{eligible.run_if.field}={actual!r}"),
                    )
                    state.transition(Status.PREFLIGHTED)
                    state.transition(Status.RUNNING)
                    state.transition(Status.VALIDATING)
                    state.transition(Status.COMPLETE)
                    atomic_write_state(path, state)
                return {
                    "status": Status.COMPLETE,
                    "task_id": eligible.task_id,
                    "skipped": True,
                    "reason": f"gate {eligible.run_if.field} was {actual!r}",
                    "outputs": {},
                }
        if path.exists():
            state = load_state(path)
            if state.status not in {Status.FAILED_RETRYABLE, Status.PREFLIGHTED}:
                raise ControllerError(f"task is not resumable from {state.status}")
        else:
            expanded_inputs = [expand_scoped_path(item, run_id) for item in eligible.inputs]
            missing = [str(item) for item in expanded_inputs if not item.exists()]
            if missing:
                raise ControllerError(f"task inputs are missing: {missing}")
            state = RunState(
                run_id=run_id,
                task_id=eligible.task_id,
                config_sha256=config.fingerprint(),
                input_hashes={
                    str(item): file_sha256(item)
                    for item in expanded_inputs
                },
                validation_command=eligible.validation_command,
            )
            state.transition(Status.PREFLIGHTED)
            atomic_write_state(path, state)
        decision = evaluate_budget(
            config.budget,
            consumed_gpu_hours=consumed_gpu_hours(state_dir),
            projected_stage_gpu_hours=eligible.estimated_gpu_hours,
        )
        if not decision.allowed or eligible.estimated_storage_gb > config.budget.storage_limit_gb:
            state.transition(Status.WAITING_FOR_HUMAN)
            state.failure_signature = (
                decision.reason if not decision.allowed else "storage limit exceeded"
            )
            atomic_write_state(path, state)
            return {"status": state.status, "reason": state.failure_signature}
        if state.status == Status.FAILED_RETRYABLE:
            state.transition(Status.RUNNING)
        else:
            state.transition(Status.RUNNING)
        atomic_write_state(path, state)
        log_path = state_dir / "logs" / f"{eligible.task_id}.log"
        env = _subprocess_env(run_id)
        exit_code = _run_with_heartbeat(
            eligible.command, state, path, log_path, heartbeat_seconds, env
        )
        if exit_code != 0:
            signature = hashlib.sha256(f"exit:{exit_code}".encode()).hexdigest()[:16]
            try:
                state.register_retry(signature)
            except ValueError:
                state.transition(Status.FAILED_TERMINAL)
            else:
                target = {
                    "retry": Status.FAILED_RETRYABLE,
                    "terminal": Status.FAILED_TERMINAL,
                    "wait_for_human": Status.WAITING_FOR_HUMAN,
                }[eligible.failure_behavior]
                state.transition(target)
            atomic_write_state(path, state)
            return {"status": state.status, "exit_code": exit_code, "log": str(log_path)}
        state.transition(Status.VALIDATING)
        atomic_write_state(path, state)
        validation = subprocess.run(
            expand_command(eligible.validation_command, env), check=False, env=env
        )
        state.validation_exit_code = validation.returncode
        expanded_outputs = [expand_scoped_path(item, run_id) for item in eligible.outputs]
        missing_outputs = [str(item) for item in expanded_outputs if not item.exists()]
        if validation.returncode or missing_outputs:
            state.failure_signature = (
                f"validation-exit-{validation.returncode}"
                if validation.returncode
                else "missing-outputs"
            )
            state.transition(Status.FAILED_TERMINAL)
        else:
            state.output_hashes = {str(item): file_sha256(item) for item in expanded_outputs}
            state.consumed_gpu_hours = eligible.estimated_gpu_hours
            state.consumed_cost_usd = (
                state.consumed_gpu_hours * config.budget.provider_rate_usd_per_hour
            )
            following = next(
                (task.task_id for task in plan.tasks if eligible.task_id in task.dependencies), None
            )
            state.next_permitted_task = following
            state.transition(Status.COMPLETE)
        atomic_write_state(path, state)
        return {
            "status": state.status,
            "task_id": eligible.task_id,
            "outputs": state.output_hashes,
            "log": str(log_path),
        }
