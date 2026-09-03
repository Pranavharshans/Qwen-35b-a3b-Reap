"""Single-writer, resumable autonomous task controller with live heartbeat."""

from __future__ import annotations

import fcntl
import hashlib
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from reverse_reap.budget import evaluate_budget
from reverse_reap.config import ExperimentConfig, StrictModel
from reverse_reap.state import RunState, Status, atomic_write_state, load_state


class ControllerError(RuntimeError):
    """Raised when an autonomous task cannot be safely dispatched."""


class TaskDefinition(StrictModel):
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    objective: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    validation_command: list[str] = Field(min_length=1)
    inputs: list[Path] = Field(default_factory=list)
    outputs: list[Path] = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    estimated_gpu_hours: float = Field(ge=0)
    estimated_storage_gb: float = Field(ge=0)
    failure_behavior: Literal["retry", "terminal", "wait_for_human"] = "retry"


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


def _run_with_heartbeat(
    command: list[str], state: RunState, state_path: Path, log_path: Path, interval: float
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                return process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                state.heartbeat_at_utc = datetime.now(UTC)
                state.updated_at_utc = state.heartbeat_at_utc
                atomic_write_state(state_path, state)


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
        if path.exists():
            state = load_state(path)
            if state.status not in {Status.FAILED_RETRYABLE, Status.PREFLIGHTED}:
                raise ControllerError(f"task is not resumable from {state.status}")
        else:
            missing = [str(item) for item in eligible.inputs if not item.exists()]
            if missing:
                raise ControllerError(f"task inputs are missing: {missing}")
            state = RunState(
                run_id=run_id,
                task_id=eligible.task_id,
                config_sha256=config.fingerprint(),
                input_hashes={str(item): file_sha256(item) for item in eligible.inputs},
                validation_command=eligible.validation_command,
            )
            state.transition(Status.PREFLIGHTED)
            atomic_write_state(path, state)
        decision = evaluate_budget(
            config.budget,
            consumed_gpu_hours=state.consumed_gpu_hours,
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
        exit_code = _run_with_heartbeat(
            eligible.command, state, path, log_path, heartbeat_seconds
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
        validation = subprocess.run(eligible.validation_command, check=False)
        state.validation_exit_code = validation.returncode
        missing_outputs = [str(item) for item in eligible.outputs if not item.exists()]
        if validation.returncode or missing_outputs:
            state.failure_signature = (
                f"validation-exit-{validation.returncode}"
                if validation.returncode
                else "missing-outputs"
            )
            state.transition(Status.FAILED_TERMINAL)
        else:
            state.output_hashes = {str(item): file_sha256(item) for item in eligible.outputs}
            state.consumed_gpu_hours += eligible.estimated_gpu_hours
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
