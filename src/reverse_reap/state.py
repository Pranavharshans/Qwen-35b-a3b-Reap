"""Atomic, resumable stage state for unattended experiments."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Status(StrEnum):
    PENDING = "PENDING"
    PREFLIGHTED = "PREFLIGHTED"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"


ALLOWED_TRANSITIONS: dict[Status, set[Status]] = {
    Status.PENDING: {Status.PREFLIGHTED, Status.WAITING_FOR_HUMAN, Status.FAILED_TERMINAL},
    Status.PREFLIGHTED: {Status.RUNNING, Status.WAITING_FOR_HUMAN, Status.FAILED_TERMINAL},
    Status.RUNNING: {
        Status.VALIDATING,
        Status.FAILED_RETRYABLE,
        Status.FAILED_TERMINAL,
        Status.WAITING_FOR_HUMAN,
    },
    Status.VALIDATING: {
        Status.COMPLETE,
        Status.FAILED_RETRYABLE,
        Status.FAILED_TERMINAL,
        Status.WAITING_FOR_HUMAN,
    },
    Status.FAILED_RETRYABLE: {Status.RUNNING, Status.FAILED_TERMINAL, Status.WAITING_FOR_HUMAN},
    Status.COMPLETE: set(),
    Status.FAILED_TERMINAL: set(),
    Status.WAITING_FOR_HUMAN: set(),
}


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    task_id: str
    status: Status = Status.PENDING
    attempt: int = Field(default=0, ge=0, le=2)
    config_sha256: str
    input_hashes: dict[str, str] = Field(default_factory=dict)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    consumed_gpu_hours: float = Field(default=0.0, ge=0)
    consumed_cost_usd: float = Field(default=0.0, ge=0)
    last_validated_chunk: str | None = None
    failure_signature: str | None = None
    updated_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def transition(self, target: Status) -> None:
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid state transition: {self.status} -> {target}")
        self.status = target
        self.updated_at_utc = datetime.now(UTC)

    def register_retry(self, signature: str) -> None:
        if self.failure_signature == signature:
            self.attempt += 1
        else:
            self.failure_signature = signature
            self.attempt = 1
        if self.attempt > 2:
            raise ValueError("identical failure signature exceeded two retries")


def atomic_write_state(path: Path, state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path: Path) -> RunState:
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))
