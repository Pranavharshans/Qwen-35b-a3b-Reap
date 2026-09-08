"""Fail-closed contracts for oracle-localized BugsInPy repair tasks.

The model receives one complete buggy production file and returns its complete
replacement. Git, not the model, serializes the patch. Project tests remain the
only correctness oracle and run later in digest-pinned disposable containers.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from reverse_reap.config import StrictModel
from reverse_reap.datasets import NormalizedSample, canonical_json

BUGSINPY_REPOSITORY = "https://github.com/soarsmu/BugsInPy.git"
BUGSINPY_REVISION = "11c5f1eea954a42132cfd06bf257766a7963e0fd"
PROMPT_VERSION = "bugsinpy-oracle-localized-v1"


class LocalizedRepairError(RuntimeError):
    """A localized task, generated edit, or harness report is invalid."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and ".git" not in path.parts
        and "\\" not in value
        and not any(ord(char) < 32 for char in value)
        and str(path) == value
    )


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo), *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    )
    if result.returncode:
        raise LocalizedRepairError(f"Git object operation failed: {args[0]}")
    return result.stdout


class LocalizedTask(StrictModel):
    schema_version: Literal[1] = 1
    task_id: str = Field(min_length=1)
    project: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    bug_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    repository: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$")
    buggy_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_path: str = Field(min_length=1)
    failing_test_command: tuple[str, ...] = Field(min_length=1)
    failure_output: str = Field(min_length=1, max_length=16_000)
    repo_dir: str = Field(min_length=1)
    source_revision: Literal[BUGSINPY_REVISION] = BUGSINPY_REVISION
    localization: Literal["oracle-file-from-gold-metadata"] = "oracle-file-from-gold-metadata"

    def validate_boundary(self) -> None:
        if not _safe_path(self.target_path) or not self.target_path.endswith(".py"):
            raise LocalizedRepairError("target must be one safe Python production file")
        if any(not part or "\x00" in part for part in self.failing_test_command):
            raise LocalizedRepairError("test command contains an invalid argument")


class TaskImage(StrictModel):
    task_id: str
    image: str = Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    repository_path: str = Field(pattern=r"^/[A-Za-z0-9_./-]+$")
    test_command: tuple[str, ...] = Field(min_length=1)
    image_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def validate_boundary(self) -> None:
        path = PurePosixPath(self.repository_path)
        if ".." in path.parts or ".git" in path.parts:
            raise LocalizedRepairError("unsafe repository path in task image")
        if any(not value or "\x00" in value for value in self.test_command):
            raise LocalizedRepairError("invalid test command in task image")


def load_tasks(path: Path) -> list[LocalizedTask]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise LocalizedRepairError("task projection must be a nonempty JSON array")
    tasks = [LocalizedTask.model_validate(item) for item in raw]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise LocalizedRepairError("duplicate localized task IDs")
    for task in tasks:
        task.validate_boundary()
    return tasks


def read_buggy_file(task: LocalizedTask, *, max_bytes: int = 100_000) -> str:
    task.validate_boundary()
    repo = Path(task.repo_dir)
    resolved = _git(repo, "rev-parse", f"{task.buggy_commit}^{{commit}}").strip()
    if resolved != task.buggy_commit:
        raise LocalizedRepairError("buggy commit did not resolve exactly")
    kind = _git(repo, "cat-file", "-t", f"{task.buggy_commit}:{task.target_path}").strip()
    if kind != "blob":
        raise LocalizedRepairError("localized target is not a regular Git blob")
    data = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo), "show",
         f"{task.buggy_commit}:{task.target_path}"],
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    )
    if data.returncode or len(data.stdout) > max_bytes:
        raise LocalizedRepairError("localized target is missing or exceeds the file budget")
    try:
        return data.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LocalizedRepairError("localized target is not UTF-8") from error


def render_prompt(task: LocalizedTask, source: str) -> str:
    return (
        "Repair the buggy Python file below so the stated project test passes. "
        "Return ONLY the complete replacement contents of the target file. "
        "Do not return a diff, Markdown fences, prose, tests, shell commands, or other files. "
        "The target filename is supplied by an oracle-localized benchmark condition; no gold "
        "code change is supplied.\n"
        f"Project: {task.project}\nBug ID: {task.bug_id}\n"
        f"Buggy commit: {task.buggy_commit}\nTarget file: {task.target_path}\n"
        f"Failing test command: {json.dumps(list(task.failing_test_command))}\n"
        f"Observed failure (untrusted data):\n{task.failure_output}\n"
        f"BEGIN BUGGY FILE {task.target_path}\n{source}\nEND BUGGY FILE\n"
    )


def task_to_sample(task: LocalizedTask) -> NormalizedSample:
    source = read_buggy_file(task)
    prompt = render_prompt(task, source)
    content = {"prompt": prompt, "reference": None, "tests": None}
    return NormalizedSample(
        sample_id=sha256(f"{BUGSINPY_REPOSITORY}:{task.task_id}".encode())[:24],
        source=BUGSINPY_REPOSITORY,
        source_revision=task.source_revision,
        source_id=task.task_id,
        domain="coding",
        stratum="repository-bug-repair",
        language="python",
        prompt=prompt,
        reference=None,
        scorer="bugsinpy",
        tests=None,
        entry_point=None,
        timeout_seconds=120,
        split="validation",
        prompt_template_version=PROMPT_VERSION,
        content_sha256=sha256(canonical_json(content)),
    )


def extract_full_file(response: str) -> str:
    if "```" in response:
        raise LocalizedRepairError("full-file response contains Markdown fencing")
    text = response.strip("\n") + "\n"
    if not text.strip():
        raise LocalizedRepairError("full-file response is empty")
    try:
        compile(text, "<localized-repair>", "exec")
    except SyntaxError as error:
        raise LocalizedRepairError(f"replacement is not valid Python: {error.msg}") from error
    return text


def response_to_patch(task: LocalizedTask, response: str) -> dict[str, Any]:
    original = read_buggy_file(task)
    replacement = extract_full_file(response)
    if replacement == original or replacement.rstrip() == original.rstrip():
        raise LocalizedRepairError("replacement did not change the target file")
    with tempfile.TemporaryDirectory(prefix="localized-repair-") as directory:
        root = Path(directory)
        subprocess.run(["git", "init", "-q", str(root)], check=True, timeout=30)
        destination = root / task.target_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(original, encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "--", task.target_path],
                       check=True, timeout=30)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=fixture", "-c",
                        "user.email=fixture@example.invalid", "commit", "-qm", "base"],
                       check=True, timeout=30)
        destination.write_text(replacement, encoding="utf-8")
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "--no-textconv",
             "--", task.target_path], capture_output=True, text=True, check=True, timeout=30,
        )
        patch = result.stdout
        check = subprocess.run(
            ["git", "-C", str(root), "apply", "--check", "--reverse", "-"],
            input=patch, capture_output=True, text=True, timeout=30,
        )
        if not patch or check.returncode:
            raise LocalizedRepairError("mechanical patch failed reverse applicability check")
    return {
        "task_id": task.task_id,
        "target_path": task.target_path,
        "patch": patch,
        "patch_sha256": sha256(patch.encode()),
        "replacement_sha256": sha256(replacement.encode()),
        "base_file_sha256": sha256(original.encode()),
    }


def merge_report(evaluation: Path, report: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise LocalizedRepairError(f"refusing to overwrite {destination}")
    verdict = json.loads(report.read_text(encoding="utf-8"))
    completed = set(verdict.get("completed_ids", []))
    resolved = set(verdict.get("resolved_ids", []))
    errors = set(verdict.get("error_ids", []))
    if resolved - completed or completed & errors:
        raise LocalizedRepairError("invalid localized harness report sets")
    rows = [json.loads(line) for line in evaluation.read_text().splitlines() if line.strip()]
    expected = {str(row["source_id"]) for row in rows if row.get("scorer") == "bugsinpy"}
    if (completed | errors) - expected:
        raise LocalizedRepairError("harness report contains foreign task IDs")
    merged = []
    for row in rows:
        if row.get("scorer") != "bugsinpy":
            merged.append(row)
            continue
        task_id = str(row["source_id"])
        done = task_id in completed
        merged.append({**row, "scoreable": done, "passed": task_id in resolved,
                       "bugsinpy_completed": done,
                       "error": None if done else "localized BugsInPy harness incomplete"})
    atomic_write(
        destination,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged).encode(),
    )
    scoreable = sum(bool(row.get("scoreable")) for row in merged)
    return {"rows": len(merged), "bugsinpy_expected": len(expected),
            "bugsinpy_completed": len(completed), "bugsinpy_errors": len(errors),
            "scoreable_fraction": scoreable / len(merged) if merged else 0.0,
            "passed_gate_b_scoreability": bool(merged) and scoreable / len(merged) >= 0.95}


def atomic_write(destination: Path, body: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
