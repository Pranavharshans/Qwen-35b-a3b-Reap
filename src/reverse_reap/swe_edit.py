"""Deterministic, fail-closed repository editing for SWE-bench model sessions.

The scaffold exposes only bounded list/search/read/exact-match-edit operations.  It
materializes regular blobs from an exact Git commit into a new repository rather
than checking out an untrusted source repository, so hooks, filters, replacement
objects, symlinks, and the source working tree cannot affect a session.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
PROTOCOL_VERSION = "swe-edit-v1"
TASK_KEYS = {"sample_id", "source_id", "repo", "base_commit", "problem_statement", "repo_dir"}
DENIED_TASK_KEYS = {
    "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text", "resolution_status",
}


class EditSessionError(ValueError):
    """The session cannot continue without weakening a declared invariant."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (rendered + "\n").encode()


def _git(repo: Path, *args: str, input_bytes: bytes | None = None, timeout: int = 60) -> bytes:
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo), *args],
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()[:500]
        raise EditSessionError(f"git {args[0]} failed: {message}")
    return result.stdout


def safe_repo_path(value: str, *, allow_root: bool = False) -> bool:
    if value == "" and allow_root:
        return True
    path = PurePosixPath(value)
    return bool(
        value
        and not path.is_absolute()
        and ".." not in path.parts
        and ".git" not in path.parts
        and "\\" not in value
        and not any(ord(char) < 32 for char in value)
        and str(path) == value
    )


def validate_task(task: dict[str, Any]) -> None:
    if set(task) != TASK_KEYS:
        denied = sorted(set(task) & DENIED_TASK_KEYS)
        suffix = f": {denied}" if denied else ""
        raise EditSessionError(f"task must contain exactly six non-answer fields{suffix}")
    if any(not isinstance(value, str) or not value for value in task.values()):
        raise EditSessionError("all task fields must be non-empty strings")
    if not re.fullmatch(r"[0-9a-f]{40}", task["base_commit"]):
        raise EditSessionError("base_commit must be a full immutable Git SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", task["repo"]):
        raise EditSessionError("invalid repository identity")


@dataclass(frozen=True)
class EditPolicy:
    max_turns: int = 8
    max_tool_calls: int = 16
    max_input_tokens: int = 24_000
    max_output_tokens: int = 8_192
    max_session_seconds: int = 900
    max_tree_files: int = 100_000
    max_materialized_bytes: int = 100_000_000
    max_file_bytes: int = 1_000_000
    max_list_entries: int = 200
    max_search_results: int = 50
    max_search_bytes: int = 30_000_000
    max_read_lines: int = 240
    max_read_bytes: int = 32_000
    max_edit_bytes: int = 128_000
    max_patch_bytes: int = 2_000_000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditPolicy:
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            raise EditSessionError("policy fields differ from the frozen schema")
        if any(type(value) is not int or value <= 0 for value in data.values()):
            raise EditSessionError("all policy values must be positive integers")
        return cls(**data)


def protocol_prompt(task: dict[str, Any], policy: EditPolicy) -> str:
    validate_task(task)
    return (
        "You are repairing a repository at one immutable base commit. Use exactly one JSON object "
        "per turn and no surrounding prose. Allowed actions:\n"
        '{"action":"list","path":"pkg"}\n'
        '{"action":"search","query":"literal text","path":"pkg"}\n'
        '{"action":"read","path":"pkg/file.py","start_line":1,"end_line":120}\n'
        '{"action":"edit","path":"pkg/file.py","old_text":"exact existing text",'
        '"new_text":"replacement text"}\n'
        '{"action":"finish"}\n'
        "Edits must uniquely match existing regular files. You cannot create/delete files, run "
        "commands, access the network, or read outside the exact base tree. The final unified diff "
        "is produced mechanically.\n"
        f"Limits: {policy.max_turns} turns, {policy.max_tool_calls} tools, "
        f"{policy.max_input_tokens} input tokens, {policy.max_output_tokens} output tokens.\n"
        f"Repository: {task['repo']}\nBase commit: {task['base_commit']}\n"
        f"Issue:\n{task['problem_statement']}"
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _workspace_diff(workspace: Path) -> bytes:
    return _git(
        workspace, "diff", "--binary", "--no-ext-diff", "--no-textconv",
        "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--",
    )


def _regular_blobs(source: Path, base: str, policy: EditPolicy) -> list[dict[str, Any]]:
    resolved = _git(source, "rev-parse", f"{base}^{{commit}}").decode().strip()
    if resolved != base:
        raise EditSessionError("base commit did not resolve exactly")
    entries = _git(source, "ls-tree", "-rlz", base).split(b"\0")
    if len(entries) - 1 > policy.max_tree_files:
        raise EditSessionError("source tree exceeds file-count budget")
    records: list[dict[str, Any]] = []
    total = 0
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid, raw_size = metadata.split()
        try:
            path = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if kind != b"blob" or mode not in (b"100644", b"100755") or not safe_repo_path(path):
            continue
        size = int(raw_size)
        if size > policy.max_file_bytes:
            continue
        total += size
        if total > policy.max_materialized_bytes:
            raise EditSessionError("source tree exceeds materialization byte budget")
        records.append({"path": path, "mode": mode.decode(), "oid": oid.decode(), "size": size})
    if not records:
        raise EditSessionError("no safe regular blobs found at base commit")
    return sorted(records, key=lambda record: record["path"])


def materialize_base(task: dict[str, Any], destination: Path, policy: EditPolicy) -> dict[str, Any]:
    """Copy exact regular Git blobs into a new inert repository and commit the snapshot."""
    validate_task(task)
    source = Path(task["repo_dir"])
    if destination.exists() and any(destination.iterdir()):
        raise EditSessionError("session workspace must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    records = _regular_blobs(source, task["base_commit"], policy)
    for record in records:
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _git(source, "cat-file", "blob", record["oid"])
        if len(data) != record["size"]:
            raise EditSessionError("Git blob size mismatch")
        target.write_bytes(data)
        target.chmod(0o755 if record["mode"] == "100755" else 0o644)
        record["sha256"] = sha256(data)
    _git(destination, "init", "-q")
    _git(destination, "add", "--all")
    _git(
        destination, "-c", "user.name=Reverse REAP", "-c", "user.email=invalid@example.invalid",
        "commit", "-qm", "exact-base-snapshot",
    )
    tree = _git(destination, "rev-parse", "HEAD^{tree}").decode().strip()
    return {"base_commit": task["base_commit"], "files": records, "snapshot_tree": tree}


def _target(workspace: Path, relative: str, *, allow_root: bool = False) -> Path:
    if not safe_repo_path(relative, allow_root=allow_root):
        raise EditSessionError("unsafe repository path")
    target = workspace if relative == "" else workspace / relative
    try:
        relative_to_root = target.resolve(strict=False).relative_to(workspace.resolve())
    except ValueError as exc:
        raise EditSessionError("path escapes session workspace") from exc
    if ".git" in relative_to_root.parts:
        raise EditSessionError("Git metadata is not accessible")
    return target


def _regular_file(workspace: Path, relative: str, policy: EditPolicy) -> Path:
    target = _target(workspace, relative)
    try:
        info = target.lstat()
    except FileNotFoundError as exc:
        raise EditSessionError("file does not exist") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or target.is_symlink()
        or info.st_size > policy.max_file_bytes
    ):
        raise EditSessionError("path is not an allowed regular file")
    return target


def _parse_action(raw: str) -> dict[str, Any]:
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EditSessionError("model response must be one JSON object") from exc
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise EditSessionError("model response must be an action object")
    name = action["action"]
    fields = {
        "list": {"action", "path"},
        "search": {"action", "query", "path"},
        "read": {"action", "path", "start_line", "end_line"},
        "edit": {"action", "path", "old_text", "new_text"},
        "finish": {"action"},
    }
    if name not in fields or set(action) != fields[name]:
        raise EditSessionError("unknown action or action fields differ from protocol")
    return action


class EditSession:
    """One deterministic session. The caller supplies tokenizer-measured token usage."""

    def __init__(self, root: Path, state: dict[str, Any], policy: EditPolicy):
        self.root = root
        self.workspace = root / "workspace"
        self.state_path = root / "state.json"
        self.state = state
        self.policy = policy

    @classmethod
    def create(
        cls,
        root: Path,
        task: dict[str, Any],
        policy: EditPolicy,
        *,
        run_id: str,
        config_sha256: str,
        model_revision: str,
        tokenizer_sha256: str,
    ) -> EditSession:
        validate_task(task)
        if root.exists() and any(root.iterdir()):
            raise EditSessionError("session root already contains data")
        if not run_id or not all(re.fullmatch(r"[0-9a-f]{64}", item) for item in (
            config_sha256, tokenizer_sha256,
        )):
            raise EditSessionError("run/config/tokenizer binding is incomplete")
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
            raise EditSessionError("model revision must be immutable")
        root.mkdir(parents=True, exist_ok=True)
        source = materialize_base(task, root / "workspace", policy)
        binding = {
            "run_id": run_id,
            "config_sha256": config_sha256,
            "task_sha256": sha256(canonical_json(task)),
            "model_revision": model_revision,
            "tokenizer_sha256": tokenizer_sha256,
            "policy_sha256": sha256(canonical_json(asdict(policy))),
            "base_commit": task["base_commit"],
            "source_snapshot_sha256": sha256(canonical_json(source)),
        }
        now = time.time()
        state = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "status": "RUNNING",
            "binding": binding,
            "task": {key: task[key] for key in sorted(TASK_KEYS) if key != "repo_dir"},
            "source": source,
            "started_unix": now,
            "updated_unix": now,
            "turns": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "transcript": [],
            "accepted_edits": [],
            "workspace_diff_sha256": sha256(b""),
            "artifacts": {},
        }
        session = cls(root, state, policy)
        session._save()
        return session

    @classmethod
    def open(cls, root: Path, policy: EditPolicy, expected_binding: dict[str, str]) -> EditSession:
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if state.get("protocol_version") != PROTOCOL_VERSION:
            raise EditSessionError("checkpoint protocol version mismatch")
        if any(state["binding"].get(key) != value for key, value in expected_binding.items()):
            raise EditSessionError("checkpoint binding drift")
        if state["binding"].get("policy_sha256") != sha256(canonical_json(asdict(policy))):
            raise EditSessionError("checkpoint policy drift")
        if state.get("workspace_diff_sha256") != sha256(_workspace_diff(root / "workspace")):
            raise EditSessionError("checkpoint workspace drift")
        return cls(root, state, policy)

    def _save(self) -> None:
        self.state["updated_unix"] = time.time()
        _atomic_json(self.state_path, self.state)

    def _check_budget(self, input_tokens: int, output_tokens: int) -> None:
        if type(input_tokens) is not int or type(output_tokens) is not int:
            raise EditSessionError("token counts must be integers")
        if input_tokens < 0 or output_tokens <= 0:
            raise EditSessionError("token counts must be non-negative input and positive output")
        if self.state["turns"] + 1 > self.policy.max_turns:
            raise EditSessionError("turn budget exhausted")
        if self.state["tool_calls"] + 1 > self.policy.max_tool_calls:
            raise EditSessionError("tool-call budget exhausted")
        if self.state["input_tokens"] + input_tokens > self.policy.max_input_tokens:
            raise EditSessionError("input-token budget exhausted")
        if self.state["output_tokens"] + output_tokens > self.policy.max_output_tokens:
            raise EditSessionError("output-token budget exhausted")
        if time.time() - self.state["started_unix"] > self.policy.max_session_seconds:
            raise EditSessionError("session wall-time budget exhausted")

    def _list(self, action: dict[str, Any]) -> dict[str, Any]:
        relative = action["path"]
        target = _target(self.workspace, relative, allow_root=True)
        if not target.is_dir() or target.is_symlink():
            raise EditSessionError("list target is not a directory")
        entries = []
        for child in sorted(target.iterdir(), key=lambda item: item.name):
            if child.name == ".git" or child.is_symlink():
                continue
            entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
            if len(entries) >= self.policy.max_list_entries:
                break
        return {"action": "list", "path": relative, "entries": entries,
                "truncated": len(entries) == self.policy.max_list_entries}

    def _search(self, action: dict[str, Any]) -> dict[str, Any]:
        query, relative = action["query"], action["path"]
        if not isinstance(query, str) or not query or len(query.encode()) > 512:
            raise EditSessionError("search query is empty or too large")
        target = _target(self.workspace, relative, allow_root=True)
        if not target.exists() or target.is_symlink():
            raise EditSessionError("search root does not exist")
        paths = [target] if target.is_file() else sorted(target.rglob("*"))
        results, scanned = [], 0
        for path in paths:
            if ".git" in path.parts or path.is_symlink() or not path.is_file():
                continue
            info = path.stat()
            if info.st_size > self.policy.max_file_bytes:
                continue
            scanned += info.st_size
            if scanned > self.policy.max_search_bytes:
                raise EditSessionError("search byte budget exhausted")
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, 1):
                if query in line:
                    results.append({"path": path.relative_to(self.workspace).as_posix(),
                                    "line": number, "text": line[:500]})
                    if len(results) >= self.policy.max_search_results:
                        return {"action": "search", "query": query, "results": results,
                                "truncated": True}
        return {"action": "search", "query": query, "results": results, "truncated": False}

    def _read(self, action: dict[str, Any]) -> dict[str, Any]:
        path = _regular_file(self.workspace, action["path"], self.policy)
        start, end = action["start_line"], action["end_line"]
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise EditSessionError("invalid read line range")
        if end - start + 1 > self.policy.max_read_lines:
            raise EditSessionError("read line budget exceeded")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise EditSessionError("file is not UTF-8 text") from exc
        selected = lines[start - 1:end]
        rendered = "\n".join(f"{index}: {line}" for index, line in enumerate(selected, start))
        if len(rendered.encode()) > self.policy.max_read_bytes:
            raise EditSessionError("read byte budget exceeded")
        return {"action": "read", "path": action["path"], "start_line": start,
                "end_line": start + len(selected) - 1, "content": rendered}

    def _edit(self, action: dict[str, Any]) -> dict[str, Any]:
        path = _regular_file(self.workspace, action["path"], self.policy)
        old, new = action["old_text"], action["new_text"]
        if not isinstance(old, str) or not isinstance(new, str) or not old or "\0" in old + new:
            raise EditSessionError("edit texts must be non-empty-match UTF-8 strings without NUL")
        if len(old.encode()) + len(new.encode()) > self.policy.max_edit_bytes:
            raise EditSessionError("edit byte budget exceeded")
        try:
            before = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise EditSessionError("edit target is not UTF-8 text") from exc
        occurrences = before.count(old)
        if occurrences != 1:
            raise EditSessionError(
                f"exact-match edit is ambiguous or absent: {occurrences} matches"
            )
        after = before.replace(old, new, 1)
        if len(after.encode()) > self.policy.max_file_bytes:
            raise EditSessionError("edited file exceeds file-size budget")
        info = path.stat()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(after)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(stat.S_IMODE(info.st_mode))
        temporary.replace(path)
        edit = {"path": action["path"], "old_sha256": sha256(old.encode()),
                "new_sha256": sha256(new.encode()), "before_sha256": sha256(before.encode()),
                "after_sha256": sha256(after.encode())}
        self.state["accepted_edits"].append(edit)
        return {"action": "edit", "path": action["path"], "accepted": True,
                "after_sha256": edit["after_sha256"]}

    def _finish(self) -> dict[str, Any]:
        if not self.state["accepted_edits"]:
            raise EditSessionError("cannot finish without an accepted edit")
        patch = _workspace_diff(self.workspace)
        if not patch or len(patch) > self.policy.max_patch_bytes:
            raise EditSessionError("mechanical patch is empty or exceeds patch budget")
        patch_path = self.root / "model.patch"
        patch_path.write_bytes(patch)
        # Check against an independently materialized exact-base snapshot.
        with tempfile.TemporaryDirectory(prefix="reverse-reap-apply-check-") as directory:
            task = dict(self.state["task"], repo_dir=str(self.root / "source-repo-link"))
            # The source path is intentionally kept outside the checkpoint task record.
            source_path = self.root / "source-path.txt"
            task["repo_dir"] = source_path.read_text(encoding="utf-8").strip()
            clean = Path(directory) / "repo"
            clean_source = materialize_base(task, clean, self.policy)
            if sha256(canonical_json(clean_source)) != self.state["binding"][
                "source_snapshot_sha256"
            ]:
                raise EditSessionError("independent apply-check source snapshot drift")
            _git(clean, "apply", "--check", "--cached", str(patch_path))
        self.state["status"] = "COMPLETE"
        return {"action": "finish", "status": "COMPLETE",
                "patch": {"path": patch_path.name, "sha256": sha256(patch)}}

    def process_turn(
        self, raw_response: str, *, input_tokens: int, output_tokens: int
    ) -> dict[str, Any]:
        if self.state["status"] != "RUNNING":
            raise EditSessionError("session is not running")
        self._check_budget(input_tokens, output_tokens)
        action = _parse_action(raw_response)
        handlers = {
            "list": self._list, "search": self._search, "read": self._read,
            "edit": self._edit, "finish": lambda _: self._finish(),
        }
        try:
            result = handlers[action["action"]](action)
        except Exception as exc:
            result = {"action": action["action"], "error": type(exc).__name__, "message": str(exc)}
        self.state["turns"] += 1
        self.state["tool_calls"] += 1
        self.state["input_tokens"] += input_tokens
        self.state["output_tokens"] += output_tokens
        record = {
            "turn": self.state["turns"], "model_response": raw_response,
            "model_response_sha256": sha256(raw_response.encode()),
            "input_tokens": input_tokens, "output_tokens": output_tokens, "tool_result": result,
        }
        record["record_sha256"] = sha256(canonical_json(record))
        self.state["transcript"].append(record)
        self.state["workspace_diff_sha256"] = sha256(_workspace_diff(self.workspace))
        if self.state["status"] == "COMPLETE":
            transcript_path = self.root / "transcript.jsonl"
            transcript = b"".join(canonical_json(row) for row in self.state["transcript"])
            transcript_path.write_bytes(transcript)
            self.state["artifacts"] = {
                "patch": result["patch"],
                "transcript": {"path": transcript_path.name, "sha256": sha256(transcript)},
            }
            result = {"action": "finish", "status": "COMPLETE",
                      "artifacts": self.state["artifacts"]}
        self._save()
        return result


def write_source_binding(root: Path, repo_dir: Path) -> None:
    """Write the local-only source path used to rebuild an independent apply-check tree."""
    root.mkdir(parents=True, exist_ok=True)
    path = repo_dir.resolve()
    if not path.exists():
        raise EditSessionError("source repository path does not exist")
    (root / "source-path.txt").write_text(str(path) + "\n", encoding="utf-8")
