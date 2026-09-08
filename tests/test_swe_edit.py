import json
import os
import subprocess
import sys

import pytest

from reverse_reap.swe_edit import (
    EditPolicy,
    EditSession,
    EditSessionError,
    canonical_json,
    protocol_prompt,
    safe_repo_path,
    sha256,
    validate_task,
    write_source_binding,
)

MODEL_REVISION = "a" * 40
HASH = "b" * 64


@pytest.fixture
def source_task(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "core.py").write_text(
        "def widget():\n    value = 1\n    return value\n", encoding="utf-8"
    )
    (repo / "pkg" / "repeat.py").write_text("same\nsame\n", encoding="utf-8")
    (repo / "outside.txt").write_text("not python but legitimate source\n", encoding="utf-8")
    (repo / "link.py").symlink_to("pkg/core.py")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "pkg" / "core.py").write_text("POST_FIX_SECRET\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "post-fix")
    task = {
        "sample_id": "sample-1",
        "source_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": base,
        "problem_statement": "widget should return two",
        "repo_dir": str(repo),
    }
    return repo, task


def create_session(tmp_path, source_task, policy=None):
    repo, task = source_task
    root = tmp_path / "session"
    policy = policy or EditPolicy()
    session = EditSession.create(
        root, task, policy, run_id="run-1", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
    )
    write_source_binding(root, repo)
    return session, policy, task


def turn(session, action, input_tokens=10, output_tokens=10):
    return session.process_turn(
        json.dumps(action), input_tokens=input_tokens, output_tokens=output_tokens,
    )


def test_protocol_and_task_allowlist(source_task):
    _, task = source_task
    prompt = protocol_prompt(task, EditPolicy())
    assert '"action":"edit"' in prompt
    assert "run commands" in prompt and task["problem_statement"] in prompt
    validate_task(task)
    with pytest.raises(EditSessionError, match="six non-answer"):
        validate_task(dict(task, patch="gold"))
    with pytest.raises(EditSessionError, match="six non-answer"):
        validate_task({key: value for key, value in task.items() if key != "source_id"})


@pytest.mark.parametrize(
    "path", ["../secret", "/etc/passwd", "a/../b", ".git/config", "a\\b", "bad\npath"],
)
def test_unsafe_paths_rejected(path):
    assert not safe_repo_path(path)


def test_exact_base_materialization_does_not_use_checkout_or_symlink(tmp_path, source_task):
    session, _, _ = create_session(tmp_path, source_task)
    core = (session.workspace / "pkg/core.py").read_text(encoding="utf-8")
    assert "return value" in core and "POST_FIX_SECRET" not in core
    assert not (session.workspace / "link.py").exists()
    assert session.state["source"]["files"] == sorted(
        session.state["source"]["files"], key=lambda row: row["path"]
    )
    assert all("sha256" in row and "oid" in row for row in session.state["source"]["files"])


def test_list_search_read_are_sorted_bounded_and_hide_git(tmp_path, source_task):
    session, _, _ = create_session(tmp_path, source_task)
    listed = turn(session, {"action": "list", "path": ""})
    assert [row["name"] for row in listed["entries"]] == ["outside.txt", "pkg"]
    searched = turn(session, {"action": "search", "query": "widget", "path": ""})
    assert searched["results"][0]["path"] == "pkg/core.py"
    read = turn(
        session, {"action": "read", "path": "pkg/core.py", "start_line": 1, "end_line": 2}
    )
    assert read["content"] == "1: def widget():\n2:     value = 1"


def test_exact_match_edit_exports_mechanical_applicable_patch(tmp_path, source_task):
    session, _, task = create_session(tmp_path, source_task)
    edited = turn(session, {
        "action": "edit",
        "path": "pkg/core.py",
        "old_text": "    value = 1\n",
        "new_text": "    value = 2\n",
    })
    assert edited["accepted"]
    finished = turn(session, {"action": "finish"})
    assert finished["status"] == "COMPLETE"
    patch = (session.root / "model.patch").read_text(encoding="utf-8")
    assert "diff --git a/pkg/core.py b/pkg/core.py" in patch
    assert "-    value = 1" in patch and "+    value = 2" in patch
    state = json.loads(session.state_path.read_text(encoding="utf-8"))
    assert state["binding"]["base_commit"] == task["base_commit"]
    assert state["artifacts"]["patch"]["sha256"] == sha256(patch.encode())
    transcript = (session.root / "transcript.jsonl").read_bytes()
    assert state["artifacts"]["transcript"]["sha256"] == sha256(transcript)
    assert len(transcript.splitlines()) == 2


def test_ambiguous_absent_symlink_and_command_actions_fail_closed(tmp_path, source_task):
    session, _, _ = create_session(tmp_path, source_task)
    ambiguous = turn(session, {
        "action": "edit", "path": "pkg/repeat.py", "old_text": "same\n", "new_text": "x\n",
    })
    assert ambiguous["error"] == "EditSessionError"
    missing = turn(session, {
        "action": "edit", "path": "link.py", "old_text": "value", "new_text": "x",
    })
    assert missing["error"] == "EditSessionError"
    with pytest.raises(EditSessionError, match="unknown action"):
        session.process_turn(
            '{"action":"shell","command":"curl example.com"}',
            input_tokens=1, output_tokens=1,
        )


def test_token_turn_tool_and_time_budgets(tmp_path, source_task, monkeypatch):
    policy = EditPolicy(max_turns=1, max_tool_calls=1, max_input_tokens=3, max_output_tokens=3)
    session, _, _ = create_session(tmp_path, source_task, policy)
    turn(session, {"action": "list", "path": ""}, input_tokens=3, output_tokens=3)
    with pytest.raises(EditSessionError, match="turn budget"):
        turn(session, {"action": "list", "path": ""}, input_tokens=1, output_tokens=1)

    other, _, _ = create_session(tmp_path / "other", source_task, EditPolicy(max_session_seconds=1))
    monkeypatch.setattr("reverse_reap.swe_edit.time.time", lambda: other.state["started_unix"] + 2)
    with pytest.raises(EditSessionError, match="wall-time"):
        turn(other, {"action": "list", "path": ""})


def test_checkpoint_binding_and_policy_drift_rejected(tmp_path, source_task):
    session, policy, _ = create_session(tmp_path, source_task)
    binding = session.state["binding"]
    reopened = EditSession.open(session.root, policy, binding)
    assert reopened.state["binding"] == binding
    with pytest.raises(EditSessionError, match="binding drift"):
        EditSession.open(session.root, policy, {**binding, "run_id": "different"})
    with pytest.raises(EditSessionError, match="policy drift"):
        EditSession.open(session.root, EditPolicy(max_turns=7), binding)
    (session.workspace / "pkg/core.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(EditSessionError, match="workspace drift"):
        EditSession.open(session.root, policy, binding)


def test_deterministic_replay_produces_same_patch(tmp_path, source_task):
    patches = []
    for index in range(2):
        session, _, _ = create_session(tmp_path / str(index), source_task)
        turn(session, {
            "action": "edit", "path": "pkg/core.py",
            "old_text": "    value = 1\n", "new_text": "    value = 2\n",
        })
        turn(session, {"action": "finish"})
        patches.append((session.root / "model.patch").read_bytes())
    assert patches[0] == patches[1]


def test_source_binding_not_committed_to_checkpoint(tmp_path, source_task):
    repo, task = source_task
    session, _, _ = create_session(tmp_path, source_task)
    state_bytes = session.state_path.read_bytes()
    assert str(repo).encode() not in state_bytes
    assert task["problem_statement"].encode() in state_bytes
    assert (session.root / "source-path.txt").read_text().strip() == str(repo.resolve())


def test_canonical_hash_stable():
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    assert sha256(canonical_json({"b": 2, "a": 1})) == sha256(canonical_json({"a": 1, "b": 2}))


def test_materialized_files_are_not_world_writable(tmp_path, source_task):
    session, _, _ = create_session(tmp_path, source_task)
    mode = (session.workspace / "pkg/core.py").stat().st_mode
    assert mode & 0o002 == 0
    assert os.path.isdir(session.workspace / ".git")


def test_cli_start_and_step(tmp_path, source_task):
    _, task = source_task
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(EditPolicy().__dict__), encoding="utf-8")
    session_dir = tmp_path / "cli-session"
    start = subprocess.run(
        [
            sys.executable, "scripts/run_swe_edit_session.py", "start",
            "--session-dir", str(session_dir), "--task", str(task_path),
            "--policy", str(policy_path), "--run-id", "cli-run",
            "--config-sha256", HASH, "--model-revision", MODEL_REVISION,
            "--tokenizer-sha256", HASH,
        ],
        capture_output=True, text=True, check=False,
    )
    assert start.returncode == 0, start.stderr
    response = tmp_path / "response.json"
    response.write_text('{"action":"list","path":""}', encoding="utf-8")
    step = subprocess.run(
        [
            sys.executable, "scripts/run_swe_edit_session.py", "step",
            "--session-dir", str(session_dir), "--policy", str(policy_path),
            "--expected-binding", str(session_dir / "binding.json"),
            "--response", str(response), "--input-tokens", "10", "--output-tokens", "5",
        ],
        capture_output=True, text=True, check=False,
    )
    assert step.returncode == 0, step.stderr
    assert json.loads(step.stdout)["entries"]
