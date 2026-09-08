import copy
import json
import subprocess

import pytest

from reverse_reap.swe_agentic import (
    SessionDriver,
    run_session,
    summaries_equal,
)
from reverse_reap.swe_edit import (
    EditPolicy,
    EditSession,
    EditSessionError,
    canonical_json,
    sha256,
    write_source_binding,
)

MODEL_REVISION = "5" * 40
HASH = "6" * 64
CONDITIONS = ["c0-baseline-a", "c0-baseline-b", "c0-noop-masked", "c2-selected"]


class FakeTokenizer:
    """Deterministic stand-in with the exact donor-tokenizer interface."""

    def apply_chat_template(self, messages, tokenize,
                            add_generation_prompt=False, enable_thinking=False):
        assert enable_thinking is False
        text = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text) if tokenize else text

    def __call__(self, text):
        return {"input_ids": self.encode(text)}

    def encode(self, text):
        return text.split()


class ScriptedModel:
    """Canned responses in order; records the messages of every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        assert self._responses, "script exhausted"
        return self._responses.pop(0)


def action(**fields):
    return json.dumps(fields)


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
    (repo / "pkg" / "other.py").write_text("def helper():\n    return 0\n", encoding="utf-8")
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


def drive(tmp_path, source_task, script, condition_id="c0-baseline-a",
          policy=None, name="session"):
    repo, task = source_task
    policy = policy or EditPolicy()
    summary = run_session(
        session_dir=tmp_path / name, task=task, policy=policy,
        tokenizer=FakeTokenizer(), generate_fn=ScriptedModel(script),
        run_id="run-1", condition_id=condition_id, config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
        model_max_input_tokens=24_000,
    )
    return summary


SUCCESS_SCRIPT = [
    action(action="list", path="pkg"),
    action(action="search", query="widget", path="pkg"),
    action(action="read", path="pkg/core.py", start_line=1, end_line=10),
    action(action="edit", path="pkg/core.py",
           old_text="    value = 1", new_text="    value = 2"),
    action(action="finish"),
]


def test_successful_session(tmp_path, source_task):
    summary = drive(tmp_path, source_task, SUCCESS_SCRIPT)
    assert summary.status == "COMPLETE"
    assert summary.patch_sha256 and summary.transcript_sha256
    assert summary.turns == 5 and summary.tool_calls == 5
    assert summary.error is None


def test_multiple_edits(tmp_path, source_task):
    script = [
        action(action="read", path="pkg/core.py", start_line=1, end_line=10),
        action(action="edit", path="pkg/core.py",
               old_text="    value = 1", new_text="    value = 2"),
        action(action="edit", path="pkg/other.py",
               old_text="    return 0", new_text="    return 1"),
        action(action="finish"),
    ]
    summary = drive(tmp_path, source_task, script)
    assert summary.status == "COMPLETE" and summary.turns == 4


def test_invalid_json_fail_closed(tmp_path, source_task):
    summary = drive(tmp_path, source_task, ["not json"])
    assert summary.status == "ERROR" and summary.patch_sha256 is None


def test_prose_markdown_wrapped_json_rejected(tmp_path, source_task):
    script = ['Here is my action:\n```json\n{"action":"finish"}\n```']
    summary = drive(tmp_path, source_task, script)
    assert summary.status == "ERROR"


def test_ambiguous_edit_rejected_and_empty_patch_fails(tmp_path, source_task):
    repo, task = source_task
    (repo / "pkg" / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    subprocess.check_output(["git", "-C", str(repo), "add", "."])
    subprocess.check_output(["git", "-C", str(repo), "commit", "-qm", "dup"])
    task = dict(task, base_commit=subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip())
    script = [
        action(action="edit", path="pkg/dup.py", old_text="x = 1", new_text="x = 2"),
        action(action="finish"),
        action(action="finish"),
    ]
    summary = run_session(
        session_dir=tmp_path / "session", task=task, policy=EditPolicy(),
        tokenizer=FakeTokenizer(), generate_fn=ScriptedModel(script),
        run_id="run-1", condition_id="c0-baseline-a", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
        model_max_input_tokens=24_000,
    )
    assert summary.status == "ERROR" and summary.patch_sha256 is None


def test_unsupported_action_and_traversal_fail_closed(tmp_path, source_task):
    unsupported = drive(tmp_path, source_task,
                        [action(action="run", command="evil")], name="s-run")
    assert unsupported.status == "ERROR" and unsupported.turns == 0
    traversal = drive(
        tmp_path, source_task,
        [action(action="read", path="../evil.py", start_line=1, end_line=5),
         action(action="read", path=".git/HEAD", start_line=1, end_line=5),
         action(action="finish")],
        name="s-trav",
    )
    assert traversal.status == "ERROR" and traversal.patch_sha256 is None
    assert "error" in traversal.history[0]["result"]
    assert "error" in traversal.history[1]["result"]
    # A traversal probe does not poison a fresh session.
    assert drive(tmp_path, source_task, SUCCESS_SCRIPT, name="clean").status == "COMPLETE"


def test_budget_exhaustion(tmp_path, source_task):
    policy = EditPolicy(max_turns=2, max_tool_calls=16, max_input_tokens=24_000,
                        max_output_tokens=8_192, max_session_seconds=900,
                        max_tree_files=100_000, max_materialized_bytes=100_000_000,
                        max_file_bytes=1_000_000, max_list_entries=200,
                        max_search_results=50, max_search_bytes=30_000_000,
                        max_read_lines=240, max_read_bytes=32_000,
                        max_edit_bytes=128_000, max_patch_bytes=2_000_000)
    summary = drive(tmp_path, source_task, SUCCESS_SCRIPT, policy=policy)
    assert summary.status == "ERROR" and "budget" in (summary.error or "")


def test_repeated_identical_errors_stop(tmp_path, source_task):
    bad = action(action="read", path="pkg/missing.py", start_line=1, end_line=5)
    summary = drive(tmp_path, source_task, [bad, bad, bad])
    assert summary.status == "ERROR"
    assert summary.stop_reason == "repeated-tool-error"
    assert summary.turns == 2


def test_resume_after_interruption(tmp_path, source_task):
    repo, task = source_task
    policy = EditPolicy()
    tokenizer = FakeTokenizer()

    class CrashingModel(ScriptedModel):
        def __call__(self, messages):
            response = super().__call__(messages)
            if len(self.calls) == 2:
                raise RuntimeError("simulated crash")
            return response

    root = tmp_path / "session"
    write_source_binding(root, repo)
    binding_raw = (root / "source-path.txt").read_bytes()
    (root / "source-path.txt").unlink()
    session = EditSession.create(
        root, task, policy, run_id="run-1", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
    )
    (root / "source-path.txt").write_bytes(binding_raw)
    driver = SessionDriver(session, policy, tokenizer, condition_id="c0-baseline-a",
                           run_id="run-1", model_max_input_tokens=24_000)
    with pytest.raises(RuntimeError, match="simulated crash"):
        driver.run(CrashingModel(SUCCESS_SCRIPT))
    binding = json.loads((root / "state.json").read_text())["binding"]
    resumed = run_session(
        session_dir=root, task=task, policy=policy, tokenizer=tokenizer,
        generate_fn=ScriptedModel(SUCCESS_SCRIPT[1:]), run_id="run-1",
        condition_id="c0-baseline-a", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
        model_max_input_tokens=24_000, expected_binding=binding,
    )
    fresh = drive(tmp_path, source_task, SUCCESS_SCRIPT, name="fresh")
    assert resumed.status == "COMPLETE"
    assert resumed.patch_sha256 == fresh.patch_sha256
    assert resumed.transcript_sha256 == fresh.transcript_sha256


def test_deterministic_replay(tmp_path, source_task):
    first = drive(tmp_path, source_task, SUCCESS_SCRIPT, name="a")
    second = drive(tmp_path, source_task, SUCCESS_SCRIPT, name="b")
    assert first.patch_sha256 == second.patch_sha256
    assert first.transcript_sha256 == second.transcript_sha256


def test_baseline_noop_plumbing_identical(tmp_path, source_task):
    baseline = drive(tmp_path, source_task, SUCCESS_SCRIPT,
                     condition_id="c0-baseline-a", name="a")
    repeat = drive(tmp_path, source_task, SUCCESS_SCRIPT,
                   condition_id="c0-baseline-b", name="b")
    noop = drive(tmp_path, source_task, SUCCESS_SCRIPT,
                 condition_id="c0-noop-masked", name="n")
    assert baseline.status == noop.status == "COMPLETE"
    assert summaries_equal(baseline, repeat)
    assert summaries_equal(baseline, noop)
    assert repeat.condition_id == "c0-baseline-b"


def test_selected_intervention_plumbing(tmp_path, source_task):
    baseline = drive(tmp_path, source_task, SUCCESS_SCRIPT,
                     condition_id="c0-baseline-a", name="a")
    selected_script = [
        action(action="read", path="pkg/core.py", start_line=1, end_line=10),
        action(action="edit", path="pkg/core.py",
               old_text="    value = 1", new_text="    value = 3"),
        action(action="finish"),
    ]
    selected = drive(tmp_path, source_task, selected_script,
                     condition_id="c2-selected", name="s")
    assert selected.condition_id == "c2-selected"
    assert selected.status == "COMPLETE"
    assert selected.patch_sha256 != baseline.patch_sha256


def test_exact_tokenizer_accounting(tmp_path, source_task):
    repo, task = source_task
    tokenizer = FakeTokenizer()
    model = ScriptedModel(SUCCESS_SCRIPT)
    summary = run_session(
        session_dir=tmp_path / "session", task=task, policy=EditPolicy(),
        tokenizer=tokenizer, generate_fn=model,
        run_id="run-1", condition_id="c0-baseline-a", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
        model_max_input_tokens=24_000,
    )
    assert summary.status == "COMPLETE"
    assert len(model.calls) == 5
    for call, record in zip(model.calls, summary.history, strict=True):
        rendered = tokenizer.apply_chat_template(
            call, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        expected_in = len(tokenizer(rendered)["input_ids"])
        assert record["input_tokens"] == expected_in
    assert summary.input_tokens == sum(r["input_tokens"] for r in summary.history)
    assert summary.output_tokens == sum(r["output_tokens"] for r in summary.history)


def test_mechanical_patch_applicability(tmp_path, source_task):
    repo, task = source_task
    summary = drive(tmp_path, source_task, SUCCESS_SCRIPT)
    assert summary.status == "COMPLETE"
    patch = tmp_path / "session" / "model.patch"
    assert patch.exists()
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", task["base_commit"]])
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (tmp_path / "base.tar").write_bytes(archive)
    subprocess.check_output(["tar", "-xf", str(tmp_path / "base.tar"), "-C", str(fresh)])
    result = subprocess.run(["git", "apply", "--check", str(patch)], cwd=str(fresh),
                            capture_output=True)
    assert result.returncode == 0, result.stderr.decode()[:500]


def test_model_context_limit(tmp_path, source_task):
    repo, task = source_task
    policy = EditPolicy()
    summary = run_session(
        session_dir=tmp_path / "session", task=task, policy=policy,
        tokenizer=FakeTokenizer(), generate_fn=ScriptedModel(SUCCESS_SCRIPT),
        run_id="run-1", condition_id="c0-baseline-a", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
        model_max_input_tokens=1,
    )
    assert summary.status == "ERROR" and "input limit" in (summary.error or "")


def test_checkpoint_drift_rejected(tmp_path, source_task):
    repo, task = source_task
    policy = EditPolicy()
    root = tmp_path / "session"
    write_source_binding(root, repo)
    binding_raw = (root / "source-path.txt").read_bytes()
    (root / "source-path.txt").unlink()
    session = EditSession.create(
        root, task, policy, run_id="run-1", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
    )
    (root / "source-path.txt").write_bytes(binding_raw)
    driver = SessionDriver(session, policy, FakeTokenizer(),
                           condition_id="c0-baseline-a", run_id="run-1",
                           model_max_input_tokens=24_000)
    incomplete = driver.run(ScriptedModel([
        action(action="list", path="pkg"),
        "not json",
    ]))
    assert incomplete.status == "ERROR" and incomplete.turns == 1
    binding = json.loads((root / "state.json").read_text())["binding"]
    tampered = dict(binding, run_id="other-run")
    with pytest.raises(EditSessionError):
        run_session(
            session_dir=root, task=task, policy=policy,
            tokenizer=FakeTokenizer(), generate_fn=ScriptedModel(SUCCESS_SCRIPT[1:]),
            run_id="other-run", condition_id="c0-baseline-a", config_sha256=HASH,
            model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
            model_max_input_tokens=24_000, expected_binding=tampered,
        )


def test_empty_patch_rejected_without_edit(tmp_path, source_task):
    # A premature finish is a recoverable tool error; repeating it identically
    # stops the session fail-closed with no patch.
    summary = drive(tmp_path, source_task,
                    [action(action="finish"), action(action="finish")])
    assert summary.status == "ERROR" and summary.patch_sha256 is None


def test_error_streak_survives_resume(tmp_path, source_task):
    repo, task = source_task
    policy = EditPolicy()
    tokenizer = FakeTokenizer()
    bad = action(action="read", path="pkg/missing.py", start_line=1, end_line=5)

    class CrashOnce(ScriptedModel):
        def __call__(self, messages):
            response = super().__call__(messages)
            if len(self.calls) == 2:
                raise RuntimeError("simulated crash")
            return response

    root = tmp_path / "session"
    write_source_binding(root, repo)
    binding_raw = (root / "source-path.txt").read_bytes()
    (root / "source-path.txt").unlink()
    session = EditSession.create(
        root, task, policy, run_id="run-1", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
    )
    (root / "source-path.txt").write_bytes(binding_raw)
    driver = SessionDriver(session, policy, tokenizer, condition_id="c0-baseline-a",
                           run_id="run-1", model_max_input_tokens=24_000)
    with pytest.raises(RuntimeError, match="simulated crash"):
        driver.run(CrashOnce([bad, bad]))
    binding = json.loads((root / "state.json").read_text())["binding"]
    resumed = run_session(
        session_dir=root, task=task, policy=policy, tokenizer=tokenizer,
        generate_fn=ScriptedModel([bad]), run_id="run-1",
        condition_id="c0-baseline-a", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
        model_max_input_tokens=24_000, expected_binding=binding,
    )
    # One identical error before the crash plus one after resumes the streak.
    assert resumed.status == "ERROR"
    assert resumed.stop_reason == "repeated-tool-error"
    assert resumed.turns == 2


def test_completed_session_does_not_call_model_again(tmp_path, source_task):
    first = drive(tmp_path, source_task, SUCCESS_SCRIPT)
    assert first.status == "COMPLETE"
    repo, task = source_task
    policy = EditPolicy()
    root = tmp_path / "session"
    binding = json.loads((root / "state.json").read_text())["binding"]
    model = ScriptedModel([action(action="finish")])
    resumed = run_session(
        session_dir=root, task=task, policy=policy,
        tokenizer=FakeTokenizer(), generate_fn=model,
        run_id="run-1", condition_id="c0-baseline-a", config_sha256=HASH,
        model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
        model_max_input_tokens=24_000, expected_binding=binding,
    )
    assert resumed.status == "ERROR" and resumed.stop_reason == "already-terminal"
    assert model.calls == []


def test_canonical_transcript_hashes(tmp_path, source_task):
    summary = drive(tmp_path, source_task, SUCCESS_SCRIPT)
    transcript = (tmp_path / "session" / "transcript.jsonl").read_bytes()
    assert sha256(transcript) == summary.transcript_sha256
    assert sha256(canonical_json(json.loads(transcript.splitlines()[0]))) is not None
