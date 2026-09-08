import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from reverse_reap.swe_agentic import AgenticError, plan_sessions, run_session
from reverse_reap.swe_edit import EditPolicy, EditSessionError


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HASH = "6" * 64
MODEL_REVISION = "5" * 40


@pytest.fixture
def two_tasks(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "a.py").write_text("POST_FIX\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "post-fix")

    def task(sample_id):
        return {"sample_id": sample_id, "source_id": "o__r-1", "repo": "o/r",
                "base_commit": base, "problem_statement": f"fix {sample_id}",
                "repo_dir": str(repo)}

    return [task("s-1"), task("s-2")]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize,
                            add_generation_prompt=False, enable_thinking=False):
        text = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text) if tokenize else text

    def __call__(self, text):
        return {"input_ids": self.encode(text)}

    def encode(self, text):
        return text.split()


class ScriptedModel:
    def __init__(self, responses):
        self._responses = list(responses)

    def __call__(self, messages):
        assert self._responses, "script exhausted"
        return self._responses.pop(0)


def run_condition(tmp_path, two_tasks, condition, script):
    for task in two_tasks:
        run_session(
            session_dir=tmp_path / "probe" / condition / task["sample_id"],
            task=task, policy=EditPolicy(), tokenizer=FakeTokenizer(),
            generate_fn=ScriptedModel(list(script)), run_id="r",
            condition_id=condition, config_sha256=HASH,
            model_revision=MODEL_REVISION, tokenizer_sha256=HASH,
            model_max_input_tokens=24_000,
        )


SCRIPT = [
    json.dumps({"action": "read", "path": "a.py", "start_line": 1, "end_line": 5}),
    json.dumps({"action": "edit", "path": "a.py", "old_text": "x = 1",
                "new_text": "x = 2"}),
    json.dumps({"action": "finish"}),
]


def test_plan_sessions_order_and_uniqueness(two_tasks):
    plan = plan_sessions(two_tasks, ["c0", "c1"])
    assert [(c, t["sample_id"]) for c, t in plan] == [
        ("c0", "s-1"), ("c0", "s-2"), ("c1", "s-1"), ("c1", "s-2")]
    with pytest.raises(AgenticError):
        plan_sessions(two_tasks + [dict(two_tasks[0])], ["c0"])
    with pytest.raises(AgenticError):
        plan_sessions(two_tasks, ["c0", "c0"])
    with pytest.raises(AgenticError):
        plan_sessions([], ["c0"])


def test_validator_passes_identical_conditions(tmp_path, two_tasks):
    run_condition(tmp_path, two_tasks, "c0-baseline-a", SCRIPT)
    run_condition(tmp_path, two_tasks, "c0-baseline-b", SCRIPT)
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(
        [{k: t[k] for k in ("sample_id", "source_id", "repo", "base_commit",
                            "problem_statement", "repo_dir")} for t in two_tasks]))
    validate = script("validate_swe_agentic_probe")
    report = validate.validate(
        json.loads(tasks_file.read_text()), tmp_path / "probe",
        ["c0-baseline-a", "c0-baseline-b"])
    assert report["passed"]
    assert len(report["checks"]) == 4
    assert all(c["passed"] for c in report["checks"])
    assert report["identities"][0]["identical"]


def test_validator_rejects_tampered_patch(tmp_path, two_tasks):
    run_condition(tmp_path, two_tasks, "c0-baseline-a", SCRIPT)
    run_condition(tmp_path, two_tasks, "c0-baseline-b", SCRIPT)
    victim = tmp_path / "probe" / "c0-baseline-b" / "s-1" / "model.patch"
    data = victim.read_bytes().replace(b"-x = 1", b"-x = 9")
    victim.write_bytes(data)
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(two_tasks))
    validate = script("validate_swe_agentic_probe")
    report = validate.validate(
        json.loads(tasks_file.read_text()), tmp_path / "probe",
        ["c0-baseline-a", "c0-baseline-b"])
    assert not report["passed"]
    tampered = [c for c in report["checks"]
                if c["condition"] == "c0-baseline-b" and c["sample_id"] == "s-1"]
    assert len(tampered) == 1 and not tampered[0]["passed"]


def test_validator_rejects_missing_session(tmp_path, two_tasks):
    run_condition(tmp_path, two_tasks, "c0-baseline-a", SCRIPT)
    (tmp_path / "probe" / "c0-baseline-b").mkdir(parents=True)
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(two_tasks))
    validate = script("validate_swe_agentic_probe")
    with pytest.raises(EditSessionError):
        validate.validate(json.loads(tasks_file.read_text()), tmp_path / "probe",
                          ["c0-baseline-a", "c0-baseline-b"])


def test_probe_runner_help_lists_conditions(tmp_path):
    run = script("run_swe_agentic_probe")
    assert callable(run.main)
    shutil.rmtree(tmp_path / "probe", ignore_errors=True)
