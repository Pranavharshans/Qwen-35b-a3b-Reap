import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from reverse_reap.localized_repair import (
    BUGSINPY_REVISION,
    LocalizedRepairError,
    LocalizedTask,
    extract_full_file,
    merge_report,
    read_buggy_file,
    render_prompt,
    response_to_patch,
    task_to_sample,
)


def _script(name):
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def localized_task(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "logic.py").write_text("def value():\n    return 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "buggy")
    buggy = _git(repo, "rev-parse", "HEAD")
    (repo / "pkg" / "logic.py").write_text("POST_FIX_SECRET = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixed")
    return LocalizedTask(
        task_id="fixture-1",
        project="fixture",
        bug_id="1",
        repository="https://github.com/example/fixture",
        buggy_commit=buggy,
        target_path="pkg/logic.py",
        failing_test_command=("python", "-m", "pytest", "tests/test_logic.py"),
        failure_output="assert 1 == 2",
        repo_dir=str(repo),
        source_revision=BUGSINPY_REVISION,
    )


def test_exact_buggy_blob_and_prompt_do_not_read_postfix(localized_task):
    source = read_buggy_file(localized_task)
    assert source == "def value():\n    return 1\n"
    assert "POST_FIX_SECRET" not in source
    prompt = render_prompt(localized_task, source)
    assert "oracle-localized benchmark condition" in prompt
    assert "complete replacement contents" in prompt
    assert "POST_FIX_SECRET" not in prompt


def test_full_file_becomes_git_applicable_patch(localized_task):
    result = response_to_patch(localized_task, "def value():\n    return 2\n")
    assert result["target_path"] == "pkg/logic.py"
    assert "-    return 1" in result["patch"]
    assert "+    return 2" in result["patch"]
    assert len(result["patch_sha256"]) == 64


@pytest.mark.parametrize(
    "response", ["", "```python\ndef value(): return 2\n```", "not python !!!"]
)
def test_invalid_full_file_fails_closed(response):
    with pytest.raises(LocalizedRepairError):
        extract_full_file(response)


@pytest.mark.parametrize("path", ["../x.py", "/x.py", ".git/config", "pkg\\x.py", "x.txt"])
def test_unsafe_or_non_python_target_rejected(localized_task, path):
    changed = localized_task.model_copy(update={"target_path": path})
    with pytest.raises(LocalizedRepairError):
        changed.validate_boundary()


def test_sample_is_explicitly_oracle_localized(localized_task):
    sample = task_to_sample(localized_task)
    assert sample.scorer == "bugsinpy"
    assert sample.split == "validation"
    assert sample.stratum == "repository-bug-repair"
    assert sample.prompt_template_version == "bugsinpy-oracle-localized-v1"


def test_merge_retains_incomplete_rows_in_denominator(tmp_path):
    evaluation = tmp_path / "eval.jsonl"
    report = tmp_path / "report.json"
    destination = tmp_path / "merged.jsonl"
    rows = [
        {"source_id": "a", "scorer": "bugsinpy"},
        {"source_id": "b", "scorer": "bugsinpy"},
        {"source_id": "control", "scorer": "exact_match", "scoreable": True,
         "passed": True},
    ]
    evaluation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report.write_text(json.dumps({"completed_ids": ["a"], "resolved_ids": ["a"],
                                  "error_ids": ["b"]}))
    summary = merge_report(evaluation, report, destination)
    merged = [json.loads(line) for line in destination.read_text().splitlines()]
    assert merged[0]["passed"] is True
    assert merged[1]["scoreable"] is False
    assert summary["scoreable_fraction"] == pytest.approx(2 / 3)
    assert summary["passed_gate_b_scoreability"] is False


def test_answer_bearing_fields_are_rejected(localized_task):
    payload = localized_task.model_dump()
    payload["gold_patch"] = "secret"
    with pytest.raises(ValidationError):
        LocalizedTask.model_validate(payload)


def test_harness_is_networkless_digest_pinned_and_distinguishes_test_failure(monkeypatch):
    harness = _script("run_bugsinpy_localized_harness")
    image = {
        "task_id": "fixture-1",
        "image": "fixture@sha256:" + "a" * 64,
        "repository_path": "/opt/repo",
        "test_command": ["python", "-m", "pytest", "-q"],
        "image_source_sha256": "b" * 64,
    }
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command, 1, stdout=harness.PATCH_APPLIED + "\nfailed\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = harness.run_task(harness.TaskImage.model_validate(image), "diff --git a/x b/x\n")
    assert result["completed"] is True
    assert result["resolved"] is False
    assert "--network=none" in captured["command"]
    assert "--read-only" in captured["command"]
    assert image["image"] in captured["command"]


def test_harness_patch_apply_failure_is_unscoreable(monkeypatch):
    harness = _script("run_bugsinpy_localized_harness")
    image = harness.TaskImage.model_validate({
        "task_id": "fixture-1", "image": "fixture@sha256:" + "a" * 64,
        "repository_path": "/opt/repo", "test_command": ["pytest"],
        "image_source_sha256": "b" * 64,
    })
    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 2, "", "bad patch"),
    )
    result = harness.run_task(image, "bad")
    assert result["completed"] is False
    assert result["resolved"] is False


def test_replacement_manifest_changes_exactly_eight_swe_rows(localized_task, tmp_path):
    prepare = _script("prepare_bugsinpy_localized")
    base = tmp_path / "base.jsonl"
    tasks_path = tmp_path / "tasks.json"
    output = tmp_path / "replacement.jsonl"
    provenance = tmp_path / "provenance.json"
    untouched = b'{"sample_id":"control","scorer":"exact_match","split":"validation"}\n'
    swe_rows = [
        json.dumps({"sample_id": f"old-{index}", "scorer": "swebench",
                    "split": "validation"}, sort_keys=True).encode() + b"\n"
        for index in range(8)
    ]
    base.write_bytes(untouched + b"".join(swe_rows))
    tasks = [
        localized_task.model_copy(
            update={"task_id": f"fixture-{index}", "bug_id": str(index)}
        ).model_dump(mode="json")
        for index in range(8)
    ]
    tasks_path.write_text(json.dumps(tasks, sort_keys=True))
    approved = prepare.sha256(tasks_path.read_bytes())
    report = prepare.prepare(base, tasks_path, approved, output, provenance)
    assert report["replaced_rows"] == 8
    assert report["unaffected_bytes_identical"] is True
    assert output.read_bytes().startswith(untouched)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(row.get("scorer") == "bugsinpy" for row in rows) == 8
    assert not any(row.get("scorer") == "swebench" for row in rows)
