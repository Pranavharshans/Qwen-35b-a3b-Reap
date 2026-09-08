import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from reverse_reap.swe_context import (
    POLICY,
    VERSION,
    ContextError,
    build_context,
    digest,
    render_prompt,
    safe_path,
    validate_prompt_contract,
)


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def task(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()
    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (repo / "engine.py").write_text("def widget():\n    return 1\n")
    (repo / "test_engine.py").write_text("GOLD_SECRET = 'widget'\n")
    (repo / "symlink.py").symlink_to("test_engine.py")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "engine.py").write_text("POST_FIX_SECRET = 'widget'\n")
    git("add", ".")
    git("commit", "-qm", "later")
    return {"sample_id": "sample", "source_id": "owner__repo-1", "repo": "owner/repo",
            "base_commit": base, "problem_statement": "widget returns incorrect value",
            "repo_dir": str(repo)}


def test_base_objects_deterministic_and_no_leakage(task):
    first = build_context(task)
    assert first == build_context(task)
    assert [c["path"] for c in first["chunks"]] == ["engine.py"]
    text = first["chunks"][0]["text"]
    assert "return 1" in text and "SECRET" not in json.dumps(first)
    assert first["chunks"][0]["chunk_sha256"] == digest(text.encode())
    assert first["chunks"][0]["file_sha256"] == digest(text.encode())
    assert "complete raw unified Git diff" in render_prompt(task, first)


@pytest.mark.parametrize("key", ["patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hint"])
def test_deny_answer_bearing_fields(task, key):
    with pytest.raises(ContextError):
        build_context(dict(task, **{key: "SECRET"}))


@pytest.mark.parametrize("path", ["../x.py", "/x.py", "a/../x.py", "a\\x.py", "x\ny.py"])
def test_path_safety(path):
    assert not safe_path(path)


def test_budget_and_revision_fail_closed(task):
    with pytest.raises(ContextError):
        build_context(dict(task, base_commit="HEAD"))
    with pytest.raises(ContextError):
        build_context(task, policy=dict(POLICY, max_context_bytes=1))
    with pytest.raises(ContextError):
        build_context(task, policy=dict(POLICY, max_scan_bytes=1))


def test_manifest_unaffected_bytes_and_token_gate(task, tmp_path):
    old, new, provenance = (tmp_path / p for p in ("old.jsonl", "new.jsonl", "context.json"))
    other = b'{"sample_id": "other", "prompt": "unchanged"}\n'
    row = {"sample_id": "sample", "source_id": task["source_id"], "scorer": "swebench",
           "split": "validation", "prompt": "old", "reference": None, "tests": None}
    old.write_bytes(other + (json.dumps(row) + "\n").encode())
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([task]))
    prepare = script("prepare_swe_context").prepare
    with pytest.raises(ContextError, match="approved SHA"):
        prepare(old, tasks, new, provenance, "0" * 64)
    prepare(old, tasks, new, provenance, digest(tasks.read_bytes()))
    assert new.read_bytes().startswith(other)
    changed = json.loads(new.read_text().splitlines()[1])
    assert changed["prompt_template_version"] == VERSION
    sample = SimpleNamespace(**changed)
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: list(range(11)))
    prov = json.loads(provenance.read_text())
    validate_prompt_contract([sample], prov, new, tokenizer, 11)
    with pytest.raises(ContextError, match="token budget"):
        validate_prompt_contract([sample], prov, new, tokenizer, 10)
    sample.prompt += "tamper"
    with pytest.raises(ContextError, match="provenance"):
        validate_prompt_contract([sample], prov, new, tokenizer, 11)


def test_patch_checks_base_not_postfix_checkout(task):
    check = script("validate_swe_context_probe").check_patch
    patch = ("--- a/engine.py\n+++ b/engine.py\n@@ -1,2 +1,2 @@\n"
             " def widget():\n-    return 1\n+    return 2\n")
    assert check(task, patch)[0]
    assert not check(task, patch.replace("return 1", "return 99"))[0]
    assert not check(task, "```diff\n" + patch + "```")[0]
    assert "POST_FIX_SECRET" in (Path(task["repo_dir"]) / "engine.py").read_text()
