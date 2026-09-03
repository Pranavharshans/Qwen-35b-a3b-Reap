import subprocess

import pytest

from reverse_reap.evaluator import EvaluationError, evaluate_java, evaluate_python


def test_requires_digest_pinned_container():
    with pytest.raises(EvaluationError, match="pinned"):
        evaluate_python("x = 1", "assert x == 1", image="python:3.12")


def test_constructs_locked_down_docker_invocation(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "passed", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = evaluate_python(
        "def add(a, b): return a + b",
        "assert add(2, 3) == 5",
        image="python@sha256:" + "a" * 64,
    )
    command = seen["command"]
    assert result.passed
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "-I" in command
    assert seen["kwargs"]["timeout"] == 10


def test_timeout_is_a_scored_failure(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 3, output="partial", stderr="stuck")

    monkeypatch.setattr(subprocess, "run", timeout)
    result = evaluate_python(
        "while True: pass",
        "",
        image="python@sha256:" + "b" * 64,
        timeout_seconds=3,
    )
    assert result.timed_out and not result.passed
    assert result.return_code is None


def test_java_uses_compiler_and_locked_down_container(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = evaluate_java(
        "class Solution {}",
        "public class Main { public static void main(String[] args) {} }",
        image="evaluator@sha256:" + "c" * 64,
    )
    assert result.passed
    assert "--network=none" in seen["command"]
    assert any("javac -d /tmp/classes" in part for part in seen["command"])
