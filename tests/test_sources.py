from reverse_reap.sources import SourceDefinition, _adapt


def source(adapter, **overrides):
    data = {
        "name": "fixture",
        "dataset_id": "fixture/data",
        "split": "test",
        "adapter": adapter,
        "domain": "coding",
        "stratum": "function-synthesis",
        "license": "MIT",
        "citation": "https://example.test/paper",
        "language": "python",
    }
    data.update(overrides)
    return SourceDefinition.model_validate(data)


def test_humaneval_adapter_constructs_executable_check():
    result = _adapt(
        source("humaneval"),
        "a" * 40,
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "canonical_solution": "    return a + b",
            "test": "def check(candidate): assert candidate(1, 2) == 3",
            "entry_point": "add",
        },
        0,
    )
    assert result["tests"].endswith("check(add)\n")
    assert result["scorer"] == "unit_tests"


def test_swebench_adapter_preserves_repository_issue_and_patch():
    result = _adapt(
        source("swebench", stratum="repository-bug-repair", language="mixed"),
        "b" * 40,
        {
            "instance_id": "project__repo-1",
            "repo": "project/repo",
            "problem_statement": "Fix the parser.",
            "patch": "diff --git a/a.py b/a.py",
        },
        0,
    )
    assert "project/repo" in result["prompt"]
    assert result["scorer"] == "swebench"
