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


def test_mbpp_adapter_uses_current_prompt_field():
    result = _adapt(
        source("mbpp"),
        "a" * 40,
        {
            "task_id": 11,
            "prompt": "Write a Python function that returns one.",
            "code": "def one(): return 1",
            "test_list": ["assert one() == 1"],
        },
        0,
    )
    assert result["prompt"].startswith("Write a Python")
    assert result["tests"] == "assert one() == 1"


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


def test_humanevalpack_java_adapter_uses_live_schema_fields():
    result = _adapt(
        source("humanevalpack", language="java"),
        "c" * 40,
        {
            "task_id": "Java/0",
            "prompt": "class Solution { public int answer() {",
            "canonical_solution": "return 42; } }",
            "test": "public class Main { public static void main(String[] x) {} }",
            "entry_point": "answer",
        },
        0,
    )
    assert result["language"] == "java"
    assert result["tests"].startswith("public class Main")


def test_mmlu_adapter_formats_choices_and_answer_letter():
    result = _adapt(
        source("mmlu", domain="control", stratum="general-knowledge", language=None),
        "d" * 40,
        {
            "question": "Which value?",
            "subject": "math",
            "choices": ["zero", "one", "two", "three"],
            "answer": 2,
        },
        7,
    )
    assert "C. two" in result["prompt"]
    assert result["reference"] == "C"


def test_source_definition_records_predeclared_exclusions():
    value = source("mbpp", exclude_source_ids=["141"])
    assert value.exclude_source_ids == ("141",)
