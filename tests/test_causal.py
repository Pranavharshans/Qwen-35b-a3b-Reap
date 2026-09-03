import json

import pytest

from reverse_reap.causal import (
    CausalError,
    causal_gate_report,
    compare_deterministic_evaluations,
    load_expert_set,
    score_response,
)
from reverse_reap.datasets import normalize_sample
from reverse_reap.evaluator import EvaluationResult


def sample(scorer="exact_match"):
    data = {
        "source": "fixture",
        "source_revision": "abc",
        "source_id": "one",
        "domain": "coding",
        "stratum": "understanding",
        "language": "python",
        "prompt": "What is the output?",
        "reference": "#### 42",
        "scorer": scorer,
    }
    if scorer == "unit_tests":
        data.update(
            {
                "prompt": "def answer():\n",
                "reference": "    return 42",
                "tests": "assert answer() == 42",
                "entry_point": "answer",
            }
        )
    return normalize_sample(data, seed=1)


def test_exact_scorer_extracts_final_answer():
    result = score_response(sample(), "Reasoning\n\\boxed{42}", evaluator_image="unused")
    assert result["passed"]


def test_multiple_choice_scorer_uses_final_standalone_letter():
    item = sample().model_copy(update={"scorer": "multiple_choice", "reference": "B"})
    result = score_response(item, "I considered A and C. Final: B", evaluator_image="unused")
    assert result["passed"]


def test_unit_test_scorer_uses_sandbox(monkeypatch):
    captured = {}

    def fake_evaluate(code, tests, **kwargs):
        captured.update(code=code, tests=tests, kwargs=kwargs)
        return EvaluationResult(True, 0, False, "", "", "a" * 64)

    monkeypatch.setattr("reverse_reap.causal.evaluate_python", fake_evaluate)
    result = score_response(
        sample("unit_tests"), "```python\n    return 42\n```", evaluator_image="image@sha256:x"
    )
    assert result["passed"]
    assert captured["code"].startswith("def answer")


def write_results(path, condition, coding_passes, control_passes):
    rows = []
    for domain, values in (("coding", coding_passes), ("control", control_passes)):
        for index, passed in enumerate(values):
            rows.append(
                {
                    "sample_id": f"{domain}-{index}",
                    "domain": domain,
                    "condition_id": condition,
                    "scoreable": True,
                    "passed": passed,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_gate_d_applies_all_preregistered_thresholds(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    selected = tmp_path / "selected.jsonl"
    replication_baseline = tmp_path / "replication-baseline.jsonl"
    replication_selected = tmp_path / "replication-selected.jsonl"
    write_results(baseline, "C0", [True] * 10, [True] * 10)
    write_results(selected, "C2", [False] * 8 + [True] * 2, [False] + [True] * 9)
    write_results(replication_baseline, "C0", [True] * 10, [True] * 10)
    write_results(replication_selected, "C2", [False] * 7 + [True] * 3, [False] + [True] * 9)
    random_paths = []
    for index in range(20):
        path = tmp_path / f"random-{index}.jsonl"
        write_results(path, "C3", [False] * 2 + [True] * 8, [True] * 10)
        random_paths.append(path)
    report = causal_gate_report(
        baseline,
        selected,
        random_paths,
        replication_baseline_path=replication_baseline,
        replication_selected_path=replication_selected,
    )
    assert report["passed"]
    assert report["label"] == "coding-critical-v0"
    assert report["coding_drop"] == pytest.approx(0.8)
    assert report["replication"]["coding_drop"] == pytest.approx(0.7)
    assert len(report["coding_drop_95ci"]) == 2


def test_expert_manifest_rejects_duplicate_identity(tmp_path):
    path = tmp_path / "experts.json"
    path.write_text(json.dumps({"experts": [{"layer": 1, "expert": 2}] * 2}))
    with pytest.raises(CausalError, match="duplicate"):
        load_expert_set(path)


def test_determinism_comparison_requires_exact_responses_and_95_percent_scoreable(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    rows = [
        {
            "sample_id": f"s-{index}",
            "domain": "coding",
            "response": "same",
            "passed": True,
            "scoreable": True,
        }
        for index in range(20)
    ]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    first.write_text(content)
    second.write_text(content)
    assert compare_deterministic_evaluations(first, second)["passed"]
    changed = list(rows)
    changed[0] = {**changed[0], "response": "different"}
    second.write_text("".join(json.dumps(row) + "\n" for row in changed))
    assert not compare_deterministic_evaluations(first, second)["passed"]
