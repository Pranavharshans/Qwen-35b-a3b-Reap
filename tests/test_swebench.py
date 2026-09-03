import json

import pytest

from reverse_reap.swebench import SwebenchError, export_predictions, merge_report


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exports_official_prediction_contract_and_merges_completed_verdict(tmp_path):
    evaluation = tmp_path / "raw.jsonl"
    write_rows(
        evaluation,
        [
            {
                "sample_id": "a",
                "source_id": "sympy__sympy-1",
                "scorer": "swebench",
                "response": "```diff\ndiff --git a/a b/a\n```",
                "scoreable": False,
                "passed": False,
            },
            {"sample_id": "b", "scorer": "exact_match", "scoreable": True, "passed": True},
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    result = export_predictions(evaluation, predictions, model_name="qwen-c0")
    assert result["predictions"] == 1
    prediction = json.loads(predictions.read_text())
    assert prediction["instance_id"] == "sympy__sympy-1"
    assert prediction["model_patch"].startswith("diff --git")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "completed_ids": ["sympy__sympy-1"],
                "resolved_ids": ["sympy__sympy-1"],
                "error_ids": [],
            }
        )
    )
    merged = tmp_path / "merged.jsonl"
    merged_report = merge_report(evaluation, report, merged)
    assert merged_report["passed_gate_b_scoreability"]
    rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert rows[0]["scoreable"] and rows[0]["passed"]


def test_merge_rejects_foreign_harness_ids(tmp_path):
    evaluation = tmp_path / "raw.jsonl"
    write_rows(
        evaluation,
        [{"sample_id": "a", "source_id": "wanted", "scorer": "swebench"}],
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"completed_ids": ["foreign"], "resolved_ids": [], "error_ids": []})
    )
    with pytest.raises(SwebenchError, match="unrequested"):
        merge_report(evaluation, report, tmp_path / "merged.jsonl")
