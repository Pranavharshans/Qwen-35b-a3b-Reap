from pathlib import Path

from reverse_reap.controller import load_plan


def test_smoke_execution_plan_is_valid_and_dependency_complete():
    root = Path(__file__).parents[1]
    plan = load_plan(root / "configs" / "execution-plan-smoke.yaml")
    assert plan.tasks[0].task_id == "gpu-preflight"
    assert {task.task_id for task in plan.tasks} >= {
        "dataset-freeze",
        "dataset-lengthmatch",
        "dataset-token-length-audit",
        "instrumentation-probe",
        "telemetry-smoke",
        "candidate-analysis",
        "single-expert-intervention-probe",
        "baseline-validation",
        "selected-ablation-validation",
        "extract-candidates",
    }
    by_id = {task.task_id: task for task in plan.tasks}
    lengthmatch = by_id["dataset-lengthmatch"]
    assert "datasets/manifests/smoke-lengthmatched.jsonl" in [
        str(path) for path in lengthmatch.outputs
    ]
    audit = by_id["dataset-token-length-audit"]
    assert "datasets/manifests/smoke-lengthmatched.jsonl" in [
        str(part) for part in audit.command
    ]
    assert "dataset-lengthmatch" in audit.dependencies
