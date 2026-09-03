from pathlib import Path

from reverse_reap.controller import load_plan


def test_smoke_execution_plan_is_valid_and_dependency_complete():
    root = Path(__file__).parents[1]
    plan = load_plan(root / "configs" / "execution-plan-smoke.yaml")
    assert plan.tasks[0].task_id == "gpu-preflight"
    assert {task.task_id for task in plan.tasks} >= {
        "dataset-freeze",
        "dataset-token-length-audit",
        "instrumentation-probe",
        "telemetry-smoke",
        "candidate-analysis",
        "single-expert-intervention-probe",
        "baseline-validation",
        "selected-ablation-validation",
        "extract-candidates",
    }
