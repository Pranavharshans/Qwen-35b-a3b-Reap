from pathlib import Path

import pytest

from reverse_reap.controller import load_plan
from reverse_reap.plans import build_full_plan


def test_full_plan_contains_all_frozen_controls_and_replication_after_validation():
    plan = build_full_plan()
    by_id = {task.task_id: task for task in plan.tasks}
    random_ids = {f"layer-random-{index:03d}" for index in range(20)}
    assert random_ids <= by_id.keys()
    assert "single-expert-intervention-probe" in by_id
    assert "dataset-token-length-audit" in by_id
    assert "single-expert-intervention-probe" in by_id["selected-validation"].dependencies
    assert random_ids <= set(by_id["baseline-replication"].dependencies)
    assert random_ids <= set(by_id["selected-replication"].dependencies)
    assert set(by_id["causal-report"].dependencies) == {
        "baseline-replication",
        "selected-replication",
    }
    assert set(by_id["final-bundle"].dependencies) == {
        "extract-candidates",
        "thinking-selected-pilot",
    }
    assert len(plan.tasks) >= 38


def test_every_full_plan_task_has_mandatory_execution_contract():
    for task in build_full_plan().tasks:
        assert task.objective
        assert task.definition_of_done
        assert task.outputs
        assert task.validation_command
        assert task.estimated_gpu_hours >= 0
        assert task.estimated_storage_gb >= 0


SMOKE_PLAN = Path(__file__).resolve().parents[1] / "configs" / "execution-plan-smoke.yaml"


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(build_full_plan(), id="full-plan"),
        pytest.param(load_plan(SMOKE_PLAN), id="smoke-plan"),
    ],
)
def test_manifest_consuming_tasks_are_gated_on_gate_c(plan):
    """Any task consuming candidate-manifest.json must be gated on gate_passed == true.

    A skipped gate transitions to COMPLETE, so dependencies alone cannot protect
    downstream intervention, ablation, replication, or extraction stages: each
    manifest consumer needs its own run_if gate on the Gate C decision.
    """
    for task in plan.tasks:
        consumes_manifest = any(
            "candidate-manifest.json" in str(item) for item in task.inputs
        )
        if not consumes_manifest:
            continue
        assert task.run_if is not None, f"{task.task_id} consumes the manifest ungated"
        assert "candidate-manifest.json" in str(task.run_if.path)
        assert task.run_if.field == "gate_passed"
        assert task.run_if.equals is True
