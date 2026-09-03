from reverse_reap.plans import build_full_plan


def test_full_plan_contains_all_frozen_controls_and_replication_after_validation():
    plan = build_full_plan()
    by_id = {task.task_id: task for task in plan.tasks}
    random_ids = {f"layer-random-{index:03d}" for index in range(20)}
    assert random_ids <= by_id.keys()
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
