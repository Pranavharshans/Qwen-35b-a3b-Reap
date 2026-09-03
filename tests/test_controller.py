import sys
from datetime import UTC, datetime

import pytest
import yaml

from reverse_reap.config import ExperimentConfig
from reverse_reap.controller import ControllerError, run_next
from reverse_reap.state import Status, load_state


def config():
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "model": {
                "id": "Qwen/Qwen3.5-35B-A3B",
                "revision": "a" * 40,
                "source_precision": "bf16",
                "execution_precision": "bf16",
                "text_only": True,
            },
            "runtime": {
                "seed": 7,
                "max_input_tokens": 128,
                "max_new_tokens": 16,
            },
            "budget": {
                "max_gpu_hours": 10,
                "max_cost_usd": 10,
                "provider_rate_usd_per_hour": 0.5,
                "storage_limit_gb": 10,
                "deadline_utc": datetime(2099, 1, 1, tzinfo=UTC),
            },
            "datasets": {"manifest": "manifest.jsonl", "split": "smoke"},
        }
    )


def write_plan(tmp_path, tasks):
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "tasks": tasks}))
    return path


def task(task_id, output, dependencies=None):
    code = f"from pathlib import Path; Path({str(output)!r}).write_text('ok')"
    return {
        "task_id": task_id,
        "objective": "fixture task",
        "command": [sys.executable, "-c", code],
        "validation_command": [sys.executable, "-c", "raise SystemExit(0)"],
        "outputs": [str(output)],
        "dependencies": dependencies or [],
        "estimated_gpu_hours": 0,
        "estimated_storage_gb": 0.001,
        "failure_behavior": "retry",
    }


def test_run_next_executes_one_dependency_ordered_task_per_call(tmp_path):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    plan = write_plan(tmp_path, [task("first", first), task("second", second, ["first"])])
    state_dir = tmp_path / "state"
    result = run_next(plan, config(), state_dir, run_id="fixture", heartbeat_seconds=0.01)
    assert result["status"] == Status.COMPLETE
    assert first.exists() and not second.exists()
    state = load_state(state_dir / "first.json")
    assert state.output_hashes[str(first)]
    assert state.validation_exit_code == 0
    assert state.next_permitted_task == "second"
    run_next(plan, config(), state_dir, run_id="fixture", heartbeat_seconds=0.01)
    assert second.exists()


def test_missing_inputs_fail_before_process_launch(tmp_path):
    output = tmp_path / "output.txt"
    item = task("blocked", output)
    item["inputs"] = [str(tmp_path / "missing.txt")]
    plan = write_plan(tmp_path, [item])
    with pytest.raises(ControllerError, match="inputs are missing"):
        run_next(plan, config(), tmp_path / "state", run_id="fixture")


def test_budget_denial_enters_waiting_for_human(tmp_path):
    output = tmp_path / "output.txt"
    item = task("expensive", output)
    item["estimated_gpu_hours"] = 9
    plan = write_plan(tmp_path, [item])
    result = run_next(plan, config(), tmp_path / "state", run_id="fixture")
    assert result["status"] == Status.WAITING_FOR_HUMAN
    assert not output.exists()
