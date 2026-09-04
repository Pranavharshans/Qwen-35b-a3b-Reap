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


def test_build_bundle_validation_commands_accept_real_bundle_payload(tmp_path):
    """Regression: run-bundle validation must read the payload key it checks.

    The pilot run's run-bundle task failed validation because the command
    asserted ``bundle['files']`` while ``build_run_bundle`` emits
    ``bundle['artifacts']``. This test builds a real bundle with
    ``build_run_bundle`` and executes every committed plan's build-bundle
    ``validation_command`` against it, so a key drift fails in CI instead of
    after a GPU run.
    """
    import json
    import subprocess
    import sys

    from reverse_reap.config import load_config
    from reverse_reap.reporting import build_run_bundle

    root = Path(__file__).parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "analysis").mkdir()
    (run_dir / "analysis" / "candidate-manifest.json").write_text(
        json.dumps({"gate_passed": True, "experts": [{"layer": 0, "expert": 1}]})
    )
    config = load_config(root / "configs" / "pinned-3090-bf16.yaml")
    bundle_path = tmp_path / "run-bundle.json"
    build_run_bundle(run_dir, tmp_path / "state", config, bundle_path)

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload and "files" not in payload

    for plan_name in ("execution-plan-smoke.yaml", "execution-plan-pilot.yaml"):
        plan = load_plan(root / "configs" / plan_name)
        for task in plan.tasks:
            if "build-bundle" not in task.command:
                continue
            argv = [
                sys.executable if part == "python" else part for part in task.validation_command
            ]
            argv = [part.replace("${RUN_ID}", "test-run") for part in argv]
            argv[-1] = str(bundle_path)
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            assert completed.returncode == 0, (
                f"{plan_name}:{task.task_id} validation failed: "
                f"{completed.stderr.strip()}"
            )
