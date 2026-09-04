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


def test_causal_pilot_plan_is_valid_and_dependency_complete():
    root = Path(__file__).parents[1]
    plan = load_plan(root / "configs" / "execution-plan-causal-pilot.yaml")
    ids = [task.task_id for task in plan.tasks]
    assert len(ids) == len(set(ids)) == 9
    by_id = {task.task_id: task for task in plan.tasks}
    # Pre-GPU freeze enforcement is terminal and precedes all GPU generation.
    assert by_id["verify-frozen-inputs"].failure_behavior == "terminal"
    assert "gpu-preflight" in by_id["gen-validation-conditions"].dependencies
    assert "verify-frozen-inputs" in by_id["gen-validation-conditions"].dependencies
    # The determinism pre-gate stops the run before the 20 random controls.
    gate = by_id["response-determinism-pre-gate"]
    assert gate.failure_behavior == "terminal"
    assert "gen-validation-conditions" in gate.dependencies
    assert "response-determinism-pre-gate" in by_id[
        "gen-random-replication-conditions"
    ].dependencies
    # Every command is wall-time bounded.
    for task in plan.tasks:
        assert task.command[0] == "timeout", task.task_id
        assert str(task.command[2]).isdigit(), task.task_id
    total_booked = sum(task.estimated_gpu_hours for task in plan.tasks)
    assert total_booked <= 6.4  # 80% of max_gpu_hours 8 in the gen config
    assert abs(total_booked - 4.76) < 1e-6


def test_causal_conditions_spec_is_frozen_and_complete():
    import json
    import re

    root = Path(__file__).parents[1]
    spec = json.loads((root / "configs" / "causal-pilot-conditions.json").read_text())
    conditions = spec["conditions"]
    assert len(conditions) == 27
    by_id = {c["condition_id"]: c for c in conditions}
    assert len(by_id) == 27
    assert by_id["c0-baseline-a"]["expert_manifest"] is None
    assert by_id["c0-baseline-b"]["expert_manifest"] is None
    expected_counts = {"validation": 5, "random-replication": 22}
    assert {phase: sum(1 for c in conditions if c["phase"] == phase) for phase in expected_counts}
    splits = {c["split"]: 0 for c in conditions}
    for c in conditions:
        splits[c["split"]] += 1
    assert splits == {"validation": 25, "replication": 2}
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    assert sum(1 for c in conditions if c["expert_manifest"] is None) == 3
    for c in conditions:
        if c["expert_manifest"] is None:
            assert c["expert_manifest_sha256"] is None
            continue
        assert c["expert_manifest"].startswith(
            "runs/pilot/20260904T100102Z-qwen35a3b-direct-503e4ee9-644a80fc/analysis/"
        )
        assert sha_pattern.match(c["expert_manifest_sha256"])
    assert by_id["c2-selected"]["expert_manifest_sha256"] == by_id[
        "c2-replication-selected"
    ]["expert_manifest_sha256"] == spec["source_candidate_manifest_sha256"]
    assert spec["splits"] == {
        "validation": {"samples": 50, "coding": 40, "control": 10},
        "replication": {"samples": 52, "coding": 40, "control": 12},
    }


def test_verify_frozen_experts_script_enforces_byte_identity(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys

    manifest = tmp_path / "experts.json"
    payload = json.dumps({"experts": [{"layer": 3, "expert": 26}]})
    manifest.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "conditions": [
                    {"condition_id": "a", "phase": "validation", "split": "validation",
                     "expert_manifest": str(manifest), "expert_manifest_sha256": digest},
                    {"condition_id": "b", "phase": "validation", "split": "validation",
                     "expert_manifest": None, "expert_manifest_sha256": None},
                ],
            }
        )
    )
    script = Path(__file__).parents[1] / "scripts" / "verify_frozen_experts.py"
    report = tmp_path / "report.json"
    ok = subprocess.run(
        [sys.executable, str(script), "--conditions", str(spec), "--report", str(report)],
        capture_output=True, text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(report.read_text())["verified_manifests"] == 1
    manifest.write_text(payload.replace("26", "27"))
    bad = subprocess.run(
        [sys.executable, str(script), "--conditions", str(spec),
         "--report", str(tmp_path / "report2.json")],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1 and "MISMATCH" in bad.stderr
