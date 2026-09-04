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
    assert "datasets/manifests/smoke-lengthmatched.jsonl" in [str(part) for part in audit.command]
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
                f"{plan_name}:{task.task_id} validation failed: {completed.stderr.strip()}"
            )


def test_causal_pilot_plan_is_valid_and_dependency_complete():
    root = Path(__file__).parents[1]
    plan = load_plan(root / "configs" / "execution-plan-causal-pilot.yaml")
    ids = [task.task_id for task in plan.tasks]
    assert len(ids) == len(set(ids)) == 12
    by_id = {task.task_id: task for task in plan.tasks}
    expected_tasks = {
        "gpu-preflight",
        "verify-frozen-inputs",
        "gen-validation-baselines",
        "response-determinism-pre-gate",
        "gen-validation-interventions",
        "gen-random-controls",
        "score-conditions",
        "docker-evaluator-prep",
        "swebench-harness-prep",
        "swebench-harness-score",
        "causal-gates",
        "run-bundle",
    }
    assert set(ids) == expected_tasks
    # Pre-GPU freeze enforcement is terminal and precedes all GPU generation.
    assert by_id["verify-frozen-inputs"].failure_behavior == "terminal"
    assert "gpu-preflight" in by_id["gen-validation-baselines"].dependencies
    assert "verify-frozen-inputs" in by_id["gen-validation-baselines"].dependencies
    # The determinism/no-op pre-gate stops the run before ANY intervention or
    # control generation, after only 150 of 1,300 generations.
    gate = by_id["response-determinism-pre-gate"]
    assert gate.failure_behavior == "terminal"
    assert "gen-validation-baselines" in gate.dependencies
    assert "response-determinism-pre-gate" in by_id["gen-validation-interventions"].dependencies
    assert "response-determinism-pre-gate" in by_id["gen-random-controls"].dependencies
    # Scoring chain: conditions -> official swebench harness -> gates -> bundle.
    assert {"gen-random-controls", "docker-evaluator-prep", "swebench-harness-prep"} == set(
        by_id["score-conditions"].dependencies
    )
    assert "score-conditions" in by_id["swebench-harness-score"].dependencies
    assert "swebench-harness-score" in by_id["causal-gates"].dependencies
    assert "causal-gates" in by_id["run-bundle"].dependencies
    # The pinned harness revision is driven with the pinned swe-bench-tasks
    # task repo (measured 2026-09-04: hosted dataset rows lack the image/
    # eval_script columns; no --report_dir/--log_dir flags exist).
    assert "--tasks-repo" in by_id["swebench-harness-score"].command
    assert "--tasks-dir" in by_id["swebench-harness-prep"].command
    assert any(
        "3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d" in part
        for part in by_id["swebench-harness-prep"].validation_command
    )
    # The pre-gate checks BOTH the baseline pair and the no-op equivalence.
    pregate_command = " ".join(gate.command)
    assert pregate_command.count("--pair") == 2
    assert "c0-baseline-a" in pregate_command and "c0-baseline-b" in pregate_command
    assert "c0-noop-masked" in pregate_command
    # Every command is wall-time bounded.
    for task in plan.tasks:
        assert task.command[0] == "timeout", task.task_id
        assert str(task.command[2]).isdigit(), task.task_id
    total_booked = sum(task.estimated_gpu_hours for task in plan.tasks)
    assert total_booked <= 6.4  # 80% of max_gpu_hours 8 in the gen config
    assert abs(total_booked - 3.91) < 1e-6
    # Only the three GPU generation tasks book host-hours; everything else is
    # CPU/container work on the scoring host selected by the pending decision.
    booked = {
        task.task_id: task.estimated_gpu_hours
        for task in plan.tasks
        if task.estimated_gpu_hours > 0
    }
    assert booked == {
        "gpu-preflight": 0.01,
        "gen-validation-baselines": 0.7,
        "gen-validation-interventions": 0.7,
        "gen-random-controls": 2.5,
    }


def test_causal_conditions_spec_is_frozen_and_validation_stage_only():
    import json
    import re

    root = Path(__file__).parents[1]
    spec = json.loads((root / "configs" / "causal-pilot-conditions.json").read_text())
    conditions = spec["conditions"]
    assert len(conditions) == 26
    by_id = {c["condition_id"]: c for c in conditions}
    assert len(by_id) == 26
    assert by_id["c0-baseline-a"]["expert_manifest"] is None
    assert by_id["c0-baseline-b"]["expert_manifest"] is None
    assert by_id["c0-noop-masked"]["expert_manifest"] is None
    assert by_id["c0-noop-masked"]["instrument_noop"] is True
    # Three generation phases: baselines -> (pre-gate) -> interventions and 20
    # random controls. NO replication conditions exist in this stage.
    phase_counts = {}
    for c in conditions:
        phase_counts[c["phase"]] = phase_counts.get(c["phase"], 0) + 1
    assert phase_counts == {
        "validation-baselines": 3,
        "validation-interventions": 3,
        "random": 20,
    }
    assert all(c["split"] == "validation" for c in conditions)
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    assert sum(1 for c in conditions if c["expert_manifest"] is None) == 3
    manifests = [
        c["expert_manifest_sha256"] for c in conditions if c["expert_manifest_sha256"] is not None
    ]
    assert len(manifests) == 23 and len(set(manifests)) == 23
    for c in conditions:
        if c["expert_manifest"] is None:
            assert c["expert_manifest_sha256"] is None
            continue
        assert c["expert_manifest"].startswith(
            "runs/pilot/20260904T100102Z-qwen35a3b-direct-503e4ee9-644a80fc/analysis/"
        )
        assert sha_pattern.match(c["expert_manifest_sha256"])
    assert (
        by_id["c2-selected"]["expert_manifest_sha256"] == spec["source_candidate_manifest_sha256"]
    )
    # The replication split is NOT opened anywhere in this stage.
    assert spec["splits"] == {"validation": {"samples": 50, "coding": 40, "control": 10}}
    assert "replication" not in spec["splits"]
    assert not any("replication" in c["condition_id"] for c in conditions)
    assert spec["swebench_instances"] == [
        "django__django-14017",
        "matplotlib__matplotlib-23563",
        "matplotlib__matplotlib-25442",
        "scikit-learn__scikit-learn-14983",
        "sympy__sympy-14396",
        "sympy__sympy-16281",
        "sympy__sympy-16988",
        "sympy__sympy-20154",
    ]


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
                    {
                        "condition_id": "a",
                        "phase": "validation",
                        "split": "validation",
                        "expert_manifest": str(manifest),
                        "expert_manifest_sha256": digest,
                    },
                    {
                        "condition_id": "b",
                        "phase": "validation",
                        "split": "validation",
                        "expert_manifest": None,
                        "expert_manifest_sha256": None,
                    },
                ],
            }
        )
    )
    script = Path(__file__).parents[1] / "scripts" / "verify_frozen_experts.py"
    report = tmp_path / "report.json"
    ok = subprocess.run(
        [sys.executable, str(script), "--conditions", str(spec), "--report", str(report)],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(report.read_text())["verified_manifests"] == 1
    manifest.write_text(payload.replace("26", "27"))
    bad = subprocess.run(
        [
            sys.executable,
            str(script),
            "--conditions",
            str(spec),
            "--report",
            str(tmp_path / "report2.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1 and "MISMATCH" in bad.stderr


# --- execution split: run-ID-scoped generation and scoring plans ------------

GEN_PLAN_TASKS = {
    "verify-frozen-inputs",
    "gpu-preflight",
    "gen-validation-baselines",
    "response-determinism-pre-gate",
    "gen-validation-interventions",
    "gen-random-controls",
    "generation-bundle",
}

SCORE_PLAN_TASKS = {
    "verify-generation-inputs",
    "docker-evaluator-prep",
    "swebench-harness-prep",
    "score-conditions",
    "swebench-harness-score",
    "causal-gates",
    "run-bundle",
}

_RANDOM_CONTROLS = [f"c3-layer-random-{index:03d}" for index in range(20)]


def _assert_all_outputs_run_scoped(plan) -> None:
    for task in plan.tasks:
        for path in list(task.outputs) + list(task.inputs):
            assert "${RUN_ID}" in str(path) or str(path).startswith(
                ("configs/", "datasets/", "evaluator/", "deploy/")
            ), f"{task.task_id}: unscoped path {path}"


def test_causal_gen_plan_is_valid_and_dependency_complete():
    root = Path(__file__).parents[1]
    plan = load_plan(root / "configs" / "execution-plan-causal-gen.yaml")
    ids = [task.task_id for task in plan.tasks]
    assert len(ids) == len(set(ids)) == 7
    assert set(ids) == GEN_PLAN_TASKS
    by_id = {task.task_id: task for task in plan.tasks}
    # Freeze enforcement and preflight are terminal and precede all GPU work.
    assert by_id["verify-frozen-inputs"].failure_behavior == "terminal"
    assert by_id["gpu-preflight"].failure_behavior == "terminal"
    assert by_id["gen-validation-baselines"].dependencies == [
        "gpu-preflight",
        "verify-frozen-inputs",
    ]
    # The terminal pre-gate stops the run before ANY intervention/control work.
    gate = by_id["response-determinism-pre-gate"]
    assert gate.failure_behavior == "terminal"
    assert gate.dependencies == ["gen-validation-baselines"]
    pregate_command = " ".join(gate.command)
    assert pregate_command.count("--pair") == 2
    assert "c0-baseline-a" in pregate_command and "c0-baseline-b" in pregate_command
    assert "c0-noop-masked" in pregate_command
    assert by_id["gen-validation-interventions"].dependencies == ["response-determinism-pre-gate"]
    assert by_id["gen-random-controls"].dependencies == ["response-determinism-pre-gate"]
    # The generation bundle depends on BOTH later generation phases and is
    # terminal: the plan stops after hashing.
    bundle = by_id["generation-bundle"]
    assert bundle.failure_behavior == "terminal"
    assert set(bundle.dependencies) == {"gen-validation-interventions", "gen-random-controls"}
    assert "generation_bundle.py" in " ".join(bundle.command)
    assert "build" in bundle.command and "verify" in bundle.validation_command
    # Docker, the evaluator, and the official SWE-bench harness are removed
    # from the GPU host: no command may reference them.
    for task in plan.tasks:
        joined = " ".join(task.command + task.validation_command)
        for banned in ("docker", "evaluator", "swebench", "harness", "replicat"):
            assert banned not in joined, f"{task.task_id}: GPU plan mentions {banned!r}"
    # Booked estimates: 3.91 four-GPU host-hours, generation tasks only.
    booked = {
        task.task_id: task.estimated_gpu_hours
        for task in plan.tasks
        if task.estimated_gpu_hours > 0
    }
    assert booked == {
        "gpu-preflight": 0.01,
        "gen-validation-baselines": 0.7,
        "gen-validation-interventions": 0.7,
        "gen-random-controls": 2.5,
    }
    assert sum(booked.values()) <= 6.4  # 80% reserve of max_gpu_hours 8
    # Every command is wall-time bounded and every run path is ${RUN_ID}-scoped.
    for task in plan.tasks:
        assert task.command[0] == "timeout", task.task_id
        assert str(task.command[2]).isdigit(), task.task_id
    _assert_all_outputs_run_scoped(plan)
    # The 20 random controls are declared as individual file outputs.
    assert [str(path) for path in by_id["gen-random-controls"].outputs] == [
        f"runs/causal-pilot/${{RUN_ID}}/generations/{condition}.jsonl"
        for condition in _RANDOM_CONTROLS
    ]


def test_causal_score_plan_is_valid_and_dependency_complete():
    root = Path(__file__).parents[1]
    plan = load_plan(root / "configs" / "execution-plan-causal-score.yaml")
    ids = [task.task_id for task in plan.tasks]
    assert len(ids) == len(set(ids)) == 7
    assert set(ids) == SCORE_PLAN_TASKS
    by_id = {task.task_id: task for task in plan.tasks}
    # Scoring runs on the CPU scoring VM: nothing books GPU-host hours.
    assert all(task.estimated_gpu_hours == 0 for task in plan.tasks)
    # Input verification is terminal and precedes all scoring.
    assert by_id["verify-generation-inputs"].failure_behavior == "terminal"
    assert by_id["verify-generation-inputs"].dependencies == []
    verify_command = " ".join(by_id["verify-generation-inputs"].command)
    assert "verify" in verify_command and "generation-bundle.json" in verify_command
    assert "26" in " ".join(by_id["verify-generation-inputs"].validation_command)
    # Scoring chain: verify -> score -> official harness -> gates -> bundle.
    assert set(by_id["score-conditions"].dependencies) == {
        "verify-generation-inputs",
        "docker-evaluator-prep",
        "swebench-harness-prep",
    }
    assert by_id["swebench-harness-score"].dependencies == ["score-conditions"]
    assert by_id["causal-gates"].dependencies == ["swebench-harness-score"]
    assert by_id["run-bundle"].dependencies == ["causal-gates"]
    assert by_id["causal-gates"].failure_behavior == "terminal"
    assert by_id["run-bundle"].failure_behavior == "terminal"
    # Pinned harness + task repo drive the SWE-bench stage (measured 2026-09-04).
    assert "--tasks-repo" in by_id["swebench-harness-score"].command
    assert "--tasks-dir" in by_id["swebench-harness-prep"].command
    assert any(
        "3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d" in part
        for part in by_id["swebench-harness-prep"].validation_command
    )
    # The bundle validation admits only validation-stage outcomes: never
    # 'positive' (replication is not run in this split).
    bundle_validation = " ".join(by_id["run-bundle"].validation_command)
    assert "'positive'" not in bundle_validation
    assert "incomplete" in bundle_validation and "'null'" in bundle_validation
    # The bundle reads the run-ID-scoped state dir it is launched with.
    assert "runs/causal-pilot/state/${RUN_ID}" in " ".join(by_id["run-bundle"].command)
    # No replication anywhere; the 26-condition denominator is untouched. The
    # frozen Gate D criterion KEY 'replication_direction' must still appear
    # (the report schema is unchanged; the criterion simply cannot hold).
    for task in plan.tasks:
        joined = " ".join(task.command + task.validation_command)
        allowed = joined.replace("replication_direction", "").replace("unreplicated-candidates", "")
        assert "replicat" not in allowed, f"{task.task_id}: score plan mentions replication work"
    # All 26 final scored files (and all 26 staging files) are declared.
    final_outputs = [str(path) for path in by_id["swebench-harness-score"].outputs]
    assert len(final_outputs) == 26
    assert all(
        f"/{condition}.jsonl" in " ".join(final_outputs)
        for condition in ["c2-selected"] + _RANDOM_CONTROLS
    )
    staging_outputs = [str(path) for path in by_id["score-conditions"].outputs]
    assert len(staging_outputs) == 26
    # Every command is wall-time bounded and every run path is ${RUN_ID}-scoped.
    for task in plan.tasks:
        assert task.command[0] == "timeout", task.task_id
        assert str(task.command[2]).isdigit(), task.task_id
    _assert_all_outputs_run_scoped(plan)


def test_combined_causal_plan_is_preserved_for_provenance():
    """The pre-split combined plan stays byte-meaningful and un-launched."""
    root = Path(__file__).parents[1]
    plan = load_plan(root / "configs" / "execution-plan-causal-pilot.yaml")
    ids = [task.task_id for task in plan.tasks]
    assert len(ids) == len(set(ids)) == 12
    # The combined plan carries both phases in one 12-task list: the generation
    # tasks (minus the phase-1 bundle) plus the scoring tasks (minus the
    # phase-2 input verification).
    assert set(ids) == (
        GEN_PLAN_TASKS - {"generation-bundle"} | SCORE_PLAN_TASKS - {"verify-generation-inputs"}
    )
    assert "docker-evaluator-prep" in ids and "swebench-harness-score" in ids
