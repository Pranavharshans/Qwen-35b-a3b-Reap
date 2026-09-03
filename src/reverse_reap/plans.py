"""Generate the complete governed v0 execution graph without runtime improvisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from reverse_reap.controller import ExecutionPlan


def _task(
    task_id: str,
    objective: str,
    done: str,
    command: list[str],
    validation: list[str],
    outputs: list[str],
    *,
    inputs: list[str] | None = None,
    dependencies: list[str] | None = None,
    gpu_hours: float = 0,
    storage_gb: float = 1,
    failure: str = "retry",
    run_if: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "objective": objective,
        "definition_of_done": done,
        "command": command,
        "validation_command": validation,
        "inputs": inputs or [],
        "outputs": outputs,
        "dependencies": dependencies or [],
        "estimated_gpu_hours": gpu_hours,
        "estimated_storage_gb": storage_gb,
        "failure_behavior": failure,
        "run_if": run_if,
    }


def build_full_plan(
    *,
    pinned_config: str = "configs/pinned-3090-bf16.yaml",
    thinking_config: str = "configs/pinned-thinking-3090-bf16.yaml",
    run_dir: str = "runs/v0",
    dataset_manifest: str = "datasets/manifests/full.jsonl",
) -> ExecutionPlan:
    rr = "reverse-reap"
    model = "/models/qwen"
    evaluator = "${REVERSE_REAP_EVALUATOR_IMAGE}"
    analysis = f"{run_dir}/analysis"
    gate_c = {
        "path": f"{analysis}/candidate-manifest.json",
        "field": "gate_passed",
        "equals": True,
    }
    tasks = [
        _task(
            "gpu-preflight",
            "Verify the approved four-RTX-3090 execution environment.",
            "Hardware report passes exact count, model, memory, capability, and disk checks.",
            ["python", "scripts/gpu_preflight.py", "--output", f"{run_dir}/preflight.json"],
            ["python", "scripts/validate_artifact.py", "preflight", f"{run_dir}/preflight.json"],
            [f"{run_dir}/preflight.json"],
            gpu_hours=0.01,
            failure="terminal",
        ),
        _task(
            "dataset-freeze",
            "Resolve, normalize, audit, and freeze all coding and control sources.",
            "At least 500 coding and 500 control rows are immutable with source provenance.",
            [rr, "fetch-datasets", "configs/dataset-sources.yaml", dataset_manifest],
            ["python", "scripts/validate_dataset.py", dataset_manifest, "--full"],
            [dataset_manifest, "datasets/manifests/full.sources.json"],
            inputs=["configs/dataset-sources.yaml"],
            storage_gb=3,
            failure="terminal",
        ),
        _task(
            "instrumentation-probe",
            "Prove exact Qwen routing capture and no-op behavioral equivalence.",
            "Exact logits, routed row count, and architecture contract all pass.",
            [rr, "probe", pinned_config, model, "--output", f"{run_dir}/probe.json"],
            ["python", "scripts/validate_artifact.py", "probe", f"{run_dir}/probe.json"],
            [f"{run_dir}/probe.json"],
            inputs=[pinned_config],
            dependencies=["gpu-preflight"],
            gpu_hours=0.5,
            failure="terminal",
        ),
    ]
    for split in ("calibration", "selection"):
        task_id = f"telemetry-{split}"
        path = f"{run_dir}/telemetry-{split}.jsonl"
        report = f"{run_dir}/telemetry-{split}-validation.json"
        tasks.append(
            _task(
                task_id,
                f"Capture token-level streaming REAP telemetry for frozen {split} rows.",
                "Every token-layer group has exactly eight unique finite routed expert rows.",
                [rr, "capture", pinned_config, model, dataset_manifest, path, "--split", split],
                [rr, "validate-telemetry", path, "--output", report],
                [path, report],
                inputs=[pinned_config, dataset_manifest],
                dependencies=["instrumentation-probe", "dataset-freeze"],
                gpu_hours=12,
                storage_gb=30,
            )
        )
    tasks.append(
        _task(
            "merge-telemetry",
            "Merge only calibration and selection routing chunks without duplicate identities.",
            "Merged telemetry validates and contains no validation or replication samples.",
            [
                rr,
                "merge-telemetry",
                f"{run_dir}/telemetry.jsonl",
                f"{run_dir}/telemetry-calibration.jsonl",
                f"{run_dir}/telemetry-selection.jsonl",
            ],
            [rr, "validate-telemetry", f"{run_dir}/telemetry.jsonl"],
            [f"{run_dir}/telemetry.jsonl"],
            inputs=[
                f"{run_dir}/telemetry-calibration.jsonl",
                f"{run_dir}/telemetry-selection.jsonl",
            ],
            dependencies=["telemetry-calibration", "telemetry-selection"],
            storage_gb=60,
            failure="terminal",
        )
    )
    tasks.append(
        _task(
            "candidate-analysis",
            "Compute differential ranking, uncertainty, permutations, and frozen controls.",
            "Gate C decision and all selected/random/frequency/negative manifests are hashed.",
            [rr, "analyze", f"{run_dir}/telemetry.jsonl", analysis],
            [
                "python",
                "scripts/validate_artifact.py",
                "candidate",
                f"{analysis}/candidate-manifest.json",
            ],
            [
                f"{analysis}/candidate-manifest.json",
                f"{analysis}/bootstrap-stability.json",
                f"{analysis}/label-permutation.json",
                f"{analysis}/control-manifests.json",
            ],
            inputs=[f"{run_dir}/telemetry.jsonl"],
            dependencies=["merge-telemetry"],
            storage_gb=2,
            failure="terminal",
        )
    )
    for suffix in ("a", "b"):
        tasks.append(
            _task(
                f"baseline-validation-{suffix}",
                "Run the intact C0 validation baseline for exact repeatability.",
                "All validation rows have explicit scores/errors and the output is immutable.",
                [
                    rr,
                    "evaluate",
                    pinned_config,
                    model,
                    dataset_manifest,
                    f"{run_dir}/baseline-validation-{suffix}.jsonl",
                    "--split",
                    "validation",
                    "--condition-id",
                    "C0",
                    "--evaluator-image",
                    evaluator,
                ],
                [
                    "python",
                    "scripts/validate_artifact.py",
                    "dataset",
                    f"{run_dir}/baseline-validation-{suffix}.jsonl",
                ],
                [f"{run_dir}/baseline-validation-{suffix}.jsonl"],
                inputs=[pinned_config, dataset_manifest],
                dependencies=["instrumentation-probe", "dataset-freeze"],
                gpu_hours=8,
                storage_gb=3,
            )
        )
    tasks.append(
        _task(
            "baseline-determinism",
            "Prove Gate B scoreability and exact repeated C0 responses.",
            "At least 95 percent are scoreable and no repeated response/score differs.",
            [
                rr,
                "compare-evaluations",
                f"{run_dir}/baseline-validation-a.jsonl",
                f"{run_dir}/baseline-validation-b.jsonl",
                "--output",
                f"{run_dir}/baseline-determinism.json",
            ],
            [
                "python",
                "scripts/validate_artifact.py",
                "probe",
                f"{run_dir}/baseline-determinism.json",
            ],
            [f"{run_dir}/baseline-determinism.json"],
            inputs=[
                f"{run_dir}/baseline-validation-a.jsonl",
                f"{run_dir}/baseline-validation-b.jsonl",
            ],
            dependencies=["baseline-validation-a", "baseline-validation-b"],
            failure="terminal",
        )
    )
    selected_validation = f"{run_dir}/selected-validation.jsonl"
    tasks.append(
        _task(
            "selected-validation",
            "Measure the frozen selected-set C2 causal intervention on validation data.",
            "Paired selected-set results exist for the unchanged validation membership.",
            [
                rr,
                "evaluate",
                pinned_config,
                model,
                dataset_manifest,
                selected_validation,
                "--split",
                "validation",
                "--condition-id",
                "C2",
                "--evaluator-image",
                evaluator,
                "--expert-manifest",
                f"{analysis}/candidate-manifest.json",
            ],
            ["python", "scripts/validate_artifact.py", "dataset", selected_validation],
            [selected_validation],
            inputs=[pinned_config, dataset_manifest, f"{analysis}/candidate-manifest.json"],
            dependencies=["candidate-analysis", "baseline-determinism"],
            gpu_hours=8,
            storage_gb=3,
            run_if=gate_c,
        )
    )
    random_task_ids = []
    random_results = []
    for index in range(20):
        task_id = f"layer-random-{index:03d}"
        manifest = f"{analysis}/controls/{task_id}.json"
        result = f"{run_dir}/{task_id}-validation.jsonl"
        random_task_ids.append(task_id)
        random_results.append(result)
        tasks.append(
            _task(
                task_id,
                "Evaluate one pre-frozen equal-cardinality layer-matched random control.",
                "Paired validation results exist without changing the frozen random manifest.",
                [
                    rr,
                    "evaluate",
                    pinned_config,
                    model,
                    dataset_manifest,
                    result,
                    "--split",
                    "validation",
                    "--condition-id",
                    "C3",
                    "--evaluator-image",
                    evaluator,
                    "--expert-manifest",
                    manifest,
                ],
                ["python", "scripts/validate_artifact.py", "dataset", result],
                [result],
                inputs=[pinned_config, dataset_manifest, manifest],
                dependencies=["candidate-analysis", "baseline-determinism"],
                gpu_hours=8,
                storage_gb=3,
                run_if=gate_c,
            )
        )
    for control_id, condition in (("frequency-matched", "C4"), ("lowest-differential", "C5")):
        result = f"{run_dir}/{control_id}-validation.jsonl"
        tasks.append(
            _task(
                control_id,
                f"Evaluate the frozen {control_id} causal control.",
                "Paired validation results exist for the unchanged control manifest.",
                [
                    rr,
                    "evaluate",
                    pinned_config,
                    model,
                    dataset_manifest,
                    result,
                    "--split",
                    "validation",
                    "--condition-id",
                    condition,
                    "--evaluator-image",
                    evaluator,
                    "--expert-manifest",
                    f"{analysis}/controls/{control_id}.json",
                ],
                ["python", "scripts/validate_artifact.py", "dataset", result],
                [result],
                inputs=[pinned_config, dataset_manifest, f"{analysis}/controls/{control_id}.json"],
                dependencies=["candidate-analysis", "baseline-determinism"],
                gpu_hours=8,
                storage_gb=3,
                run_if=gate_c,
            )
        )
    for mode in ("baseline", "selected"):
        result = f"{run_dir}/{mode}-replication.jsonl"
        command = [
            rr,
            "evaluate",
            pinned_config,
            model,
            dataset_manifest,
            result,
            "--split",
            "replication",
            "--condition-id",
            "C0" if mode == "baseline" else "C2",
            "--evaluator-image",
            evaluator,
        ]
        inputs = [pinned_config, dataset_manifest]
        if mode == "selected":
            command.extend(["--expert-manifest", f"{analysis}/candidate-manifest.json"])
            inputs.append(f"{analysis}/candidate-manifest.json")
        tasks.append(
            _task(
                f"{mode}-replication",
                f"Run untouched {mode} replication only after validation controls finish.",
                "Every frozen replication member has a score or explicit failure record.",
                command,
                ["python", "scripts/validate_artifact.py", "dataset", result],
                [result],
                inputs=inputs,
                dependencies=[
                    "selected-validation",
                    *random_task_ids,
                    "frequency-matched",
                    "lowest-differential",
                ],
                gpu_hours=8,
                storage_gb=3,
                run_if=gate_c,
            )
        )
    causal_report = f"{run_dir}/causal-report.json"
    tasks.append(
        _task(
            "causal-report",
            "Apply Gate D from paired validation, random controls, and untouched replication.",
            "The report computes every Gate D criterion and assigns only an allowed "
            "evidence label.",
            [
                rr,
                "causal-report",
                f"{run_dir}/baseline-validation-a.jsonl",
                selected_validation,
                causal_report,
                *random_results,
                "--replication-baseline",
                f"{run_dir}/baseline-replication.jsonl",
                "--replication-selected",
                f"{run_dir}/selected-replication.jsonl",
            ],
            [
                "python",
                "scripts/validate_artifact.py",
                "causal",
                causal_report,
            ],
            [causal_report],
            inputs=[
                f"{run_dir}/baseline-validation-a.jsonl",
                selected_validation,
                *random_results,
                f"{run_dir}/baseline-replication.jsonl",
                f"{run_dir}/selected-replication.jsonl",
            ],
            dependencies=["baseline-replication", "selected-replication"],
            failure="terminal",
            run_if=gate_c,
        )
    )
    tasks.append(
        _task(
            "extract-candidates",
            "Copy every frozen selected expert tensor and independently verify source bytes.",
            "Complete extraction, source map, checksums, verification, and dependency "
            "warning exist.",
            [
                rr,
                "extract",
                pinned_config,
                model,
                f"{analysis}/candidate-manifest.json",
                f"{run_dir}/extraction",
            ],
            [
                "python",
                "scripts/validate_artifact.py",
                "extraction",
                f"{run_dir}/extraction/extraction-manifest.json",
            ],
            [
                f"{run_dir}/extraction/extraction-manifest.json",
                f"{run_dir}/extraction/experts.safetensors",
                f"{run_dir}/extraction/verification-report.json",
            ],
            inputs=[pinned_config, f"{analysis}/candidate-manifest.json"],
            dependencies=["causal-report"],
            gpu_hours=0.2,
            storage_gb=20,
            failure="terminal",
        )
    )
    tasks.append(
        _task(
            "final-bundle",
            "Inventory all evidence, hashes, costs, gates, and limitations for human review.",
            "A conservative terminal or incomplete classification and full artifact "
            "inventory exist.",
            [
                rr,
                "build-bundle",
                pinned_config,
                run_dir,
                f"{run_dir}/state",
                f"{run_dir}/manifest.json",
            ],
            ["python", "-m", "json.tool", f"{run_dir}/manifest.json"],
            [f"{run_dir}/manifest.json"],
            inputs=[pinned_config, f"{run_dir}/extraction/extraction-manifest.json"],
            dependencies=["extract-candidates", "thinking-selected-pilot"],
            failure="terminal",
        )
    )
    # C1/C6 use their own pinned config and are intentionally separate from the C0 causal gate.
    tasks.extend(
        [
            _task(
                "thinking-baseline-pilot",
                "Run a bounded C1 thinking-enabled pilot without pooling C0 evidence.",
                "A separately identified thinking baseline artifact exists.",
                [
                    rr,
                    "evaluate",
                    thinking_config,
                    model,
                    dataset_manifest,
                    f"{run_dir}/thinking-baseline.jsonl",
                    "--split",
                    "validation",
                    "--condition-id",
                    "C1",
                    "--evaluator-image",
                    evaluator,
                    "--limit",
                    "20",
                ],
                [
                    "python",
                    "scripts/validate_artifact.py",
                    "dataset",
                    f"{run_dir}/thinking-baseline.jsonl",
                ],
                [f"{run_dir}/thinking-baseline.jsonl"],
                inputs=[thinking_config, dataset_manifest],
                dependencies=["causal-report"],
                gpu_hours=1,
                storage_gb=1,
                run_if=gate_c,
            ),
            _task(
                "thinking-selected-pilot",
                "Run the same bounded thinking pilot with the frozen C6 selected intervention.",
                "A separately identified thinking selected-set artifact exists.",
                [
                    rr,
                    "evaluate",
                    thinking_config,
                    model,
                    dataset_manifest,
                    f"{run_dir}/thinking-selected.jsonl",
                    "--split",
                    "validation",
                    "--condition-id",
                    "C6",
                    "--evaluator-image",
                    evaluator,
                    "--expert-manifest",
                    f"{analysis}/candidate-manifest.json",
                    "--limit",
                    "20",
                ],
                [
                    "python",
                    "scripts/validate_artifact.py",
                    "dataset",
                    f"{run_dir}/thinking-selected.jsonl",
                ],
                [f"{run_dir}/thinking-selected.jsonl"],
                inputs=[thinking_config, dataset_manifest, f"{analysis}/candidate-manifest.json"],
                dependencies=["thinking-baseline-pilot"],
                gpu_hours=1,
                storage_gb=1,
                run_if=gate_c,
            ),
        ]
    )
    return ExecutionPlan.model_validate({"schema_version": 1, "tasks": tasks})


def write_full_plan(destination: Path, **kwargs: Any) -> ExecutionPlan:
    plan = build_full_plan(**kwargs)
    body = yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False, width=120)
    if destination.exists() and destination.read_text(encoding="utf-8") != body:
        raise ValueError(f"refusing to overwrite execution plan: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(body, encoding="utf-8")
    return plan
