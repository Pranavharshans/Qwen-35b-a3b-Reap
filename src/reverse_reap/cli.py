"""Deterministic command-line entrypoint for Reverse-REAP."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from reverse_reap.causal import (
    causal_gate_report,
    compare_deterministic_evaluations,
    evaluate_condition,
)
from reverse_reap.config import load_config
from reverse_reap.controller import run_all, run_next, run_status
from reverse_reap.datasets import freeze_tiers
from reverse_reap.extraction import (
    architecture_from_weight_index,
    extract_experts,
    verify_extraction,
)
from reverse_reap.model_preflight import download_verified_weights, preflight_model
from reverse_reap.pipeline import analyze_telemetry
from reverse_reap.plans import write_full_plan
from reverse_reap.reporting import build_run_bundle
from reverse_reap.runtime import capture_manifest, probe_instrumentation
from reverse_reap.sources import fetch_and_freeze
from reverse_reap.telemetry import merge_telemetry, validate_telemetry


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def validate_config(path: Path) -> int:
    config = load_config(path)
    output = {
        "valid": True,
        "config_sha256": config.fingerprint(),
        "resolved_run_id": config.run_id or config.resolve_run_id(git_sha()),
        "thinking_enabled": config.runtime.enable_thinking,
        "chat_template_kwargs": {"enable_thinking": config.runtime.enable_thinking},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def emit_json(output: object, destination: Path | None = None) -> None:
    rendered = json.dumps(output, indent=2, sort_keys=True, default=str) + "\n"
    print(rendered, end="")
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reverse-reap")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    run = subparsers.add_parser("run-next")
    run.add_argument("config", type=Path)
    run.add_argument("plan", type=Path)
    run.add_argument("state_dir", type=Path)
    run.add_argument("--heartbeat-seconds", type=float, default=30)
    run_all_parser = subparsers.add_parser("run-all")
    run_all_parser.add_argument("config", type=Path)
    run_all_parser.add_argument("plan", type=Path)
    run_all_parser.add_argument("state_dir", type=Path)
    run_all_parser.add_argument("--heartbeat-seconds", type=float, default=30)
    run_all_parser.add_argument("--stale-after-seconds", type=float, default=180)
    status = subparsers.add_parser("status")
    status.add_argument("state_dir", type=Path)
    capture = subparsers.add_parser("capture")
    capture.add_argument("config", type=Path)
    capture.add_argument("model_path", type=Path)
    capture.add_argument("manifest", type=Path)
    capture.add_argument("destination", type=Path)
    capture.add_argument("--split", default="calibration")
    capture.add_argument("--limit", type=int)
    probe = subparsers.add_parser("probe")
    probe.add_argument("config", type=Path)
    probe.add_argument("model_path", type=Path)
    probe.add_argument(
        "--prompt", default="Write a Python function that returns the sum of two integers."
    )
    probe.add_argument("--output", type=Path)
    fetch = subparsers.add_parser("fetch-datasets")
    fetch.add_argument("catalog", type=Path)
    fetch.add_argument("destination", type=Path)
    tiers = subparsers.add_parser("freeze-dataset-tiers")
    tiers.add_argument("full_manifest", type=Path)
    tiers.add_argument("destination_dir", type=Path)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("telemetry", type=Path)
    analyze.add_argument("output_dir", type=Path)
    analyze.add_argument("--top-n", type=int, default=32)
    analyze.add_argument("--bootstrap-iterations", type=int, default=1000)
    analyze.add_argument("--permutation-iterations", type=int, default=1000)
    analyze.add_argument("--seed", type=int, default=20260903)
    analyze.add_argument("--cardinality-grid", type=int, nargs="+")
    extract = subparsers.add_parser("extract")
    extract.add_argument("config", type=Path)
    extract.add_argument("model_path", type=Path)
    extract.add_argument("candidate_manifest", type=Path)
    extract.add_argument("destination", type=Path)
    verify = subparsers.add_parser("verify-extraction")
    verify.add_argument("model_path", type=Path)
    verify.add_argument("destination", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("model_path", type=Path)
    evaluate.add_argument("dataset_manifest", type=Path)
    evaluate.add_argument("destination", type=Path)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--condition-id", required=True)
    evaluate.add_argument("--evaluator-image", required=True)
    evaluate.add_argument("--expert-manifest", type=Path)
    evaluate.add_argument("--limit", type=int)
    gate = subparsers.add_parser("causal-report")
    gate.add_argument("baseline", type=Path)
    gate.add_argument("selected", type=Path)
    gate.add_argument("destination", type=Path)
    gate.add_argument("random", type=Path, nargs="+")
    gate.add_argument("--replication-baseline", type=Path)
    gate.add_argument("--replication-selected", type=Path)
    bundle = subparsers.add_parser("build-bundle")
    bundle.add_argument("config", type=Path)
    bundle.add_argument("run_dir", type=Path)
    bundle.add_argument("state_dir", type=Path)
    bundle.add_argument("destination", type=Path)
    telemetry = subparsers.add_parser("validate-telemetry")
    telemetry.add_argument("path", type=Path)
    telemetry.add_argument("--output", type=Path)
    model_preflight = subparsers.add_parser("preflight-model")
    model_preflight.add_argument("template_config", type=Path)
    model_preflight.add_argument("pinned_config", type=Path)
    model_preflight.add_argument("metadata_dir", type=Path)
    model_preflight.add_argument("report", type=Path)
    weights = subparsers.add_parser("download-weights")
    weights.add_argument("preflight_report", type=Path)
    weights.add_argument("destination", type=Path)
    compare = subparsers.add_parser("compare-evaluations")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--output", type=Path)
    merge = subparsers.add_parser("merge-telemetry")
    merge.add_argument("destination", type=Path)
    merge.add_argument("inputs", type=Path, nargs="+")
    make_plan = subparsers.add_parser("make-full-plan")
    make_plan.add_argument("destination", type=Path)
    make_plan.add_argument("--pinned-config", default="configs/pinned-3090-bf16.yaml")
    make_plan.add_argument("--thinking-config", default="configs/pinned-thinking-3090-bf16.yaml")
    make_plan.add_argument("--run-dir", default="runs/v0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-config":
        return validate_config(args.config)
    if args.command == "run-next":
        config = load_config(args.config)
        resolved_run_id = config.run_id or config.resolve_run_id(git_sha())
        output = run_next(
            args.plan,
            config,
            args.state_dir,
            run_id=resolved_run_id,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        emit_json(output)
        return 0 if output["status"] == "COMPLETE" else 2
    if args.command == "run-all":
        config = load_config(args.config)
        output = run_all(
            args.plan,
            config,
            args.state_dir,
            run_id=config.run_id or config.resolve_run_id(git_sha()),
            heartbeat_seconds=args.heartbeat_seconds,
            stale_after_seconds=args.stale_after_seconds,
        )
        emit_json(output)
        return 0 if output["status"] == "COMPLETE" else 2
    if args.command == "status":
        emit_json(run_status(args.state_dir))
        return 0
    if args.command == "capture":
        output = capture_manifest(
            args.model_path,
            args.manifest,
            args.destination,
            load_config(args.config),
            split=args.split,
            limit=args.limit,
        )
        emit_json(output)
        return 0
    if args.command == "probe":
        output = probe_instrumentation(args.model_path, load_config(args.config), args.prompt)
        emit_json(output, args.output)
        return 0 if output["passed"] else 3
    if args.command == "fetch-datasets":
        output = fetch_and_freeze(args.catalog, args.destination)
        emit_json(output)
        return 0
    if args.command == "freeze-dataset-tiers":
        emit_json(freeze_tiers(args.full_manifest, args.destination_dir))
        return 0
    if args.command == "analyze":
        output = analyze_telemetry(
            args.telemetry,
            args.output_dir,
            top_n=args.top_n,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            seed=args.seed,
            cardinality_grid=(
                tuple(args.cardinality_grid) if args.cardinality_grid is not None else None
            ),
        )
        emit_json(output)
        return 0
    if args.command == "extract":
        config = load_config(args.config)
        candidates = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
        selected = [(int(item["layer"]), int(item["expert"])) for item in candidates["experts"]]
        output = extract_experts(
            args.model_path,
            architecture_from_weight_index(args.model_path),
            selected,
            args.destination,
            model_id=config.model.id,
            model_revision=config.model.revision,
            run_id=config.run_id or config.resolve_run_id(git_sha()),
            selection_status=(
                "domain-differential candidate"
                if candidates.get("gate_passed")
                else "observational-candidates"
            ),
            selection_metrics={
                "method": candidates.get("selection_method"),
                "thresholds": candidates.get("thresholds"),
            },
            tool_git_revision=git_sha(),
        )
        emit_json(output)
        return 0
    if args.command == "verify-extraction":
        output = verify_extraction(args.destination, args.model_path)
        emit_json(output)
        return 0
    if args.command == "evaluate":
        output = evaluate_condition(
            args.model_path,
            args.dataset_manifest,
            args.destination,
            load_config(args.config),
            split=args.split,
            condition_id=args.condition_id,
            evaluator_image=args.evaluator_image,
            expert_manifest=args.expert_manifest,
            limit=args.limit,
        )
        emit_json(output)
        return 0
    if args.command == "causal-report":
        output = causal_gate_report(
            args.baseline,
            args.selected,
            args.random,
            replication_baseline_path=args.replication_baseline,
            replication_selected_path=args.replication_selected,
        )
        emit_json(output, args.destination)
        return 0
    if args.command == "build-bundle":
        output = build_run_bundle(
            args.run_dir, args.state_dir, load_config(args.config), args.destination
        )
        emit_json(output)
        return 0
    if args.command == "validate-telemetry":
        output = validate_telemetry(args.path)
        emit_json(output, args.output)
        return 0
    if args.command == "preflight-model":
        output = preflight_model(
            args.template_config, args.pinned_config, args.metadata_dir, args.report
        )
        emit_json(output)
        return 0 if output["passed"] else 2
    if args.command == "download-weights":
        output = download_verified_weights(args.preflight_report, args.destination)
        emit_json(output)
        return 0
    if args.command == "compare-evaluations":
        output = compare_deterministic_evaluations(args.first, args.second)
        emit_json(output, args.output)
        return 0 if output["passed"] else 2
    if args.command == "merge-telemetry":
        emit_json(merge_telemetry(args.inputs, args.destination))
        return 0
    if args.command == "make-full-plan":
        plan = write_full_plan(
            args.destination,
            pinned_config=args.pinned_config,
            thinking_config=args.thinking_config,
            run_dir=args.run_dir,
        )
        emit_json({"valid": True, "tasks": len(plan.tasks), "path": str(args.destination)})
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
