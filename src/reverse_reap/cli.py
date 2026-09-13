"""Deterministic command-line entrypoint for Reverse-REAP."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from reverse_reap.bridge_benchmark import (
    fetch_and_freeze_humanevalplus,
    run_bridge_benchmark,
    score_bridge_benchmark,
)
from reverse_reap.bridge_capture import (
    build_target_handoff,
    freeze_bridge_manifest,
    tokenizer_fingerprint,
    validate_target_handoff,
    validate_target_shard,
)
from reverse_reap.bridge_rescue import (
    fit_linear_gate_manifest,
    freeze_mbpp_strength_screen,
    load_rescue_experiment_config,
    run_strength_screen,
    score_strength_screen,
)
from reverse_reap.bridge_training import (
    BridgeExpertMapping,
    capture_host_hidden_states,
    estimate_bridge_memory,
    freeze_host_state_manifest,
    load_bridge_training_config,
    repair_bridge_manifest,
    train_bridge,
    validate_bridge_training_config,
)
from reverse_reap.causal import (
    causal_gate_report,
    compare_deterministic_evaluations,
    evaluate_condition,
)
from reverse_reap.config import load_config
from reverse_reap.controller import run_all, run_next, run_status
from reverse_reap.datasets import audit_manifest_token_lengths, freeze_tiers
from reverse_reap.extraction import (
    architecture_from_weight_index,
    extract_experts,
    verify_extraction,
)
from reverse_reap.mbpp_bridge_benchmark import (
    load_mbpp_bridge_config,
    run_mbpp_bridge_benchmark,
    score_mbpp_bridge_benchmark,
    validate_mbpp_generation,
)
from reverse_reap.model_preflight import download_verified_weights, preflight_model
from reverse_reap.pipeline import analyze_telemetry
from reverse_reap.plans import write_full_plan
from reverse_reap.reporting import build_run_bundle
from reverse_reap.runtime import (
    capture_manifest,
    capture_targeted_manifest,
    probe_instrumentation,
    probe_single_expert_intervention,
)
from reverse_reap.sources import fetch_and_freeze
from reverse_reap.swebench import export_predictions, merge_report
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
    capture.add_argument("--report", type=Path)
    bridge_freeze = subparsers.add_parser("freeze-bridge-manifest")
    bridge_freeze.add_argument("full_manifest", type=Path)
    bridge_freeze.add_argument("tokenizer_path", type=Path)
    bridge_freeze.add_argument("destination", type=Path)
    bridge_freeze.add_argument("--config", type=Path, required=True)
    bridge_freeze.add_argument("--candidate-manifest", type=Path, required=True)
    bridge_freeze.add_argument("--model-revision", required=True)
    bridge_freeze.add_argument("--run-id", required=True)
    bridge_freeze.add_argument("--target-tokens", type=int, default=200_000)
    bridge_freeze.add_argument("--hard-token-ceiling", type=int, default=500_000)
    bridge_freeze.add_argument("--max-input-tokens", type=int, default=1024)
    bridge_freeze.add_argument("--seed", type=int, default=20260903)
    bridge_freeze.add_argument("--report", type=Path)
    bridge_capture = subparsers.add_parser("capture-targets")
    bridge_capture.add_argument("config", type=Path)
    bridge_capture.add_argument("model_path", type=Path)
    bridge_capture.add_argument("capture_manifest", type=Path)
    bridge_capture.add_argument("destination", type=Path)
    bridge_capture.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8))
    bridge_capture.add_argument("--left-padding", action="store_true")
    bridge_capture.add_argument("--shard-max-records", type=int, default=4096)
    bridge_capture.add_argument("--run-id")
    bridge_capture.add_argument("--report", type=Path)
    bridge_validate = subparsers.add_parser("validate-target-shard")
    bridge_validate.add_argument("shard", type=Path)
    bridge_validate.add_argument("--hidden-size", type=int, default=2048)
    bridge_bundle = subparsers.add_parser("build-target-handoff")
    bridge_bundle.add_argument("capture_root", type=Path)
    bridge_bundle.add_argument("capture_manifest", type=Path)
    bridge_bundle.add_argument("destination", type=Path)
    bridge_bundle.add_argument("--extraction-dir", type=Path)
    bridge_bundle.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="bundle a hash-valid but coverage-incomplete capture with an "
        "explicit coverage-incomplete classification instead of failing",
    )
    bridge_bundle_validate = subparsers.add_parser("validate-target-handoff")
    bridge_bundle_validate.add_argument("handoff", type=Path)
    bridge_config = subparsers.add_parser("validate-bridge-config")
    bridge_config.add_argument("config", type=Path)
    bridge_config.add_argument("--output", type=Path)
    bridge_repair = subparsers.add_parser("repair-bridge-manifest")
    bridge_repair.add_argument("config", type=Path)
    bridge_repair.add_argument("--output", type=Path)
    bridge_repair_raw = subparsers.add_parser("repair-bridge-manifest-raw")
    bridge_repair_raw.add_argument("handoff", type=Path)
    bridge_repair_raw.add_argument("host_states", type=Path)
    bridge_repair_raw.add_argument("destination", type=Path)
    bridge_repair_raw.add_argument(
        "--mapping",
        action="append",
        required=True,
        metavar="DONOR_LAYER:DONOR_EXPERT:HOST_LAYER",
    )
    bridge_repair_raw.add_argument("--seed", type=int, default=20260909)
    bridge_repair_raw.add_argument("--min-samples-per-cell", type=int, default=1)
    bridge_repair_raw.add_argument("--min-events-per-cell", type=int, default=32)
    bridge_repair_raw.add_argument("--max-rows-per-sample", type=int, default=128)
    bridge_repair_raw.add_argument("--allow-observational-coverage-incomplete", action="store_true")
    host_capture = subparsers.add_parser("capture-host-states")
    host_capture.add_argument("host_model", type=Path)
    host_capture.add_argument("tokenizer", type=Path)
    host_capture.add_argument("handoff", type=Path)
    host_capture.add_argument("capture_manifest", type=Path)
    host_capture.add_argument("destination", type=Path)
    host_capture.add_argument("--host-revision", required=True)
    host_capture.add_argument("--run-id", required=True)
    host_capture.add_argument("--allow-observational-coverage-incomplete", action="store_true")
    host_capture.add_argument(
        "--mapping",
        action="append",
        required=True,
        metavar="DONOR_LAYER:DONOR_EXPERT:HOST_LAYER",
    )
    bridge_preflight = subparsers.add_parser("bridge-preflight")
    bridge_preflight.add_argument("vram_bytes", type=int)
    bridge_preflight.add_argument("--host-parameter-count", type=int, default=2_000_000_000)
    bridge_preflight.add_argument("--sequence-tokens", type=int, default=1024)
    bridge_preflight.add_argument("--mapped-experts", type=int, default=4)
    bridge_preflight.add_argument("--output", type=Path)
    host_states = subparsers.add_parser("freeze-host-states")
    host_states.add_argument("tensors", type=Path)
    host_states.add_argument("records", type=Path)
    host_states.add_argument("destination", type=Path)
    host_states.add_argument("--run-id", required=True)
    host_states.add_argument("--host-revision", required=True)
    bridge_train = subparsers.add_parser("train-bridge")
    bridge_train.add_argument("config", type=Path)
    bridge_train.add_argument("--resume-checkpoint", type=Path)
    rescue_validate = subparsers.add_parser("validate-bridge-rescue-config")
    rescue_validate.add_argument("config", type=Path)
    rescue_gate = subparsers.add_parser("fit-bridge-rescue-gate")
    rescue_gate.add_argument("manifest", type=Path)
    rescue_gate.add_argument("destination", type=Path)
    rescue_gate.add_argument("--manifest-sha256", required=True)
    rescue_gate.add_argument("--max-generated-tokens", type=int, required=True)
    rescue_freeze = subparsers.add_parser("freeze-bridge-strength-screen")
    rescue_freeze.add_argument("benchmark_config", type=Path)
    rescue_freeze.add_argument("destination", type=Path)
    rescue_run = subparsers.add_parser("run-bridge-strength-screen")
    rescue_run.add_argument("config", type=Path)
    rescue_score = subparsers.add_parser("score-bridge-strength-screen")
    rescue_score.add_argument("config", type=Path)
    rescue_score.add_argument("--evalplus-image", required=True)
    bridge_benchmark = subparsers.add_parser("run-bridge-benchmark")
    bridge_benchmark.add_argument("config", type=Path)
    bridge_dataset = subparsers.add_parser("freeze-humanevalplus")
    bridge_dataset.add_argument("revision")
    bridge_dataset.add_argument("destination", type=Path)
    bridge_dataset.add_argument("--seed", type=int, default=20260909)
    bridge_score = subparsers.add_parser("score-bridge-benchmark")
    bridge_score.add_argument("config", type=Path)
    bridge_score.add_argument("--evaluator-image", required=True)
    bridge_score.add_argument("--exclusion-manifest", type=Path, default=None)
    mbpp_bridge = subparsers.add_parser("run-mbpp-bridge-benchmark")
    mbpp_bridge.add_argument("config", type=Path)
    mbpp_bridge_validate = subparsers.add_parser("validate-mbpp-bridge-benchmark")
    mbpp_bridge_validate.add_argument("config", type=Path)
    mbpp_bridge_score = subparsers.add_parser("score-mbpp-bridge-benchmark")
    mbpp_bridge_score.add_argument("config", type=Path)
    mbpp_bridge_score.add_argument("--evalplus-image", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("config", type=Path)
    probe.add_argument("model_path", type=Path)
    probe.add_argument(
        "--prompt", default="Write a Python function that returns the sum of two integers."
    )
    probe.add_argument("--output", type=Path)
    intervention_probe = subparsers.add_parser("probe-intervention")
    intervention_probe.add_argument("config", type=Path)
    intervention_probe.add_argument("model_path", type=Path)
    intervention_probe.add_argument("dataset_manifest", type=Path)
    intervention_probe.add_argument("candidate_manifest", type=Path)
    intervention_probe.add_argument("output", type=Path)
    intervention_probe.add_argument("--split", default="selection")
    intervention_probe.add_argument("--limit", type=int, default=20)
    fetch = subparsers.add_parser("fetch-datasets")
    fetch.add_argument("catalog", type=Path)
    fetch.add_argument("destination", type=Path)
    tiers = subparsers.add_parser("freeze-dataset-tiers")
    tiers.add_argument("full_manifest", type=Path)
    tiers.add_argument("destination_dir", type=Path)
    rebalance = subparsers.add_parser("rebalance-controls")
    rebalance.add_argument("full_manifest", type=Path)
    rebalance.add_argument("destination_dir", type=Path)
    rebalance.add_argument("tokenizer_path", type=Path)
    rebalance.add_argument("--seed", type=int, default=20260903)
    lengths = subparsers.add_parser("audit-token-lengths")
    lengths.add_argument("manifest", type=Path)
    lengths.add_argument("tokenizer_path", type=Path)
    lengths.add_argument("output", type=Path)
    lengths.add_argument("--max-input-tokens", type=int, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("telemetry", type=Path)
    analyze.add_argument("output_dir", type=Path)
    analyze.add_argument("--top-n", type=int, default=32)
    analyze.add_argument("--bootstrap-iterations", type=int, default=1000)
    analyze.add_argument("--permutation-iterations", type=int, default=1000)
    analyze.add_argument("--seed", type=int, default=20260903)
    analyze.add_argument("--cardinality-grid", type=int, nargs="+")
    analyze.add_argument(
        "--engine",
        choices=("fast", "reference"),
        default="fast",
        help="analysis engine: vectorized streaming (fast) or dict-based reference oracle",
    )
    analyze.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="directory for the telemetry-SHA-keyed aggregate cache (default .cache/analysis)",
    )
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
    telemetry.add_argument("--num-layers", type=int, default=40)
    telemetry.add_argument("--num-experts", type=int, default=256)
    telemetry.add_argument("--top-k", type=int, default=8)
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
    swe_export = subparsers.add_parser("export-swebench")
    swe_export.add_argument("evaluation", type=Path)
    swe_export.add_argument("destination", type=Path)
    swe_export.add_argument("--model-name", required=True)
    swe_merge = subparsers.add_parser("merge-swebench")
    swe_merge.add_argument("evaluation", type=Path)
    swe_merge.add_argument("report", type=Path)
    swe_merge.add_argument("destination", type=Path)
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
        emit_json(output, args.report)
        return 0
    if args.command == "freeze-bridge-manifest":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_path), local_files_only=True, trust_remote_code=False
        )
        config = load_config(args.config)
        if config.model.revision != args.model_revision:
            raise SystemExit("--model-revision differs from the selected config")
        output = freeze_bridge_manifest(
            args.full_manifest,
            tokenizer,
            args.destination,
            model_id=config.model.id,
            model_revision=args.model_revision,
            tokenizer_fingerprint_value=tokenizer_fingerprint(args.tokenizer_path),
            config_sha256=config.fingerprint(),
            candidate_manifest=args.candidate_manifest,
            run_id=args.run_id,
            target_tokens=args.target_tokens,
            hard_token_ceiling=args.hard_token_ceiling,
            max_input_tokens=args.max_input_tokens,
            seed=args.seed,
        )
        emit_json(output, args.report)
        return 0
    if args.command == "capture-targets":
        config = load_config(args.config)
        output = capture_targeted_manifest(
            args.model_path,
            args.capture_manifest,
            args.destination,
            config,
            batch_size=args.batch_size,
            left_padding=args.left_padding,
            shard_max_records=args.shard_max_records,
            run_id=args.run_id,
        )
        emit_json(output, args.report)
        return 0
    if args.command == "validate-target-shard":
        emit_json(validate_target_shard(args.shard, hidden_size=args.hidden_size))
        return 0
    if args.command == "build-target-handoff":
        emit_json(
            build_target_handoff(
                args.capture_root,
                args.capture_manifest,
                args.destination,
                extraction_dir=args.extraction_dir,
                allow_incomplete=args.allow_incomplete,
            )
        )
        return 0
    if args.command == "validate-target-handoff":
        emit_json(validate_target_handoff(args.handoff))
        return 0
    if args.command == "validate-bridge-config":
        emit_json(validate_bridge_training_config(args.config), args.output)
        return 0
    if args.command == "repair-bridge-manifest":
        config = load_bridge_training_config(args.config)
        destination = args.output or config.data.training_manifest
        output = repair_bridge_manifest(
            config.data.handoff_manifest,
            config.data.host_states_manifest,
            destination,
            mappings=config.mappings,
            seed=config.runtime.seed,
            allow_observational_coverage_incomplete=(
                config.data.allow_observational_coverage_incomplete
            ),
        )
        emit_json(output)
        return 0
    if args.command == "repair-bridge-manifest-raw":
        try:
            mappings = [
                BridgeExpertMapping(
                    donor_layer=int(value.split(":")[0]),
                    donor_expert=int(value.split(":")[1]),
                    host_layer=int(value.split(":")[2]),
                )
                for value in args.mapping
                if len(value.split(":")) == 3
            ]
            if len(mappings) != len(args.mapping):
                raise ValueError("each --mapping must be LAYER:EXPERT:HOST_LAYER")
        except (TypeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        output = repair_bridge_manifest(
            args.handoff,
            args.host_states,
            args.destination,
            mappings=mappings,
            seed=args.seed,
            min_samples_per_cell=args.min_samples_per_cell,
            min_events_per_cell=args.min_events_per_cell,
            max_rows_per_sample=args.max_rows_per_sample,
            allow_observational_coverage_incomplete=(args.allow_observational_coverage_incomplete),
        )
        emit_json(output)
        return 0
    if args.command == "capture-host-states":
        try:
            mappings = [
                BridgeExpertMapping(
                    donor_layer=int(value.split(":")[0]),
                    donor_expert=int(value.split(":")[1]),
                    host_layer=int(value.split(":")[2]),
                )
                for value in args.mapping
                if len(value.split(":")) == 3
            ]
            if len(mappings) != len(args.mapping):
                raise ValueError("each --mapping must be LAYER:EXPERT:HOST_LAYER")
        except (TypeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        output = capture_host_hidden_states(
            args.host_model,
            args.tokenizer,
            args.handoff,
            args.capture_manifest,
            args.destination,
            mappings=mappings,
            host_revision=args.host_revision,
            run_id=args.run_id,
            allow_observational_coverage_incomplete=(args.allow_observational_coverage_incomplete),
        )
        emit_json(output)
        return 0
    if args.command == "bridge-preflight":
        report = estimate_bridge_memory(
            vram_bytes=args.vram_bytes,
            host_parameter_count=args.host_parameter_count,
            sequence_tokens=args.sequence_tokens,
            mapped_expert_count=args.mapped_experts,
        )
        emit_json(report, args.output)
        return 0 if report["passed"] else 2
    if args.command == "freeze-host-states":
        output = freeze_host_state_manifest(
            args.tensors,
            args.records,
            args.destination,
            run_id=args.run_id,
            host_revision=args.host_revision,
        )
        emit_json(output)
        return 0
    if args.command == "train-bridge":
        output = train_bridge(args.config, resume_checkpoint=args.resume_checkpoint)
        emit_json(output)
        return 0 if output["status"] == "PASS" else 2
    if args.command == "validate-bridge-rescue-config":
        config = load_rescue_experiment_config(args.config)
        emit_json(
            {
                "valid": True,
                "run_id": config.run_id,
                "experiment": config.experiment,
                "split": config.split,
                "expected_tasks": config.expected_tasks,
                "thinking_enabled": config.thinking_enabled,
                "policy_count": len(config.policies),
                "fingerprint": config.fingerprint(),
            }
        )
        return 0
    if args.command == "fit-bridge-rescue-gate":
        emit_json(
            fit_linear_gate_manifest(
                args.manifest,
                args.destination,
                expected_sha256=args.manifest_sha256,
                max_generated_tokens=args.max_generated_tokens,
            )
        )
        return 0
    if args.command == "freeze-bridge-strength-screen":
        emit_json(freeze_mbpp_strength_screen(args.benchmark_config, args.destination))
        return 0
    if args.command == "run-bridge-strength-screen":
        output = run_strength_screen(args.config)
        emit_json(output)
        return 0 if output["status"] == "PASS" else 2
    if args.command == "score-bridge-strength-screen":
        output = score_strength_screen(args.config, evalplus_image=args.evalplus_image)
        emit_json(output)
        return 0 if output["status"] == "PASS" else 2
    if args.command == "run-bridge-benchmark":
        output = run_bridge_benchmark(args.config)
        emit_json(output)
        return 0 if output["status"] == "PASS" else 2
    if args.command == "freeze-humanevalplus":
        output = fetch_and_freeze_humanevalplus(
            args.destination, revision=args.revision, seed=args.seed
        )
        emit_json(output)
        return 0
    if args.command == "score-bridge-benchmark":
        output = score_bridge_benchmark(
            args.config,
            evaluator_image=args.evaluator_image,
            exclusion_manifest=args.exclusion_manifest,
        )
        emit_json(output)
        return 0 if output["status"] == "PASS" else 2
    if args.command == "run-mbpp-bridge-benchmark":
        output = run_mbpp_bridge_benchmark(args.config)
        emit_json(output)
        return 0 if output["status"] == "PASS" else 2
    if args.command == "validate-mbpp-bridge-benchmark":
        output = validate_mbpp_generation(load_mbpp_bridge_config(args.config, allow_expired=True))
        emit_json(output)
        return 0 if output["passed"] else 2
    if args.command == "score-mbpp-bridge-benchmark":
        output = score_mbpp_bridge_benchmark(args.config, evalplus_image=args.evalplus_image)
        emit_json(output)
        return 0 if output["status"] == "PASS" else 2
    if args.command == "probe":
        output = probe_instrumentation(args.model_path, load_config(args.config), args.prompt)
        emit_json(output, args.output)
        return 0 if output["passed"] else 3
    if args.command == "probe-intervention":
        output = probe_single_expert_intervention(
            args.model_path,
            args.dataset_manifest,
            args.candidate_manifest,
            load_config(args.config),
            split=args.split,
            limit=args.limit,
        )
        emit_json(output, args.output)
        return 0 if output["passed"] else 3
    if args.command == "fetch-datasets":
        output = fetch_and_freeze(args.catalog, args.destination)
        emit_json(output)
        return 0
    if args.command == "freeze-dataset-tiers":
        emit_json(freeze_tiers(args.full_manifest, args.destination_dir))
        return 0
    if args.command == "rebalance-controls":
        from transformers import AutoTokenizer

        from reverse_reap.datasets import rebalance_controls_by_length

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_path), local_files_only=True, trust_remote_code=False
        )
        emit_json(
            rebalance_controls_by_length(
                args.full_manifest, args.destination_dir, tokenizer, seed=args.seed
            )
        )
        return 0
    if args.command == "audit-token-lengths":
        output = audit_manifest_token_lengths(
            args.manifest,
            args.tokenizer_path,
            max_input_tokens=args.max_input_tokens,
        )
        emit_json(output, args.output)
        return 0 if output["passed"] else 2
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
            engine=args.engine,
            cache_dir=args.cache_dir,
        )
        emit_json(output)
        return 0
    if args.command == "extract":
        config = load_config(args.config)
        candidates = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
        selected = [(int(item["layer"]), int(item["expert"])) for item in candidates["experts"]]
        output = extract_experts(
            args.model_path,
            architecture_from_weight_index(args.model_path, config.model.id),
            selected,
            args.destination,
            model_id=config.model.id,
            model_revision=config.model.revision,
            run_id=config.run_id or config.resolve_run_id(git_sha()),
            # Gate C only supplies observational evidence.  Causal validation
            # and replication are separate gates, so extraction from the
            # existing candidate artifact must never upgrade its label.
            selection_status="observational-candidates",
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
        output = validate_telemetry(
            args.path,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
        )
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
    if args.command == "export-swebench":
        emit_json(export_predictions(args.evaluation, args.destination, model_name=args.model_name))
        return 0
    if args.command == "merge-swebench":
        output = merge_report(args.evaluation, args.report, args.destination)
        emit_json(output)
        return 0 if output["passed_gate_b_scoreability"] else 2
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
