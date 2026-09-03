"""Benchmark the reference versus optimized CPU analysis engines.

Runs both engines on the same telemetry file in subprocess-isolated workers
(peak RSS via ``resource.getrusage``), reports parsing, aggregation, ranking,
bootstrap, permutation, freeze, and total wall-clock time per engine, verifies
artifact equivalence between the engines, and prints the speedup.

Usage:
    uv run python scripts/benchmark_analysis.py \
        --telemetry /path/to/telemetry.jsonl \
        --output-dir /tmp/analysis-bench \
        --top-n 8 --grid 4 8 16 \
        --bootstrap-iterations 200 --permutation-iterations 200 --seed 20260903

The telemetry file is opened read-only. Nothing is merged, launched, or
mutated: the only writes are the benchmark's own output directory and the
telemetry-SHA-keyed aggregate cache under ``--cache-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

SPLIT_FILTER_TOLERANCE = 1e-9


def _peak_rss_mb() -> float:
    """Peak RSS in MB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1048576 if sys.platform == "darwin" else raw / 1024


def _worker_reference(args: argparse.Namespace) -> dict:
    """Reference engine stage timings (mirrors pipeline.analyze_telemetry)."""
    from reverse_reap.analysis import bootstrap_stability, differential_ranking, label_permutation
    from reverse_reap.pipeline import _freeze_analysis_artifacts

    started_total = time.perf_counter()

    parse_started = time.perf_counter()
    # Exactly the reference engine's materialization:
    # telemetry_path.read_text().splitlines() + one json.loads per line.
    raw_rows = [
        json.loads(line)
        for line in args.telemetry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parse_seconds = time.perf_counter() - parse_started

    aggregate_started = time.perf_counter()
    token_rows = [
        row
        for row in raw_rows
        if row["split"] in args.splits
        and (args.segment == "joint" or row["segment"] == args.segment)
    ]
    if not token_rows:
        raise ValueError("no telemetry rows match the requested splits and segment")
    sample_token_counts: dict[str, int] = {}
    token_identities: dict[str, set[int]] = {}
    for row in token_rows:
        if "token_index" in row:
            token_identities.setdefault(row["sample_id"], set()).add(int(row["token_index"]))
    sample_token_counts.update({key: len(value) for key, value in token_identities.items()})
    grouped: dict[tuple[str, str, str, int, int], dict[str, float]] = {}
    for row in token_rows:
        layer = int(row.get("layer_index", row.get("layer")))
        expert = int(row.get("expert_index", row.get("expert")))
        key = (row["sample_id"], row["domain"], row["stratum"], layer, expert)
        values = grouped.setdefault(
            key,
            {
                "count": 0.0,
                "router_weight_sum": 0.0,
                "expert_output_norm_sum": 0.0,
                "weighted_norm": 0.0,
            },
        )
        routed_count = int(row.get("routed_count", 1))
        values["count"] += routed_count
        if "expert_output_l2" in row:
            weight = float(row["router_weight"])
            norm = float(row["expert_output_l2"])
            values["router_weight_sum"] += weight
            values["expert_output_norm_sum"] += norm
            values["weighted_norm"] += weight * norm
        else:
            values["router_weight_sum"] += float(row.get("router_mass", 0))
            values["weighted_norm"] += float(row["reap_saliency"]) * routed_count
    observations = [
        {
            "sample_id": key[0],
            "domain": key[1],
            "stratum": key[2],
            "layer": key[3],
            "expert": key[4],
            "routed_count": int(value["count"]),
            "token_count": sample_token_counts.get(key[0], int(value["count"])),
            "router_weight_sum": value["router_weight_sum"],
            "expert_output_norm_sum": value["expert_output_norm_sum"],
            "weighted_norm_sum": value["weighted_norm"],
            "reap_saliency": value["weighted_norm"] / value["count"],
        }
        for key, value in grouped.items()
    ]
    aggregate_seconds = time.perf_counter() - aggregate_started

    ranking_started = time.perf_counter()
    ranking = differential_ranking(observations)
    ranking_seconds = time.perf_counter() - ranking_started

    configured_top_n = args.top_n
    cardinalities = sorted(set(args.grid or (args.top_n,)))
    if configured_top_n not in cardinalities:
        cardinalities.append(configured_top_n)
        cardinalities.sort()
    bootstrap_seconds = 0.0
    permutation_seconds = 0.0
    analyses = []
    for cardinality in cardinalities:
        started = time.perf_counter()
        current_bootstrap = bootstrap_stability(
            observations, top_n=cardinality, iterations=args.bootstrap_iterations, seed=args.seed
        )
        bootstrap_seconds += time.perf_counter() - started
        started = time.perf_counter()
        current_permutation = label_permutation(
            observations,
            top_n=cardinality,
            iterations=args.permutation_iterations,
            seed=args.seed,
        )
        permutation_seconds += time.perf_counter() - started
        analyses.append(
            {
                "top_n": cardinality,
                "gate_passed": current_bootstrap["median_jaccard"] >= 0.60
                and current_permutation["p_value"] <= 0.05,
                "median_bootstrap_jaccard": current_bootstrap["median_jaccard"],
                "permutation_p_value": current_permutation["p_value"],
                "bootstrap": current_bootstrap,
                "permutation": current_permutation,
            }
        )
    passing = [item for item in analyses if item["gate_passed"]]
    chosen = (
        passing[0]
        if passing
        else next(item for item in analyses if item["top_n"] == configured_top_n)
    )
    top_n = chosen["top_n"]

    import hashlib

    freeze_started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.telemetry.read_bytes()).hexdigest()
    candidates = _freeze_analysis_artifacts(
        args.output,
        ranking=ranking,
        bootstrap=chosen["bootstrap"],
        permutation=chosen["permutation"],
        analyses=analyses,
        chosen_top_n=top_n,
        configured_top_n=configured_top_n,
        any_gate_passed=bool(passing),
        source_hash=source_hash,
        seed=args.seed,
    )
    freeze_seconds = time.perf_counter() - freeze_started
    total_seconds = time.perf_counter() - started_total
    return {
        "engine": "reference",
        "stages": {
            "parse_s": parse_seconds,
            "aggregate_s": aggregate_seconds,
            "ranking_s": ranking_seconds,
            "bootstrap_s": bootstrap_seconds,
            "permutation_s": permutation_seconds,
            "freeze_s": freeze_seconds,
            "total_s": total_seconds,
        },
        "result": _summarize(candidates, ranking, chosen, top_n),
        "counts": {"routing_rows": len(token_rows), "observations": len(observations)},
        "peak_rss_mb": _peak_rss_mb(),
    }


def _summarize(
    candidates: dict, ranking: list[dict], chosen: dict, top_n: int
) -> dict:
    selected = [(item["layer"], item["expert"]) for item in candidates["experts"]]
    return {
        "selected_top_n": top_n,
        "gate_passed": candidates["gate_passed"],
        "median_bootstrap_jaccard": chosen["bootstrap"]["median_jaccard"],
        "permutation_p_value": chosen["permutation"]["p_value"],
        "selected_experts": selected,
        "ranked_order": [
            (row["layer"], row["expert"])
            for row in ranking
            if row.get("observed", True)
        ][:top_n],
    }


def _worker_fast(args: argparse.Namespace) -> dict:
    """Optimized engine stage timings (mirrors pipeline._analyze_telemetry_fast)."""
    from reverse_reap.analysis_fast import (
        fast_analysis_outputs,
        load_or_build_cells,
        telemetry_sha256,
    )
    from reverse_reap.pipeline import _freeze_analysis_artifacts

    started_total = time.perf_counter()

    load_started = time.perf_counter()
    table = load_or_build_cells(
        args.telemetry, splits=args.splits, segment=args.segment, cache_dir=args.cache_dir
    )
    load_seconds = time.perf_counter() - load_started
    # stream_cells records its own stage timings when it actually ran; a cache
    # hit never touches the parser, so both stay at their defaults (0.0).
    from reverse_reap.analysis_fast import stream_cells

    parse_seconds = float(getattr(stream_cells, "last_parse_seconds", 0.0))
    aggregate_seconds = float(getattr(stream_cells, "last_aggregate_seconds", 0.0))
    cache_hit = parse_seconds == 0.0 and load_seconds > 0

    outputs = fast_analysis_outputs(
        table,
        top_n=args.top_n,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
        cardinality_grid=args.grid,
    )
    ranking = outputs["ranking"]
    chosen = outputs["chosen"]
    top_n = chosen["top_n"]


    freeze_started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = telemetry_sha256(args.telemetry)
    candidates = _freeze_analysis_artifacts(
        args.output,
        ranking=ranking,
        bootstrap=chosen["bootstrap"],
        permutation=chosen["permutation"],
        analyses=outputs["analyses"],
        chosen_top_n=top_n,
        configured_top_n=outputs["configured_top_n"],
        any_gate_passed=any(item["gate_passed"] for item in outputs["analyses"]),
        source_hash=source_hash,
        seed=args.seed,
    )
    freeze_seconds = time.perf_counter() - freeze_started
    total_seconds = time.perf_counter() - started_total
    stages = {
        "parse_s": parse_seconds,
        "aggregate_s": aggregate_seconds,
        "ranking_s": outputs["timings"]["baseline_s"],
        "bootstrap_s": outputs["timings"]["bootstrap_s"],
        "permutation_s": outputs["timings"]["permutation_s"],
        "freeze_s": freeze_seconds,
        "total_s": total_seconds,
    }
    return {
        "engine": "fast",
        "stages": stages,
        "result": _summarize(candidates, ranking, chosen, top_n),
        "counts": {
            "routing_rows": table.routing_rows,
            "observations": table.n_cells,
            "cache_hit": bool(cache_hit),
        },
        "peak_rss_mb": _peak_rss_mb(),
    }


def _compare(reference: dict, fast: dict, reference_output: Path, fast_output: Path) -> list[str]:
    """Return a list of equivalence failures (empty when equivalent)."""
    failures = []
    ref_result = reference["result"]
    fast_result = fast["result"]
    if reference["counts"]["routing_rows"] != fast["counts"]["routing_rows"]:
        failures.append("routing row counts differ")
    if reference["counts"]["observations"] != fast["counts"]["observations"]:
        failures.append("observation (cell) counts differ")
    if ref_result["selected_top_n"] != fast_result["selected_top_n"]:
        failures.append("selected cardinality differs")
    if ref_result["gate_passed"] != fast_result["gate_passed"]:
        failures.append("candidate gate differs")
    if ref_result["selected_experts"] != fast_result["selected_experts"]:
        failures.append("selected expert set differs")
    if ref_result["ranked_order"] != fast_result["ranked_order"]:
        failures.append("ranked order differs")
    if not math.isclose(
        ref_result["median_bootstrap_jaccard"],
        fast_result["median_bootstrap_jaccard"],
        rel_tol=SPLIT_FILTER_TOLERANCE,
    ):
        failures.append("median bootstrap jaccard differs beyond 1e-9")
    if abs(ref_result["permutation_p_value"] - fast_result["permutation_p_value"]) > 0:
        failures.append("permutation p-value differs")
    # Full artifact comparison across the frozen outputs.
    for name in (
        "expert-ranking.json",
        "bootstrap-stability.json",
        "label-permutation.json",
        "cardinality-grid.json",
        "control-manifests.json",
        "candidate-manifest.json",
    ):
        ref_doc = json.loads((reference_output / name).read_text())
        fast_doc = json.loads((fast_output / name).read_text())
        if name == "expert-ranking.json":
            if [(r["layer"], r["expert"], r["observed"]) for r in ref_doc] != [
                (r["layer"], r["expert"], r["observed"]) for r in fast_doc
            ]:
                failures.append(f"{name}: expert identities/order/observed differ")
            for ref_row, fast_row in zip(ref_doc, fast_doc, strict=True):
                for field, value in ref_row.items():
                    if isinstance(value, float) and not math.isclose(
                        fast_row[field], value, rel_tol=SPLIT_FILTER_TOLERANCE
                    ):
                        failures.append(f"{name}: float field {field} differs")
                        break
        elif name == "candidate-manifest.json":
            if [(e["layer"], e["expert"]) for e in ref_doc["experts"]] != [
                (e["layer"], e["expert"]) for e in fast_doc["experts"]
            ]:
                failures.append(f"{name}: selected experts differ")
            if ref_doc["gate_passed"] != fast_doc["gate_passed"]:
                failures.append(f"{name}: gate_passed differs")
        elif name == "control-manifests.json":
            if ref_doc != fast_doc:
                failures.append(f"{name}: control memberships differ")
        elif name == "label-permutation.json":
            if ref_doc["p_value"] != fast_doc["p_value"]:
                failures.append(f"{name}: p_value differs")
            if sorted(ref_doc["null_scores"]) != sorted(fast_doc["null_scores"]) and not all(
                math.isclose(a, b, rel_tol=SPLIT_FILTER_TOLERANCE, abs_tol=SPLIT_FILTER_TOLERANCE)
                for a, b in zip(
                    sorted(ref_doc["null_scores"]),
                    sorted(fast_doc["null_scores"]),
                    strict=True,
                )
            ):
                failures.append(f"{name}: null statistic multiset differs")
        elif name == "bootstrap-stability.json":
            if ref_doc["jaccards"] != fast_doc["jaccards"]:
                failures.append(f"{name}: replicate jaccards differ")
        elif name == "cardinality-grid.json":
            if [i["top_n"] for i in ref_doc["grid"]] != [i["top_n"] for i in fast_doc["grid"]]:
                failures.append(f"{name}: grid cardinalities differ")
            if [i["gate_passed"] for i in ref_doc["grid"]] != [
                i["gate_passed"] for i in fast_doc["grid"]
            ]:
                failures.append(f"{name}: grid gate decisions differ")
    return failures


def _run_worker(engine: str, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        engine,
        "--telemetry",
        str(args.telemetry),
        "--output",
        str(args.output),
        "--top-n",
        str(args.top_n),
        "--bootstrap-iterations",
        str(args.bootstrap_iterations),
        "--permutation-iterations",
        str(args.permutation_iterations),
        "--seed",
        str(args.seed),
    ]
    if args.grid:
        command += ["--grid"] + [str(value) for value in args.grid]
    if args.cache_dir:
        command += ["--cache-dir", str(args.cache_dir)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{engine} worker failed (exit {completed.returncode}):\n{completed.stderr[-4000:]}"
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload["subprocess_wall_s"] = wall_seconds
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--grid", type=int, nargs="*", default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--permutation-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--worker", choices=("reference", "fast"), default=None)
    args = parser.parse_args()
    args.splits = ("calibration", "selection")
    args.segment = "joint"

    if args.worker:
        worker = _worker_reference if args.worker == "reference" else _worker_fast
        print(json.dumps(worker(args)))
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = args.output / "cache"

    print(f"telemetry: {args.telemetry}")
    print(
        f"params: top_n={args.top_n} grid={args.grid or (args.top_n,)} "
        f"bootstrap={args.bootstrap_iterations} permutation={args.permutation_iterations} "
        f"seed={args.seed}"
    )
    print()

    reference_args, fast_args = argparse.Namespace(**vars(args)), argparse.Namespace(**vars(args))
    reference_args.output = args.output / "reference"
    fast_args.output = args.output / "fast"
    reference = _run_worker("reference", reference_args)
    fast_cold = _run_worker("fast", fast_args)
    fast_warm = _run_worker("fast", fast_args)

    print("=== stage timings (seconds) ===")
    stages = (
        "parse_s",
        "aggregate_s",
        "ranking_s",
        "bootstrap_s",
        "permutation_s",
        "freeze_s",
        "total_s",
    )
    header = f"{'stage':<14} {'reference':>12} {'fast cold':>12} {'fast warm':>12}"
    print(header)
    for stage in stages:
        print(
            f"{stage:<14} {reference['stages'][stage]:>12.3f} "
            f"{fast_cold['stages'][stage]:>12.3f} {fast_warm['stages'][stage]:>12.3f}"
        )
    print(f"{'subprocess':<14} {reference['subprocess_wall_s']:>12.3f} "
          f"{fast_cold['subprocess_wall_s']:>12.3f} {fast_warm['subprocess_wall_s']:>12.3f}")
    print()
    print("=== peak RSS (MB) ===")
    print(
        f"reference {reference['peak_rss_mb']:.1f}  fast cold {fast_cold['peak_rss_mb']:.1f}  "
        f"fast warm {fast_warm['peak_rss_mb']:.1f}"
    )
    print()
    speedup_cold = reference["stages"]["total_s"] / fast_cold["stages"]["total_s"]
    speedup_warm = reference["stages"]["total_s"] / fast_warm["stages"]["total_s"]
    print(f"total speedup cold: {speedup_cold:.1f}x   warm: {speedup_warm:.1f}x")
    print()

    failures = _compare(reference, fast_cold, args.output / "reference", args.output / "fast")
    print("=== equivalence reference vs fast (cold artifacts) ===")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
    else:
        print("PASS: identities, orderings, selections, p-values, grid decisions, "
              "controls, and floats within 1e-9 all agree")

    report = {
        "telemetry": str(args.telemetry),
        "params": {
            "top_n": args.top_n,
            "grid": args.grid or [args.top_n],
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "seed": args.seed,
        },
        "reference": reference,
        "fast_cold": fast_cold,
        "fast_warm": fast_warm,
        "speedup_cold": speedup_cold,
        "speedup_warm": speedup_warm,
        "equivalence_failures": failures,
    }
    report_path = args.output / "benchmark-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nreport written to {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
