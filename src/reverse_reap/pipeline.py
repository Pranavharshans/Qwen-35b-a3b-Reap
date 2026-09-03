"""File-oriented CPU analysis stage for telemetry-to-frozen-candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reverse_reap.analysis import (
    bootstrap_stability,
    build_control_sets,
    differential_ranking,
    freeze_candidates,
    label_permutation,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def analyze_telemetry(
    telemetry_path: Path,
    output_dir: Path,
    *,
    top_n: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
    splits: tuple[str, ...] = ("calibration", "selection"),
    segment: str = "joint",
) -> dict[str, Any]:
    observations = [
        row
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["split"] in splits and row["segment"] == segment
    ]
    if not observations:
        raise ValueError("no telemetry rows match the requested splits and segment")
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking = differential_ranking(observations)
    bootstrap = bootstrap_stability(
        observations, top_n=top_n, iterations=bootstrap_iterations, seed=seed
    )
    permutation = label_permutation(
        observations, top_n=top_n, iterations=permutation_iterations, seed=seed
    )
    source_hash = hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
    candidate_path = output_dir / "candidate-manifest.json"
    candidates = freeze_candidates(
        ranking,
        bootstrap,
        permutation,
        top_n=top_n,
        source_hashes={"telemetry": source_hash},
        destination=candidate_path,
    )
    selected = [(item["layer"], item["expert"]) for item in candidates["experts"]]
    controls = build_control_sets(ranking, selected, random_sets=20, seed=seed)
    _write_json(output_dir / "expert-ranking.json", ranking)
    _write_json(output_dir / "bootstrap-stability.json", bootstrap)
    _write_json(output_dir / "label-permutation.json", permutation)
    _write_json(output_dir / "control-manifests.json", controls)
    controls_dir = output_dir / "controls"
    controls_dir.mkdir(exist_ok=True)
    for item in controls["layer_matched_random_sets"]:
        _write_json(controls_dir / f"{item['control_id']}.json", item)
    _write_json(
        controls_dir / "frequency-matched.json",
        {"control_id": "frequency-matched", "experts": controls["frequency_matched_set"]},
    )
    _write_json(
        controls_dir / "lowest-differential.json",
        {"control_id": "lowest-differential", "experts": controls["lowest_differential_set"]},
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        parquet_written = False
    else:
        pq.write_table(pa.Table.from_pylist(ranking), output_dir / "expert-ranking.parquet")
        parquet_written = True
    return {
        "observations": len(observations),
        "experts_ranked": len(ranking),
        "candidate_gate_passed": candidates["gate_passed"],
        "median_bootstrap_jaccard": bootstrap["median_jaccard"],
        "permutation_p_value": permutation["p_value"],
        "parquet_written": parquet_written,
        "output_dir": str(output_dir),
    }
