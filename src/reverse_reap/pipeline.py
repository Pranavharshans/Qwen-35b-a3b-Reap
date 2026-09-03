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
    token_rows = [
        row
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["split"] in splits
        and (segment == "joint" or row["segment"] == segment)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking = differential_ranking(observations)
    bootstrap = bootstrap_stability(
        observations, top_n=top_n, iterations=bootstrap_iterations, seed=seed
    )
    permutation = label_permutation(
        observations, top_n=top_n, iterations=permutation_iterations, seed=seed
    )
    intervals = {
        (item["layer"], item["expert"]): item
        for item in bootstrap["differential_intervals"]
    }
    p_values = {
        (item["layer"], item["expert"]): item for item in permutation["expert_p_values"]
    }
    for row in ranking:
        key = (row["layer"], row["expert"])
        interval = intervals[key]
        row["differential_bootstrap_95ci"] = [interval["low"], interval["high"]]
        row["label_permutation_p_value"] = p_values[key]["p_value"]
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
    for item in controls["frequency_matched_random_sets"]:
        _write_json(controls_dir / f"{item['control_id']}.json", item)
    _write_json(
        controls_dir / "frequency-matched.json",
        {
            "control_id": "frequency-matched",
            "experts": controls["frequency_matched_random_sets"][0]["experts"],
            "source_control_id": controls["frequency_matched_random_sets"][0]["control_id"],
        },
    )
    _write_json(
        controls_dir / "highest-frequency.json",
        {"control_id": "highest-frequency", "experts": controls["highest_frequency_set"]},
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
        "routing_rows": len(token_rows),
        "observations": len(observations),
        "experts_ranked": len(ranking),
        "candidate_gate_passed": candidates["gate_passed"],
        "median_bootstrap_jaccard": bootstrap["median_jaccard"],
        "permutation_p_value": permutation["p_value"],
        "parquet_written": parquet_written,
        "output_dir": str(output_dir),
    }
