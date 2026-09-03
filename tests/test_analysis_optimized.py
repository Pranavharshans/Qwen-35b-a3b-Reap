"""Reference-versus-optimized equivalence tests for the CPU analysis engine.

The optimized engine (:mod:`reverse_reap.analysis_fast`) streams telemetry into
compact per-(sample, layer, expert) aggregates, caches them by telemetry
SHA-256, and vectorizes the bootstrap/permutation inference. These tests hold
the dict-based reference implementation in :mod:`reverse_reap.analysis` as the
oracle and require identical integer/set outputs and float outputs within a
tight documented tolerance (summation order differs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from reverse_reap.analysis import (
    AnalysisError,
    bootstrap_stability,
    differential_ranking,
    label_permutation,
)
from reverse_reap.analysis_fast import (
    _baseline_arrays,
    cell_table_from_observations,
    fast_baseline_ranking,
    fast_bootstrap,
    fast_label_permutation,
    load_or_build_cells,
    stream_cells,
)
from reverse_reap.pipeline import analyze_telemetry

FLOAT_TOLERANCE = 1e-9


def _router_weights(domain: str, layer: int, expert: int, rng_value: float) -> float:
    """Asymmetric routing with a shared expert universe and no exact ties.

    Both domains route the same favored experts (so the shared universe is
    non-empty) but with different strengths (so differentials are informative).
    Expert 9 is coding-only and expert 5 is control-favoured at layer 1,
    producing genuinely unranked single-domain keys as well. The minimum
    favoured weight (0.15 + 0.02 base) always exceeds the maximum jitter-only
    weight (0.02 + 0.10), so top-k membership is deterministic.
    """
    base = 0.02
    coding_bias = {0: {1: 0.50, 3: 0.35, 5: 0.20}, 1: {1: 0.30, 3: 0.45, 9: 0.15}}
    control_bias = {0: {1: 0.15, 3: 0.55, 5: 0.30}, 1: {1: 0.50, 3: 0.20, 5: 0.30}}
    if domain == "coding":
        weight = coding_bias[layer].get(expert, 0.0)
    else:
        weight = control_bias[layer].get(expert, 0.0)
    return base + weight + 0.01 * rng_value


def _telemetry_rows(
    *,
    coding_samples: int = 4,
    control_samples: int = 4,
    strata: dict[str, list[str]] | None = None,
    layers: int = 2,
    experts: int = 16,
    tokens: int = 3,
    top_k: int = 3,
) -> list[dict]:
    if strata is None:
        strata = {
            "mixed-a": ["coding", "control", "coding", "control"],
            "mixed-b": ["coding", "control", "coding", "control"],
        }
    rows: list[dict] = []
    sample_number = 0
    domain_counts = {"coding": coding_samples, "control": control_samples}
    stratum_cycle: list[tuple[str, str]] = []
    for name, domains in strata.items():
        for _position, domain in enumerate(domains):
            stratum_cycle.append((domain, name))
    for domain, stratum in stratum_cycle:
        sample_id = f"sample-{sample_number:04d}-{domain}"
        sample_number += 1
        for token_index in range(tokens):
            for layer in range(layers):
                # Deterministic pseudo-router: strongest coding/control biases win.
                scored = sorted(
                    (
                        (
                            -_router_weights(
                                domain,
                                layer,
                                expert,
                                ((token_index * 7 + expert * 13 + layer * 5) % 11) / 110.0,
                            )
                        ),
                        expert,
                    )
                    for expert in range(experts)
                )
                for rank, (_, expert) in enumerate(scored[:top_k]):
                    weight = -scored[rank][0]
                    norm = 0.4 + 0.05 * ((token_index + expert + layer) % 6)
                    rows.append(
                        {
                            "schema_version": 1,
                            "chunk_id": sample_id,
                            "condition_id": "C0",
                            "sample_id": sample_id,
                            "domain": domain,
                            "stratum": stratum,
                            "split": "calibration",
                            "segment": "prompt",
                            "language": "python",
                            "run_id": "fixture",
                            "token_id": 100 + token_index,
                            "token_index": token_index,
                            "layer_index": layer,
                            "expert_index": expert,
                            "route_rank": rank,
                            "router_weight": weight,
                            "expert_output_l2": norm,
                        }
                    )
    # Off-window rows (validation/replication) that both engines must exclude.
    for split, domain in (("validation", "coding"), ("replication", "control")):
        sample_id = f"sample-off-{split}"
        for token_index in range(tokens):
            for layer in range(layers):
                for expert in (0, 1):
                    rows.append(
                        {
                            "schema_version": 1,
                            "chunk_id": sample_id,
                            "condition_id": "C0",
                            "sample_id": sample_id,
                            "domain": domain,
                            "stratum": "off-window",
                            "split": split,
                            "segment": "prompt",
                            "language": "python",
                            "run_id": "fixture",
                            "token_id": 100 + token_index,
                            "token_index": token_index,
                            "layer_index": layer,
                            "expert_index": expert,
                            "route_rank": 0,
                            "router_weight": 0.9,
                            "expert_output_l2": 0.5,
                        }
                    )
    assert sample_number == sum(domain_counts.values())
    return rows


def _write_telemetry(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _reference_observations(rows: list[dict]) -> list[dict]:
    """Exactly the reference pipeline's materialize-then-group stage."""
    splits = ("calibration", "selection")
    segment = "joint"
    token_rows = [
        row
        for row in rows
        if row["split"] in splits and (segment == "joint" or row["segment"] == segment)
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
    return [
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


# ---------------------------------------------------------------------------
# Streaming aggregation
# ---------------------------------------------------------------------------


def test_streaming_cells_match_reference_observations(tmp_path):
    rows = _telemetry_rows()
    path = tmp_path / "telemetry.jsonl"
    _write_telemetry(path, rows)
    table = stream_cells(path)
    reference = _reference_observations(rows)
    in_window_rows = sum(1 for row in rows if row["split"] in ("calibration", "selection"))
    assert table.routing_rows == in_window_rows  # 96: every fixture row is in-window
    assert table.n_cells == len(reference)
    reference_by_key = {
        (row["sample_id"], row["layer"], row["expert"]): row for row in reference
    }
    table_by_key = {
        (table.sample_ids[table.sample_index[position]],
         int(table.layer_of_key[table.key_index[position]]),
         int(table.expert_of_key[table.key_index[position]])): position
        for position in range(table.n_cells)
    }
    assert set(table_by_key) == set(reference_by_key)
    for key, row in reference_by_key.items():
        position = table_by_key[key]
        assert math.isclose(
            table.saliency[position], row["reap_saliency"], rel_tol=0, abs_tol=1e-12
        )
        assert table.routed_count[position] == row["routed_count"]
        assert math.isclose(
            table.router_weight_sum[position], row["router_weight_sum"], rel_tol=0, abs_tol=1e-12
        )
        assert math.isclose(
            table.weighted_norm_sum[position], row["weighted_norm_sum"], rel_tol=0, abs_tol=1e-12
        )
    token_expectations = {
        sample_id: 3 for sample_id in {row["sample_id"] for row in reference}
    }
    assert {
        table.sample_ids[index]: table.token_count_of_sample[index]
        for index in range(table.n_samples)
    } == token_expectations


def test_streaming_cells_exclude_off_window_splits(tmp_path):
    rows = _telemetry_rows()
    path = tmp_path / "telemetry.jsonl"
    _write_telemetry(path, rows)
    table = stream_cells(path)
    sample_ids = set(table.sample_ids)
    assert all("off" not in sample_id for sample_id in sample_ids)


# ---------------------------------------------------------------------------
# Baseline ranking
# ---------------------------------------------------------------------------


def test_baseline_ranking_matches_reference(tmp_path):
    rows = _telemetry_rows()
    path = tmp_path / "telemetry.jsonl"
    _write_telemetry(path, rows)
    table = load_or_build_cells(path, cache_dir=tmp_path / "cache")
    reference = differential_ranking(_reference_observations(rows))
    optimized = fast_baseline_ranking(table)
    assert [(r["layer"], r["expert"]) for r in optimized] == [
        (r["layer"], r["expert"]) for r in reference
    ]
    for fast_row, ref_row in zip(optimized, reference, strict=False):
        assert fast_row["observed"] == ref_row["observed"]
        assert fast_row["observed_in"] == ref_row["observed_in"]
        for field in (
            "differential",
            "coding_z",
            "control_z",
            "coding_mean_reap",
            "control_mean_reap",
        ):
            if ref_row[field] is None:
                assert fast_row[field] is None
            else:
                assert math.isclose(fast_row[field], ref_row[field], rel_tol=FLOAT_TOLERANCE)
        assert math.isclose(
            fast_row["routing_frequency"], ref_row["routing_frequency"], rel_tol=FLOAT_TOLERANCE
        )
        for prefix in ("coding", "control"):
            if not ref_row["observed"]:
                # reference unranked rows carry no per-domain metric fields
                assert f"{prefix}_routing_count" not in ref_row
                assert f"{prefix}_routing_count" not in fast_row
                continue
            assert fast_row[f"{prefix}_routing_count"] == ref_row[f"{prefix}_routing_count"]
            for field in (
                "routing_rate",
                "router_weight_sum",
                "router_weight_mean",
                "expert_output_norm_mean",
                "standard_reap_saliency",
            ):
                fast_value = fast_row[f"{prefix}_{field}"]
                ref_value = ref_row[f"{prefix}_{field}"]
                if ref_value is None:
                    assert fast_value is None
                else:
                    assert math.isclose(fast_value, ref_value, rel_tol=FLOAT_TOLERANCE)


# ---------------------------------------------------------------------------
# Bootstrap and permutation equivalence
# ---------------------------------------------------------------------------


def test_bootstrap_matches_reference_for_every_cardinality(tmp_path):
    rows = _telemetry_rows()
    path = tmp_path / "telemetry.jsonl"
    _write_telemetry(path, rows)
    table = load_or_build_cells(path, cache_dir=tmp_path / "cache")
    observations = _reference_observations(rows)
    _, differential, order = _baseline_arrays(table)
    grid = [2, 4, 6]
    optimized = fast_bootstrap(
        table, top_ns=grid, iterations=25, seed=17, baseline_order=order
    )
    for top_n in grid:
        reference = bootstrap_stability(observations, top_n=top_n, iterations=25, seed=17)
        fast = optimized[top_n]
        assert fast["iterations"] == reference["iterations"]
        assert fast["top_n"] == reference["top_n"]
        assert fast["seed"] == reference["seed"]
        assert fast["jaccards"] == reference["jaccards"]
        assert fast["median_jaccard"] == reference["median_jaccard"]
        ref_selection = {
            (i["layer"], i["expert"]): i["frequency"] for i in reference["selection_frequency"]
        }
        fast_selection = {
            (i["layer"], i["expert"]): i["frequency"] for i in fast["selection_frequency"]
        }
        assert fast_selection == ref_selection
        ref_intervals = {
            (i["layer"], i["expert"]): i for i in reference["differential_intervals"]
        }
        fast_intervals = {
            (i["layer"], i["expert"]): i for i in fast["differential_intervals"]
        }
        assert set(fast_intervals) == set(ref_intervals)
        for key, ref_item in ref_intervals.items():
            fast_item = fast_intervals[key]
            assert fast_item["observations"] == ref_item["observations"]
            assert math.isclose(fast_item["low"], ref_item["low"], rel_tol=FLOAT_TOLERANCE)
            assert math.isclose(fast_item["high"], ref_item["high"], rel_tol=FLOAT_TOLERANCE)


def test_permutation_matches_reference_single_domain_strata(tmp_path):
    strata = {
        "only-coding": ["coding", "coding", "coding", "coding"],
        "only-control": ["control", "control", "control", "control"],
    }
    rows = _telemetry_rows(strata=strata)
    path = tmp_path / "telemetry.jsonl"
    _write_telemetry(path, rows)
    table = load_or_build_cells(path, cache_dir=tmp_path / "cache")
    observations = _reference_observations(rows)
    _, differential, order = _baseline_arrays(table)
    grid = [2, 4]
    optimized = fast_label_permutation(
        table,
        top_ns=grid,
        iterations=40,
        seed=23,
        baseline_order=order,
        baseline_differential=differential,
    )
    for top_n in grid:
        reference = label_permutation(observations, top_n=top_n, iterations=40, seed=23)
        fast = optimized[top_n]
        for key in (
            "method",
            "assignment_mode",
            "permutation_design",
            "attainable_assignments",
            "assignments_evaluated",
            "unique_assignments_evaluated",
            "permutations_changed_labels",
            "unique_null_statistics",
            "permutation_design_valid",
            "global_fallback_limitation",
            "iterations_requested",
            "iterations_valid",
            "seed",
            "p_value",
        ):
            assert fast[key] == reference[key], (top_n, key)
        assert fast["method"] == "global-count-preserving-exact-enumeration"
        assert math.isclose(
            fast["observed_top_sum"], reference["observed_top_sum"], rel_tol=FLOAT_TOLERANCE
        )
        assert sorted(fast["null_scores"]) == pytest.approx(
            sorted(reference["null_scores"]), abs=FLOAT_TOLERANCE
        )
        ref_p = {(i["layer"], i["expert"]): i["p_value"] for i in reference["expert_p_values"]}
        fast_p = {(i["layer"], i["expert"]): i["p_value"] for i in fast["expert_p_values"]}
        assert fast_p == ref_p


def test_permutation_matches_reference_mixed_strata(tmp_path):
    rows = _telemetry_rows()  # default fixture: every stratum is mixed
    path = tmp_path / "telemetry.jsonl"
    _write_telemetry(path, rows)
    table = load_or_build_cells(path, cache_dir=tmp_path / "cache")
    observations = _reference_observations(rows)
    _, differential, order = _baseline_arrays(table)
    optimized = fast_label_permutation(
        table,
        top_ns=[4],
        iterations=40,
        seed=29,
        baseline_order=order,
        baseline_differential=differential,
    )
    reference = label_permutation(observations, top_n=4, iterations=40, seed=29)
    fast = optimized[4]
    assert fast["method"] == reference["method"] == "stratified-count-preserving-exact-enumeration"
    for key in (
        "p_value",
        "attainable_assignments",
        "assignments_evaluated",
        "permutation_design",
    ):
        assert fast[key] == reference[key], key
    # unique_null_statistics is a summation-order-sensitive diagnostic: the
    # dict-based reference splits mathematical ties into last-bit FP variants,
    # while the vectorized engine computes bit-identical statistics for
    # symmetric assignments. The vectorized count can only be <= the
    # reference's, and the multiset of statistics must still agree exactly.
    assert fast["unique_null_statistics"] <= reference["unique_null_statistics"]
    assert len({round(value, 6) for value in fast["null_scores"]}) == len(
        {round(value, 6) for value in reference["null_scores"]}
    )
    assert sorted(fast["null_scores"]) == pytest.approx(
        sorted(reference["null_scores"]), abs=FLOAT_TOLERANCE
    )


def test_permutation_boundary_tie_limitation_is_documented():
    """Symmetric fixtures can cancel the statistic to ~0, where the >= comparison
    is sensitive to summation-order noise. The implementations still agree on
    the null multiset; only boundary ties may shift the exceedance count. The
    same noise splits mathematical ties in the reference's
    unique_null_statistics diagnostic, which the vectorized engine collapses."""
    rows = []
    for domain in ("coding", "control"):
        for sample_number in range(4):
            for expert in range(3):
                base = [8.0, 3.0, 1.0][expert] if domain == "coding" else [1.0, 3.0, 8.0][expert]
                rows.append(
                    {
                        "sample_id": f"{domain}-{sample_number}",
                        "domain": domain,
                        "stratum": "shared-stratum",
                        "layer": 0,
                        "expert": expert,
                        "reap_saliency": base + sample_number * 0.01,
                    }
                )
    table = cell_table_from_observations(rows)
    _, differential, order = _baseline_arrays(table)
    fast = fast_label_permutation(
        table, top_ns=[2], iterations=56, seed=13,
        baseline_order=order, baseline_differential=differential,
    )[2]
    reference = label_permutation(rows, top_n=2, iterations=56, seed=13)
    assert sorted(fast["null_scores"]) == pytest.approx(
        sorted(reference["null_scores"]), abs=FLOAT_TOLERANCE
    )
    boundary = sum(
        1 for score in fast["null_scores"] if abs(score - fast["observed_top_sum"]) < 1e-9
    )
    # Any p-value disagreement is fully explained by those boundary ties.
    if fast["p_value"] != reference["p_value"]:
        assert boundary > 0


# ---------------------------------------------------------------------------
# End-to-end engine equivalence on identical artifacts
# ---------------------------------------------------------------------------


def _engine_outputs(tmp_path, engine, rows_path, cache_dir):
    output_dir = tmp_path / f"analysis-{engine}"
    result = analyze_telemetry(
        rows_path,
        output_dir,
        top_n=4,
        bootstrap_iterations=25,
        permutation_iterations=40,
        seed=17,
        cardinality_grid=(2, 4),
        engine=engine,
        cache_dir=cache_dir,
    )
    def load(name):
        return json.loads((output_dir / name).read_text())

    return result, load


def test_engines_produce_equivalent_artifacts(tmp_path):
    rows = _telemetry_rows()
    rows_path = tmp_path / "telemetry.jsonl"
    _write_telemetry(rows_path, rows)
    reference_result, reference_load = _engine_outputs(
        tmp_path, "reference", rows_path, tmp_path / "cache-reference"
    )
    fast_result, fast_load = _engine_outputs(
        tmp_path, "fast", rows_path, tmp_path / "cache-fast"
    )
    assert reference_result["routing_rows"] == fast_result["routing_rows"]
    assert reference_result["observations"] == fast_result["observations"]
    assert reference_result["selected_top_n"] == fast_result["selected_top_n"]
    assert reference_result["candidate_gate_passed"] == fast_result["candidate_gate_passed"]
    assert reference_result["experts_ranked"] == fast_result["experts_ranked"]
    assert (
        reference_result["experts_unranked_single_domain"]
        == fast_result["experts_unranked_single_domain"]
    )

    ref_ranking = reference_load("expert-ranking.json")
    fast_ranking = fast_load("expert-ranking.json")
    assert [(r["layer"], r["expert"], r["observed"]) for r in fast_ranking] == [
        (r["layer"], r["expert"], r["observed"]) for r in ref_ranking
    ]
    for fast_row, ref_row in zip(fast_ranking, ref_ranking, strict=False):
        for field, value in ref_row.items():
            if isinstance(value, float):
                assert math.isclose(fast_row[field], value, rel_tol=FLOAT_TOLERANCE), field
            elif isinstance(value, list) and value and isinstance(value[0], float):
                assert len(fast_row[field]) == len(value), field
                for fast_item, ref_item in zip(fast_row[field], value, strict=False):
                    assert math.isclose(fast_item, ref_item, rel_tol=FLOAT_TOLERANCE), field
            elif not isinstance(value, dict):
                assert fast_row[field] == value, field

    ref_manifest = reference_load("candidate-manifest.json")
    fast_manifest = fast_load("candidate-manifest.json")
    assert [(e["layer"], e["expert"]) for e in fast_manifest["experts"]] == [
        (e["layer"], e["expert"]) for e in ref_manifest["experts"]
    ]
    for fast_expert, ref_expert in zip(
        fast_manifest["experts"], ref_manifest["experts"], strict=True
    ):
        for field, value in ref_expert.items():
            if isinstance(value, float):
                assert math.isclose(fast_expert[field], value, rel_tol=FLOAT_TOLERANCE), field
            elif isinstance(value, list):
                assert fast_expert[field] == value or all(
                    math.isclose(a, b, rel_tol=FLOAT_TOLERANCE)
                    for a, b in zip(fast_expert[field], value, strict=False)
                ), field
            elif not isinstance(value, dict):
                assert fast_expert[field] == value, field
    assert fast_manifest["gate_passed"] == ref_manifest["gate_passed"]
    assert fast_manifest["experts_ranked"] == ref_manifest["experts_ranked"]
    assert (
        fast_manifest["experts_unranked_single_domain"]
        == ref_manifest["experts_unranked_single_domain"]
    )

    ref_grid = reference_load("cardinality-grid.json")
    fast_grid = fast_load("cardinality-grid.json")
    assert ref_grid["selected_top_n"] == fast_grid["selected_top_n"]
    assert [item["gate_passed"] for item in ref_grid["grid"]] == [
        item["gate_passed"] for item in fast_grid["grid"]
    ]
    assert [item["top_n"] for item in ref_grid["grid"]] == [
        item["top_n"] for item in fast_grid["grid"]
    ]

    # Control manifests are produced by the same reference code from equivalent
    # rankings with the same seed, so they must be byte-identical structures.
    assert fast_load("control-manifests.json") == reference_load("control-manifests.json")
    assert fast_load("unobserved-experts.json")["unranked"] == (
        reference_load("unobserved-experts.json")["unranked"]
    )


def test_engines_agree_on_single_domain_strata_end_to_end(tmp_path):
    rows = _telemetry_rows(
        strata={
            "only-coding": ["coding", "coding", "coding", "coding"],
            "only-control": ["control", "control", "control", "control"],
        }
    )
    rows_path = tmp_path / "telemetry.jsonl"
    _write_telemetry(rows_path, rows)
    reference_result, reference_load = _engine_outputs(
        tmp_path, "reference", rows_path, tmp_path / "cache-reference"
    )
    fast_result, fast_load = _engine_outputs(
        tmp_path, "fast", rows_path, tmp_path / "cache-fast"
    )
    assert fast_result["selected_top_n"] == reference_result["selected_top_n"]
    ref_perm = reference_load("label-permutation.json")
    fast_perm = fast_load("label-permutation.json")
    assert fast_perm["method"] == ref_perm["method"]
    assert fast_perm["global_fallback_limitation"] == ref_perm["global_fallback_limitation"]
    assert fast_perm["p_value"] == ref_perm["p_value"]


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_cache_roundtrip_is_transparent(tmp_path):
    rows = _telemetry_rows()
    rows_path = tmp_path / "telemetry.jsonl"
    _write_telemetry(rows_path, rows)
    cache_dir = tmp_path / "cache"
    first = load_or_build_cells(rows_path, cache_dir=cache_dir)
    cache_files = list(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1
    second = load_or_build_cells(rows_path, cache_dir=cache_dir)
    assert second.sample_ids == first.sample_ids
    assert np.array_equal(second.saliency, first.saliency)
    assert np.array_equal(second.key_index, first.key_index)
    assert second.routing_rows == first.routing_rows
    # Warm-cache artifacts equal cold-cache artifacts.
    output_a = tmp_path / "analysis-cold"
    output_b = tmp_path / "analysis-warm"
    common = dict(
        top_n=4, bootstrap_iterations=20, permutation_iterations=30, seed=17,
        cardinality_grid=(2, 4), engine="fast",
    )
    analyze_telemetry(rows_path, output_a, cache_dir=tmp_path / "cold-cache", **common)
    analyze_telemetry(rows_path, output_b, cache_dir=cache_dir, **common)
    assert json.loads((output_b / "candidate-manifest.json").read_text()) == json.loads(
        (output_a / "candidate-manifest.json").read_text()
    )


def test_cache_never_reuses_a_different_telemetry(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_telemetry(first_path, _telemetry_rows())
    _write_telemetry(second_path, _telemetry_rows(tokens=4))
    cache_dir = tmp_path / "cache"
    first = load_or_build_cells(first_path, cache_dir=cache_dir)
    second = load_or_build_cells(second_path, cache_dir=cache_dir)
    assert len(list(cache_dir.glob("*.npz"))) == 2
    # different telemetry (extra token) -> different SHA -> independently built table
    assert second.routing_rows != first.routing_rows
    assert not np.array_equal(second.saliency, first.saliency)


# ---------------------------------------------------------------------------
# Determinism and fail-closed parity
# ---------------------------------------------------------------------------


def test_fast_engine_is_seed_deterministic(tmp_path):
    rows = _telemetry_rows()
    rows_path = tmp_path / "telemetry.jsonl"
    _write_telemetry(rows_path, rows)
    results = []
    payloads = []
    for run in range(2):
        output_dir = tmp_path / f"analysis-{run}"
        analyze_telemetry(
            rows_path,
            output_dir,
            top_n=4,
            bootstrap_iterations=20,
            permutation_iterations=30,
            seed=17,
            cardinality_grid=(2, 4),
            engine="fast",
            cache_dir=tmp_path / f"cache-{run}",
        )
        payloads.append(json.loads((output_dir / "label-permutation.json").read_text()))
        results.append(payloads[-1]["p_value"])
    assert results[0] == results[1]
    assert payloads[0]["null_scores"] == payloads[1]["null_scores"]


def test_engines_fail_closed_together_on_degenerate_null(tmp_path):
    rows = []
    for domain in ("coding", "control"):
        for sample_number in range(3):
            for expert in range(2):
                rows.append(
                    {
                        "sample_id": f"{domain}-{sample_number}",
                        "domain": domain,
                        "stratum": "only",
                        "layer": 0,
                        "expert": expert,
                        "reap_saliency": 5.0,
                    }
                )
    table = cell_table_from_observations(rows)
    _, differential, order = _baseline_arrays(table)
    with pytest.raises(AnalysisError, match="degenerate"):
        fast_label_permutation(
            table, top_ns=[1], iterations=10, seed=3,
            baseline_order=order, baseline_differential=differential,
        )
    with pytest.raises(AnalysisError, match="degenerate"):
        label_permutation(rows, top_n=1, iterations=10, seed=3)


def test_engines_raise_identically_without_matching_rows(tmp_path):
    rows = _telemetry_rows()
    for row in rows:
        row["split"] = "validation"
    path = tmp_path / "telemetry.jsonl"
    _write_telemetry(path, rows)
    for engine in ("reference", "fast"):
        with pytest.raises(ValueError, match="no telemetry rows match"):
            analyze_telemetry(
                path,
                tmp_path / f"out-{engine}",
                top_n=4,
                bootstrap_iterations=10,
                permutation_iterations=10,
                seed=1,
                engine=engine,
            )
