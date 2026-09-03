import json

import pytest

from reverse_reap.analysis import (
    AnalysisError,
    bootstrap_stability,
    build_control_sets,
    differential_ranking,
    freeze_candidates,
    label_permutation,
)


def observations():
    rows = []
    for domain in ("coding", "control"):
        for sample_number in range(8):
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
    return rows


def test_differential_ranking_identifies_coding_specific_expert():
    ranking = differential_ranking(observations())
    assert (ranking[0]["layer"], ranking[0]["expert"]) == (0, 0)
    assert ranking[0]["differential"] > 0
    assert ranking[-1]["expert"] == 2


def test_bootstrap_and_permutation_are_seed_deterministic():
    rows = observations()
    first = bootstrap_stability(rows, top_n=1, iterations=20, seed=13)
    second = bootstrap_stability(rows, top_n=1, iterations=20, seed=13)
    assert first == second
    assert first["median_jaccard"] == 1.0
    permutation = label_permutation(rows, top_n=1, iterations=30, seed=13)
    assert permutation["iterations_valid"] == 30
    assert 0 < permutation["p_value"] <= 1


def test_ranking_rejects_missing_control_evidence():
    with pytest.raises(AnalysisError, match="missing a coding or control"):
        differential_ranking([observations()[0]])


def test_candidate_manifest_is_hashed_and_cannot_be_mutated(tmp_path):
    rows = observations()
    ranking = differential_ranking(rows)
    bootstrap = bootstrap_stability(rows, top_n=1, iterations=10, seed=3)
    permutation = label_permutation(rows, top_n=1, iterations=20, seed=3)
    path = tmp_path / "candidates.json"
    manifest = freeze_candidates(
        ranking,
        bootstrap,
        permutation,
        top_n=1,
        source_hashes={"telemetry": "a" * 64},
        destination=path,
    )
    assert json.loads(path.read_text()) == manifest
    assert manifest["experts"][0]["expert"] == 0
    with pytest.raises(AnalysisError, match="refusing to overwrite"):
        freeze_candidates(
            ranking,
            bootstrap,
            permutation,
            top_n=2,
            source_hashes={"telemetry": "a" * 64},
            destination=path,
        )


def test_builds_predeclared_causal_control_sets():
    ranking = []
    for layer in range(2):
        for expert in range(6):
            ranking.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "differential": 10 - expert,
                    "routing_frequency": expert / 10,
                }
            )
    controls = build_control_sets(ranking, [(0, 0), (1, 0)], random_sets=20, seed=9)
    assert len(controls["layer_matched_random_sets"]) == 20
    assert all(len(item["experts"]) == 2 for item in controls["layer_matched_random_sets"])
    assert len(controls["frequency_matched_random_sets"]) == 20
    assert all(len(item["experts"]) == 2 for item in controls["frequency_matched_random_sets"])
    assert len(controls["highest_frequency_set"]) == 2
    assert len(controls["lowest_differential_set"]) == 2


def test_rejects_too_few_random_controls():
    with pytest.raises(AnalysisError, match="at least 20"):
        build_control_sets([], [], random_sets=19, seed=1)
