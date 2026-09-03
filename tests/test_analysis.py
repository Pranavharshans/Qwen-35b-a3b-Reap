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


def test_ranking_records_single_domain_experts_without_ranking_them():
    rows = observations()
    # Drop every control observation of expert 2 so it is coding-only.
    partial = [row for row in rows if not (row["domain"] == "control" and row["expert"] == 2)]
    ranking = differential_ranking(partial)
    ranked = [row for row in ranking if row["observed"]]
    unranked = [row for row in ranking if not row["observed"]]
    assert {(row["layer"], row["expert"]) for row in ranked} == {(0, 0), (0, 1)}
    assert ranking[:2] == ranked  # ranked intersection first, unranked trailing
    assert [(row["layer"], row["expert"]) for row in unranked] == [(0, 2)]
    assert unranked[0]["observed_in"] == ["coding"]
    assert unranked[0]["differential"] is None
    assert unranked[0]["exclusion_reason"]


def test_ranking_rejects_empty_shared_universe():
    with pytest.raises(AnalysisError, match="no experts observed in both"):
        differential_ranking([observations()[0]])


def _stratified_rows(strata_assignments, saliency_by_domain, sample_offset=0.0):
    """Build observations: strata_assignments maps stratum -> list of domains."""
    rows = []
    sample_number = 0
    for stratum, domains in strata_assignments.items():
        for domain in domains:
            for expert in range(2):
                rows.append(
                    {
                        "sample_id": f"s{sample_number}",
                        "domain": domain,
                        "stratum": stratum,
                        "layer": 0,
                        "expert": expert,
                        "reap_saliency": (
                            saliency_by_domain[domain][expert]
                            + sample_number * sample_offset
                        ),
                    }
                )
            sample_number += 1
    return rows


def test_permutation_single_domain_strata_switch_to_global_fallback():
    rows = _stratified_rows(
        {"only-coding": ["coding", "coding"], "only-control": ["control", "control"]},
        {"coding": [8.0, 7.0], "control": [1.0, 2.0]},
    )
    report = label_permutation(rows, top_n=1, iterations=200, seed=7)
    assert report["method"] == "global-count-preserving-exact-enumeration"
    assert "stratified" not in report["method"]
    assert report["assignment_mode"] == "exact-enumeration"
    assert report["permutation_design"]["mixed_strata"] == 0
    assert report["permutation_design"]["single_domain_strata"] == 2
    assert report["permutation_design"]["coding_samples"] == 2
    assert report["permutation_design"]["control_samples"] == 2
    assert report["global_fallback_limitation"]
    assert report["attainable_assignments"] == 6  # C(4, 2)
    assert report["assignments_evaluated"] == 6
    assert report["permutations_changed_labels"] is True
    assert report["permutation_design_valid"] is True
    assert report["unique_null_statistics"] > 1
    assert 0 < report["p_value"] <= 1


def test_permutation_mixed_strata_stay_stratified():
    rows = _stratified_rows(
        {"shared": ["coding", "control", "coding", "control"]},
        {"coding": [8.0, 7.0], "control": [1.0, 2.0]},
    )
    report = label_permutation(rows, top_n=1, iterations=200, seed=7)
    assert report["method"] == "stratified-count-preserving-exact-enumeration"
    assert report["permutation_design"]["mixed_strata"] == 1
    assert report["global_fallback_limitation"] is None
    assert report["attainable_assignments"] == 6  # C(4, 2) in the single stratum
    assert report["assignments_evaluated"] == 6
    assert report["unique_null_statistics"] > 1


def test_permutation_exact_enumeration_is_deterministic():
    rows = _stratified_rows(
        {"only-coding": ["coding", "coding"], "only-control": ["control", "control"]},
        {"coding": [8.0, 7.0], "control": [1.0, 2.0]},
        sample_offset=0.01,
    )
    def strip_seed(report):
        return {key: value for key, value in report.items() if key != "seed"}

    first = label_permutation(rows, top_n=1, iterations=200, seed=7)
    second = label_permutation(rows, top_n=1, iterations=200, seed=99)
    assert strip_seed(first) == strip_seed(second)  # enumeration ignores the seed
    assert label_permutation(rows, top_n=1, iterations=200, seed=7) == first


def test_permutation_monte_carlo_is_deterministic_and_deduplicates():
    rows = _stratified_rows(
        {"big": ["coding"] * 12 + ["control"] * 12},
        {"coding": [8.0, 7.0], "control": [1.0, 2.0]},
        sample_offset=0.01,
    )
    first = label_permutation(rows, top_n=1, iterations=25, seed=11)
    second = label_permutation(rows, top_n=1, iterations=25, seed=11)
    assert first == second
    assert first["assignment_mode"] == "monte-carlo"
    assert first["method"] == "stratified-count-preserving-monte-carlo"
    assert first["assignments_evaluated"] <= 25
    # Monte Carlo never repeats an assignment: every evaluated assignment is unique.
    assert first["unique_assignments_evaluated"] == first["assignments_evaluated"]
    assert first["unique_null_statistics"] > 1
    assert first["permutation_design_valid"] is True


def test_permutation_fails_closed_on_degenerate_null():
    rows = _stratified_rows(
        {"only-coding": ["coding"], "only-control": ["control"]},
        {"coding": [5.0, 5.0], "control": [5.0, 5.0]},
    )
    with pytest.raises(AnalysisError, match="degenerate"):
        label_permutation(rows, top_n=1, iterations=200, seed=3)


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
    assert len(controls["task_agnostic_reap_set"]) == 2
    assert len(controls["lowest_differential_set"]) == 2


def test_per_expert_reports_include_uncertainty_and_empirical_null():
    rows = observations()
    bootstrap = bootstrap_stability(rows, top_n=1, iterations=20, seed=3)
    permutation = label_permutation(rows, top_n=1, iterations=20, seed=3)
    assert bootstrap["differential_intervals"]
    assert all(item["observations"] == 20 for item in bootstrap["differential_intervals"])
    assert permutation["expert_p_values"]
    assert all(0 < item["p_value"] <= 1 for item in permutation["expert_p_values"])


def test_rejects_too_few_random_controls():
    with pytest.raises(AnalysisError, match="at least 20"):
        build_control_sets([], [], random_sets=19, seed=1)
