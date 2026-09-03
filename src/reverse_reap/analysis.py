"""Differential expert ranking with deterministic bootstrap and permutation inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np


class AnalysisError(ValueError):
    """Raised when observational evidence is incomplete or malformed."""


def _zscore_by_layer(values: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    layers: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (layer, expert), value in values.items():
        layers[layer].append((expert, value))
    result: dict[tuple[int, int], float] = {}
    for layer, expert_values in layers.items():
        raw = np.asarray([value for _, value in expert_values], dtype=np.float64)
        mean, std = float(raw.mean()), float(raw.std())
        for expert, value in expert_values:
            result[(layer, expert)] = (value - mean) / std if std > 0 else 0.0
    return result


def differential_ranking(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank coding-control effects from sample-level REAP observations."""
    grouped: dict[str, dict[tuple[int, int], list[float]]] = {
        "coding": defaultdict(list),
        "control": defaultdict(list),
    }
    routing_counts: dict[tuple[int, int], list[float]] = defaultdict(list)
    domain_metrics: dict[str, dict[tuple[int, int], dict[str, float]]] = {
        "coding": defaultdict(lambda: defaultdict(float)),
        "control": defaultdict(lambda: defaultdict(float)),
    }
    sample_token_counts: dict[tuple[str, str], float] = {}
    for row in observations:
        domain = row.get("domain")
        if domain not in grouped:
            raise AnalysisError(f"invalid domain: {domain!r}")
        value = float(row["reap_saliency"])
        if not np.isfinite(value):
            raise AnalysisError("REAP observations must be finite")
        sample_key = (domain, str(row["sample_id"]))
        token_count = float(row.get("token_count", 0))
        sample_token_counts[sample_key] = max(sample_token_counts.get(sample_key, 0), token_count)
        grouped[domain][(int(row["layer"]), int(row["expert"]))].append(value)
        routing_counts[(int(row["layer"]), int(row["expert"]))].append(
            float(row.get("routed_count", 0))
        )
        key = (int(row["layer"]), int(row["expert"]))
        metrics = domain_metrics[domain][key]
        metrics["routing_count"] += float(row.get("routed_count", 0))
        metrics["router_weight_sum"] += float(row.get("router_weight_sum", 0))
        metrics["expert_output_norm_sum"] += float(row.get("expert_output_norm_sum", 0))
        metrics["weighted_norm_sum"] += float(
            row.get(
                "weighted_norm_sum",
                float(row["reap_saliency"]) * float(row.get("routed_count", 0)),
            )
        )
    shared = set(grouped["coding"]) & set(grouped["control"])
    if not shared:
        raise AnalysisError("no experts observed in both coding and control")
    means = {
        domain: {
            key: float(np.mean(values)) for key, values in domain_values.items() if key in shared
        }
        for domain, domain_values in grouped.items()
    }
    domain_token_counts = {
        domain: sum(
            count for (sample_domain, _), count in sample_token_counts.items()
            if sample_domain == domain
        )
        for domain in ("coding", "control")
    }
    coding_z = _zscore_by_layer(means["coding"])
    control_z = _zscore_by_layer(means["control"])
    rows = []
    for layer, expert in shared:
        key = (layer, expert)
        row = {
            "layer": layer,
            "expert": expert,
            "observed": True,
            "observed_in": ["coding", "control"],
            "coding_mean_reap": means["coding"][key],
            "control_mean_reap": means["control"][key],
            "coding_z": coding_z[key],
            "control_z": control_z[key],
            "differential": coding_z[key] - control_z[key],
            "routing_frequency": float(np.mean(routing_counts[key])),
        }
        for domain in ("coding", "control"):
            metric = domain_metrics[domain][key]
            count = metric["routing_count"]
            tokens = domain_token_counts[domain]
            prefix = f"{domain}_"
            row.update(
                {
                    f"{prefix}routing_count": int(count),
                    f"{prefix}routing_rate": count / tokens if tokens else None,
                    f"{prefix}router_weight_sum": metric["router_weight_sum"],
                    f"{prefix}router_weight_mean": (
                        metric["router_weight_sum"] / count if count else None
                    ),
                    f"{prefix}expert_output_norm_mean": (
                        metric["expert_output_norm_sum"] / count if count else None
                    ),
                    f"{prefix}standard_reap_saliency": (
                        metric["weighted_norm_sum"] / count if count else None
                    ),
                }
            )
        rows.append(row)
    for layer, expert in sorted((set(grouped["coding"]) | set(grouped["control"])) - shared):
        key = (layer, expert)
        observed_in = sorted(
            domain for domain in ("coding", "control") if key in grouped[domain]
        )
        rows.append(
            {
                "layer": layer,
                "expert": expert,
                "observed": False,
                "observed_in": observed_in,
                "coding_mean_reap": means["coding"].get(key),
                "control_mean_reap": means["control"].get(key),
                "coding_z": None,
                "control_z": None,
                "differential": None,
                "routing_frequency": float(np.mean(routing_counts[key])),
                "exclusion_reason": "observed in only one domain; not ranked",
            }
        )
    ranked = sorted(
        (row for row in rows if row["observed"]),
        key=lambda row: (-row["differential"], row["layer"], row["expert"]),
    )
    unranked = sorted(
        (row for row in rows if not row["observed"]),
        key=lambda row: (row["layer"], row["expert"]),
    )
    return ranked + unranked


def unobserved_experts(
    ranking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the recorded-but-unranked experts excluded from the differential."""
    return [row for row in ranking if not row.get("observed", True)]


def _sample_groups(observations: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in observations:
        groups[(str(row["domain"]), str(row.get("stratum", "default")))].add(str(row["sample_id"]))
    return {key: sorted(value) for key, value in groups.items()}


def bootstrap_stability(
    observations: list[dict[str, Any]], *, top_n: int, iterations: int, seed: int
) -> dict[str, Any]:
    if top_n <= 0 or iterations <= 0:
        raise ValueError("top_n and iterations must be positive")
    baseline = [row for row in differential_ranking(observations) if row["observed"]]
    if not baseline:
        raise AnalysisError("no experts observed in both coding and control")
    target = {(row["layer"], row["expert"]) for row in baseline[:top_n]}
    groups = _sample_groups(observations)
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        rows_by_sample[str(row["sample_id"])].append(row)
    rng = np.random.default_rng(seed)
    jaccards = []
    selection_counts: dict[tuple[int, int], int] = defaultdict(int)
    differential_samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for ids in groups.values():
            chosen = rng.choice(ids, size=len(ids), replace=True)
            for sample_id in chosen:
                sampled.extend(rows_by_sample[str(sample_id)])
        ranked = [row for row in differential_ranking(sampled) if row["observed"]]
        selected = {(row["layer"], row["expert"]) for row in ranked[:top_n]}
        for row in ranked:
            differential_samples[(row["layer"], row["expert"])].append(row["differential"])
        for key in selected:
            selection_counts[key] += 1
        union = target | selected
        jaccards.append(len(target & selected) / len(union) if union else 1.0)
    return {
        "iterations": iterations,
        "top_n": top_n,
        "seed": seed,
        "median_jaccard": float(np.median(jaccards)),
        "jaccards": jaccards,
        "selection_frequency": [
            {"layer": key[0], "expert": key[1], "frequency": count / iterations}
            for key, count in sorted(selection_counts.items())
        ],
        "differential_intervals": [
            {
                "layer": key[0],
                "expert": key[1],
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "observations": len(values),
            }
            for key, values in sorted(differential_samples.items())
        ],
    }


def _sample_fields(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    labels: dict[str, str] = {}
    stratum_of: dict[str, str] = {}
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        sample_id = str(row["sample_id"])
        labels[sample_id] = str(row["domain"])
        stratum_of[sample_id] = str(row.get("stratum", "default"))
        rows_by_sample[sample_id].append(row)
    members: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(labels):
        members[stratum_of[sample_id]].append(sample_id)
    return labels, dict(sorted(members.items())), rows_by_sample


def _design_report(labels: dict[str, str], strata: dict[str, list[str]]) -> dict[str, Any]:
    entries = []
    for name, ids in strata.items():
        coding = sum(1 for sample_id in ids if labels[sample_id] == "coding")
        control = len(ids) - coding
        entries.append(
            {
                "stratum": name,
                "coding_samples": coding,
                "control_samples": control,
                "mixed": coding > 0 and control > 0,
            }
        )
    mixed = sum(1 for entry in entries if entry["mixed"])
    return {
        "strata": entries,
        "mixed_strata": mixed,
        "single_domain_strata": len(entries) - mixed,
        "samples": len(labels),
        "coding_samples": sum(1 for label in labels.values() if label == "coding"),
        "control_samples": sum(1 for label in labels.values() if label == "control"),
    }


PERMUTATION_ENUMERATION_LIMIT = 4096
_PERMUTATION_ATTEMPT_FACTOR = 50
_GLOBAL_FALLBACK_LIMITATION = (
    "global fallback: domain labels are exchanged across all samples ignoring strata "
    "because every declared stratum is single-domain; stratum balance is therefore "
    "not preserved under the null, so this test is a count-preserving global "
    "permutation, not a stratified one"
)


def _permuted_statistic(
    assignment: dict[str, str],
    rows_by_sample: dict[str, list[dict[str, Any]]],
    *,
    top_n: int,
    observed_by_key: dict[tuple[int, int], float],
    expert_exceedances: dict[tuple[int, int], int],
    expert_null_observations: dict[tuple[int, int], int],
) -> float:
    permuted = [
        {**row, "domain": assignment[str(row["sample_id"])]}
        for rows in rows_by_sample.values()
        for row in rows
    ]
    ranked = [row for row in differential_ranking(permuted) if row["observed"]]
    for row in ranked:
        key = (row["layer"], row["expert"])
        expert_null_observations[key] += 1
        if key in observed_by_key and float(row["differential"]) >= observed_by_key[key]:
            expert_exceedances[key] += 1
    return float(sum(row["differential"] for row in ranked[:top_n]))


def label_permutation(
    observations: list[dict[str, Any]], *, top_n: int, iterations: int, seed: int
) -> dict[str, Any]:
    """Permutation test with a mandatory design guard and fail-closed semantics.

    The design guard inspects sample-level strata. When every declared stratum
    is single-domain (strata confounded with domain), within-stratum shuffling
    never changes any label, so the test switches explicitly to a global
    count-preserving sample-level permutation that preserves the original
    coding/control counts. The fallback is never described as stratified.

    Assignments are exact-enumerated when the attainable assignment count is
    tractable (``<= PERMUTATION_ENUMERATION_LIMIT``); otherwise a deterministic
    seeded Monte Carlo runs without duplicate assignments where practical.

    Fails closed via :class:`AnalysisError` when labels can never change, when
    the null has only one attainable assignment/statistic, or when no
    assignment can be evaluated.
    """
    baseline = [row for row in differential_ranking(observations) if row["observed"]]
    if not baseline:
        raise AnalysisError("no experts observed in both coding and control")
    observed = float(sum(row["differential"] for row in baseline[:top_n]))
    observed_by_key = {
        (row["layer"], row["expert"]): float(row["differential"]) for row in baseline
    }
    sample_labels, strata, rows_by_sample = _sample_fields(observations)
    design = _design_report(sample_labels, strata)
    rng = np.random.default_rng(seed)
    unique_assignments = 0
    null_scores: list[float] = []
    expert_exceedances: dict[tuple[int, int], int] = defaultdict(int)
    expert_null_observations: dict[tuple[int, int], int] = defaultdict(int)

    saw_changed_labels = False

    def evaluate(assignment: dict[str, str]) -> float:
        nonlocal saw_changed_labels
        if assignment != sample_labels:
            saw_changed_labels = True
        return _permuted_statistic(
            assignment,
            rows_by_sample,
            top_n=top_n,
            observed_by_key=observed_by_key,
            expert_exceedances=expert_exceedances,
            expert_null_observations=expert_null_observations,
        )

    global_fallback = design["mixed_strata"] == 0
    if global_fallback:
        all_ids = sorted(sample_labels)
        coding_ids = sorted(
            sample_id for sample_id in all_ids if sample_labels[sample_id] == "coding"
        )
        attainable = math.comb(len(all_ids), len(coding_ids))
        if attainable <= 1:
            raise AnalysisError(
                "permutation design invalid: with these domain counts no assignment "
                "can change any label; the requested test cannot be evaluated"
            )
        limitation: str | None = _GLOBAL_FALLBACK_LIMITATION
        if attainable <= PERMUTATION_ENUMERATION_LIMIT:
            method = "global-count-preserving-exact-enumeration"
            mode = "exact-enumeration"
            for coding_subset in combinations(all_ids, len(coding_ids)):
                assignment = {
                    sample_id: ("coding" if sample_id in set(coding_subset) else "control")
                    for sample_id in all_ids
                }
                null_scores.append(evaluate(assignment))
            unique_assignments = len(null_scores)
        else:
            method = "global-count-preserving-monte-carlo"
            mode = "monte-carlo"
            seen: set[tuple[int, ...]] = set()
            attempts = 0
            while (
                len(null_scores) < min(iterations, attainable)
                and attempts < iterations * _PERMUTATION_ATTEMPT_FACTOR
            ):
                attempts += 1
                chosen = tuple(
                    sorted(
                        rng.choice(len(all_ids), size=len(coding_ids), replace=False).tolist()
                    )
                )
                if chosen in seen:
                    continue
                seen.add(chosen)
                assignment = {
                    sample_id: ("coding" if index in chosen else "control")
                    for index, sample_id in enumerate(all_ids)
                }
                null_scores.append(evaluate(assignment))
            unique_assignments = len(seen)
    else:
        attainable = 1
        mixed_names = [
            name for name, ids in strata.items()
            if any(sample_labels[s] == "coding" for s in ids)
            and any(sample_labels[s] == "control" for s in ids)
        ]
        for name in mixed_names:
            ids = strata[name]
            coding_in_stratum = sum(1 for s in ids if sample_labels[s] == "coding")
            attainable *= math.comb(len(ids), coding_in_stratum)
        limitation = None
        if attainable <= 1:
            raise AnalysisError(
                "permutation design invalid: no stratum can exchange labels; "
                "the requested test cannot be evaluated"
            )
        if attainable <= PERMUTATION_ENUMERATION_LIMIT:
            method = "stratified-count-preserving-exact-enumeration"
            mode = "exact-enumeration"
            fixed = {}
            per_stratum_options = []
            for name, ids in strata.items():
                coding_in_stratum = {
                    s for s in ids if sample_labels[s] == "coding"
                }
                if name in mixed_names:
                    per_stratum_options.append(
                        [
                            set(combo)
                            for combo in combinations(ids, len(coding_in_stratum))
                        ]
                    )
                else:
                    fixed.update(
                        {sample_id: sample_labels[sample_id] for sample_id in ids}
                    )
            for combo_set in product(*per_stratum_options):
                coding_subset = set().union(*combo_set) if combo_set else set()
                assignment = {
                    **fixed,
                    **{
                        sample_id: ("coding" if sample_id in coding_subset else "control")
                        for name in mixed_names
                        for sample_id in strata[name]
                    },
                }
                null_scores.append(evaluate(assignment))
            unique_assignments = len(null_scores)
        else:
            method = "stratified-count-preserving-monte-carlo"
            mode = "monte-carlo"
            seen = set()
            attempts = 0
            strata_plan = []
            for name in mixed_names:
                ids = strata[name]
                coding_in_stratum = sum(1 for s in ids if sample_labels[s] == "coding")
                strata_plan.append((name, ids, coding_in_stratum))
            while (
                len(null_scores) < min(iterations, attainable)
                and attempts < iterations * _PERMUTATION_ATTEMPT_FACTOR
            ):
                attempts += 1
                coding_subset: set[str] = set()
                key_parts = []
                for name, ids, coding_in_stratum in strata_plan:
                    chosen = tuple(
                        sorted(
                            rng.permutation(len(ids))[:coding_in_stratum].tolist()
                        )
                    )
                    key_parts.append((name, chosen))
                    coding_subset.update(ids[index] for index in chosen)
                key = tuple(key_parts)
                if key in seen:
                    continue
                seen.add(key)
                assignment = {
                    sample_id: ("coding" if sample_id in coding_subset else "control")
                    for name, ids, _ in strata_plan
                    for sample_id in ids
                }
                null_scores.append(evaluate(assignment))
            unique_assignments = len(seen)
    if not null_scores:
        raise AnalysisError("permutation test cannot be evaluated: no assignments ran")
    if not saw_changed_labels:
        raise AnalysisError(
            "permutation test invalid: no evaluated assignment changed any label"
        )
    if len(set(null_scores)) <= 1:
        raise AnalysisError(
            "permutation null is degenerate: every attainable assignment yields the "
            "same statistic, so the test carries no information"
        )
    labels_changed = saw_changed_labels
    exceedances = sum(score >= observed for score in null_scores)
    if mode == "exact-enumeration":
        p_value = exceedances / len(null_scores)
    else:
        p_value = (exceedances + 1) / (len(null_scores) + 1)
    return {
        "iterations_requested": iterations,
        "iterations_valid": len(null_scores),
        "top_n": top_n,
        "seed": seed,
        "observed_top_sum": observed,
        "p_value": p_value,
        "null_scores": null_scores,
        "expert_p_values": [
            {
                "layer": key[0],
                "expert": key[1],
                "p_value": (expert_exceedances[key] + 1)
                / (expert_null_observations[key] + 1),
                "null_observations": expert_null_observations[key],
            }
            for key in sorted(expert_null_observations)
        ],
        "method": method,
        "assignment_mode": mode,
        "permutation_design": design,
        "attainable_assignments": attainable,
        "assignments_evaluated": len(null_scores),
        "unique_assignments_evaluated": unique_assignments,
        "permutations_changed_labels": labels_changed,
        "unique_null_statistics": len(set(null_scores)),
        "permutation_design_valid": True,
        "global_fallback_limitation": limitation,
    }


def freeze_candidates(
    ranking: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    permutation: dict[str, Any],
    *,
    top_n: int,
    source_hashes: dict[str, str],
    destination: Path,
) -> dict[str, Any]:
    ranked = [row for row in ranking if row.get("observed", True)]
    unranked = [row for row in ranking if not row.get("observed", True)]
    if len(ranked) < top_n:
        raise AnalysisError(
            f"only {len(ranked)} experts observed in both domains; cannot select top-{top_n}"
        )
    payload = {
        "schema_version": 1,
        "status": "domain-differential candidate",
        "selection_method": "within-layer-zscore-coding-minus-control",
        "selection_universe": "experts-observed-in-both-domains",
        "top_n": top_n,
        "thresholds": {"median_bootstrap_jaccard": 0.60, "permutation_p_value": 0.05},
        "gate_passed": bootstrap["median_jaccard"] >= 0.60 and permutation["p_value"] <= 0.05,
        "experts_ranked": len(ranked),
        "experts_unranked_single_domain": len(unranked),
        "experts": [
            {"layer": row["layer"], "expert": row["expert"], "differential": row["differential"]}
            for row in ranked[:top_n]
        ],
        "source_hashes": dict(sorted(source_hashes.items())),
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(unsigned).hexdigest()
    body = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    if destination.exists() and destination.read_bytes() != body:
        raise AnalysisError(f"refusing to overwrite frozen candidate manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return payload


def build_control_sets(
    ranking: list[dict[str, Any]],
    selected: list[tuple[int, int]],
    *,
    random_sets: int = 20,
    seed: int,
) -> dict[str, Any]:
    """Precompute layer-matched, frequency-matched, and negative controls.

    Control pools use only ranked (both-domain-observed) experts so every
    control is a valid differential comparator for a selected candidate.
    """
    if random_sets < 20:
        raise AnalysisError("v0 requires at least 20 random control sets")
    selected_set = set(selected)
    by_layer: dict[int, list[int]] = defaultdict(list)
    row_by_key = {}
    for row in ranking:
        if not row.get("observed", True):
            continue
        key = (int(row["layer"]), int(row["expert"]))
        row_by_key[key] = row
        if key not in selected_set:
            by_layer[key[0]].append(key[1])
    required_by_layer: dict[int, int] = defaultdict(int)
    for layer, _ in selected:
        required_by_layer[layer] += 1
    rng = np.random.default_rng(seed)
    layer_matched = []
    for index in range(random_sets):
        members = []
        for layer, count in sorted(required_by_layer.items()):
            pool = sorted(by_layer[layer])
            if len(pool) < count:
                raise AnalysisError(f"insufficient non-selected experts in layer {layer}")
            experts = rng.choice(pool, size=count, replace=False)
            members.extend({"layer": layer, "expert": int(expert)} for expert in experts)
        layer_matched.append({"control_id": f"layer-random-{index:03d}", "experts": members})

    frequency_matched_sets = []
    for index in range(random_sets):
        available = set(row_by_key) - selected_set
        members = []
        for target in selected:
            target_frequency = float(row_by_key[target].get("routing_frequency", 0.0))
            same_layer = [key for key in available if key[0] == target[0]]
            if not same_layer:
                raise AnalysisError(f"no frequency control available for {target}")
            ordered = sorted(
                same_layer,
                key=lambda key: (
                    abs(float(row_by_key[key].get("routing_frequency", 0.0)) - target_frequency),
                    key,
                ),
            )
            # Randomize only within the closest deterministic candidate window. This
            # produces independently frozen sets without sacrificing the matching goal.
            window = ordered[: min(random_sets, len(ordered))]
            chosen = window[int(rng.integers(0, len(window)))]
            available.remove(chosen)
            members.append({"layer": chosen[0], "expert": chosen[1]})
        frequency_matched_sets.append(
            {"control_id": f"frequency-random-{index:03d}", "experts": members}
        )
    eligible = [
        row
        for row in ranking
        if row.get("observed", True) and (row["layer"], row["expert"]) not in selected_set
    ]
    lowest = sorted(
        eligible,
        key=lambda row: (row["differential"], row["layer"], row["expert"]),
    )[: len(selected)]
    highest_frequency = sorted(
        eligible,
        key=lambda row: (-float(row.get("routing_frequency", 0.0)), row["layer"], row["expert"]),
    )[: len(selected)]
    task_agnostic_reap = sorted(
        eligible,
        key=lambda row: (
            -(
                float(row.get("coding_mean_reap", 0))
                + float(row.get("control_mean_reap", 0))
            ),
            row["layer"],
            row["expert"],
        ),
    )[: len(selected)]
    return {
        "schema_version": 1,
        "seed": seed,
        "layer_matched_random_sets": layer_matched,
        "frequency_matched_random_sets": frequency_matched_sets,
        # Kept as an explicit negative control, distinct from differential ranking.
        "highest_frequency_set": [
            {"layer": row["layer"], "expert": row["expert"]} for row in highest_frequency
        ],
        "task_agnostic_reap_set": [
            {"layer": row["layer"], "expert": row["expert"]}
            for row in task_agnostic_reap
        ],
        "lowest_differential_set": [
            {"layer": row["layer"], "expert": row["expert"]} for row in lowest
        ],
    }
