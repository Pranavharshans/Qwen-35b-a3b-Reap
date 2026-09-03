"""Differential expert ranking with deterministic bootstrap and permutation inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
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
    for row in observations:
        domain = row.get("domain")
        if domain not in grouped:
            raise AnalysisError(f"invalid domain: {domain!r}")
        value = float(row["reap_saliency"])
        if not np.isfinite(value):
            raise AnalysisError("REAP observations must be finite")
        grouped[domain][(int(row["layer"]), int(row["expert"]))].append(value)
    keys = set(grouped["coding"]) | set(grouped["control"])
    incomplete = [
        key for key in keys if key not in grouped["coding"] or key not in grouped["control"]
    ]
    if incomplete:
        raise AnalysisError(f"experts missing a coding or control observation: {incomplete[:5]}")
    means = {
        domain: {key: float(np.mean(values)) for key, values in domain_values.items()}
        for domain, domain_values in grouped.items()
    }
    coding_z = _zscore_by_layer(means["coding"])
    control_z = _zscore_by_layer(means["control"])
    rows = [
        {
            "layer": layer,
            "expert": expert,
            "coding_mean_reap": means["coding"][(layer, expert)],
            "control_mean_reap": means["control"][(layer, expert)],
            "coding_z": coding_z[(layer, expert)],
            "control_z": control_z[(layer, expert)],
            "differential": coding_z[(layer, expert)] - control_z[(layer, expert)],
        }
        for layer, expert in keys
    ]
    return sorted(rows, key=lambda row: (-row["differential"], row["layer"], row["expert"]))


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
    baseline = differential_ranking(observations)
    target = {(row["layer"], row["expert"]) for row in baseline[:top_n]}
    groups = _sample_groups(observations)
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        rows_by_sample[str(row["sample_id"])].append(row)
    rng = np.random.default_rng(seed)
    jaccards = []
    selection_counts: dict[tuple[int, int], int] = defaultdict(int)
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for ids in groups.values():
            chosen = rng.choice(ids, size=len(ids), replace=True)
            for sample_id in chosen:
                sampled.extend(rows_by_sample[str(sample_id)])
        ranked = differential_ranking(sampled)
        selected = {(row["layer"], row["expert"]) for row in ranked[:top_n]}
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
    }


def label_permutation(
    observations: list[dict[str, Any]], *, top_n: int, iterations: int, seed: int
) -> dict[str, Any]:
    """Permute domain labels at sample level while preserving group sizes."""
    baseline = differential_ranking(observations)
    observed = float(sum(row["differential"] for row in baseline[:top_n]))
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_labels: dict[str, str] = {}
    strata: dict[str, list[str]] = defaultdict(list)
    for row in observations:
        sample_id = str(row["sample_id"])
        rows_by_sample[sample_id].append(row)
        sample_labels[sample_id] = str(row["domain"])
        strata[str(row.get("stratum", "default"))].append(sample_id)
    strata = {key: sorted(set(ids)) for key, ids in strata.items()}
    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(iterations):
        permuted_labels: dict[str, str] = {}
        for ids in strata.values():
            labels = [sample_labels[sample_id] for sample_id in ids]
            shuffled = rng.permutation(labels)
            permuted_labels.update(zip(ids, shuffled, strict=True))
        permuted = [
            {**row, "domain": permuted_labels[str(row["sample_id"])]}
            for row in observations
        ]
        try:
            ranking = differential_ranking(permuted)
        except AnalysisError:
            continue
        null_scores.append(float(sum(row["differential"] for row in ranking[:top_n])))
    if not null_scores:
        raise AnalysisError("no valid label permutations; ensure both domains exist within strata")
    exceedances = sum(score >= observed for score in null_scores)
    return {
        "iterations_requested": iterations,
        "iterations_valid": len(null_scores),
        "top_n": top_n,
        "seed": seed,
        "observed_top_sum": observed,
        "p_value": (exceedances + 1) / (len(null_scores) + 1),
        "null_scores": null_scores,
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
    payload = {
        "schema_version": 1,
        "status": "domain-differential candidate",
        "selection_method": "within-layer-zscore-coding-minus-control",
        "top_n": top_n,
        "thresholds": {"median_bootstrap_jaccard": 0.60, "permutation_p_value": 0.05},
        "gate_passed": bootstrap["median_jaccard"] >= 0.60 and permutation["p_value"] <= 0.05,
        "experts": [
            {"layer": row["layer"], "expert": row["expert"], "differential": row["differential"]}
            for row in ranking[:top_n]
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
