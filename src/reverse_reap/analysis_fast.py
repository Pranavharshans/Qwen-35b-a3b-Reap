"""Vectorized CPU analysis engine: streaming aggregation and NumPy inference.

The reference engine in :mod:`reverse_reap.analysis` materializes every
observation as a Python dictionary and re-runs the full dict-based ranking for
every bootstrap replicate and every permutation assignment. This module keeps
the same scientific semantics while:

1. streaming telemetry once into compact per-(sample, layer, expert) aggregates
   instead of retaining all JSONL rows as dictionaries,
2. caching those aggregates keyed by the telemetry SHA-256,
3. vectorizing the per-replicate ranking with NumPy,
4. computing one ranking per replicate at the maximum candidate cardinality and
   reusing it for every smaller cardinality in the grid,
5. batching exact permutation enumeration.

Differences from the reference are limited to floating-point summation order
(NumPy ``bincount`` versus per-key ``np.mean``), so float outputs agree to
within ~1e-9 relative while integer outputs (selection sets, exceedance
counts, p-value numerators) are identical. The reference implementation stays
importable and is the oracle in ``tests/test_analysis_optimized.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np

from reverse_reap.analysis import (
    _GLOBAL_FALLBACK_LIMITATION,
    _PERMUTATION_ATTEMPT_FACTOR,
    PERMUTATION_ENUMERATION_LIMIT,
    AnalysisError,
    _design_report,
)

CODING = 0
CONTROL = 1
DOMAIN_NAMES = ("coding", "control")
_CACHE_FORMAT_VERSION = 2
_DEFAULT_CACHE_DIR = Path(".cache/analysis")


def telemetry_sha256(path: Path) -> str:
    """Stream the file in chunks; equals sha256 of the full byte content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CellTable:
    """Compact per-(sample, layer, expert) aggregates over filtered telemetry.

    ``domain`` and ``stratum`` are per-sample properties; ``saliency`` is the
    per-cell REAP saliency (weighted norm divided by routed count) exactly as
    the reference pipeline derives it before calling the analysis functions.
    """

    sample_ids: list[str]
    domain_of_sample: np.ndarray
    stratum_index_of_sample: np.ndarray
    stratum_names: list[str]
    token_count_of_sample: np.ndarray
    group_order: tuple[tuple[int, int], ...]
    sample_index: np.ndarray
    key_index: np.ndarray
    layer_of_key: np.ndarray
    expert_of_key: np.ndarray
    saliency: np.ndarray
    routed_count: np.ndarray
    router_weight_sum: np.ndarray
    expert_output_norm_sum: np.ndarray
    weighted_norm_sum: np.ndarray
    routing_rows: int

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_cells(self) -> int:
        return len(self.sample_index)

    @property
    def n_keys(self) -> int:
        return len(self.layer_of_key)


def _empty_weights(table: CellTable) -> np.ndarray:
    return np.ones(table.n_samples, dtype=np.float64)


def _cells_from_maps(
    cells: dict[tuple[Any, ...], list[float]],
    saliency_of_cell: dict[tuple[Any, ...], float],
    sample_domain: dict[str, str],
    sample_stratum: dict[str, str],
    token_counts: dict[str, float],
    group_order: list[tuple[str, str]],
    routing_rows: int,
) -> CellTable:
    if not cells:
        raise AnalysisError("no experts observed in both coding and control")
    sample_ids = sorted(sample_domain)
    sample_pos = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    stratum_names = sorted({name for _, name in group_order})
    stratum_pos = {name: index for index, name in enumerate(stratum_names)}
    domain_of_sample = np.array(
        [CODING if sample_domain[s] == "coding" else CONTROL for s in sample_ids], dtype=np.int64
    )
    stratum_index_of_sample = np.array(
        [stratum_pos[sample_stratum[s]] for s in sample_ids], dtype=np.int64
    )
    token_count_of_sample = np.array([token_counts.get(s, 0.0) for s in sample_ids])
    group_order_encoded = tuple(
        (DOMAIN_NAMES.index(domain), stratum_pos[name]) for domain, name in group_order
    )
    keys = sorted({(cell_key[3], cell_key[4]) for cell_key in cells})
    key_pos = {key: index for index, key in enumerate(keys)}
    n_cells = len(cells)
    sample_index = np.empty(n_cells, dtype=np.int64)
    key_index = np.empty(n_cells, dtype=np.int64)
    routed_count = np.empty(n_cells, dtype=np.float64)
    router_weight_sum = np.empty(n_cells, dtype=np.float64)
    expert_output_norm_sum = np.empty(n_cells, dtype=np.float64)
    weighted_norm_sum = np.empty(n_cells, dtype=np.float64)
    saliency = np.empty(n_cells, dtype=np.float64)
    for position, cell_key in enumerate(
        sorted(cells, key=lambda k: (k[0], k[3], k[4]))
    ):
        values = cells[cell_key]
        sample_index[position] = sample_pos[cell_key[0]]
        key_index[position] = key_pos[(cell_key[3], cell_key[4])]
        routed_count[position] = values[0]
        router_weight_sum[position] = values[1]
        expert_output_norm_sum[position] = values[2]
        weighted_norm_sum[position] = values[3]
        saliency[position] = saliency_of_cell[cell_key]
    if not np.all(np.isfinite(saliency)):
        raise AnalysisError("REAP observations must be finite")
    layer_of_key = np.array([key[0] for key in keys], dtype=np.int64)
    expert_of_key = np.array([key[1] for key in keys], dtype=np.int64)
    return CellTable(
        sample_ids=sample_ids,
        domain_of_sample=domain_of_sample,
        stratum_index_of_sample=stratum_index_of_sample,
        stratum_names=stratum_names,
        token_count_of_sample=token_count_of_sample,
        group_order=group_order_encoded,
        sample_index=sample_index,
        key_index=key_index,
        layer_of_key=layer_of_key,
        expert_of_key=expert_of_key,
        saliency=saliency,
        routed_count=routed_count,
        router_weight_sum=router_weight_sum,
        expert_output_norm_sum=expert_output_norm_sum,
        weighted_norm_sum=weighted_norm_sum,
        routing_rows=routing_rows,
    )


def stream_cells(
    telemetry_path: Path,
    *,
    splits: tuple[str, ...] = ("calibration", "selection"),
    segment: str = "joint",
) -> CellTable:
    """One-pass streaming aggregation of telemetry into a :class:`CellTable`.

    Never materializes more than one JSONL row at a time. Applies exactly the
    same split/segment filter and per-cell aggregation branches as the
    reference pipeline.
    """
    parse_started = time.perf_counter()
    cells: dict[tuple[str, str, str, int, int], list[float]] = {}
    token_sets: dict[str, set[int]] = defaultdict(set)
    sample_domain: dict[str, str] = {}
    sample_stratum: dict[str, str] = {}
    group_order: list[tuple[str, str]] = []
    seen_groups: set[tuple[str, str]] = set()
    routing_rows = 0
    with telemetry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] not in splits:
                continue
            if segment != "joint" and row["segment"] != segment:
                continue
            routing_rows += 1
            domain = row["domain"]
            if domain not in DOMAIN_NAMES:
                raise AnalysisError(f"invalid domain: {domain!r}")
            sample_id = str(row["sample_id"])
            stratum = str(row.get("stratum", "default"))
            previous = sample_domain.get(sample_id)
            if previous is None:
                sample_domain[sample_id] = domain
                sample_stratum[sample_id] = stratum
            elif previous != domain or sample_stratum[sample_id] != stratum:
                raise AnalysisError(
                    f"sample {sample_id!r} appears with inconsistent domain/stratum"
                )
            group = (domain, stratum)
            if group not in seen_groups:
                seen_groups.add(group)
                group_order.append(group)
            if "token_index" in row:
                token_sets[sample_id].add(int(row["token_index"]))
            layer = int(row.get("layer_index", row.get("layer")))
            expert = int(row.get("expert_index", row.get("expert")))
            routed_count = int(row.get("routed_count", 1))
            cell_key = (sample_id, domain, stratum, layer, expert)
            cell = cells.get(cell_key)
            if cell is None:
                cells[cell_key] = cell = [0.0, 0.0, 0.0, 0.0]
            cell[0] += routed_count
            if "expert_output_l2" in row:
                weight = float(row["router_weight"])
                norm = float(row["expert_output_l2"])
                cell[1] += weight
                cell[2] += norm
                cell[3] += weight * norm
            else:
                cell[1] += float(row.get("router_mass", 0))
                cell[3] += float(row["reap_saliency"]) * routed_count
    if routing_rows == 0:
        raise ValueError("no telemetry rows match the requested splits and segment")
    parse_seconds = time.perf_counter() - parse_started
    aggregate_started = time.perf_counter()
    token_counts = {sample_id: float(len(ids)) for sample_id, ids in token_sets.items()}
    if len(token_counts) < len(sample_domain):
        # Samples without any token_index row fall back to their max cell count,
        # mirroring the reference per-observation token_count fallback.
        max_cell_count: dict[str, float] = defaultdict(float)
        for cell_key, values in cells.items():
            max_cell_count[cell_key[0]] = max(max_cell_count[cell_key[0]], values[0])
        for sample_id in sample_domain:
            token_counts.setdefault(sample_id, max_cell_count[sample_id])
    saliency_of_cell = {
        cell_key: values[3] / values[0]
        for cell_key, values in cells.items()
        if values[0] > 0
    }
    if len(saliency_of_cell) != len(cells):
        raise AnalysisError("REAP observations must be finite")
    table = _cells_from_maps(
        cells,
        saliency_of_cell,
        sample_domain,
        sample_stratum,
        token_counts,
        group_order,
        routing_rows,
    )
    stream_cells.last_parse_seconds = parse_seconds
    stream_cells.last_aggregate_seconds = time.perf_counter() - aggregate_started
    return table


def cell_table_from_observations(observations: list[dict[str, Any]]) -> CellTable:
    """Build the same compact table from reference-style observation dicts.

    Each observation row is treated exactly as :func:`reverse_reap.analysis.differential_ranking`
    treats it: ``reap_saliency`` is the per-row value (mean over repeated rows
    of the same cell), and the metric sums use the same fallbacks.
    """
    cells: dict[tuple[str, str, str, int, int], list[float]] = {}
    saliency_accumulator: dict[tuple[str, str, str, int, int], list[float]] = {}
    token_counts: dict[str, float] = {}
    sample_domain: dict[str, str] = {}
    sample_stratum: dict[str, str] = {}
    group_order: list[tuple[str, str]] = []
    seen_groups: set[tuple[str, str]] = set()
    routing_rows = 0
    for row in observations:
        routing_rows += 1
        domain = row.get("domain")
        if domain not in DOMAIN_NAMES:
            raise AnalysisError(f"invalid domain: {domain!r}")
        value = float(row["reap_saliency"])
        if not np.isfinite(value):
            raise AnalysisError("REAP observations must be finite")
        sample_id = str(row["sample_id"])
        stratum = str(row.get("stratum", "default"))
        previous = sample_domain.get(sample_id)
        if previous is None:
            sample_domain[sample_id] = domain
            sample_stratum[sample_id] = stratum
        elif previous != domain or sample_stratum[sample_id] != stratum:
            raise AnalysisError(
                f"sample {sample_id!r} appears with inconsistent domain/stratum"
            )
        group = (domain, stratum)
        if group not in seen_groups:
            seen_groups.add(group)
            group_order.append(group)
        token_count = float(row.get("token_count", 0))
        token_counts[sample_id] = max(token_counts.get(sample_id, 0.0), token_count)
        layer = int(row["layer"])
        expert = int(row["expert"])
        cell_key = (sample_id, domain, stratum, layer, expert)
        cell = cells.get(cell_key)
        if cell is None:
            cells[cell_key] = cell = [0.0, 0.0, 0.0, 0.0]
            saliency_accumulator[cell_key] = [0.0, 0]
        cell[0] += float(row.get("routed_count", 0))
        cell[1] += float(row.get("router_weight_sum", 0))
        cell[2] += float(row.get("expert_output_norm_sum", 0))
        cell[3] += float(
            row.get("weighted_norm_sum", value * float(row.get("routed_count", 0)))
        )
        accumulator = saliency_accumulator[cell_key]
        accumulator[0] += value
        accumulator[1] += 1
    if routing_rows == 0:
        raise AnalysisError("no experts observed in both coding and control")
    saliency_of_cell = {
        cell_key: values[0] / values[1]
        for cell_key, values in saliency_accumulator.items()
    }
    return _cells_from_maps(
        cells,
        saliency_of_cell,
        sample_domain,
        sample_stratum,
        token_counts,
        group_order,
        routing_rows,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_payload(
    table: CellTable, sha: str, splits: tuple[str, ...], segment: str
) -> dict[str, Any]:
    return {
        "meta": json.dumps(
            {
                "format_version": _CACHE_FORMAT_VERSION,
                "telemetry_sha256": sha,
                "splits": list(splits),
                "segment": segment,
                "n_samples": table.n_samples,
                "n_cells": table.n_cells,
                "n_keys": table.n_keys,
                "routing_rows": table.routing_rows,
            }
        ),
        "sample_ids": np.array(table.sample_ids),
        "domain_of_sample": table.domain_of_sample,
        "stratum_index_of_sample": table.stratum_index_of_sample,
        "stratum_names": np.array(table.stratum_names),
        "token_count_of_sample": table.token_count_of_sample,
        "group_order": np.array(table.group_order, dtype=np.int64).reshape(-1, 2),
        "sample_index": table.sample_index,
        "key_index": table.key_index,
        "layer_of_key": table.layer_of_key,
        "expert_of_key": table.expert_of_key,
        "saliency": table.saliency,
        "routed_count": table.routed_count,
        "router_weight_sum": table.router_weight_sum,
        "expert_output_norm_sum": table.expert_output_norm_sum,
        "weighted_norm_sum": table.weighted_norm_sum,
    }


def _table_from_cache(
    payload: dict[str, Any], sha: str, splits: tuple[str, ...], segment: str
) -> CellTable:
    meta = json.loads(str(payload["meta"]))
    expected = {
        "format_version": _CACHE_FORMAT_VERSION,
        "telemetry_sha256": sha,
        "splits": list(splits),
        "segment": segment,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise AnalysisError(f"cache metadata mismatch on {key}")
    group_array = np.asarray(payload["group_order"], dtype=np.int64).reshape(-1, 2)
    return CellTable(
        sample_ids=[str(item) for item in payload["sample_ids"].tolist()],
        domain_of_sample=np.asarray(payload["domain_of_sample"], dtype=np.int64),
        stratum_index_of_sample=np.asarray(payload["stratum_index_of_sample"], dtype=np.int64),
        stratum_names=[str(item) for item in payload["stratum_names"].tolist()],
        token_count_of_sample=np.asarray(payload["token_count_of_sample"], dtype=np.float64),
        group_order=tuple((int(a), int(b)) for a, b in group_array),
        sample_index=np.asarray(payload["sample_index"], dtype=np.int64),
        key_index=np.asarray(payload["key_index"], dtype=np.int64),
        layer_of_key=np.asarray(payload["layer_of_key"], dtype=np.int64),
        expert_of_key=np.asarray(payload["expert_of_key"], dtype=np.int64),
        saliency=np.asarray(payload["saliency"], dtype=np.float64),
        routed_count=np.asarray(payload["routed_count"], dtype=np.float64),
        router_weight_sum=np.asarray(payload["router_weight_sum"], dtype=np.float64),
        expert_output_norm_sum=np.asarray(payload["expert_output_norm_sum"], dtype=np.float64),
        weighted_norm_sum=np.asarray(payload["weighted_norm_sum"], dtype=np.float64),
        routing_rows=int(meta["routing_rows"]),
    )


def load_or_build_cells(
    telemetry_path: Path,
    *,
    splits: tuple[str, ...] = ("calibration", "selection"),
    segment: str = "joint",
    cache_dir: Path | None = None,
) -> CellTable:
    """Load the SHA-256-keyed aggregate cache or rebuild it from telemetry.

    The cache key is the telemetry file SHA-256, so a hash mismatch can never
    reuse a stale cache; a metadata mismatch (splits/segment/format) falls
    through to a rebuild that atomically replaces the cache file.
    """
    sha = telemetry_sha256(telemetry_path)
    directory = _DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache_path = directory / f"{sha}.npz"
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                table = _table_from_cache(dict(payload), sha, splits, segment)
            if (
                table.n_samples == int(table.domain_of_sample.size)
                and table.n_cells == int(table.saliency.size)
                and table.n_keys == int(table.layer_of_key.size)
            ):
                return table
        except (OSError, ValueError, KeyError, AnalysisError, json.JSONDecodeError):
            pass  # fall through to rebuild
    table = stream_cells(telemetry_path, splits=splits, segment=segment)
    directory.mkdir(parents=True, exist_ok=True)
    payload = _cache_payload(table, sha, splits, segment)
    fd, temporary = tempfile.mkstemp(prefix=f".{cache_path.name}.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return table


# ---------------------------------------------------------------------------
# Vectorized replicate ranking
# ---------------------------------------------------------------------------


def _z_within_layer(values: np.ndarray, layers: np.ndarray, n_layers: int) -> np.ndarray:
    """Per-layer population z-scores; 0.0 wherever the layer std is zero."""
    counts = np.bincount(layers, minlength=n_layers).astype(np.float64)
    total = np.bincount(layers, weights=values, minlength=n_layers).astype(np.float64)
    total_sq = np.bincount(layers, weights=values * values, minlength=n_layers).astype(
        np.float64
    )
    zeros = np.zeros(n_layers, dtype=np.float64)
    mean = np.divide(total, counts, out=zeros.copy(), where=counts > 0)
    variance = np.divide(total_sq, counts, out=zeros, where=counts > 0)
    variance = variance - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    z = np.zeros_like(values)
    ok = std[layers] > 0
    z[ok] = (values[ok] - mean[layers[ok]]) / std[layers[ok]]
    return z


def _replicate_ranking(
    table: CellTable,
    weights: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized differential ranking for one replicate.

    Returns ``(shared_mask, differential, order)`` where ``differential`` is a
    full-length key array (NaN off the shared universe) and ``order`` lists
    shared key indices sorted by (-differential, layer, expert).
    """
    n_keys = table.n_keys
    cell_label = labels[table.sample_index]
    cell_weight = weights[table.sample_index]
    bins = cell_label * n_keys + table.key_index
    counts = np.bincount(bins, weights=cell_weight, minlength=2 * n_keys)
    sums = np.bincount(bins, weights=cell_weight * table.saliency, minlength=2 * n_keys)
    coding_n, control_n = counts[:n_keys], counts[n_keys:]
    coding_sum, control_sum = sums[:n_keys], sums[n_keys:]
    shared = (coding_n > 0) & (control_n > 0)
    differential = np.full(n_keys, np.nan)
    index = np.flatnonzero(shared)
    if index.size == 0:
        return shared, differential, index
    coding_mean = coding_sum[index] / coding_n[index]
    control_mean = control_sum[index] / control_n[index]
    n_layers = int(table.layer_of_key.max()) + 1
    coding_z = _z_within_layer(coding_mean, table.layer_of_key[index], n_layers)
    control_z = _z_within_layer(control_mean, table.layer_of_key[index], n_layers)
    values = coding_z - control_z
    differential[index] = values
    order = index[
        np.lexsort((table.expert_of_key[index], table.layer_of_key[index], -values))
    ]
    return shared, differential, order


def fast_baseline_ranking(table: CellTable) -> list[dict[str, Any]]:
    """Reference-equivalent baseline ranking rows from the compact table."""
    n_keys = table.n_keys
    labels = table.domain_of_sample
    bins = labels[table.sample_index] * n_keys + table.key_index
    ones = np.ones(table.n_cells)
    n_layers = int(table.layer_of_key.max()) + 1

    def per_domain(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        aggregated = np.bincount(bins, weights=values, minlength=2 * n_keys)
        return aggregated[:n_keys], aggregated[n_keys:]

    cell_n_coding, cell_n_control = per_domain(ones)
    count_coding, count_control = per_domain(table.routed_count)
    weight_coding, weight_control = per_domain(table.router_weight_sum)
    norm_coding, norm_control = per_domain(table.expert_output_norm_sum)
    weighted_coding, weighted_control = per_domain(table.weighted_norm_sum)
    saliency_coding, saliency_control = per_domain(table.saliency)
    total_cells = np.bincount(table.key_index, weights=ones, minlength=n_keys)
    total_count = np.bincount(table.key_index, weights=table.routed_count, minlength=n_keys)
    domain_tokens = (
        float(table.token_count_of_sample[labels == CODING].sum()),
        float(table.token_count_of_sample[labels == CONTROL].sum()),
    )
    shared, differential, order = _replicate_ranking(table, _empty_weights(table), labels)
    index = np.flatnonzero(shared)
    coding_z = np.full(n_keys, np.nan)
    control_z = np.full(n_keys, np.nan)
    coding_z[index] = _z_within_layer(
        saliency_coding[index] / cell_n_coding[index], table.layer_of_key[index], n_layers
    )
    control_z[index] = _z_within_layer(
        saliency_control[index] / cell_n_control[index], table.layer_of_key[index], n_layers
    )
    rows: list[dict[str, Any]] = []
    for key in order:
        row: dict[str, Any] = {
            "layer": int(table.layer_of_key[key]),
            "expert": int(table.expert_of_key[key]),
            "observed": True,
            "observed_in": ["coding", "control"],
            "coding_mean_reap": float(saliency_coding[key] / cell_n_coding[key]),
            "control_mean_reap": float(saliency_control[key] / cell_n_control[key]),
            "coding_z": float(coding_z[key]),
            "control_z": float(control_z[key]),
            "differential": float(differential[key]),
            "routing_frequency": float(total_count[key] / total_cells[key]),
        }
        domain_rows = (
            ("coding", count_coding, weight_coding, norm_coding, weighted_coding),
            ("control", count_control, weight_control, norm_control, weighted_control),
        )
        for domain_index, (prefix, count_v, weight_v, norm_v, weighted_v) in enumerate(
            domain_rows
        ):
            tokens = domain_tokens[domain_index]
            row.update(
                {
                    f"{prefix}_routing_count": int(count_v[key]),
                    f"{prefix}_routing_rate": (count_v[key] / tokens if tokens else None),
                    f"{prefix}_router_weight_sum": float(weight_v[key]),
                    f"{prefix}_router_weight_mean": (
                        float(weight_v[key] / count_v[key]) if count_v[key] else None
                    ),
                    f"{prefix}_expert_output_norm_mean": (
                        float(norm_v[key] / count_v[key]) if count_v[key] else None
                    ),
                    f"{prefix}_standard_reap_saliency": (
                        float(weighted_v[key] / count_v[key]) if count_v[key] else None
                    ),
                }
            )
        rows.append(row)
    for key in np.flatnonzero(~shared):
        observed_in = [
            name
            for name, counts in (
                (DOMAIN_NAMES[CODING], cell_n_coding),
                (DOMAIN_NAMES[CONTROL], cell_n_control),
            )
            if counts[key] > 0
        ]
        rows.append(
            {
                "layer": int(table.layer_of_key[key]),
                "expert": int(table.expert_of_key[key]),
                "observed": False,
                "observed_in": observed_in,
                "coding_mean_reap": None,
                "control_mean_reap": None,
                "coding_z": None,
                "control_z": None,
                "differential": None,
                "routing_frequency": float(total_count[key] / total_cells[key]),
                "exclusion_reason": "observed in only one domain; not ranked",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Bootstrap (one shared replicate loop across all grid cardinalities)
# ---------------------------------------------------------------------------


def _baseline_arrays(table: CellTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(shared_mask, differential, order) for the unweighted original labels."""
    return _replicate_ranking(table, _empty_weights(table), table.domain_of_sample)


def fast_bootstrap(
    table: CellTable,
    *,
    top_ns: list[int],
    iterations: int,
    seed: int,
    baseline_order: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Reference-equivalent bootstrap artifacts for every requested top_n.

    Replicate k draws the same resample for every cardinality (identical RNG
    consumption to a per-cardinality reference call with the same seed), so one
    replicate loop serves the whole grid: the full per-replicate ranking is
    computed once and each top_n is a prefix of the same ranked order.
    """
    if iterations <= 0 or any(top_n <= 0 for top_n in top_ns):
        raise ValueError("top_n and iterations must be positive")
    n_samples = table.n_samples
    n_keys = table.n_keys
    sample_pos = {sample_id: index for index, sample_id in enumerate(table.sample_ids)}
    groups: list[np.ndarray] = []
    id_arrays: list[np.ndarray] = []
    for domain_index, stratum_index in table.group_order:
        members = [
            position
            for position in range(n_samples)
            if table.domain_of_sample[position] == domain_index
            and table.stratum_index_of_sample[position] == stratum_index
        ]
        members.sort(key=lambda position: table.sample_ids[position])
        array = np.array(members, dtype=np.int64)
        groups.append(array)
        id_arrays.append(np.array([table.sample_ids[m] for m in members]))
    labels = table.domain_of_sample
    targets = [set(int(key) for key in baseline_order[:top_n]) for top_n in top_ns]
    rng = np.random.default_rng(seed)
    replicate_differentials = np.full((iterations, n_keys), np.nan)
    jaccards: dict[int, list[float]] = {top_n: [] for top_n in top_ns}
    selection_counts: dict[int, np.ndarray] = {
        top_n: np.zeros(n_keys, dtype=np.int64) for top_n in top_ns
    }
    for replicate in range(iterations):
        weights = np.zeros(n_samples)
        for _group_index, id_array in enumerate(id_arrays):
            chosen = rng.choice(id_array, size=len(id_array), replace=True)
            indices = np.fromiter(
                (sample_pos[str(sample_id)] for sample_id in chosen),
                dtype=np.int64,
                count=len(chosen),
            )
            weights += np.bincount(indices, minlength=n_samples)
        _, differential, order = _replicate_ranking(table, weights, labels)
        replicate_differentials[replicate, order] = differential[order]
        for position, top_n in enumerate(top_ns):
            selected = order[:top_n]
            selection_counts[top_n][selected] += 1
            selected_set = set(selected.tolist())
            union = targets[position] | selected_set
            jaccards[top_n].append(
                len(targets[position] & selected_set) / len(union) if union else 1.0
            )
    artifacts: dict[int, dict[str, Any]] = {}
    for top_n in top_ns:
        selection_frequency = [
            {
                "layer": int(table.layer_of_key[key]),
                "expert": int(table.expert_of_key[key]),
                "frequency": selection_counts[top_n][key] / iterations,
            }
            for key in np.flatnonzero(selection_counts[top_n])
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            percentiles = np.nanpercentile(replicate_differentials, [2.5, 97.5], axis=0)
            observation_counts = np.isfinite(replicate_differentials).sum(axis=0)
        differential_intervals = [
            {
                "layer": int(table.layer_of_key[key]),
                "expert": int(table.expert_of_key[key]),
                "low": float(percentiles[0][key]),
                "high": float(percentiles[1][key]),
                "observations": int(observation_counts[key]),
            }
            for key in range(n_keys)
            if observation_counts[key] > 0
        ]
        artifacts[top_n] = {
            "iterations": iterations,
            "top_n": top_n,
            "seed": seed,
            "median_jaccard": float(np.median(jaccards[top_n])),
            "jaccards": jaccards[top_n],
            "selection_frequency": selection_frequency,
            "differential_intervals": differential_intervals,
        }
    return artifacts


# ---------------------------------------------------------------------------
# Permutation test (one shared assignment loop across all grid cardinalities)
# ---------------------------------------------------------------------------


def fast_label_permutation(
    table: CellTable,
    *,
    top_ns: list[int],
    iterations: int,
    seed: int,
    baseline_order: np.ndarray,
    baseline_differential: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Reference-equivalent permutation artifacts for every requested top_n.

    Assignment enumeration / Monte Carlo draws consume the RNG identically to
    the reference, and each evaluated assignment produces one full ranking that
    is reused for every cardinality (the top_n statistic is a prefix sum of the
    same ranked order).
    """
    if iterations <= 0 or any(top_n <= 0 for top_n in top_ns):
        raise ValueError("top_n and iterations must be positive")
    n_keys = table.n_keys
    sample_labels = {
        sample_id: DOMAIN_NAMES[int(table.domain_of_sample[position])]
        for position, sample_id in enumerate(table.sample_ids)
    }
    strata: dict[str, list[str]] = {name: [] for name in table.stratum_names}
    for position, sample_id in enumerate(table.sample_ids):
        strata[table.stratum_names[int(table.stratum_index_of_sample[position])]].append(sample_id)
    strata = {name: sorted(ids) for name, ids in sorted(strata.items())}
    design = _design_report(sample_labels, strata)
    original_labels = table.domain_of_sample
    observed_top_sum = {
        top_n: float(np.cumsum(baseline_differential[baseline_order[:top_n]])[-1])
        for top_n in top_ns
    }
    observed_by_key = np.where(np.isfinite(baseline_differential), baseline_differential, np.nan)
    rng = np.random.default_rng(seed)
    null_scores: dict[int, list[float]] = {top_n: [] for top_n in top_ns}
    expert_exceedances = np.zeros(n_keys, dtype=np.int64)
    expert_null_observations = np.zeros(n_keys, dtype=np.int64)
    state = {"saw_changed_labels": False}
    ones = _empty_weights(table)

    def evaluate(labels: np.ndarray) -> None:
        if not np.array_equal(labels, original_labels):
            state["saw_changed_labels"] = True
        _, differential, order = _replicate_ranking(table, ones, labels)
        expert_null_observations[order] += 1
        eligible = order[np.isfinite(observed_by_key[order])]
        hits = eligible[differential[eligible] >= observed_by_key[eligible]]
        expert_exceedances[hits] += 1
        for top_n in top_ns:
            null_scores[top_n].append(float(np.cumsum(differential[order[:top_n]])[-1]))

    global_fallback = design["mixed_strata"] == 0
    if global_fallback:
        all_ids = sorted(table.sample_ids)
        position_of = {sample_id: index for index, sample_id in enumerate(all_ids)}
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
                labels = np.full(len(all_ids), CONTROL, dtype=np.int64)
                for sample_id in coding_subset:
                    labels[position_of[sample_id]] = CODING
                evaluate(labels)
            unique_assignments = len(null_scores[top_ns[0]])
        else:
            method = "global-count-preserving-monte-carlo"
            mode = "monte-carlo"
            seen: set[tuple[int, ...]] = set()
            attempts = 0
            while (
                len(null_scores[top_ns[0]]) < min(iterations, attainable)
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
                labels = np.full(len(all_ids), CONTROL, dtype=np.int64)
                labels[list(chosen)] = CODING
                evaluate(labels)
            unique_assignments = len(seen)
    else:
        attainable = 1
        mixed_names = [
            name
            for name, ids in strata.items()
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
        position_of = {sample_id: index for index, sample_id in enumerate(table.sample_ids)}
        if attainable <= PERMUTATION_ENUMERATION_LIMIT:
            method = "stratified-count-preserving-exact-enumeration"
            mode = "exact-enumeration"
            fixed = np.full(table.n_samples, CONTROL, dtype=np.int64)
            for name, ids in strata.items():
                if name in mixed_names:
                    continue
                for sample_id in ids:
                    fixed[position_of[sample_id]] = (
                        CODING if sample_labels[sample_id] == "coding" else CONTROL
                    )
            per_stratum_options = []
            for name in mixed_names:
                ids = strata[name]
                coding_in_stratum = sum(1 for s in ids if sample_labels[s] == "coding")
                per_stratum_options.append(
                    [
                        [position_of[sample_id] for sample_id in combo]
                        for combo in combinations(ids, coding_in_stratum)
                    ]
                )
            for combo_sets in product(*per_stratum_options):
                labels = fixed.copy()
                for positions in combo_sets:
                    labels[list(positions)] = CODING
                evaluate(labels)
            unique_assignments = len(null_scores[top_ns[0]])
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
                len(null_scores[top_ns[0]]) < min(iterations, attainable)
                and attempts < iterations * _PERMUTATION_ATTEMPT_FACTOR
            ):
                attempts += 1
                key_parts = []
                coding_positions: list[int] = []
                for name, ids, coding_in_stratum in strata_plan:
                    chosen = tuple(
                        sorted(rng.permutation(len(ids))[:coding_in_stratum].tolist())
                    )
                    key_parts.append((name, chosen))
                    coding_positions.extend(position_of[ids[index]] for index in chosen)
                key = tuple(key_parts)
                if key in seen:
                    continue
                seen.add(key)
                labels = np.full(table.n_samples, CONTROL, dtype=np.int64)
                labels[coding_positions] = CODING
                evaluate(labels)
            unique_assignments = len(seen)
    artifacts: dict[int, dict[str, Any]] = {}
    for top_n in top_ns:
        scores = null_scores[top_n]
        if not scores:
            raise AnalysisError("permutation test cannot be evaluated: no assignments ran")
        if not state["saw_changed_labels"]:
            raise AnalysisError(
                "permutation test invalid: no evaluated assignment changed any label"
            )
        if len(set(scores)) <= 1:
            raise AnalysisError(
                "permutation null is degenerate: every attainable assignment yields the "
                "same statistic, so the test carries no information"
            )
        exceedances = sum(score >= observed_top_sum[top_n] for score in scores)
        if mode == "exact-enumeration":
            p_value = exceedances / len(scores)
        else:
            p_value = (exceedances + 1) / (len(scores) + 1)
        expert_p_values = [
            {
                "layer": int(table.layer_of_key[key]),
                "expert": int(table.expert_of_key[key]),
                "p_value": (int(expert_exceedances[key]) + 1)
                / (int(expert_null_observations[key]) + 1),
                "null_observations": int(expert_null_observations[key]),
            }
            for key in range(n_keys)
            if expert_null_observations[key] > 0
        ]
        artifacts[top_n] = {
            "iterations_requested": iterations,
            "iterations_valid": len(scores),
            "top_n": top_n,
            "seed": seed,
            "observed_top_sum": observed_top_sum[top_n],
            "p_value": p_value,
            "null_scores": scores,
            "expert_p_values": expert_p_values,
            "method": method,
            "assignment_mode": mode,
            "permutation_design": design,
            "attainable_assignments": attainable,
            "assignments_evaluated": len(scores),
            "unique_assignments_evaluated": unique_assignments,
            "permutations_changed_labels": state["saw_changed_labels"],
            "unique_null_statistics": len(set(scores)),
            "permutation_design_valid": True,
            "global_fallback_limitation": limitation,
        }
    return artifacts


# ---------------------------------------------------------------------------
# Orchestrated fast analysis (reference pipeline semantics)
# ---------------------------------------------------------------------------


def fast_analysis_outputs(
    table: CellTable,
    *,
    top_n: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
    cardinality_grid: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Compute the full analysis state the pipeline needs to freeze artifacts."""

    started = time.perf_counter()
    rows = fast_baseline_ranking(table)
    baseline_seconds = time.perf_counter() - started
    ranked_count = sum(1 for row in rows if row.get("observed", True))
    unranked_count = len(rows) - ranked_count
    if ranked_count < top_n:
        raise ValueError(
            f"only {ranked_count} experts observed in both domains; "
            f"cannot evaluate top-{top_n}"
        )
    configured_top_n = top_n
    cardinalities = sorted(set(cardinality_grid or (top_n,)))
    if not cardinalities or any(value <= 0 for value in cardinalities):
        raise ValueError("candidate cardinalities must be positive")
    if top_n not in cardinalities:
        cardinalities.append(top_n)
        cardinalities.sort()
    _, baseline_differential, baseline_order = _baseline_arrays(table)
    started = time.perf_counter()
    bootstraps = fast_bootstrap(
        table,
        top_ns=cardinalities,
        iterations=bootstrap_iterations,
        seed=seed,
        baseline_order=baseline_order,
    )
    bootstrap_seconds = time.perf_counter() - started
    started = time.perf_counter()
    permutations = fast_label_permutation(
        table,
        top_ns=cardinalities,
        iterations=permutation_iterations,
        seed=seed,
        baseline_order=baseline_order,
        baseline_differential=baseline_differential,
    )
    permutation_seconds = time.perf_counter() - started
    analyses = []
    for cardinality in cardinalities:
        bootstrap = bootstraps[cardinality]
        permutation = permutations[cardinality]
        analyses.append(
            {
                "top_n": cardinality,
                "gate_passed": bootstrap["median_jaccard"] >= 0.60
                and permutation["p_value"] <= 0.05,
                "median_bootstrap_jaccard": bootstrap["median_jaccard"],
                "permutation_p_value": permutation["p_value"],
                "bootstrap": bootstrap,
                "permutation": permutation,
            }
        )
    passing = [item for item in analyses if item["gate_passed"]]
    chosen = passing[0] if passing else next(item for item in analyses if item["top_n"] == top_n)
    return {
        "ranking": rows,
        "ranked_count": ranked_count,
        "unranked_count": unranked_count,
        "configured_top_n": configured_top_n,
        "cardinalities": cardinalities,
        "analyses": analyses,
        "chosen": chosen,
        "timings": {
            "baseline_s": baseline_seconds,
            "bootstrap_s": bootstrap_seconds,
            "permutation_s": permutation_seconds,
        },
    }
