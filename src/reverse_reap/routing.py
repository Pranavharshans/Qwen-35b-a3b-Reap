"""Router decoding and bounded, streaming Reverse-REAP aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class RoutingError(ValueError):
    """Raised for malformed or unsafe routing telemetry."""


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class RouterBatch:
    indices: np.ndarray
    weights: np.ndarray

    @property
    def tokens(self) -> int:
        return self.indices.shape[0]

    @property
    def top_k(self) -> int:
        return self.indices.shape[1]


def validate_router_batch(
    indices: Any, weights: Any, *, num_experts: int, expected_top_k: int | None = None
) -> RouterBatch:
    idx = as_numpy(indices)
    wgt = as_numpy(weights).astype(np.float64, copy=False)
    if idx.ndim != 2 or wgt.shape != idx.shape:
        raise RoutingError(
            f"indices/weights must share [tokens, top_k], got {idx.shape}/{wgt.shape}"
        )
    if expected_top_k is not None and idx.shape[1] != expected_top_k:
        raise RoutingError(f"expected top_k={expected_top_k}, got {idx.shape[1]}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise RoutingError("expert indices must be integers")
    idx = idx.astype(np.int64, copy=False)
    if np.any(idx < 0) or np.any(idx >= num_experts):
        raise RoutingError("expert index outside configured range")
    if not np.all(np.isfinite(wgt)) or np.any(wgt < 0):
        raise RoutingError("router weights must be finite and non-negative")
    if idx.shape[0] and not np.allclose(wgt.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise RoutingError("top-k router weights must be normalized per token")
    if any(len(set(row.tolist())) != len(row) for row in idx):
        raise RoutingError("a token cannot route to the same expert twice")
    return RouterBatch(idx, wgt)


class StreamingReapAccumulator:
    """O(num_experts) sufficient statistics; never retains token activations."""

    def __init__(self, num_experts: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        self.count = np.zeros(num_experts, dtype=np.int64)
        self.router_mass = np.zeros(num_experts, dtype=np.float64)
        self.output_norm_sum = np.zeros(num_experts, dtype=np.float64)
        self.weighted_norm_sum = np.zeros(num_experts, dtype=np.float64)

    def update(self, batch: RouterBatch, output_norms: Any) -> None:
        norms = as_numpy(output_norms).astype(np.float64, copy=False)
        if norms.shape != batch.indices.shape:
            raise RoutingError(
                f"output norms must have shape {batch.indices.shape}, got {norms.shape}"
            )
        if not np.all(np.isfinite(norms)) or np.any(norms < 0):
            raise RoutingError("expert output norms must be finite and non-negative")
        flat_idx = batch.indices.ravel()
        np.add.at(self.count, flat_idx, 1)
        np.add.at(self.router_mass, flat_idx, batch.weights.ravel())
        np.add.at(self.output_norm_sum, flat_idx, norms.ravel())
        np.add.at(self.weighted_norm_sum, flat_idx, (batch.weights * norms).ravel())

    def merge(self, other: StreamingReapAccumulator) -> None:
        if other.num_experts != self.num_experts:
            raise ValueError("cannot merge accumulators with different expert counts")
        for name in ("count", "router_mass", "output_norm_sum", "weighted_norm_sum"):
            getattr(self, name)[:] += getattr(other, name)

    def records(self) -> list[dict[str, int | float]]:
        result = []
        for expert in range(self.num_experts):
            count = int(self.count[expert])
            result.append(
                {
                    "expert": expert,
                    "routed_count": count,
                    "router_mass": float(self.router_mass[expert]),
                    "mean_router_weight": float(self.router_mass[expert] / count) if count else 0.0,
                    "mean_output_norm": (
                        float(self.output_norm_sum[expert] / count) if count else 0.0
                    ),
                    "reap_saliency": (
                        float(self.weighted_norm_sum[expert] / count) if count else 0.0
                    ),
                }
            )
        return result
