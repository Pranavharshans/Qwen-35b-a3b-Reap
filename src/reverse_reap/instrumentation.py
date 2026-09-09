"""Exact PyTorch-path expert telemetry and zero-contribution interventions."""

from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reverse_reap.qwen35 import Qwen35Architecture
from reverse_reap.routing import RouterBatch, StreamingReapAccumulator

# Frozen top-four selected experts from the Gate C pilot
# (run 20260904T100102Z-qwen35a3b-direct-503e4ee9-644a80fc, smallest-passing
# rule). Convenience reference for the causal intervention path — the
# function itself accepts any mask; the frozen manifest on disk remains the
# governing artifact.
FROZEN_SELECTED_TOP4: frozenset[tuple[int, int]] = frozenset(
    {(35, 239), (3, 26), (7, 18), (37, 5)}
)


@dataclass
class CaptureState:
    num_layers: int
    num_experts: int
    accumulators: list[StreamingReapAccumulator] = field(init=False)

    def __post_init__(self) -> None:
        self.accumulators = [
            StreamingReapAccumulator(self.num_experts) for _ in range(self.num_layers)
        ]


@dataclass(frozen=True)
class TargetedRouteObservation:
    """Route-level vectors for one selected expert.

    ``token_indices`` refer to the flattened token dimension passed to the
    native expert module.  The runtime converts those indices back to
    ``sample_id``/position after the forward pass; keeping that mapping outside
    the hook lets the hook remain independent of padding conventions.
    """

    layer_index: int
    expert_index: int
    token_indices: Any
    route_ranks: Any
    router_weights: Any
    expert_inputs: Any
    replayed_expert_output: Any
    weighted_replayed_expert_output: Any


@contextmanager
def instrument_qwen35_targeted(
    architecture: Qwen35Architecture,
    targets: frozenset[tuple[int, int]],
    *,
    observer: Any | None = None,
) -> Iterator[None]:
    """Observe only selected expert vectors while preserving native outputs.

    The original fused expert forward executes first and its result is always
    returned.  The selected expert is replayed only on a detached side path so
    capture cannot perturb logits, routing, or the native grouped-matmul
    accumulation.  This context manager is deliberately separate from
    :func:`instrument_qwen35`, whose norm-only behavior is part of the v0
    telemetry contract.
    """
    import torch
    import torch.nn.functional as F

    if not targets:
        raise ValueError("targeted instrumentation requires at least one target")
    by_layer: list[frozenset[int]] = [
        frozenset(expert for layer, expert in targets if layer == layer_index)
        for layer_index in range(architecture.num_layers)
    ]
    originals: list[tuple[Any, Any]] = []
    for layer_index, layer in enumerate(architecture.layers):
        experts = layer.mlp.experts
        original = experts.forward
        originals.append((experts, original))
        layer_targets = by_layer[layer_index]

        def forward(
            this: Any,
            hidden_states: Any,
            top_k_index: Any,
            top_k_weights: Any,
            *,
            _layer: int = layer_index,
            _original: Any = original,
            _targets: frozenset[int] = layer_targets,
        ) -> Any:
            final = _original(hidden_states, top_k_index, top_k_weights)
            if observer is None or not _targets:
                return final
            with torch.no_grad():
                for expert in sorted(_targets):
                    route_mask = top_k_index == expert
                    route_tokens, route_ranks = torch.where(route_mask)
                    if not route_tokens.numel():
                        continue
                    expert_input = hidden_states[route_tokens].detach()
                    gate_up = F.linear(expert_input, this.gate_up_proj[expert])
                    gate, up = gate_up.chunk(2, dim=-1)
                    expert_output = F.linear(this.act_fn(gate) * up, this.down_proj[expert])
                    weights = top_k_weights[route_tokens, route_ranks].detach()
                    observer(
                        TargetedRouteObservation(
                            layer_index=_layer,
                            expert_index=expert,
                            token_indices=route_tokens.detach(),
                            route_ranks=route_ranks.detach(),
                            router_weights=weights,
                            expert_inputs=expert_input,
                            replayed_expert_output=expert_output.detach(),
                            weighted_replayed_expert_output=(
                                expert_output * weights[:, None]
                            ).detach(),
                        )
                    )
            return final

        experts.forward = types.MethodType(forward, experts)

    try:
        yield None
    finally:
        for experts, original in originals:
            experts.forward = original


def _selected_output_norms(
    route_outputs: Any, token_indices: Any, expert_indices: Any, tokens: int, top_k: int
) -> Any:
    """Scatter per-route norms back to [tokens, top_k] order."""
    import torch

    result = torch.zeros((tokens, top_k), dtype=torch.float64, device=route_outputs.device)
    result[token_indices, expert_indices] = torch.linalg.vector_norm(
        route_outputs.float(), dim=-1
    ).double()
    return result


@contextmanager
def instrument_qwen35(
    architecture: Qwen35Architecture,
    *,
    masked: frozenset[tuple[int, int]] = frozenset(),
    observer: Any | None = None,
) -> Iterator[CaptureState]:
    """Observe the native expert path temporarily, restoring it even after failure.

    The native ``experts.forward`` (Transformers ``grouped_mm`` kernel under the
    pinned ``_experts_implementation``) always computes the returned output, so
    capture-on logits stay bitwise identical to capture-off. Routing rows and
    pre-weighting expert-output norms are recomputed on a detached side path used
    only for telemetry — never for the returned tensor. Masking replays the
    weighted per-expert contributions on the side path with the selected
    (layer, expert) contributions zeroed and without router renormalization.
    """
    import torch
    import torch.nn.functional as F

    capture = CaptureState(architecture.num_layers, architecture.num_experts)
    originals: list[tuple[Any, Any]] = []

    for layer_index, layer in enumerate(architecture.layers):
        experts = layer.mlp.experts
        original = experts.forward
        originals.append((experts, original))

        def forward(this: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any,
                    *, _layer: int = layer_index, _original: Any = original) -> Any:
            final = _original(hidden_states, top_k_index, top_k_weights)
            tokens = hidden_states.shape[0]
            top_k = top_k_index.shape[1]
            with torch.no_grad():
                expert_mask = F.one_hot(
                    top_k_index, num_classes=architecture.num_experts
                ).permute(2, 1, 0)
                hit = torch.nonzero(
                    expert_mask.sum(dim=(1, 2)), as_tuple=False
                ).flatten()
                norms = torch.zeros(
                    (tokens, top_k), dtype=torch.float64, device=hidden_states.device
                )
                masked_total: Any | None = None
                if masked:
                    masked_total = torch.zeros_like(final)
                for expert_tensor in hit:
                    expert = int(expert_tensor.item())
                    rank_indices, token_indices = torch.where(
                        expert_mask[expert_tensor]
                    )
                    current = hidden_states[token_indices]
                    gate_up = F.linear(current, this.gate_up_proj[expert])
                    gate, up = gate_up.chunk(2, dim=-1)
                    current = this.act_fn(gate) * up
                    current = F.linear(current, this.down_proj[expert])
                    norms[token_indices, rank_indices] = torch.linalg.vector_norm(
                        current.float(), dim=-1
                    ).double()
                    if masked_total is not None and (_layer, expert) not in masked:
                        weighted = current * top_k_weights[
                            token_indices, rank_indices, None
                        ]
                        masked_total.index_add_(
                            0, token_indices, weighted.to(final.dtype)
                        )

            batch = RouterBatch(
                top_k_index.detach().cpu().numpy().astype(np.int64, copy=False),
                top_k_weights.detach().float().cpu().numpy().astype(np.float64, copy=False),
            )
            norm_values = norms.detach().cpu().numpy()
            capture.accumulators[_layer].update(batch, norm_values)
            if observer is not None:
                observer(_layer, batch, norm_values)
            if masked_total is not None:
                return masked_total
            return final

        experts.forward = types.MethodType(forward, experts)

    try:
        yield capture
    finally:
        for experts, original in originals:
            experts.forward = original


@contextmanager
def intervene_qwen35(
    architecture: Qwen35Architecture,
    *,
    masked: frozenset[tuple[int, int]] = frozenset(),
) -> Iterator[None]:
    """Zero selected expert contributions via the native fused forward only.

    Optimized intervention-only path for causal generation. Semantics match
    the slow telemetry/replay path (zero-weighted contribution without router
    renormalization) but without the telemetry side path:

    - Empty mask: the original fused expert forward is called directly with
      zero additional tensor work (no clone, no scan).
    - Selected mask: the router-weight tensor is cloned, only routes whose
      (layer, expert) identity is in ``masked`` are set to zero, and the
      original fused forward is called exactly once with the modified
      weights.
    - Expert indices are never changed, tokens are never rerouted, surviving
      weights are never renormalized, routed experts are never recomputed,
      and no expert norms, CPU copies, observers, or telemetry accumulators
      are touched.

    The existing :func:`instrument_qwen35` capture implementation is
    preserved unchanged for telemetry and Gate A.
    """
    originals: list[tuple[Any, Any]] = []
    targets_per_layer: list[frozenset[int]] = [
        frozenset(expert for (layer, expert) in masked if layer == layer_index)
        for layer_index in range(architecture.num_layers)
    ]

    for layer_index, layer in enumerate(architecture.layers):
        experts = layer.mlp.experts
        original = experts.forward
        originals.append((experts, original))
        targets = targets_per_layer[layer_index]

        def forward(
            this: Any,
            hidden_states: Any,
            top_k_index: Any,
            top_k_weights: Any,
            *,
            _original: Any = original,
            _targets: frozenset[int] = targets,
        ) -> Any:
            if not _targets:
                return _original(hidden_states, top_k_index, top_k_weights)
            modified = top_k_weights.clone()
            for expert in _targets:
                modified[top_k_index == expert] = 0
            return _original(hidden_states, top_k_index, modified)

        experts.forward = types.MethodType(forward, experts)

    try:
        yield None
    finally:
        for experts, original in originals:
            experts.forward = original
