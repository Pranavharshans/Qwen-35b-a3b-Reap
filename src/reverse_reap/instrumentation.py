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


@dataclass
class CaptureState:
    num_layers: int
    num_experts: int
    accumulators: list[StreamingReapAccumulator] = field(init=False)

    def __post_init__(self) -> None:
        self.accumulators = [
            StreamingReapAccumulator(self.num_experts) for _ in range(self.num_layers)
        ]


def _selected_output_norms(
    expert_outputs: Any, token_indices: Any, expert_indices: Any, tokens: int, top_k: int
) -> Any:
    """Scatter per-route norms back to [tokens, top_k] order."""
    import torch

    result = torch.zeros((tokens, top_k), dtype=torch.float64, device=expert_outputs.device)
    result[token_indices, expert_indices] = torch.linalg.vector_norm(
        expert_outputs.float(), dim=-1
    ).double()
    return result


@contextmanager
def instrument_qwen35(
    architecture: Qwen35Architecture,
    *,
    masked: frozenset[tuple[int, int]] = frozenset(),
    observer: Any | None = None,
) -> Iterator[CaptureState]:
    """Patch the reference expert path temporarily, restoring it even after failure.

    The implementation mirrors the Qwen3.5 fused-expert loop, records the norm before
    router weighting, and applies the primary v0 intervention after expert computation.
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
                    *, _layer: int = layer_index) -> Any:
            tokens, hidden = hidden_states.shape
            top_k = top_k_index.shape[1]
            final = torch.zeros_like(hidden_states)
            norms = torch.zeros((tokens, top_k), dtype=torch.float64, device=hidden_states.device)
            expert_mask = F.one_hot(
                top_k_index, num_classes=architecture.num_experts
            ).permute(2, 1, 0)
            hit = torch.nonzero(expert_mask.sum(dim=(1, 2)), as_tuple=False).flatten()
            for expert_tensor in hit:
                expert = int(expert_tensor.item())
                rank_indices, token_indices = torch.where(expert_mask[expert_tensor])
                current = hidden_states[token_indices]
                gate_up = F.linear(current, this.gate_up_proj[expert])
                gate, up = gate_up.chunk(2, dim=-1)
                current = this.act_fn(gate) * up
                current = F.linear(current, this.down_proj[expert])
                norms[token_indices, rank_indices] = torch.linalg.vector_norm(
                    current.float(), dim=-1
                ).double()
                weighted = current * top_k_weights[token_indices, rank_indices, None]
                if (_layer, expert) in masked:
                    weighted = torch.zeros_like(weighted)
                final.index_add_(0, token_indices, weighted.to(hidden_states.dtype))

            batch = RouterBatch(
                top_k_index.detach().cpu().numpy().astype(np.int64, copy=False),
                top_k_weights.detach().float().cpu().numpy().astype(np.float64, copy=False),
            )
            norm_values = norms.detach().cpu().numpy()
            capture.accumulators[_layer].update(batch, norm_values)
            if observer is not None:
                observer(_layer, batch, norm_values)
            return final

        experts.forward = types.MethodType(forward, experts)

    try:
        yield capture
    finally:
        for experts, original in originals:
            experts.forward = original
