"""Validated structural adapter for supported fused Qwen sparse-MoE text towers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ArchitectureError(RuntimeError):
    """Raised when a loaded checkpoint does not match the supported architecture."""


@dataclass(frozen=True)
class ExpertTensorSpec:
    layer: int
    expert: int
    gate_up_key: str
    down_key: str


@dataclass(frozen=True)
class Qwen35Architecture:
    layers: tuple[Any, ...]
    num_experts: int
    experts_per_token: int
    hidden_size: int
    expert_intermediate_size: int
    state_prefix: str
    layer_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        indices = self.layer_indices
        if indices is None:
            object.__setattr__(self, "layer_indices", tuple(range(len(self.layers))))
        elif len(indices) != len(self.layers) or len(set(indices)) != len(indices):
            raise ArchitectureError("layer_indices must uniquely identify every MoE layer")

    @property
    def num_layers(self) -> int:
        assert self.layer_indices is not None
        return max(self.layer_indices, default=-1) + 1

    @property
    def num_moe_layers(self) -> int:
        return len(self.layers)

    def tensor_spec(self, layer: int, expert: int) -> ExpertTensorSpec:
        assert self.layer_indices is not None
        if layer not in self.layer_indices:
            raise IndexError(f"layer {layer} is not a routed MoE layer")
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"expert {expert} outside [0, {self.num_experts})")
        stem = f"{self.state_prefix}.{layer}.mlp.experts"
        return ExpertTensorSpec(layer, expert, f"{stem}.gate_up_proj", f"{stem}.down_proj")


_LAYER_PATHS = (
    ("model", "language_model", "layers"),
    ("model", "language_model", "model", "layers"),
    ("language_model", "layers"),
    ("language_model", "model", "layers"),
    ("model", "layers"),
    ("layers",),
)


def _resolve(root: Any, path: tuple[str, ...]) -> Any:
    current = root
    for part in path:
        current = getattr(current, part)
    return current


def inspect_qwen35_moe(model: Any) -> Qwen35Architecture:
    """Resolve and strictly validate the shared fused Qwen sparse-MoE layout.

    The historical name remains public for compatibility. Qwen3.5 and
    Qwen3.8-Flash-Next expose the same routed-expert tensor/module interface;
    the exact family dimensions are enforced separately by the donor contract.
    """
    found: tuple[tuple[str, ...], Any] | None = None
    for path in _LAYER_PATHS:
        try:
            candidate = _resolve(model, path)
        except AttributeError:
            continue
        if len(candidate) and hasattr(candidate[0], "mlp"):
            found = (path, candidate)
            break
    if found is None:
        raise ArchitectureError("could not locate the supported Qwen language-model decoder layers")

    path, layer_list = found
    all_layers = tuple(layer_list)
    sparse = tuple(
        (index, layer) for index, layer in enumerate(all_layers)
        if hasattr(getattr(layer, "mlp", None), "experts")
    )
    if not sparse:
        raise ArchitectureError("decoder contains no instrumentable sparse-MoE blocks")
    layer_indices = tuple(index for index, _ in sparse)
    layers = tuple(layer for _, layer in sparse)
    first = layers[0].mlp
    required = ("gate", "experts")
    missing = [name for name in required if not hasattr(first, name)]
    if missing:
        raise ArchitectureError(f"first MoE block is missing: {', '.join(missing)}")

    experts = first.experts
    if not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
        raise ArchitectureError("expected fused gate_up_proj and down_proj expert tensors")
    gate_shape = tuple(experts.gate_up_proj.shape)
    down_shape = tuple(experts.down_proj.shape)
    if len(gate_shape) != 3 or len(down_shape) != 3:
        raise ArchitectureError(f"expert tensors must be rank 3, got {gate_shape} and {down_shape}")
    num_experts, twice_intermediate, hidden_size = gate_shape
    if twice_intermediate % 2:
        raise ArchitectureError("gate_up projection dimension must be even")
    intermediate = twice_intermediate // 2
    if down_shape != (num_experts, hidden_size, intermediate):
        raise ArchitectureError(
            "inconsistent fused expert shapes: " f"gate_up={gate_shape}, down={down_shape}"
        )

    top_k = getattr(first, "top_k", None) or getattr(first.gate, "top_k", None)
    if not isinstance(top_k, int) or not 0 < top_k <= num_experts:
        raise ArchitectureError(f"invalid or missing router top_k: {top_k!r}")

    for index, layer in sparse:
        block = getattr(layer, "mlp", None)
        if block is None or not hasattr(block, "experts") or not hasattr(block, "gate"):
            raise ArchitectureError(f"layer {index} is not an instrumentable sparse-MoE block")
        if tuple(block.experts.gate_up_proj.shape) != gate_shape:
            raise ArchitectureError(f"layer {index} gate_up shape differs from layer 0")
        if tuple(block.experts.down_proj.shape) != down_shape:
            raise ArchitectureError(f"layer {index} down shape differs from layer 0")

    prefix = ".".join(path)
    return Qwen35Architecture(
        layers=layers,
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_size=hidden_size,
        expert_intermediate_size=intermediate,
        state_prefix=prefix,
        layer_indices=layer_indices,
    )


# Model-neutral names for new code without breaking existing callers or artifacts.
QwenMoEArchitecture = Qwen35Architecture
inspect_qwen_moe = inspect_qwen35_moe
