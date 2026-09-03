from types import SimpleNamespace

import pytest

from reverse_reap.qwen35 import ArchitectureError, inspect_qwen35_moe


class Tensor:
    def __init__(self, shape):
        self.shape = shape


def model_with_shapes(gate=(256, 1024, 2048), down=(256, 2048, 512), layers=3):
    blocks = []
    for _ in range(layers):
        experts = SimpleNamespace(gate_up_proj=Tensor(gate), down_proj=Tensor(down))
        mlp = SimpleNamespace(
            gate=SimpleNamespace(top_k=8),
            experts=experts,
            shared_expert=object(),
            shared_expert_gate=object(),
        )
        blocks.append(SimpleNamespace(mlp=mlp))
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=blocks)))


def test_inspects_fused_qwen_layout_and_tensor_keys():
    architecture = inspect_qwen35_moe(model_with_shapes())
    assert architecture.num_layers == 3
    assert architecture.num_experts == 256
    assert architecture.experts_per_token == 8
    assert architecture.hidden_size == 2048
    assert architecture.expert_intermediate_size == 512
    spec = architecture.tensor_spec(2, 17)
    assert spec.gate_up_key == "model.language_model.layers.2.mlp.experts.gate_up_proj"
    assert spec.down_key.endswith("layers.2.mlp.experts.down_proj")


def test_rejects_inconsistent_expert_layout():
    with pytest.raises(ArchitectureError, match="inconsistent fused expert shapes"):
        inspect_qwen35_moe(model_with_shapes(down=(256, 2048, 511)))


def test_rejects_missing_sparse_moe_parts():
    model = model_with_shapes()
    del model.model.language_model.layers[0].mlp.shared_expert
    with pytest.raises(ArchitectureError, match="shared_expert"):
        inspect_qwen35_moe(model)
