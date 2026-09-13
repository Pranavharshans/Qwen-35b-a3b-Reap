from types import SimpleNamespace

import pytest

from reverse_reap.qwen35 import ArchitectureError, inspect_qwen35_moe


class Tensor:
    def __init__(self, shape):
        self.shape = shape


def model_with_shapes(gate=(256, 1024, 2048), down=(256, 2048, 512), layers=3, top_k=8):
    blocks = []
    for _ in range(layers):
        experts = SimpleNamespace(gate_up_proj=Tensor(gate), down_proj=Tensor(down))
        mlp = SimpleNamespace(
            gate=SimpleNamespace(top_k=top_k),
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


def test_runtime_inspector_leaves_shared_path_validation_to_donor_contract():
    model = model_with_shapes()
    del model.model.language_model.layers[0].mlp.shared_expert
    assert inspect_qwen35_moe(model).num_moe_layers == 3


def test_inspects_qwen38_shared_fused_layout():
    architecture = inspect_qwen35_moe(
        model_with_shapes(gate=(512, 1280, 2560), down=(512, 2560, 640), layers=48, top_k=10)
    )
    assert architecture.num_layers == 48
    assert architecture.num_experts == 512
    assert architecture.experts_per_token == 10
    assert architecture.hidden_size == 2560
    assert architecture.expert_intermediate_size == 640


def test_inspects_glm_sparse_layers_with_absolute_layer_indices():
    model = model_with_shapes(
        gate=(288, 4096, 4096), down=(288, 4096, 2048), layers=45, top_k=8
    )
    for layer in model.model.language_model.layers[:3]:
        layer.mlp = SimpleNamespace(gate_proj=object(), up_proj=object(), down_proj=object())
    architecture = inspect_qwen35_moe(model)
    assert architecture.num_layers == 45
    assert architecture.num_moe_layers == 42
    assert architecture.layer_indices == tuple(range(3, 45))
    assert architecture.tensor_spec(3, 17).gate_up_key.endswith(
        "layers.3.mlp.experts.gate_up_proj"
    )
    with pytest.raises(IndexError, match="not a routed MoE layer"):
        architecture.tensor_spec(2, 17)
