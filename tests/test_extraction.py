import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from reverse_reap.extraction import (
    ExtractionError,
    architecture_from_weight_index,
    extract_experts,
    verify_extraction,
)
from reverse_reap.qwen35 import Qwen35Architecture


def fixture_model(tmp_path):
    key_gate = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    key_down = "model.language_model.layers.0.mlp.experts.down_proj"
    arrays = {
        key_gate: np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4),
        key_down: np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2),
    }
    save_file(arrays, tmp_path / "model-00001-of-00001.safetensors")
    index = {"weight_map": {key: "model-00001-of-00001.safetensors" for key in arrays}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    architecture = Qwen35Architecture(
        layers=(SimpleNamespace(),),
        num_experts=3,
        experts_per_token=2,
        hidden_size=4,
        expert_intermediate_size=2,
        state_prefix="model.language_model.layers",
    )
    return architecture, arrays


def test_extracts_selected_slices_and_verifies_source_bytes(tmp_path):
    architecture, arrays = fixture_model(tmp_path)
    destination = tmp_path / "extracted"
    manifest = extract_experts(
        tmp_path,
        architecture,
        [(0, 2)],
        destination,
        model_id="fixture/qwen",
        model_revision="f" * 40,
    )
    output = load_file(destination / "experts.safetensors")
    assert np.array_equal(
        output["layers.0.experts.2.gate_up_proj"],
        arrays["model.language_model.layers.0.mlp.experts.gate_up_proj"][2],
    )
    assert manifest["total_parameter_bytes"] == 96
    verification = verify_extraction(destination, tmp_path)
    assert verification == {
        "valid": True,
        "tensor_count": 2,
        "source_weight_index_hash_valid": True,
    }
    assert manifest["source_model_id"] == "fixture/qwen"
    assert manifest["tensors"][0]["source_shard"] == "model-00001-of-00001.safetensors"
    for filename in (
        "source-to-extracted-map.json",
        "checksums.sha256",
        "verification-report.json",
        "README.md",
    ):
        assert (destination / filename).exists()


def test_refuses_to_overwrite_extraction(tmp_path):
    architecture, _ = fixture_model(tmp_path)
    destination = tmp_path / "already-there"
    destination.mkdir()
    with pytest.raises(ExtractionError, match="refusing to overwrite"):
        extract_experts(
            tmp_path,
            architecture,
            [(0, 1)],
            destination,
            model_id="fixture/qwen",
            model_revision="f" * 40,
        )


def test_infers_approved_40_layer_tensor_prefix(tmp_path):
    prefix = "model.language_model.layers"
    weight_map = {
        f"{prefix}.{layer}.mlp.experts.gate_up_proj": "shard.safetensors"
        for layer in range(40)
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    architecture = architecture_from_weight_index(tmp_path)
    assert architecture.num_layers == 40
    assert architecture.state_prefix == prefix
