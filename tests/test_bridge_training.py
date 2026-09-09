import hashlib
import json
from pathlib import Path

import pytest

from reverse_reap.bridge_training import (
    BridgeExpertMapping,
    BridgeHostStateRecord,
    BridgeModel,
    BridgeTrainingError,
    BridgeTrainingRow,
    EqualParameterAdapter,
    _assert_token_alignment,
    _assign_sample_groups,
    bridge_control_keys,
    estimate_bridge_memory,
    load_bridge_checkpoint,
    save_bridge_checkpoint,
)


def test_b1_preflight_is_explicit_and_requires_empirical_probe():
    report = estimate_bridge_memory(vram_bytes=16 * 1024**3)
    assert report["batch_size"] == 1
    assert report["safety_ceiling_bytes"] == int(16 * 1024**3 * 0.92)
    assert report["requires_empirical_probe"] is True
    assert estimate_bridge_memory(vram_bytes=1 * 1024**3)["passed"] is False
    with pytest.raises(BridgeTrainingError, match="B1"):
        estimate_bridge_memory(vram_bytes=16 * 1024**3, batch_size=2)


def test_random_expert_control_is_fail_closed():
    with pytest.raises(BridgeTrainingError, match="unavailable"):
        bridge_control_keys(["3:26"], condition="random-expert", seed=9)


def test_shuffled_pair_control_is_a_deterministic_derangement():
    keys = ["3:26", "7:18", "35:239", "37:5"]
    first = bridge_control_keys(keys, condition="shuffled-pair", seed=9)
    second = bridge_control_keys(keys, condition="shuffled-pair", seed=9)
    assert first == second
    assert first is not None
    assert set(first) == set(keys)
    assert all(original != shuffled for original, shuffled in zip(keys, first, strict=True))
    repeated = ["3:26", "3:26", "7:18", "7:18"]
    repeated_result = bridge_control_keys(repeated, condition="shuffled-pair", seed=9)
    assert repeated_result is not None
    assert all(
        original != shuffled
        for original, shuffled in zip(repeated, repeated_result, strict=True)
    )
    with pytest.raises(BridgeTrainingError, match="at least two"):
        bridge_control_keys(["3:26"], condition="shuffled-pair", seed=9)


def test_iterative_group_stratification_is_deterministic_and_complete():
    labels = {
        f"sample-{index}": {"coding:3:26" if index % 2 == 0 else "control:3:26"}
        for index in range(12)
    }
    groups = {key: () for key in labels}
    first = _assign_sample_groups(groups, labels, seed=9, min_samples_per_cell=1)
    second = _assign_sample_groups(groups, labels, seed=9, min_samples_per_cell=1)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    for split in ("train", "validation", "test"):
        for label in ("coding:3:26", "control:3:26"):
            assert sum(first[key] == split and label in labels[key] for key in labels) >= 1


def test_group_stratification_fails_when_cell_has_too_few_groups():
    groups = {f"sample-{index}": () for index in range(2)}
    labels = {key: {"coding:3:26"} for key in groups}
    with pytest.raises(BridgeTrainingError, match="represent cell"):
        _assign_sample_groups(groups, labels, seed=1, min_samples_per_cell=1)


def test_bridge_model_freezes_experts_and_starts_as_identity():
    torch = pytest.importorskip("torch")
    from reverse_reap.bridge_training import FrozenSwiGLUExpert

    mapping = BridgeExpertMapping(donor_layer=3, donor_expert=26, host_layer=2)
    gate_up = torch.randn(4, 2048, dtype=torch.bfloat16)
    down = torch.randn(2048, 2, dtype=torch.bfloat16)
    expert = FrozenSwiGLUExpert(gate_up, down)
    model = BridgeModel(
        [mapping],
        {mapping.key: expert},
        bottleneck_size=4,
        gate_cap=0.25,
        gate_init_bias=-6.0,
    )
    assert all(not parameter.requires_grad for parameter in model.experts.parameters())
    assert torch.count_nonzero(model.output_adapters["l3_e26"][-1].weight) == 0
    hidden = torch.randn(2048, dtype=torch.float32)
    assert torch.equal(model(hidden, mapping.key, enabled=False), hidden)
    # Zero final output projection plus finite gate bias makes the initial
    # sidecar an identity, while the trainable output projection still gets a
    # gradient on a nonzero objective.
    initial = model(hidden, mapping.key)
    assert torch.equal(initial, hidden)
    # The zero final projection preserves identity at initialization.  Once a
    # tiny output signal exists, the input-dependent gate also receives a
    # useful gradient rather than being a dead scalar.
    with torch.no_grad():
        model.output_adapters["l3_e26"][-1].weight[0, 0] = 1e-3
    output = model(hidden, mapping.key)
    loss = output.square().mean()
    loss.backward()
    assert model.output_adapters["l3_e26"][-1].weight.grad is not None
    assert model.input_adapters["l3_e26"][0].weight.grad is not None
    assert model.gate_heads["l3_e26"].weight.grad is not None
    assert torch.isfinite(model.gate_heads["l3_e26"].weight.grad).all()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


def _two_expert_model(torch):
    from reverse_reap.bridge_training import FrozenSwiGLUExpert

    mappings = [
        BridgeExpertMapping(donor_layer=3, donor_expert=26, host_layer=2),
        BridgeExpertMapping(donor_layer=7, donor_expert=18, host_layer=4),
    ]
    experts = {
        mappings[0].key: FrozenSwiGLUExpert(
            torch.randn(4, 2048, dtype=torch.bfloat16),
            torch.randn(2048, 2, dtype=torch.bfloat16),
        ),
        mappings[1].key: FrozenSwiGLUExpert(
            torch.randn(4, 2048, dtype=torch.bfloat16),
            torch.randn(2048, 2, dtype=torch.bfloat16),
        ),
    }
    return mappings, BridgeModel(mappings, experts, bottleneck_size=4)


def test_equal_parameter_control_matches_bridge_trainable_budget():
    torch = pytest.importorskip("torch")
    mappings, model = _two_expert_model(torch)
    control = EqualParameterAdapter(bottleneck_size=4, mapped_expert_count=len(mappings))
    bridge_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    control_count = sum(parameter.numel() for parameter in control.parameters())
    assert control_count == bridge_count


def test_expert_specific_sidecars_isolate_parameters_and_preserve_batch_shape():
    torch = pytest.importorskip("torch")
    mappings, model = _two_expert_model(torch)
    hidden = torch.randn(2, 2048)
    routed = model.forward_routes(hidden, [mappings[0].key, mappings[1].key])
    assert routed.shape == hidden.shape
    assert routed.dtype == hidden.dtype
    # The same host vector receives distinct gates for the two expert heads.
    with torch.no_grad():
        model.gate_heads["l3_e26"].weight.zero_()
        model.gate_heads["l3_e26"].weight[0, 0] = 0.1
        model.gate_heads["l7_e18"].weight.zero_()
        model.gate_heads["l7_e18"].weight[0, 0] = -0.1
    _, _, _, _, gate_a = model.components(hidden[:1], mappings[0].key)
    _, _, _, _, gate_b = model.components(hidden[:1], mappings[1].key)
    assert not torch.equal(gate_a, gate_b)
    # A route through one expert cannot touch another expert's trainable path.
    with torch.no_grad():
        model.output_adapters["l3_e26"][-1].weight[0, 0] = 1e-3
    model(hidden[:1], mappings[0].key).square().mean().backward()
    assert model.input_adapters["l3_e26"][0].weight.grad is not None
    assert model.input_adapters["l7_e18"][0].weight.grad is None


def test_frozen_swiglu_matches_reference_math():
    torch = pytest.importorskip("torch")

    from torch.nn import functional as F

    from reverse_reap.bridge_training import FrozenSwiGLUExpert

    gate_up = torch.arange(4 * 2048, dtype=torch.float32).reshape(4, 2048).to(torch.bfloat16)
    down = torch.arange(2048 * 2, dtype=torch.float32).reshape(2048, 2).to(torch.bfloat16)
    hidden = torch.randn(3, 2048, dtype=torch.float32)
    expert = FrozenSwiGLUExpert(gate_up, down)
    native = hidden.to(torch.bfloat16)
    gate, up = F.linear(native, gate_up).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, down).to(torch.float32)
    torch.testing.assert_close(expert(hidden), expected, rtol=0, atol=0)


def test_training_manifest_rejects_sample_content_leakage(tmp_path: Path):
    mapping = BridgeExpertMapping(donor_layer=3, donor_expert=26, host_layer=2)
    shared_content = "a" * 64
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        rows.append(
            BridgeTrainingRow(
                event_id=f"{index + 1:064x}",
                shard_path="shard/records.jsonl",
                row_index=index,
                host_state_row=index,
                sample_id=f"sample-{index}",
                sample_content_sha256=shared_content if split != "test" else "b" * 64,
                domain="coding",
                split=split,
                token_position=0,
                token_id=10,
                donor_layer=3,
                donor_expert=26,
                host_layer=2,
                route_rank=0,
                router_weight=0.5,
            )
        )
    body = {
        "schema_version": 1,
        "kind": "bridge-training-manifest",
        "handoff_manifest_sha256": "c" * 64,
        "capture_manifest_sha256": "d" * 64,
        "source_manifest_sha256": "e" * 64,
        "candidate_manifest_sha256": "f" * 64,
        "extraction_manifest_sha256": "0" * 64,
        "host_states_manifest_sha256": "1" * 64,
        "mappings": [mapping.model_dump(mode="json")],
        "split_counts": {"train": 1, "validation": 1, "test": 1},
        "cell_counts": {
            "coding:3:26": {"train": 1, "validation": 1, "test": 1}
        },
        "rows": [row.model_dump(mode="json") for row in rows],
        "unused_rows": [],
        "seed": 1,
        "min_samples_per_cell": 1,
        "max_rows_per_sample": 1,
        "min_events_per_cell": 1,
        "capture_outcome": "coverage-complete",
    }
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    from reverse_reap.bridge_training import BridgeTrainingError, load_training_manifest

    with pytest.raises(BridgeTrainingError, match="sample content leaks"):
        load_training_manifest(path)


def test_token_alignment_mismatch_is_fail_closed():
    from reverse_reap.bridge_capture import BridgeTargetRecord

    record = BridgeTargetRecord(
        schema_version=1,
        run_id="bridge",
        row_index=0,
        event_id="a" * 64,
        sample_ordinal=0,
        sample_id="sample-0",
        sample_content_sha256="b" * 64,
        domain="coding",
        split="validation",
        token_position=3,
        token_id=17,
        donor_layer=3,
        expert_id=26,
        route_rank=0,
        router_weight=0.5,
        source_manifest_hash="c" * 64,
        model_revision="d" * 40,
        tokenizer_fingerprint="tokenizer",
        config_sha256="e" * 64,
        candidate_manifest_sha256="f" * 64,
        condition_id="C0",
        chunk_id="chunk-0",
    )
    host = BridgeHostStateRecord(
        row_index=0,
        sample_id="sample-0",
        sample_content_sha256="b" * 64,
        token_position=3,
        token_id=18,
        host_layer=2,
        tensor_key="hidden_states",
    )
    with pytest.raises(BridgeTrainingError, match="token alignment"):
        _assert_token_alignment(record, host, context=record.event_id)


def test_checkpoint_hash_drift_and_corruption_fail_closed(tmp_path: Path):
    torch = pytest.importorskip("torch")
    mappings, model = _two_expert_model(torch)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    checkpoint_path = tmp_path / "checkpoint-0000"
    save_bridge_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config_sha256="a" * 64,
        training_manifest_sha256="b" * 64,
        epoch=0,
        step=1,
        metrics={"validation": {"total": 1.0}},
    )
    load_bridge_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config_sha256="a" * 64,
        training_manifest_sha256="b" * 64,
    )
    with pytest.raises(BridgeTrainingError, match="config differs"):
        load_bridge_checkpoint(
            checkpoint_path,
            model,
            config_sha256="c" * 64,
            training_manifest_sha256="b" * 64,
        )
    bridge_file = checkpoint_path / "bridge.safetensors"
    bridge_file.write_bytes(bridge_file.read_bytes() + b"tamper")
    with pytest.raises(BridgeTrainingError, match="bridge hash"):
        load_bridge_checkpoint(
            checkpoint_path,
            model,
            config_sha256="a" * 64,
            training_manifest_sha256="b" * 64,
        )
