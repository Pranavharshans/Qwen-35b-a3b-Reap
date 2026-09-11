from __future__ import annotations

import hashlib

import numpy as np
import pytest

from reverse_reap.bridge_rescue import (
    BridgePolicyController,
    BridgePolicyState,
    BridgeRescueError,
    BridgeRuntimePolicy,
    GenerationPolicyTracker,
    LinearGateCheckpoint,
    RescueExperimentConfig,
    fit_linear_gate,
    fit_linear_gate_manifest,
    load_linear_gate,
    load_rescue_experiment_config,
    save_linear_gate,
)


class Mapping:
    key = "7:18"
    host_layer = 4


def policy(**overrides):
    values = {
        "name": "candidate",
        "mode": "fixed",
        "residual_multiplier": 0.1,
    }
    values.update(overrides)
    return BridgeRuntimePolicy.model_validate(values)


def test_fixed_policy_and_base_control():
    state = BridgePolicyState()
    assert BridgePolicyController(policy(), state)(Mapping(), None, None) == 0.1
    assert BridgePolicyController(policy(name="base"), state)(Mapping(), None, None) == 0.0


def test_late_and_ramp_boundaries():
    state = BridgePolicyState(generated_tokens=1023)
    late = BridgePolicyController(policy(mode="late", start_token=1024), state)
    assert late(Mapping(), None, None) == 0.0
    state.advance(generated_tokens=1024, thinking_open=True)
    assert late(Mapping(), None, None) == 0.1
    ramp = BridgePolicyController(
        policy(mode="ramp", start_token=1024, full_strength_token=2048), state
    )
    assert ramp(Mapping(), None, None) == 0.0
    state.advance(generated_tokens=1536, thinking_open=True)
    assert ramp(Mapping(), None, None) == pytest.approx(0.05)


def test_expert_layer_and_thinking_masks():
    state = BridgePolicyState(thinking_open=False)
    controller = BridgePolicyController(
        policy(active_experts=("7:18",), active_host_layers=(4,)), state
    )
    assert controller(Mapping(), None, None) == 0.0
    state.advance(generated_tokens=0, thinking_open=True)
    assert controller(Mapping(), None, None) == 0.1
    assert BridgePolicyController(policy(active_experts=("3:26",)), state)(
        Mapping(), None, None
    ) == 0.0


def test_state_rejects_backwards_or_invalid_updates():
    state = BridgePolicyState(generated_tokens=5)
    with pytest.raises(BridgeRescueError, match="backwards"):
        state.advance(generated_tokens=4, thinking_open=True)
    with pytest.raises(BridgeRescueError, match="invalid"):
        state.advance(generated_tokens=6, thinking_open=True, repetition_rate=float("nan"))


def test_learned_gate_fit_save_load_and_hash(tmp_path):
    checkpoint = fit_linear_gate(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.5]],
        [0, 1],
        training_manifest_sha256="a" * 64,
        max_generated_tokens=4096,
        steps=20,
    )
    destination = tmp_path / "controller.json"
    digest = save_linear_gate(checkpoint, destination)
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert load_linear_gate(destination, digest) == checkpoint
    with pytest.raises(BridgeRescueError, match="hash differs"):
        load_linear_gate(destination, "b" * 64)


def test_learned_policy_is_bounded():
    checkpoint = LinearGateCheckpoint(
        weights=(0.0, 1.0, 0.0, 0.0, 0.0),
        max_generated_tokens=100,
        training_manifest_sha256="a" * 64,
    )
    configured = policy(
        mode="learned",
        controller_checkpoint_sha256=checkpoint.fingerprint(),
        residual_multiplier=0.15,
    )
    value = BridgePolicyController(configured, BridgePolicyState(50), learned=checkpoint)(
        Mapping(), None, None
    )
    assert 0.0 < value < 0.15


def test_experiment_contract_requires_thinking_and_exact_split_size(tmp_path):
    config = RescueExperimentConfig(
        run_id="run",
        experiment="strength",
        dataset_manifest=tmp_path / "tasks.jsonl",
        dataset_manifest_sha256="a" * 64,
        split="development",
        expected_tasks=64,
        policies=(policy(name="base", residual_multiplier=0.0), policy()),
        output_dir=tmp_path / "out",
    )
    assert config.thinking_enabled is True
    assert len(config.fingerprint()) == 64
    with pytest.raises(ValueError, match="exactly 64"):
        config.model_copy(update={"expected_tasks": 63}).model_validate(
            config.model_copy(update={"expected_tasks": 63}).model_dump()
        )


def test_fit_gate_rejects_bad_shapes_and_labels():
    with pytest.raises(BridgeRescueError, match="N x 4"):
        fit_linear_gate(
            [[1.0]], [1], training_manifest_sha256="a" * 64, max_generated_tokens=10
        )


def test_generation_tracker_advances_and_closes_thinking():
    class Tensor:
        ndim = 2
        shape = (1, 4)

        class Row:
            def __getitem__(self, _slice):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [10, 11]

        def __getitem__(self, _index):
            return self.Row()

    class Tokenizer:
        def decode(self, _ids, **_kwargs):
            return "reasoning </think> answer"

    state = BridgePolicyState()
    scores = object()
    assert GenerationPolicyTracker(Tokenizer(), state, prompt_tokens=2)(Tensor(), scores) is scores
    assert state.generated_tokens == 2
    assert state.thinking_open is False


def test_config_loader_and_manifest_trainer(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """schema_version: 1
run_id: rescue-1
experiment: strength
dataset_manifest: tasks.jsonl
dataset_manifest_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
split: development
expected_tasks: 64
thinking_enabled: true
policies:
  - {name: base, mode: fixed, residual_multiplier: 0.0}
  - {name: weak, mode: fixed, residual_multiplier: 0.05}
output_dir: runs/rescue-1
""",
        encoding="utf-8",
    )
    assert load_rescue_experiment_config(config_path).experiment == "strength"

    manifest = tmp_path / "training.jsonl"
    manifest.write_text(
        '{"token_fraction":0.1,"repetition_rate":0.0,"gate_mean":0.1,'
        '"residual_ratio":0.1,"bridge_helpful":0}\n'
        '{"token_fraction":0.9,"repetition_rate":0.8,"gate_mean":0.2,'
        '"residual_ratio":0.2,"bridge_helpful":1}\n',
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = fit_linear_gate_manifest(
        manifest,
        tmp_path / "controller.json",
        expected_sha256=digest,
        max_generated_tokens=4096,
    )
    assert report["status"] == "PASS"
    assert report["training_rows"] == 2
    with pytest.raises(BridgeRescueError, match="binary"):
        fit_linear_gate(
            np.zeros((1, 4)), [2], training_manifest_sha256="a" * 64, max_generated_tokens=10
        )
