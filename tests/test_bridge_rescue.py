from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

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
    run_strength_screen,
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
        benchmark_config=tmp_path / "benchmark.yaml",
        benchmark_config_sha256="b" * 64,
        dataset_manifest=tmp_path / "tasks.jsonl",
        dataset_manifest_sha256="a" * 64,
        split="screening",
        expected_tasks=12,
        policies=(policy(name="base", residual_multiplier=0.0), policy()),
        output_dir=tmp_path / "out",
    )
    assert config.thinking_enabled is True
    assert len(config.fingerprint()) == 64
    with pytest.raises(ValueError, match="exactly 12"):
        config.model_copy(update={"expected_tasks": 11}).model_validate(
            config.model_copy(update={"expected_tasks": 11}).model_dump()
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
benchmark_config: benchmark.yaml
benchmark_config_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
dataset_manifest: tasks.jsonl
dataset_manifest_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
split: screening
expected_tasks: 12
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


def test_strength_screen_runs_six_thinking_conditions(tmp_path, monkeypatch):
    import reverse_reap.bridge_benchmark as bridge_benchmark
    import reverse_reap.bridge_training as bridge_training
    import reverse_reap.mbpp_bridge_benchmark as mbpp

    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text("pinned: true\n")
    benchmark_sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    tasks = [
        {
            "task_id": f"Mbpp/{index}",
            "prompt": f"problem {index}",
            "entry_point": "solve",
            "source_row_sha256": f"{index:064x}",
        }
        for index in range(12)
    ]
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in tasks))
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    policies = [
        {"name": "base", "mode": "fixed", "residual_multiplier": 0.0},
        {"name": "current", "mode": "fixed", "residual_multiplier": 1.0},
        {"name": "strength-0.025", "mode": "fixed", "residual_multiplier": 0.1},
        {"name": "strength-0.05", "mode": "fixed", "residual_multiplier": 0.2},
        {"name": "strength-0.10", "mode": "fixed", "residual_multiplier": 0.4},
        {"name": "strength-0.15", "mode": "fixed", "residual_multiplier": 0.6},
    ]
    config_path = tmp_path / "rescue.yaml"
    import yaml

    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "run_id": "strength-run",
                "experiment": "strength",
                "benchmark_config": str(benchmark_path),
                "benchmark_config_sha256": benchmark_sha,
                "dataset_manifest": str(manifest),
                "dataset_manifest_sha256": manifest_sha,
                "split": "screening",
                "expected_tasks": 12,
                "thinking_enabled": True,
                "policies": policies,
                "output_dir": str(tmp_path / "out"),
            }
        )
    )

    class Model:
        def parameters(self):
            yield SimpleNamespace(device="cpu")

    class Telemetry:
        def snapshot(self):
            return {"7:18": {"calls": 1}}

    monkeypatch.setattr(
        mbpp,
        "load_mbpp_bridge_config",
        lambda _path: SimpleNamespace(runtime=SimpleNamespace(seed=1)),
    )
    monkeypatch.setattr(mbpp, "_check_budget", lambda *_args: None)
    monkeypatch.setattr(mbpp, "_require_deterministic_cuda", lambda: None)
    monkeypatch.setattr(
        mbpp,
        "_generate_one",
        lambda _model, _tokenizer, task, _config, *, condition, **_kwargs: {
            "task_id": task["task_id"],
            "condition": condition,
        },
    )
    monkeypatch.setattr(bridge_benchmark, "BridgeGenerationTelemetry", Telemetry)
    monkeypatch.setattr(bridge_benchmark, "_seed_runtime", lambda _seed: None)
    monkeypatch.setattr(bridge_benchmark, "_verify_host_files", lambda _config: {"ok": True})
    monkeypatch.setattr(bridge_benchmark, "_load_host", lambda _config: (Model(), object()))
    monkeypatch.setattr(
        bridge_benchmark,
        "_load_bridge",
        lambda _config, _device: (object(), [Mapping()]),
    )
    monkeypatch.setattr(bridge_training, "install_bridge_sidecars", lambda *_args, **_kwargs: [])
    report = run_strength_screen(config_path)
    assert report["status"] == "PASS"
    assert len(report["conditions"]) == 6
    assert all(value["rows"] == 12 for value in report["conditions"].values())
    with pytest.raises(BridgeRescueError, match="binary"):
        fit_linear_gate(
            np.zeros((1, 4)), [2], training_manifest_sha256="a" * 64, max_generated_tokens=10
        )
