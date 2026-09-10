import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from reverse_reap.bridge_benchmark import (
    BridgeBenchmarkConfig,
    BridgeBenchmarkError,
    _clean_code,
    _exclusion_summary,
    _freeze_artifact_hashes,
    _paired_report,
    _score_rows,
    _validate_repeats,
    freeze_bridge_benchmark_tasks,
    load_bridge_benchmark_config,
    load_scoring_exclusions,
)
from reverse_reap.datasets import normalize_sample


def _sample(index: int):
    return normalize_sample(
        {
            "source": "openai/openai_humaneval",
            "source_revision": "a" * 40,
            "source_id": f"HumanEval/{index}",
            "domain": "coding",
            "stratum": "function-synthesis",
            "language": "python",
            "prompt": f"def function_{index}():\n    pass",
            "reference": "    return 1",
            "tests": f"assert function_{index}() == 1",
            "entry_point": f"function_{index}",
            "scorer": "unit_tests",
        },
        seed=9,
    )


def _config(tmp_path: Path, count: int = 5) -> BridgeBenchmarkConfig:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(_sample(index).model_dump(mode="json"), sort_keys=True) + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return BridgeBenchmarkConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "benchmark-run",
            "host_model": str(tmp_path / "host"),
            "host_revision": "b" * 40,
            "host_files_manifest": str(tmp_path / "host-files.json"),
            "host_files_manifest_sha256": "c" * 64,
            "bridge_config": str(tmp_path / "bridge.yaml"),
            "bridge_config_sha256": "d" * 64,
            "bridge_checkpoint": str(tmp_path / "bridge.safetensors"),
            "bridge_checkpoint_sha256": "e" * 64,
            "dataset": {
                "source_manifest": str(manifest),
                "source_manifest_sha256": digest,
                "source_id": "openai/openai_humaneval",
                "pilot_items": 2,
                "full_items": count,
                "selection_seed": 9,
            },
            "runtime": {
                "seed": 9,
                "deterministic": True,
                "enable_thinking": False,
                "batch_size": 1,
                "max_input_tokens": 1024,
                "max_new_tokens": 256,
                "repeats_per_condition": 2,
                "heartbeat_seconds": 300,
            },
            "scoring_mode": "deferred",
            "budget": {
                "max_wall_hours": 1,
                "max_cost_usd": 1,
                "provider_rate_usd_per_hour": 0.1,
                "usable_fraction": 0.8,
                "deadline_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            },
            "output_dir": str(tmp_path / "output"),
            "proceed_full_regardless_of_pilot_score": True,
        }
    )


def test_task_freeze_is_deterministic_and_pilot_is_full_prefix(tmp_path: Path):
    config = _config(tmp_path)
    first_pilot, first_full, first_report = freeze_bridge_benchmark_tasks(config)
    second_pilot, second_full, second_report = freeze_bridge_benchmark_tasks(config)
    assert [item.sample_id for item in first_pilot] == [
        item.sample_id for item in second_pilot
    ]
    assert [item.sample_id for item in first_full] == [
        item.sample_id for item in second_full
    ]
    assert first_full[: len(first_pilot)] == first_pilot
    assert first_report == second_report


def test_task_freeze_fails_on_source_hash_drift(tmp_path: Path):
    config = _config(tmp_path)
    config.dataset.source_manifest.write_text("tamper\n", encoding="utf-8")
    with pytest.raises(BridgeBenchmarkError, match="hash mismatch"):
        freeze_bridge_benchmark_tasks(config)


def test_repeat_gate_compares_generated_token_ids():
    first = [{"sample_id": "a", "generated_token_ids": [1, 2]}]
    second = [{"sample_id": "a", "generated_token_ids": [1, 2]}]
    assert _validate_repeats(first, second, "base")["passed"] is True
    second[0]["generated_token_ids"] = [1, 3]
    with pytest.raises(BridgeBenchmarkError, match="nondeterminism"):
        _validate_repeats(first, second, "base")


def test_paired_report_preserves_item_transitions_and_uncertainty():
    base = [
        {"sample_id": "a", "passed": False},
        {"sample_id": "b", "passed": True},
        {"sample_id": "c", "passed": True},
        {"sample_id": "d", "passed": False},
    ]
    bridge = [
        {"sample_id": "a", "passed": True},
        {"sample_id": "b", "passed": False},
        {"sample_id": "c", "passed": True},
        {"sample_id": "d", "passed": True},
    ]
    report = _paired_report(base, bridge)
    assert report["base_passes"] == 2
    assert report["bridge_passes"] == 3
    assert report["transitions"] == {
        "both-pass": 1,
        "bridge-breaks": 1,
        "bridge-fixes": 2,
    }
    assert 0 <= report["paired_exact_p_value"] <= 1
    assert len(report["bootstrap_95pct_interval"]) == 2


def test_markdown_fence_cleanup_is_bounded():
    assert _clean_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
    assert _clean_code("    return 1\n") == "    return 1"
    assert _clean_code("```python\n    return 1\n```") == "    return 1"
    prose = "Here is code:\ndef f(): return 1"
    assert _clean_code(prose) == prose


def test_scoring_reconstructs_continuation_from_raw_without_mutating_row(
    monkeypatch: pytest.MonkeyPatch,
):
    import reverse_reap.bridge_benchmark as benchmark

    sample = _sample(1)
    sample = sample.model_copy(update={"prompt": "def function_1():\n"})
    row = {
        "sample_id": sample.sample_id,
        "raw_completion": "    return 1\n",
        "completion": "return 1",
    }
    original = dict(row)
    seen = {}

    class Result:
        passed = True
        timed_out = False
        return_code = 0
        stdout = ""
        stderr = ""
        program_sha256 = "a" * 64

    def fake_evaluate(program, _tests, **_kwargs):
        seen["program"] = program
        return Result()

    monkeypatch.setattr(benchmark, "evaluate_python", fake_evaluate)
    scored = _score_rows([row], {sample.sample_id: sample}, "image@sha256:" + "b" * 64)
    assert row == original
    assert seen["program"] == "def function_1():\n    return 1"
    assert scored[0]["completion"] == "return 1"
    assert scored[0]["scored_completion"] == "    return 1"
    assert scored[0]["completion_reconstructed_from"] == "raw_completion"
    assert scored[0]["completion_normalizer_version"] == "continuation-preserving-v2"
    assert scored[0]["normalization_changed_from_stored_completion"] is True
    assert len(scored[0]["raw_completion_sha256"]) == 64


def test_scoring_fails_closed_without_raw_completion():
    sample = _sample(1)
    with pytest.raises(BridgeBenchmarkError, match="lacks immutable raw_completion"):
        _score_rows(
            [{"sample_id": sample.sample_id, "completion": "return 1"}],
            {sample.sample_id: sample},
            "image@sha256:" + "b" * 64,
        )


def test_versioned_artifact_manifest_preserves_original(tmp_path: Path):
    original = tmp_path / "artifact-manifest.json"
    original.write_text('{"original": true}\n', encoding="utf-8")
    (tmp_path / "evidence.json").write_text("{}\n", encoding="utf-8")
    _freeze_artifact_hashes(tmp_path, manifest_name="artifact-manifest-v2.json")
    assert original.read_text(encoding="utf-8") == '{"original": true}\n'
    assert (tmp_path / "artifact-manifest-v2.json").is_file()


def test_config_requires_image_only_for_local_docker(tmp_path: Path):
    payload = _config(tmp_path).model_dump(mode="json")
    payload["scoring_mode"] = "docker-local"
    payload["evaluator_image"] = None
    with pytest.raises(ValueError, match="pinned evaluator"):
        BridgeBenchmarkConfig.model_validate(payload)


def test_expired_generation_is_rejected_but_deferred_scoring_can_load(
    tmp_path: Path,
):
    import yaml

    payload = _config(tmp_path).model_dump(mode="json")
    payload["budget"]["deadline_utc"] = (
        datetime.now(UTC) - timedelta(minutes=1)
    ).isoformat()
    path = tmp_path / "expired.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(BridgeBenchmarkError, match="deadline has expired"):
        load_bridge_benchmark_config(path)
    loaded = load_bridge_benchmark_config(path, allow_expired=True)
    assert loaded.run_id == "benchmark-run"


def test_single_command_runs_four_pilot_and_four_full_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import yaml

    import reverse_reap.bridge_benchmark as benchmark

    config = _config(tmp_path, count=3)
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"pilot_items": 1})}
    )
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")

    class Parameter:
        device = "cpu"

    class Model:
        bridged = False

        def parameters(self):
            yield Parameter()

    class Handle:
        def __init__(self, model):
            self.model = model

        def remove(self):
            self.model.bridged = False

    model = Model()
    monkeypatch.setattr(benchmark, "_seed_runtime", lambda _seed: None)
    monkeypatch.setattr(benchmark, "_verify_host_files", lambda _config: {"verified": 1})
    monkeypatch.setattr(benchmark, "_load_host", lambda _config: (model, object()))
    monkeypatch.setattr(benchmark, "_load_bridge", lambda _config, _device: (object(), []))
    monkeypatch.setattr(
        benchmark,
        "install_bridge_sidecars",
        lambda current, _bridge, _mappings, telemetry=None: (
            setattr(current, "bridged", True) or [Handle(current)]
        ),
    )

    def fake_generate(current, _tokenizer, sample, _config):
        token = 2 if current.bridged else 1
        return {
            "sample_id": sample.sample_id,
            "generated_token_ids": [token],
            "completion": str(token),
        }

    monkeypatch.setattr(benchmark, "_generate_one", fake_generate)
    result = benchmark.run_bridge_benchmark(config_path)
    assert result["classification"] == "generation-complete-scoring-deferred"
    for tier, expected in (("pilot", 1), ("full", 3)):
        for name in ("base-a", "base-b", "bridge-a", "bridge-b"):
            path = config.output_dir / tier / f"{name}.jsonl"
            assert len(path.read_text(encoding="utf-8").splitlines()) == expected
        assert result["reports"][tier]["base_determinism"]["passed"] is True
        assert result["reports"][tier]["bridge_determinism"]["passed"] is True
    state = json.loads((config.output_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "COMPLETE"


def test_single_command_records_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import yaml

    import reverse_reap.bridge_benchmark as benchmark

    config = _config(tmp_path)
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "freeze_bridge_benchmark_tasks",
        lambda _config: (_ for _ in ()).throw(BridgeBenchmarkError("frozen failure")),
    )
    with pytest.raises(BridgeBenchmarkError, match="frozen failure"):
        benchmark.run_bridge_benchmark(config_path)
    state = json.loads((config.output_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED_TERMINAL"
    assert "frozen failure" in state["failure"]


def _exclusion_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "scoring-exclusions.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _valid_exclusion_payload():
    return {
        "schema_version": 1,
        "kind": "scoring-exclusions",
        "exclusions": [
            {"sample_id": "aaa", "reason": "malformed-upstream-test"},
        ],
    }


def test_scoring_exclusions_load_with_deterministic_hash(tmp_path: Path):
    first = load_scoring_exclusions(_exclusion_manifest(tmp_path, _valid_exclusion_payload()))
    second = load_scoring_exclusions(tmp_path / "scoring-exclusions.json")
    assert first[0] == frozenset({"aaa"})
    assert first[1] == second[1]
    assert len(first[1]) == 64


def test_scoring_exclusions_fail_closed(tmp_path: Path):
    with pytest.raises(BridgeBenchmarkError, match="unexpected kind"):
        load_scoring_exclusions(_exclusion_manifest(tmp_path, {"kind": "other", "exclusions": []}))
    with pytest.raises(BridgeBenchmarkError, match="list no excluded tasks"):
        load_scoring_exclusions(
            _exclusion_manifest(tmp_path, {"kind": "scoring-exclusions", "exclusions": []})
        )
    with pytest.raises(BridgeBenchmarkError, match="duplicate"):
        load_scoring_exclusions(
            _exclusion_manifest(
                tmp_path,
                {
                    "kind": "scoring-exclusions",
                    "exclusions": [
                        {"sample_id": "aaa", "reason": "x"},
                        {"sample_id": "aaa", "reason": "x"},
                    ],
                },
            )
        )
    with pytest.raises(BridgeBenchmarkError, match="lacks sample_id or reason"):
        load_scoring_exclusions(
            _exclusion_manifest(
                tmp_path,
                {"kind": "scoring-exclusions", "exclusions": [{"sample_id": "aaa"}]},
            )
        )
    with pytest.raises(BridgeBenchmarkError):
        load_scoring_exclusions(tmp_path / "missing.json")


def test_exclusion_summary_rejects_unknown_ids_and_reports_denominators():
    summary = _exclusion_summary({"aaa", "bbb", "ccc"}, frozenset({"aaa"}))
    assert summary["original_tasks"] == 3
    assert summary["excluded_tasks"] == 1
    assert summary["eligible_tasks"] == 2
    assert summary["coverage"] == "2/3"
    assert summary["eligible_sample_ids"] == ["bbb", "ccc"]
    with pytest.raises(BridgeBenchmarkError, match="unknown sample IDs"):
        _exclusion_summary({"aaa"}, frozenset({"aaa", "zzz"}))


def test_exclusions_apply_symmetrically_without_mutating_inputs():
    universe = {"aaa", "bbb", "ccc"}
    excluded = frozenset({"aaa"})
    before = (set(universe), set(excluded))
    eligible = {
        condition: _exclusion_summary(universe, excluded)["eligible_sample_ids"]
        for condition in ("base-a", "base-b", "bridge-a", "bridge-b")
    }
    assert all(rows == ["bbb", "ccc"] for rows in eligible.values())
    assert (set(universe), set(excluded)) == before
