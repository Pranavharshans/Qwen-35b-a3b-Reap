import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import reverse_reap.mbpp_bridge_benchmark as benchmark
from reverse_reap.bridge_benchmark import BridgeBenchmarkBudget, BridgeBenchmarkError
from reverse_reap.mbpp_bridge_benchmark import (
    CONDITIONS,
    EvalPlusContract,
    MbppBridgeBenchmarkConfig,
    MbppDataset,
    MbppRuntime,
    _official_result_rows,
    _render_prompt,
    _validate_bridge_engagement,
    freeze_mbpp_tasks,
    score_mbpp_bridge_benchmark,
    validate_mbpp_generation,
)


def _task(index: int) -> dict:
    return {
        "task_id": f"Mbpp/{index}",
        "prompt": f'"""Return, task {index}."""\n',
        "entry_point": f"solve_{index}",
        "canonical_solution": f"def solve_{index}():\n    return {index}\n",
        "base_input": [[]],
        "plus_input": [[index]],
        "atol": 0,
    }


def _config(tmp_path: Path, *, count: int = 2) -> MbppBridgeBenchmarkConfig:
    archive = tmp_path / "MbppPlus.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps(_task(index)) + "\n")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dataset = MbppDataset.model_construct(
        archive=archive,
        archive_sha256=archive_sha,
        version="v0.2.0",
        expected_tasks=count,
        pilot_items=1,
        selection_seed=9,
    )
    return MbppBridgeBenchmarkConfig.model_construct(
        schema_version=1,
        run_id="mbpp-test",
        host_model=tmp_path / "host",
        host_revision="a" * 40,
        host_files_manifest=tmp_path / "host-files.json",
        host_files_manifest_sha256="b" * 64,
        bridge_config=tmp_path / "bridge.yaml",
        bridge_config_sha256="c" * 64,
        bridge_checkpoint=tmp_path / "bridge.safetensors",
        bridge_checkpoint_sha256="d" * 64,
        dataset=dataset,
        runtime=MbppRuntime(),
        evalplus=EvalPlusContract(parallel_workers=2, cpus=2),
        budget=BridgeBenchmarkBudget(
            max_wall_hours=1,
            max_cost_usd=1,
            provider_rate_usd_per_hour=0.1,
            usable_fraction=0.8,
            deadline_utc=datetime.now(UTC) + timedelta(hours=1),
        ),
        output_dir=tmp_path / "output",
        proceed_full_regardless_of_pilot_score=True,
    )


def test_config_rejects_unapproved_mbpp_archive_hash(tmp_path: Path):
    payload = _config(tmp_path).model_dump(mode="json")
    payload["dataset"]["expected_tasks"] = 378
    payload["dataset"]["archive_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="approved v0.2.0"):
        MbppBridgeBenchmarkConfig.model_validate(payload)


def test_freeze_validates_archive_and_is_deterministic(tmp_path: Path):
    config = _config(tmp_path, count=3)
    first, first_report = freeze_mbpp_tasks(config)
    second, second_report = freeze_mbpp_tasks(config)
    assert first == second
    assert first_report == second_report
    assert len(first) == 3
    assert len({row["task_id"] for row in first}) == 3
    config.dataset.archive.write_bytes(b"tampered")
    with pytest.raises(BridgeBenchmarkError, match="hash mismatch"):
        freeze_mbpp_tasks(config)


def test_render_prompt_passes_real_thinking_switch_to_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"thinking={kwargs['enable_thinking']}::{messages[-1]['content']}"

        def __call__(self, text, **_kwargs):
            return {"input_ids": [[len(text)]]}

    off, _ = _render_prompt(Tokenizer(), "problem", enable_thinking=False)
    on, _ = _render_prompt(Tokenizer(), "problem", enable_thinking=True)
    assert off != on
    assert "thinking=False" in off
    assert "thinking=True" in on


def test_generation_command_emits_exactly_four_conditions_and_bridge_engagement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _config(tmp_path)
    tasks, freeze = freeze_mbpp_tasks(config)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused", encoding="utf-8")

    class Parameter:
        device = "cpu"

    class Model:
        def parameters(self):
            yield Parameter()

    mappings = [SimpleNamespace(key=f"{i}:{i}") for i in range(4)]

    class Telemetry:
        def snapshot(self):
            return {
                item.key: {
                    "calls": 1,
                    "tokens": 2,
                    "gate_mean": 0.1,
                    "residual_l2_mean": 0.2,
                }
                for item in mappings
            }

    class Handle:
        def remove(self):
            return None

    monkeypatch.setattr(benchmark, "load_mbpp_bridge_config", lambda *_a, **_kw: config)
    monkeypatch.setattr(benchmark, "freeze_mbpp_tasks", lambda _config: (tasks, freeze))
    monkeypatch.setattr(benchmark, "_verify_host_files", lambda _config: {"verified": 1})
    monkeypatch.setattr(benchmark, "_seed_runtime", lambda _seed: None)
    monkeypatch.setattr(benchmark, "_load_host", lambda _config: (Model(), object()))
    monkeypatch.setattr(benchmark, "_load_bridge", lambda _config, _device: (object(), mappings))
    monkeypatch.setattr(benchmark, "_bridge_load_evidence", lambda *_a: {"passed": True})
    monkeypatch.setattr(benchmark, "BridgeGenerationTelemetry", Telemetry)
    monkeypatch.setattr(
        benchmark,
        "install_bridge_sidecars",
        lambda *_a, **_kw: [Handle()],
    )
    monkeypatch.setattr(
        benchmark,
        "_render_prompt",
        lambda _tok, _prompt, enable_thinking: (
            f"thinking={enable_thinking}",
            [[1]],
        ),
    )

    def fake_generate(_model, _tokenizer, task, _config, *, condition):
        return {
            "task_id": task["task_id"],
            "solution": "def f(): return 1",
            "condition": condition,
            "thinking_enabled": condition.endswith("thinking-on"),
            "bridge_enabled": condition.startswith("bridge-"),
            "prompt_sha256": hashlib.sha256(condition.encode()).hexdigest(),
        }

    monkeypatch.setattr(benchmark, "_generate_one", fake_generate)
    monkeypatch.setattr(
        benchmark,
        "validate_mbpp_generation",
        lambda _config, **_kwargs: {"passed": True, "files": {}},
    )
    report = benchmark.run_mbpp_bridge_benchmark(config_path)
    assert report["status"] == "PASS"
    assert set(report["bridge_engagement"]) == {
        "bridge-thinking-off",
        "bridge-thinking-on",
    }
    assert sorted(path.name for path in (config.output_dir / "conditions").glob("*.jsonl")) == [
        f"{condition}.jsonl" for condition in sorted(CONDITIONS)
    ]


def test_official_result_parser_requires_full_pass_at_one_universe(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "eval": {
                    "Mbpp/1": [
                        {"base_status": "pass", "plus_status": "fail"}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert _official_result_rows(path, {"Mbpp/1"}) == {
        "Mbpp/1": {"base": True, "plus": False}
    }
    with pytest.raises(BridgeBenchmarkError, match="universe differs"):
        _official_result_rows(path, {"Mbpp/1", "Mbpp/2"})


def test_generation_validation_binds_sources_prompts_and_modes(tmp_path: Path):
    config = _config(tmp_path)
    tasks, _ = freeze_mbpp_tasks(config)
    conditions = config.output_dir / "conditions"
    conditions.mkdir(parents=True)
    for condition in CONDITIONS:
        thinking = condition.endswith("thinking-on")
        rows = [
            {
                "task_id": task["task_id"],
                "condition": condition,
                "thinking_enabled": thinking,
                "bridge_enabled": condition.startswith("bridge-"),
                "source_row_sha256": task["source_row_sha256"],
                "prompt_sha256": f"{task['task_id']}:{thinking}",
            }
            for task in tasks
        ]
        (conditions / f"{condition}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    assert validate_mbpp_generation(config)["passed"] is True
    path = conditions / "bridge-thinking-on.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["source_row_sha256"] = "tampered"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(BridgeBenchmarkError, match="source task hash differs"):
        validate_mbpp_generation(config)


def test_bridge_engagement_requires_real_residual_activity():
    mappings = [SimpleNamespace(key=f"mapping-{index}") for index in range(4)]
    snapshot = {
        item.key: {
            "calls": 1,
            "tokens": 2,
            "gate_mean": 0.1,
            "residual_l2_mean": 0.2,
        }
        for item in mappings
    }
    assert _validate_bridge_engagement(snapshot, mappings, condition="bridge") == snapshot
    snapshot["mapping-2"]["residual_l2_mean"] = 0.0
    with pytest.raises(BridgeBenchmarkError, match="emitted no residual"):
        _validate_bridge_engagement(snapshot, mappings, condition="bridge")


def test_scoring_uses_official_sanitize_and_evaluate_for_all_four_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _config(tmp_path)
    tasks, freeze = freeze_mbpp_tasks(config)
    conditions = config.output_dir / "conditions"
    conditions.mkdir(parents=True)
    for condition in CONDITIONS:
        (conditions / f"{condition}.jsonl").write_text(
            "".join(
                json.dumps({"task_id": task["task_id"], "solution": "def f(): pass"})
                + "\n"
                for task in tasks
            ),
            encoding="utf-8",
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused", encoding="utf-8")
    monkeypatch.setattr(benchmark, "load_mbpp_bridge_config", lambda *_a, **_kw: config)
    monkeypatch.setattr(benchmark, "validate_mbpp_generation", lambda _config: {"passed": True})
    monkeypatch.setattr(benchmark, "freeze_mbpp_tasks", lambda _config: (tasks, freeze))
    monkeypatch.setattr(benchmark, "_verify_evalplus_image", lambda *_a: None)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if "evalplus.sanitize" in command:
            source = command[command.index("--samples") + 1].removeprefix("/work/")
            source_path = config.output_dir / source
            source_path.with_name(source_path.stem + "-sanitized.jsonl").write_text(
                source_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        if "evalplus.evaluate" in command:
            output = command[command.index("--output-file") + 1].removeprefix("/work/")
            (config.output_dir / output).write_text(
                json.dumps(
                    {
                        "eval": {
                            task["task_id"]: [
                                {"base_status": "pass", "plus_status": "pass"}
                            ]
                            for task in tasks
                        }
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    report = score_mbpp_bridge_benchmark(
        config_path, evalplus_image="local/image@sha256:" + "a" * 64
    )
    assert report["status"] == "PASS"
    assert sum("evalplus.sanitize" in command for command in commands) == 4
    assert sum("evalplus.evaluate" in command for command in commands) == 4
    assert all("--network=none" in command for command in commands)
