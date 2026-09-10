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
    _classify_output,
    _official_result_rows,
    _render_prompt,
    _validate_bridge_engagement,
    condition_output_statistics,
    evaluate_pilot_safety_gate,
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


def _config(
    tmp_path: Path, *, count: int = 2, pilot_items: int = 1
) -> MbppBridgeBenchmarkConfig:
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
        pilot_items=pilot_items,
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


class _RecordingTokenizer:
    """Fake host tokenizer honouring the native template contract."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, dict(kwargs)))
        mode = kwargs.get("enable_thinking")
        gen = kwargs.get("add_generation_prompt")
        return f"RENDERED thinking={mode} gen={gen} header::<think>\n::{messages[-1]['content']}"

    def __call__(self, text, **_kwargs):
        return {"input_ids": [[len(text)]]}


def test_native_prompt_differs_by_thinking_mode():
    tokenizer = _RecordingTokenizer()
    off, off_ids = _render_prompt(tokenizer, "problem", enable_thinking=False)
    on, on_ids = _render_prompt(tokenizer, "problem", enable_thinking=True)
    assert off != on
    assert "thinking=False" in off
    assert "thinking=True" in on
    for _messages, kwargs in tokenizer.calls:
        assert kwargs.get("add_generation_prompt") is True
    assert off_ids == [[len(off)]]
    assert on_ids == [[len(on)]]


def test_prompt_injects_no_assistant_message_or_think_tags():
    tokenizer = _RecordingTokenizer()
    _render_prompt(tokenizer, "problem", enable_thinking=True)
    ((messages, _kwargs),) = tokenizer.calls
    assert [message["role"] for message in messages] == ["user"]
    assert "<think>" not in messages[0]["content"]
    assert "</think>" not in messages[0]["content"]
    assert "```python" not in messages[0]["content"]


def test_base_and_bridge_prompts_identical_within_each_mode():
    tokenizer = _RecordingTokenizer()
    for mode in (False, True):
        base, _ = _render_prompt(tokenizer, "problem", enable_thinking=mode)
        bridge, _ = _render_prompt(tokenizer, "problem", enable_thinking=mode)
        assert base == bridge


def test_generation_boundary_is_end_of_native_prompt():
    tokenizer = _RecordingTokenizer()
    rendered, ids = _render_prompt(tokenizer, "problem", enable_thinking=False)
    # The whole native prompt is encoded, so generation starts at its end.
    assert ids == [[len(rendered)]]


def test_sanitize_recovers_code_from_thinking_on_reasoning():
    sanitize = pytest.importorskip("evalplus.sanitize").sanitize
    raw = (
        "<think>\nThe task needs a sum, so I will write a helper.\n</think>\n"
        "Here is the solution:\n```python\ndef solve_0():\n    return 0\n```\n"
    )
    cleaned = sanitize(raw, "solve_0")
    assert "def solve_0():" in cleaned
    assert "<think>" not in cleaned
    compile(cleaned, "<thinking-on>", "exec")


def test_sanitize_recovers_code_from_thinking_off_output():
    sanitize = pytest.importorskip("evalplus.sanitize").sanitize
    raw = "<think>\n\n</think>\n\n```python\ndef solve_0():\n    return 0\n```\n"
    cleaned = sanitize(raw, "solve_0")
    assert "def solve_0():" in cleaned
    compile(cleaned, "<thinking-off>", "exec")


def test_deterministic_cuda_requires_cublas_workspace(monkeypatch: pytest.MonkeyPatch):
    import sys

    state = {"deterministic": True, "cuda": True}
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: state["cuda"]),
        are_deterministic_algorithms_enabled=lambda: state["deterministic"],
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(BridgeBenchmarkError, match="CUBLAS_WORKSPACE_CONFIG"):
        benchmark._require_deterministic_cuda()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "bogus")
    with pytest.raises(BridgeBenchmarkError, match="CUBLAS_WORKSPACE_CONFIG"):
        benchmark._require_deterministic_cuda()
    for value in (":4096:8", ":16:8"):
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
        benchmark._require_deterministic_cuda()
    state["deterministic"] = False
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    benchmark._require_deterministic_cuda()
    state.update(deterministic=True, cuda=False)
    benchmark._require_deterministic_cuda()


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
        json.dumps({"eval": {"Mbpp/1": [{"base_status": "pass", "plus_status": "fail"}]}}),
        encoding="utf-8",
    )
    assert _official_result_rows(path, {"Mbpp/1"}) == {"Mbpp/1": {"base": True, "plus": False}}
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
                json.dumps({"task_id": task["task_id"], "solution": "def f(): pass"}) + "\n"
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
                            task["task_id"]: [{"base_status": "pass", "plus_status": "pass"}]
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


def _policy_row(
    task: dict,
    condition: str,
    *,
    cap_hit: bool = False,
    failure_reason: str | None = None,
    generated_tokens: int = 12,
) -> dict:
    thinking = condition.endswith("thinking-on")
    sanitizable = failure_reason != "sanitization_failure"
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "solution": "```python\ndef f(): pass\n```",
        "condition": condition,
        "bridge_enabled": condition.startswith("bridge-"),
        "thinking_enabled": thinking,
        "source_row_sha256": task["source_row_sha256"],
        "prompt_sha256": hashlib.sha256(("on" if thinking else "off").encode()).hexdigest(),
        "input_token_ids": [1],
        "generated_token_ids": [2] * generated_tokens,
        "input_tokens": 1,
        "generated_tokens": generated_tokens,
        "hit_max_new_tokens": cap_hit,
        "generation_seconds": 0.1,
        "raw_solution_sha256": hashlib.sha256(b"raw").hexdigest(),
        "cap_hit": cap_hit,
        "reasoning_opened": thinking,
        "reasoning_closed": (not cap_hit) if thinking else True,
        "final_answer_present": not cap_hit,
        "sanitizable": sanitizable,
        "score_eligible": sanitizable,
        "failure_reason": failure_reason,
        "structurally_invalid": failure_reason is not None,
        "sanitized_solution": "def f(): pass" if sanitizable else "",
        "sanitized_solution_sha256": "0" * 64,
        "sanitize_error": None,
    }


def test_output_classification_keeps_cap_hit_item_in_denominator():
    truncated = "reasoning that never closes\n```python\ndef solve_0():\n    return 0"
    classified = _classify_output(
        truncated,
        thinking=True,
        entry_point="solve_0",
        cap_hit=True,
        sanitize_func=lambda code, entry: "def solve_0():\n    return 0",
    )
    assert classified["cap_hit"] is True
    assert classified["reasoning_opened"] is True
    assert classified["reasoning_closed"] is False
    assert classified["failure_reason"] == "unclosed_reasoning"
    assert classified["sanitizable"] is True
    assert classified["score_eligible"] is True
    assert classified["structurally_invalid"] is True


def test_output_classification_marks_unrecoverable_item_score_ineligible():
    classified = _classify_output(
        "reasoning only, no code",
        thinking=True,
        entry_point="solve_0",
        cap_hit=True,
        sanitize_func=lambda code, entry: "",
    )
    assert classified["sanitizable"] is False
    assert classified["score_eligible"] is False
    assert classified["failure_reason"] == "sanitization_failure"
    assert classified["sanitized_solution"] == ""


def _run_fake_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
    pilot_items: int,
    cap_hits: int,
):
    config = _config(tmp_path, count=count, pilot_items=pilot_items)
    tasks, freeze = freeze_mbpp_tasks(config)
    index = {task["task_id"]: position for position, task in enumerate(tasks)}
    calls: list[tuple[str, str]] = []

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

    def fake_generate(_model, _tokenizer, task, _config, *, condition):
        calls.append((task["task_id"], condition))
        hit = index[task["task_id"]] < cap_hits
        return _policy_row(
            task,
            condition,
            cap_hit=hit,
            failure_reason="unclosed_reasoning" if hit else None,
        )

    monkeypatch.setattr(benchmark, "load_mbpp_bridge_config", lambda *_a, **_kw: config)
    monkeypatch.setattr(benchmark, "freeze_mbpp_tasks", lambda _config: (tasks, freeze))
    monkeypatch.setattr(benchmark, "_verify_host_files", lambda _config: {"verified": 1})
    monkeypatch.setattr(benchmark, "_seed_runtime", lambda _seed: None)
    monkeypatch.setattr(benchmark, "_load_host", lambda _config: (Model(), object()))
    monkeypatch.setattr(benchmark, "_load_bridge", lambda _config, _device: (object(), mappings))
    monkeypatch.setattr(benchmark, "_bridge_load_evidence", lambda *_a: {"passed": True})
    monkeypatch.setattr(benchmark, "BridgeGenerationTelemetry", Telemetry)
    monkeypatch.setattr(benchmark, "install_bridge_sidecars", lambda *_a, **_kw: [Handle()])
    monkeypatch.setattr(
        benchmark,
        "_render_prompt",
        lambda _tok, _prompt, enable_thinking: (
            f"thinking={enable_thinking}",
            [[1]],
        ),
    )
    monkeypatch.setattr(benchmark, "_generate_one", fake_generate)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused", encoding="utf-8")
    return config, tasks, calls, config_path


def test_single_cap_hit_row_does_not_terminate_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, tasks, calls, config_path = _run_fake_generation(
        tmp_path, monkeypatch, count=20, pilot_items=20, cap_hits=1
    )
    report = benchmark.run_mbpp_bridge_benchmark(config_path)
    assert report["status"] == "PASS"
    gate = json.loads((config.output_dir / "pilot-safety-gate.json").read_text())
    assert gate["passed"] is True
    assert len(calls) == 4 * 20
    for condition in CONDITIONS:
        rows = [
            json.loads(line)
            for line in (config.output_dir / "conditions" / f"{condition}.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(rows) == 20
        truncated = [row for row in rows if row["failure_reason"] is not None]
        assert len(truncated) == 1
        assert truncated[0]["task_id"] == tasks[0]["task_id"]
        assert truncated[0]["cap_hit"] is True
        assert len(truncated[0]["generated_token_ids"]) == 12
    assert report["output_statistics"]["base-thinking-on"]["cap_hit_count"] == 1
    assert report["output_statistics"]["base-thinking-off"]["cap_hit_count"] == 1


def test_pilot_safety_gate_blocks_full_tier_when_rate_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, _tasks, calls, config_path = _run_fake_generation(
        tmp_path, monkeypatch, count=40, pilot_items=20, cap_hits=2
    )
    with pytest.raises(BridgeBenchmarkError, match="pilot safety gate exceeded"):
        benchmark.run_mbpp_bridge_benchmark(config_path)
    assert len(calls) == 4 * 20
    assert not (config.output_dir / "full-generation-report.json").exists()
    state = json.loads((config.output_dir / "state.json").read_text())
    assert state["status"] == "FAILED_TERMINAL"
    assert state["stage"] == "pilot-safety-gate"
    gate = json.loads((config.output_dir / "pilot-safety-gate.json").read_text())
    assert gate["passed"] is False
    assert any("cap_hit_rate_exceeded" in violation for violation in gate["violations"])


def test_pilot_safety_gate_rejects_duplicate_rows(tmp_path: Path):
    config = _config(tmp_path, count=20, pilot_items=20)
    tasks, _freeze = freeze_mbpp_tasks(config)
    expected_ids = [task["task_id"] for task in tasks]
    rows = [_policy_row(task, "base-thinking-off") for task in tasks]
    rows[-1] = dict(rows[0])
    statistics = condition_output_statistics(rows, expected_ids, condition="base-thinking-off")
    assert statistics["duplicate_ids"] == [expected_ids[0]]
    assert statistics["denominator"] == 20
    rows_by_condition = {
        condition: [_policy_row(task, condition) for task in tasks] for condition in CONDITIONS
    }
    rows_by_condition["base-thinking-off"] = rows
    engagement = {
        condition: {
            f"{i}:{i}": {"calls": 1, "tokens": 2, "gate_mean": 0.1, "residual_l2_mean": 0.2}
            for i in range(4)
        }
        for condition in ("bridge-thinking-off", "bridge-thinking-on")
    }
    gate = evaluate_pilot_safety_gate(
        config, rows_by_condition, expected_ids, bridge_engagement=engagement
    )
    assert gate["passed"] is False
    assert "base-thinking-off:duplicate_rows" in gate["violations"]


def _score_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, drop_last: bool, empty_last: bool
):
    config = _config(tmp_path, count=2)
    tasks, freeze = freeze_mbpp_tasks(config)
    conditions = config.output_dir / "conditions"
    conditions.mkdir(parents=True)
    for condition in CONDITIONS:
        (conditions / f"{condition}.jsonl").write_text(
            "".join(
                json.dumps({"task_id": task["task_id"], "solution": "def f(): pass"}) + "\n"
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
    return config, tasks, config_path


def test_scoring_rejects_sanitizer_row_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, tasks, config_path = _score_fixture(
        tmp_path, monkeypatch, drop_last=True, empty_last=False
    )

    def fake_run(command, **_kwargs):
        if "evalplus.sanitize" in command:
            source = command[command.index("--samples") + 1].removeprefix("/work/")
            source_path = config.output_dir / source
            rows = [json.loads(line) for line in source_path.read_text().splitlines()]
            kept = rows[:-1]
            source_path.with_name(source_path.stem + "-sanitized.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8"
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    with pytest.raises(BridgeBenchmarkError, match="dropped, added or reordered"):
        score_mbpp_bridge_benchmark(config_path, evalplus_image="local/image@sha256:" + "a" * 64)


def test_scoring_reports_unrecoverable_items_and_keeps_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, tasks, config_path = _score_fixture(
        tmp_path, monkeypatch, drop_last=False, empty_last=True
    )

    def fake_run(command, **_kwargs):
        if "evalplus.sanitize" in command:
            source = command[command.index("--samples") + 1].removeprefix("/work/")
            source_path = config.output_dir / source
            rows = [json.loads(line) for line in source_path.read_text().splitlines()]
            out_rows = [
                {"task_id": row["task_id"], "solution": "def f(): pass" if index == 0 else ""}
                for index, row in enumerate(rows)
            ]
            source_path.with_name(source_path.stem + "-sanitized.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in out_rows), encoding="utf-8"
            )
        if "evalplus.evaluate" in command:
            output = command[command.index("--output-file") + 1].removeprefix("/work/")
            result = {
                "eval": {
                    task["task_id"]: [
                        {
                            "base_status": "pass" if index == 0 else "fail",
                            "plus_status": "pass" if index == 0 else "fail",
                        }
                    ]
                    for index, task in enumerate(tasks)
                }
            }
            (config.output_dir / output).write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    report = score_mbpp_bridge_benchmark(
        config_path, evalplus_image="local/image@sha256:" + "a" * 64
    )
    assert report["status"] == "PASS"
    for condition in CONDITIONS:
        stats = report["sanitization_statistics"][condition]
        assert stats["rows"] == 2
        assert stats["unrecoverable_items"] == [tasks[1]["task_id"]]
        assert stats["unrecoverable_rate"] == 0.5
    assert report["comparisons"]["thinking-off"]["full"]["base"]["samples"] == 2

