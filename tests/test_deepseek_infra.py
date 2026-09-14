"""Hardware-free contract tests for the isolated DeepSeek V4 Flash launcher."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from reverse_reap import deepseek_infra as infra
from reverse_reap.cli import build_parser, main

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
ROOT = Path(__file__).parents[1]


def _paths(root: Path) -> dict[str, str]:
    return {
        "model_cache_dir": str(root / "model-cache"),
        "run_dir": str(root / "run"),
        "output_dir": str(root / "output"),
        "cache_dir": str(root / "cache"),
        "log_dir": str(root / "logs"),
    }


def _mapping(root: Path, mode: str = "direct", *, placeholders: bool = False) -> dict:
    values = _paths(root)
    if placeholders:
        values = {name: "/absolute/path/to/replace" for name in values}
    result: dict = {
        "schema_version": 1,
        "run_id": "deepseek-v4-test",
        "mode": mode,
        "model": {"id": MODEL_ID, "revision": REVISION, "trust_remote_code": True},
        "serving": {
            "vllm_min_version": "0.20.0",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "gpu_count": 8,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.9,
            "host": "127.0.0.1",
            "port": 8000,
            "extra_args": ["--max-num-seqs", "2"],
        },
        "paths": values,
    }
    if mode == "direct":
        result["direct"] = {"working_directory": str(root)}
    else:
        result["fau_slurm"] = {
            "job_name": "deepseek-v4-test",
            "account": "project-123",
            "partition": "gpu",
            "constraint": "a100_80",
            "nodes": 1,
            "tasks_per_node": 1,
            "gpus_per_node": 8,
            "cpus_per_task": 32,
            "memory_gb": 256,
            "time_limit": "1-12:00:00",
            "output_path": str(root / "slurm logs" / "job.out"),
            "error_path": str(root / "slurm logs" / "job.err"),
            "module_commands": ["module load cuda/12.8", "module load python/3.12"],
            "environment_activation": "source /opt/venvs/deepseek/bin/activate",
        }
    if placeholders and mode == "fau_slurm":
        result["fau_slurm"]["account"] = "YOUR_FAU_ACCOUNT"
        result["fau_slurm"]["partition"] = "YOUR_PARTITION"
        result["fau_slurm"]["constraint"] = "YOUR_CONSTRAINT"
    return result


def _config(root: Path, mode: str = "direct", **kwargs) -> infra.DeepSeekLaunchConfig:
    return infra.DeepSeekLaunchConfig.model_validate(_mapping(root, mode, **kwargs))


def test_exact_pin_and_canonical_workload_are_shared_by_both_modes(tmp_path: Path) -> None:
    direct = _config(tmp_path / "shared")
    fau = _config(tmp_path / "shared", "fau_slurm")

    argv = infra.render_vllm_argv(direct)
    script = infra.render_fau_slurm_script(fau)

    assert argv[:3] == ["vllm", "serve", MODEL_ID]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert "--trust-remote-code" in argv
    assert argv[argv.index("--tokenizer-mode") + 1] == "deepseek_v4"
    assert argv[argv.index("--reasoning-parser") + 1] == "deepseek_v4"
    assert argv[argv.index("--tool-call-parser") + 1] == "deepseek_v4"
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8"
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "1"
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"
    assert shlex.join(argv) in script
    assert "#SBATCH --account=project-123" in script
    assert "module load cuda/12.8" in script
    assert "source /opt/venvs/deepseek/bin/activate" in script
    assert "mkdir -p" in script
    assert str(fau.paths.run_dir) in script
    assert str(fau.paths.cache_dir) in script
    assert str(fau.paths.output_dir) in script


def test_yaml_loader_and_nested_config_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "deepseek.yaml"
    path.write_text(yaml.safe_dump(_mapping(tmp_path)), encoding="utf-8")
    loaded = infra.load_deepseek_config(path)
    assert loaded.model.id == MODEL_ID
    assert loaded.model.revision == REVISION
    assert loaded.paths.model_cache_dir == tmp_path / "model-cache"

    unknown = _mapping(tmp_path)
    unknown["serving"]["unexpected"] = True
    with pytest.raises(ValidationError):
        infra.DeepSeekLaunchConfig.model_validate(unknown)

    coerced = _mapping(tmp_path)
    coerced["serving"]["tensor_parallel_size"] = "8"
    with pytest.raises(ValidationError):
        infra.DeepSeekLaunchConfig.model_validate(coerced)


def test_checked_in_examples_share_model_serving_paths_and_are_preview_only(
    monkeypatch,
) -> None:
    direct = infra.load_deepseek_config(ROOT / "configs/deepseek-v4-flash-direct.yaml")
    fau = infra.load_deepseek_config(ROOT / "configs/deepseek-v4-flash-fau-slurm.yaml")

    assert direct.mode == "direct"
    assert fau.mode == "fau_slurm"
    assert direct.model.model_dump() == fau.model.model_dump()
    assert direct.serving.model_dump() == fau.serving.model_dump()
    assert direct.paths.model_dump() == fau.paths.model_dump()
    assert infra.render_vllm_argv(direct) == infra.render_vllm_argv(fau)

    monkeypatch.setattr(infra.subprocess, "run", lambda *args, **kwargs: pytest.fail("ran"))
    for config in (direct, fau):
        checked = infra.deepseek_launch(config, check=True)
        assert checked["preflight"]["valid"] is True
        assert checked["preflight"]["warnings"]
    with pytest.raises(infra.DeepSeekInfrastructureError, match="placeholder"):
        infra.deepseek_launch(direct, execute=True)
    with pytest.raises(infra.DeepSeekInfrastructureError, match="placeholder"):
        infra.deepseek_launch(fau, submit=True)


@pytest.mark.parametrize("revision", ["a" * 39, "g" * 40, "0" * 40, "b" * 40])
def test_revision_must_be_the_exact_non_placeholder_pin(tmp_path: Path, revision: str) -> None:
    values = _mapping(tmp_path)
    values["model"]["revision"] = revision
    if (
        len(revision) != 40
        or revision == "0" * 40
        or not all(char in "0123456789abcdef" for char in revision)
    ):
        with pytest.raises(ValidationError):
            infra.DeepSeekLaunchConfig.model_validate(values)
        return
    config = infra.DeepSeekLaunchConfig.model_validate(values)
    report = infra.preflight_deepseek_config(config)
    assert report["valid"] is False
    with pytest.raises(infra.DeepSeekInfrastructureError, match="approved official 0731 pin"):
        infra.render_vllm_argv(config)


def test_protected_vllm_overrides_are_rejected(tmp_path: Path) -> None:
    values = _mapping(tmp_path)
    values["serving"]["extra_args"] = ["--revision=other-sha"]
    config = infra.DeepSeekLaunchConfig.model_validate(values)
    with pytest.raises(infra.ProtectedVLLMArgumentError, match="--revision"):
        infra.render_vllm_argv(config)
    with pytest.raises(infra.ProtectedVLLMArgumentError, match="--revision"):
        infra.deepseek_launch(config, check=True)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("paths", "run_dir", "/tmp/bad\npath"),
        ("serving", "host", "127.0.0.1\x00bad"),
        ("fau_slurm", "account", "project\n--partition=other"),
        ("fau_slurm", "module_commands", ["module load cuda; touch /tmp/pwned"]),
        ("fau_slurm", "environment_activation", "source /tmp/venv; touch /tmp/pwned"),
    ],
)
def test_newline_control_and_shell_injection_are_rejected(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    values = _mapping(tmp_path, "fau_slurm" if section == "fau_slurm" else "direct")
    values[section][field] = value
    with pytest.raises(ValidationError):
        infra.DeepSeekLaunchConfig.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account", "project with spaces"),
        ("partition", "gpu partition"),
        ("constraint", "a100 80"),
    ],
)
def test_fau_site_directives_are_single_slurm_tokens(
    tmp_path: Path, field: str, value: str
) -> None:
    values = _mapping(tmp_path, "fau_slurm")
    values["fau_slurm"][field] = value
    with pytest.raises(ValidationError, match="one Slurm-safe token"):
        infra.DeepSeekLaunchConfig.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("module_commands", ["load cuda/12.8"], "must begin with 'module '"),
        (
            "environment_activation",
            "activate /opt/venvs/deepseek",
            "must begin with 'source ' or 'conda activate '",
        ),
    ],
)
def test_fau_wrapper_commands_have_explicit_shell_intent(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    values = _mapping(tmp_path, "fau_slurm")
    values["fau_slurm"][field] = value
    with pytest.raises(ValidationError, match=message):
        infra.DeepSeekLaunchConfig.model_validate(values)


def test_default_preview_and_check_never_call_a_process(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple] = []

    def fail_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("preview/check must not execute a subprocess")

    monkeypatch.setattr(infra.subprocess, "run", fail_run)
    direct = _config(tmp_path / "direct")
    fau = _config(tmp_path / "fau", "fau_slurm")
    preview = infra.deepseek_launch(direct)
    check = infra.deepseek_launch(fau, check=True)
    assert preview["action"] == "preview"
    assert preview["mode"] == "direct"
    assert check["action"] == "check"
    assert check["preflight"]["valid"] is True
    assert calls == []


def test_explicit_actions_must_match_transport(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(infra.subprocess, "run", lambda *args, **kwargs: None)
    direct = _config(tmp_path / "direct")
    fau = _config(tmp_path / "fau", "fau_slurm")
    with pytest.raises(infra.DeepSeekModeError):
        infra.deepseek_launch(direct, submit=True)
    with pytest.raises(infra.DeepSeekModeError):
        infra.deepseek_launch(fau, execute=True)
    with pytest.raises(infra.DeepSeekInfrastructureError, match="mutually exclusive"):
        infra.deepseek_launch(direct, check=True, execute=True)


@pytest.mark.parametrize("field", ["nodes", "tasks_per_node"])
def test_fau_wrapper_rejects_unsupported_multi_node_or_multi_task_layout(
    tmp_path: Path, field: str
) -> None:
    values = _mapping(tmp_path, "fau_slurm")
    values["fau_slurm"][field] = 2
    with pytest.raises(ValidationError, match="Input should be 1"):
        infra.DeepSeekLaunchConfig.model_validate(values)


def test_direct_execute_uses_argv_without_a_shell_and_creates_run_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(infra.subprocess, "run", fake_run)
    config = _config(tmp_path / "direct")
    result = infra.deepseek_launch(config, execute=True)

    assert result["action"] == "execute"
    assert result["returncode"] == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == infra.render_vllm_argv(config)
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == str(config.direct.working_directory)
    for directory in (
        config.paths.model_cache_dir,
        config.paths.run_dir,
        config.paths.output_dir,
        config.paths.cache_dir,
        config.paths.log_dir,
    ):
        assert directory.is_dir()


def test_direct_directory_failure_is_clean_and_does_not_start_vllm(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path / "direct")
    mkdir_calls: list[Path] = []
    process_calls: list[tuple] = []

    def fail_mkdir(path: Path, *args, **kwargs):
        mkdir_calls.append(path)
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    monkeypatch.setattr(
        infra.subprocess,
        "run",
        lambda *args, **kwargs: process_calls.append((args, kwargs)),
    )

    with pytest.raises(
        infra.DeepSeekInfrastructureError, match="prepare direct launch directories"
    ):
        infra.deepseek_launch(config, execute=True)
    assert len(mkdir_calls) == 1
    assert process_calls == []


def test_direct_process_failure_is_clean_and_not_retried(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path / "direct")
    process_calls: list[tuple] = []

    def fail_process(*args, **kwargs):
        process_calls.append((args, kwargs))
        raise OSError("vllm not found")

    monkeypatch.setattr(infra.subprocess, "run", fail_process)

    with pytest.raises(infra.DeepSeekInfrastructureError, match="start direct vLLM process"):
        infra.deepseek_launch(config, execute=True)
    assert len(process_calls) == 1


def test_fau_submit_uses_only_explicit_sbatch_and_same_workload(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123\n", stderr="")

    monkeypatch.setattr(infra.subprocess, "run", fake_run)
    config = _config(tmp_path / "fau", "fau_slurm")
    config.fau_slurm.output_path.parent.mkdir(parents=True)
    result = infra.deepseek_launch(config, submit=True)

    assert result["action"] == "submit"
    assert result["returncode"] == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (["sbatch"],)
    assert kwargs["shell"] is False
    assert kwargs["text"] is True
    assert kwargs["input"] == infra.render_fau_slurm_script(config)
    assert shlex.join(infra.render_vllm_argv(config)) in kwargs["input"]
    assert not config.paths.run_dir.exists()


def test_fau_submit_failure_is_clean_and_not_retried(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path / "fau", "fau_slurm")
    config.fau_slurm.output_path.parent.mkdir(parents=True)
    submit_calls: list[tuple] = []

    def fail_submit(*args, **kwargs):
        submit_calls.append((args, kwargs))
        raise OSError("sbatch not found")

    monkeypatch.setattr(infra.subprocess, "run", fail_submit)

    with pytest.raises(infra.DeepSeekInfrastructureError, match="invoke sbatch"):
        infra.deepseek_launch(config, submit=True)
    assert len(submit_calls) == 1


def test_fau_submit_requires_log_parents_but_preview_only_warns(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path / "fau", "fau_slurm")
    calls: list[tuple] = []
    monkeypatch.setattr(
        infra.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    preview = infra.deepseek_launch(config)
    assert preview["preflight"]["valid"] is True
    assert any("output_path.parent" in warning for warning in preview["preflight"]["warnings"])
    with pytest.raises(infra.DeepSeekInfrastructureError, match="output_path.parent"):
        infra.deepseek_launch(config, submit=True)
    assert calls == []


def test_fau_site_placeholders_block_submit_after_log_parents_are_ready(
    tmp_path: Path, monkeypatch
) -> None:
    values = _mapping(tmp_path / "fau", "fau_slurm")
    values["fau_slurm"]["account"] = "YOUR_FAU_ACCOUNT"
    values["fau_slurm"]["partition"] = "YOUR_FAU_PARTITION"
    values["fau_slurm"]["constraint"] = "YOUR_FAU_CONSTRAINT"
    config = infra.DeepSeekLaunchConfig.model_validate(values)
    config.fau_slurm.output_path.parent.mkdir(parents=True)
    calls: list[tuple] = []
    monkeypatch.setattr(
        infra.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    checked = infra.deepseek_launch(config, check=True)
    assert any("fau_slurm.account" in warning for warning in checked["preflight"]["warnings"])
    with pytest.raises(infra.DeepSeekInfrastructureError, match="fau_slurm.account"):
        infra.deepseek_launch(config, submit=True)
    assert calls == []


def test_example_run_id_is_previewable_but_blocks_direct_execute(
    tmp_path: Path, monkeypatch
) -> None:
    values = _mapping(tmp_path / "direct")
    values["run_id"] = "deepseek-v4-flash-example"
    config = infra.DeepSeekLaunchConfig.model_validate(values)
    calls: list[tuple] = []
    monkeypatch.setattr(
        infra.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    checked = infra.deepseek_launch(config, check=True)
    assert any("run_id" in warning for warning in checked["preflight"]["warnings"])
    with pytest.raises(infra.DeepSeekInfrastructureError, match="run_id"):
        infra.deepseek_launch(config, execute=True)
    assert calls == []


@pytest.mark.parametrize("mode, action", [("direct", "execute"), ("fau_slurm", "submit")])
def test_placeholders_are_previewable_but_block_state_changes(
    tmp_path: Path, monkeypatch, mode: str, action: str
) -> None:
    called = False

    def fail_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("placeholder configs must fail before process launch")

    monkeypatch.setattr(infra.subprocess, "run", fail_run)
    config = _config(tmp_path / mode, mode, placeholders=True)
    preview = infra.deepseek_launch(config)
    assert preview["action"] == "preview"
    assert preview["preflight"]["warnings"]
    with pytest.raises(infra.DeepSeekInfrastructureError, match="placeholder"):
        infra.deepseek_launch(config, **{action: True})
    assert called is False


def test_cli_preview_is_json_and_has_no_subprocess_side_effect(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "direct.yaml"
    config_path.write_text(yaml.safe_dump(_mapping(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(infra.subprocess, "run", lambda *args, **kwargs: pytest.fail("ran"))
    monkeypatch.setattr(sys, "argv", ["reverse-reap", "deepseek-launch", str(config_path)])

    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "preview"
    assert output["mode"] == "direct"


def test_cli_action_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["deepseek-launch", "config.yaml", "--execute", "--submit"])
