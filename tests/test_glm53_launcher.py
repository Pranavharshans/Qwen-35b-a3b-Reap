import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "launch_glm53.py"


def module():
    spec = importlib.util.spec_from_file_location("launch_glm53", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


def valid_inputs(tmp_path):
    return {
        "config_path": ROOT / "configs" / "smoke-glm53-flash-bf16.yaml",
        "plan_path": ROOT / "configs" / "execution-plan-smoke.yaml",
        "model_dir": tmp_path,
        "state_root": tmp_path / "state-root",
    }


def test_direct_preview_renders_glm_plan_without_writing_or_scheduler_lookup(tmp_path, capsys):
    launcher = module()
    inputs = valid_inputs(tmp_path)
    context = launcher.prepare_launch(mode="direct", **inputs)
    command = launcher.render_command(context)

    assert command[:2] == ["reverse-reap", "run-all"]
    assert str(context.materialized_plan) in command
    assert "/models/qwen" not in " ".join(command).lower()
    assert not context.state_dir.exists()
    assert not context.materialized_plan.exists()

    assert launcher.main(
        [
            "--mode",
            "direct",
            "--config",
            str(inputs["config_path"]),
            "--plan",
            str(inputs["plan_path"]),
            "--model-dir",
            str(inputs["model_dir"]),
            "--state-root",
            str(inputs["state_root"]),
        ]
    ) == 0
    output = capsys.readouterr().out.lower()
    assert "action: preview" in output
    assert "sbatch" not in output
    assert not inputs["state_root"].exists()


def test_fau_preview_has_no_submit_flag_or_scheduler_call(tmp_path, capsys):
    launcher = module()
    inputs = valid_inputs(tmp_path)
    context = launcher.prepare_launch(mode="fau-slurm", **inputs)
    command = launcher.render_command(context)

    assert command[0].endswith("scripts/fau/submit_reverse_reap.sh")
    assert "--submit" not in command
    assert "--job-name" in command
    assert "--run-root" in command
    assert launcher.main(
        [
            "--fau-slurm",
            "--config",
            str(inputs["config_path"]),
            "--plan",
            str(inputs["plan_path"]),
            "--model-dir",
            str(inputs["model_dir"]),
            "--state-root",
            str(inputs["state_root"]),
        ]
    ) == 0
    output = capsys.readouterr().out.lower()
    assert "action: preview" in output
    assert "--submit" not in output
    assert "sbatch" not in output
    assert not inputs["state_root"].exists()


def test_execute_mapping_uses_fake_commands_without_gpu_or_slurm(tmp_path):
    launcher = module()
    marker = tmp_path / "invocation.json"
    fake = tmp_path / "fake-runner"
    fake.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {marker}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    inputs = valid_inputs(tmp_path)
    context = launcher.prepare_launch(
        mode="direct",
        **inputs,
        execute=True,
        reverse_reap_command=str(fake),
        run_id="glm53-test-direct",
    )
    assert launcher.execute_launch(context) == 0
    invoked = marker.read_text(encoding="utf-8").splitlines()
    assert invoked[0] == "run-all"
    assert str(context.materialized_plan) in invoked
    assert context.state_dir == tmp_path / "state-root" / "glm53-test-direct" / "state"
    assert (context.run_root / "config" / "glm53-config.yaml").is_file()

    fake_submit = tmp_path / "fake-submit"
    submit_marker = tmp_path / "submit-invocation.json"
    fake_submit.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {submit_marker}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_submit.chmod(0o755)
    fau_inputs = valid_inputs(tmp_path / "fau-model")
    fau_inputs["model_dir"].mkdir()
    fau_context = launcher.prepare_launch(
        mode="fau-slurm",
        **fau_inputs,
        execute=True,
        fau_submitter=fake_submit,
        run_id="glm53-test-fau",
    )
    assert launcher.execute_launch(fau_context) == 0
    submit_args = submit_marker.read_text(encoding="utf-8").splitlines()
    assert submit_args[0] == "--submit"
    assert "--job-name" in submit_args
    assert "--run-root" in submit_args


def test_wrong_model_is_rejected_before_rendering(tmp_path):
    launcher = module()
    wrong = tmp_path / "qwen.yaml"
    wrong.write_text(
        (ROOT / "configs" / "pinned-3090-bf16.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    inputs = valid_inputs(tmp_path)
    inputs["config_path"] = wrong
    with pytest.raises(launcher.LauncherError, match="model.id must be exactly"):
        launcher.prepare_launch(mode="direct", **inputs)


def test_missing_model_path_is_rejected(tmp_path):
    launcher = module()
    inputs = valid_inputs(tmp_path)
    inputs["model_dir"] = tmp_path / "does-not-exist"
    with pytest.raises(launcher.LauncherError, match="model directory"):
        launcher.prepare_launch(mode="direct", **inputs)


def test_execute_requires_a_reviewed_pinned_run_id(tmp_path):
    launcher = module()
    with pytest.raises(launcher.LauncherError, match="execute requires --run-id"):
        launcher.prepare_launch(mode="direct", **valid_inputs(tmp_path), execute=True)


def test_glm_plan_materialization_rewrites_identity_paths_and_profile(tmp_path):
    from reverse_reap.plan_materializer import render_plan

    inputs = valid_inputs(tmp_path)
    rendered = render_plan(
        inputs["plan_path"],
        config=inputs["config_path"],
        model_dir=inputs["model_dir"] / "glm-weights",
        execution_mode="direct",
        run_root=tmp_path / "run",
    )
    serialized = yaml.safe_dump(rendered)
    assert "/models/qwen" not in serialized.lower()
    assert "configs/pinned-3090-bf16.yaml" not in serialized
    assert "top-8" in serialized
    assert "42 sparse MoE layers (absolute layers 3-44)" in serialized
    preflight = next(task for task in rendered["tasks"] if task["task_id"] == "gpu-preflight")
    assert preflight["command"][-2:] == ["--profile", "glm53-direct"]
