import importlib.util
from pathlib import Path

import yaml


def module():
    path = Path(__file__).parents[1] / "scripts" / "fau" / "materialize_plan.py"
    spec = importlib.util.spec_from_file_location("materialize_plan", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_materializes_fau_plan_without_submitting(tmp_path):
    destination = tmp_path / "plan.yaml"
    module().materialize(
        Path("configs/execution-plan-smoke.yaml"),
        destination,
        config=Path("/cluster/repo/configs/q38.yaml"),
        model_dir=Path("/cluster/models/q38"),
    )
    payload = yaml.safe_load(destination.read_text())
    preflight = next(task for task in payload["tasks"] if task["task_id"] == "gpu-preflight")
    assert preflight["command"][-2:] == ["--profile", "alex-8x-pro6000"]
    commands = [part for task in payload["tasks"] for part in task["command"]]
    assert "/models/qwen" not in commands
    assert "/cluster/models/q38" in commands
    assert "configs/pinned-3090-bf16.yaml" not in commands
    assert "/cluster/repo/configs/q38.yaml" in commands


def test_materializes_full_qwen38_plan_with_separate_thinking_config(tmp_path):
    destination = tmp_path / "full-plan.yaml"
    module().materialize(
        Path("configs/execution-plan-v0.yaml"),
        destination,
        config=Path("/cluster/repo/configs/q38-direct.yaml"),
        thinking_config=Path("/cluster/repo/configs/q38-thinking.yaml"),
        model_dir=Path("/cluster/models/q38"),
    )
    rendered = destination.read_text()
    assert "configs/pinned-3090-bf16.yaml" not in rendered
    assert "configs/pinned-thinking-3090-bf16.yaml" not in rendered
    assert "/cluster/repo/configs/q38-direct.yaml" in rendered
    assert "/cluster/repo/configs/q38-thinking.yaml" in rendered


def test_materializes_generation_pass_only_through_candidate_analysis(tmp_path):
    destination = tmp_path / "generation-plan.yaml"
    module().materialize(
        Path("configs/execution-plan-v0.yaml"),
        destination,
        config=Path("/cluster/repo/configs/q38-direct.yaml"),
        model_dir=Path("/cluster/models/q38"),
        through_task="candidate-analysis",
    )
    task_ids = [task["task_id"] for task in yaml.safe_load(destination.read_text())["tasks"]]
    assert task_ids[-1] == "candidate-analysis"
    assert "baseline-validation-a" not in task_ids
    assert "causal-report" not in task_ids


def test_materializes_qwen38_bridge_capture_placeholders(tmp_path):
    destination = tmp_path / "capture-plan.yaml"
    module().materialize(
        Path("configs/execution-plan-qwen38-bridge-capture.yaml"),
        destination,
        config=Path("/cluster/repo/configs/q38-capture.yaml"),
        model_dir=Path("/cluster/models/q38"),
    )
    rendered = destination.read_text()
    assert "__EXPERIMENT_CONFIG__" not in rendered
    assert "__MODEL_DIR__" not in rendered
    assert "/cluster/repo/configs/q38-capture.yaml" in rendered
    assert "/cluster/models/q38" in rendered


def test_slurm_script_matches_fau_batch_contract():
    body = Path("scripts/fau/reverse_reap.slurm").read_text()
    assert body.startswith("#!/bin/bash -l")
    assert "#SBATCH --partition=rtxpro6k" in body
    assert "#SBATCH --gres=gpu:rtxpro6k:8" in body
    assert "#SBATCH --export=NONE" in body
    assert "unset SLURM_EXPORT_ENV" in body
    assert "module load cuda/12.8" in body
    assert "srun uv run --frozen --no-sync reverse-reap run-all" in body
