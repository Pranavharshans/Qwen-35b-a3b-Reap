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


def test_slurm_script_matches_fau_batch_contract():
    body = Path("scripts/fau/reverse_reap.slurm").read_text()
    assert body.startswith("#!/bin/bash -l")
    assert "#SBATCH --partition=rtxpro6k" in body
    assert "#SBATCH --gres=gpu:rtxpro6k:8" in body
    assert "#SBATCH --export=NONE" in body
    assert "unset SLURM_EXPORT_ENV" in body
    assert "module load cuda/12.8" in body
    assert "srun uv run --frozen --no-sync reverse-reap run-all" in body
