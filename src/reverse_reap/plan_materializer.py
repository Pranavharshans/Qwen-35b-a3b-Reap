"""Render a governed execution plan for a concrete donor and filesystem.

The source plans intentionally use Qwen-era placeholders because the same
controller is shared across donors.  This module performs the bounded,
deterministic substitution at launch time and validates the resulting plan;
it does not alter the source plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from reverse_reap.controller import ExecutionPlan
from reverse_reap.donors import (
    GLM53_BF16_MODEL_ID,
    QWEN38_MODEL_ID,
    donor_contract,
)

ExecutionMode = Literal["direct", "fau-slurm"]


def _configured_identity(config: Path) -> tuple[str, str]:
    """Read the donor identity without requiring a local config path."""
    configured_model_id = QWEN38_MODEL_ID
    configured_revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    if config.is_file():
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise ValueError(f"config has no model object: {config}")
        configured_model_id = str(payload["model"]["id"])
        configured_revision = str(payload["model"]["revision"])
    return configured_model_id, configured_revision


def _replace_text(
    value: str,
    *,
    model_id: str,
    revision: str,
    model_dir: Path,
    config: Path,
    thinking_config: Path | None,
    run_root: Path | None,
    execution_mode: ExecutionMode,
) -> str:
    if value in {"/models/qwen", "__MODEL_DIR__"}:
        return str(model_dir)
    if value.endswith("configs/pinned-thinking-3090-bf16.yaml"):
        if thinking_config is None:
            raise ValueError("source plan requires --thinking-config")
        return str(thinking_config)
    if value == "__EXPERIMENT_CONFIG__" or value.endswith("configs/pinned-3090-bf16.yaml"):
        return str(config)
    rendered = value
    rendered = rendered.replace("Qwen/Qwen3.8-Flash-Next", model_id)
    rendered = rendered.replace("de4b8e4d43b917e7706784d8bb445c9af86a3540", revision)
    if model_id == QWEN38_MODEL_ID:
        rendered = rendered.replace("four-RTX-3090", "eight-RTX-PRO-6000")
        rendered = rendered.replace("top-8", "top-10")
        rendered = rendered.replace("all 40 layers", "all 48 layers")
    elif model_id == GLM53_BF16_MODEL_ID:
        rendered = rendered.replace(
            "four-RTX-3090",
            "GLM direct aggregate-VRAM GPU profile"
            if execution_mode == "direct"
            else "eight-RTX-PRO-6000",
        )
        rendered = rendered.replace(
            "all 40 layers",
            "45 decoder layers with 42 sparse MoE layers (absolute layers 3-44)",
        )
        rendered = rendered.replace("Qwen-tokenizer", "GLM tokenizer")
    if run_root is not None:
        rendered = rendered.replace("runs/smoke/${RUN_ID}", str(run_root / "outputs"))
    return rendered


def render_plan(
    source: Path,
    *,
    config: Path,
    model_dir: Path,
    thinking_config: Path | None = None,
    through_task: str | None = None,
    execution_mode: ExecutionMode = "fau-slurm",
    run_root: Path | None = None,
) -> dict[str, object]:
    """Return a validated materialized plan payload without writing it."""
    if execution_mode not in {"direct", "fau-slurm"}:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError(f"source plan has no tasks list: {source}")
    configured_model_id, configured_revision = _configured_identity(config)
    if through_task is not None:
        task_ids = [task["task_id"] for task in payload["tasks"]]
        if through_task not in task_ids:
            raise ValueError(f"--through-task is not in source plan: {through_task}")
        payload["tasks"] = payload["tasks"][: task_ids.index(through_task) + 1]

    def replace(value: object) -> object:
        if isinstance(value, str):
            return _replace_text(
                value,
                model_id=configured_model_id,
                revision=configured_revision,
                model_dir=model_dir,
                config=config,
                thinking_config=thinking_config,
                run_root=run_root,
                execution_mode=execution_mode,
            )
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    rendered = replace(payload)
    assert isinstance(rendered, dict)
    profile = "alex-8x-pro6000" if execution_mode == "fau-slurm" else "glm53-direct"
    for task in rendered["tasks"]:
        if task["task_id"] != "gpu-preflight":
            continue
        if configured_model_id == GLM53_BF16_MODEL_ID:
            task["objective"] = f"{task['objective']} Donor: {configured_model_id}."
        command = task["command"]
        if "--profile" in command:
            profile_index = command.index("--profile") + 1
            if profile_index >= len(command):
                raise ValueError("gpu-preflight has an incomplete --profile option")
            command[profile_index] = profile
        else:
            command.extend(["--profile", profile])
    ExecutionPlan.model_validate(rendered)
    # The contract lookup is intentional: a future donor must be registered
    # before this shared materializer is allowed to render it.
    donor_contract(configured_model_id)
    return rendered


def materialize(
    source: Path,
    destination: Path,
    *,
    config: Path,
    model_dir: Path,
    thinking_config: Path | None = None,
    through_task: str | None = None,
    execution_mode: ExecutionMode = "fau-slurm",
    run_root: Path | None = None,
) -> None:
    """Render and atomically-ish write a plan, preserving the source file."""
    rendered = render_plan(
        source,
        config=config,
        model_dir=model_dir,
        thinking_config=thinking_config,
        through_task=through_task,
        execution_mode=execution_mode,
        run_root=run_root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
