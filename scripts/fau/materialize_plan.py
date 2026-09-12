#!/usr/bin/env python3
"""Materialize an existing governed plan for an FAU filesystem/model path."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from reverse_reap.controller import ExecutionPlan


def materialize(
    source: Path,
    destination: Path,
    *,
    config: Path,
    model_dir: Path,
    thinking_config: Path | None = None,
    through_task: str | None = None,
) -> None:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if through_task is not None:
        task_ids = [task["task_id"] for task in payload["tasks"]]
        if through_task not in task_ids:
            raise ValueError(f"--through-task is not in source plan: {through_task}")
        payload["tasks"] = payload["tasks"][: task_ids.index(through_task) + 1]

    def replace(value: object) -> object:
        if isinstance(value, str):
            if value in {"/models/qwen", "__MODEL_DIR__"}:
                return str(model_dir)
            if value.endswith("configs/pinned-thinking-3090-bf16.yaml"):
                if thinking_config is None:
                    raise ValueError("source plan requires --thinking-config")
                return str(thinking_config)
            if value == "__EXPERIMENT_CONFIG__" or value.endswith("configs/pinned-3090-bf16.yaml"):
                return str(config)
            return (
                value.replace("four-RTX-3090", "eight-RTX-PRO-6000")
                .replace("top-8", "top-10")
                .replace("all 40 layers", "all 48 layers")
            )
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    rendered = replace(payload)
    for task in rendered["tasks"]:
        if task["task_id"] == "gpu-preflight":
            command = task["command"]
            if "--profile" not in command:
                command.extend(["--profile", "alex-8x-pro6000"])
    ExecutionPlan.model_validate(rendered)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--thinking-config", type=Path)
    parser.add_argument("--through-task")
    args = parser.parse_args()
    materialize(
        args.source,
        args.destination,
        config=args.config,
        model_dir=args.model_dir,
        thinking_config=args.thinking_config,
        through_task=args.through_task,
    )


if __name__ == "__main__":
    main()
