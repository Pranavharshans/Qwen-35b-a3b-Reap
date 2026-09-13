#!/usr/bin/env python3
"""Materialize an existing governed plan for a concrete filesystem/model path."""

from __future__ import annotations

import argparse
from pathlib import Path

from reverse_reap.plan_materializer import materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--thinking-config", type=Path)
    parser.add_argument("--through-task")
    parser.add_argument("--execution-mode", choices=("direct", "fau-slurm"), default="fau-slurm")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    materialize(
        args.source,
        args.destination,
        config=args.config,
        model_dir=args.model_dir,
        thinking_config=args.thinking_config,
        through_task=args.through_task,
        execution_mode=args.execution_mode,
        run_root=args.run_root,
    )


if __name__ == "__main__":
    main()
