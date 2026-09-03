"""Deterministic command-line entrypoint for Reverse-REAP."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from reverse_reap.config import load_config
from reverse_reap.controller import run_next
from reverse_reap.runtime import capture_manifest, probe_instrumentation


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def validate_config(path: Path) -> int:
    config = load_config(path)
    output = {
        "valid": True,
        "config_sha256": config.fingerprint(),
        "resolved_run_id": config.run_id or config.resolve_run_id(git_sha()),
        "thinking_enabled": config.runtime.enable_thinking,
        "chat_template_kwargs": {"enable_thinking": config.runtime.enable_thinking},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reverse-reap")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    run = subparsers.add_parser("run-next")
    run.add_argument("config", type=Path)
    run.add_argument("plan", type=Path)
    run.add_argument("state_dir", type=Path)
    run.add_argument("--heartbeat-seconds", type=float, default=30)
    capture = subparsers.add_parser("capture")
    capture.add_argument("config", type=Path)
    capture.add_argument("model_path", type=Path)
    capture.add_argument("manifest", type=Path)
    capture.add_argument("destination", type=Path)
    capture.add_argument("--split", default="calibration")
    capture.add_argument("--limit", type=int)
    probe = subparsers.add_parser("probe")
    probe.add_argument("config", type=Path)
    probe.add_argument("model_path", type=Path)
    probe.add_argument(
        "--prompt", default="Write a Python function that returns the sum of two integers."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-config":
        return validate_config(args.config)
    if args.command == "run-next":
        config = load_config(args.config)
        resolved_run_id = config.run_id or config.resolve_run_id(git_sha())
        output = run_next(
            args.plan,
            config,
            args.state_dir,
            run_id=resolved_run_id,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if output["status"] == "COMPLETE" else 2
    if args.command == "capture":
        output = capture_manifest(
            args.model_path,
            args.manifest,
            args.destination,
            load_config(args.config),
            split=args.split,
            limit=args.limit,
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "probe":
        output = probe_instrumentation(args.model_path, load_config(args.config), args.prompt)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if output["passed"] else 3
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
