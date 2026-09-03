"""Deterministic command-line entrypoint for Reverse-REAP."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from reverse_reap.config import load_config


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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-config":
        return validate_config(args.config)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

