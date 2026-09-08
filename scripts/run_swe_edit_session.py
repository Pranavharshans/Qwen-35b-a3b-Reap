"""Create or advance one CPU-side deterministic SWE repository-editing session.

This driver never calls a model or the network. A generation loop supplies one
tokenizer-counted JSON action at a time through ``step``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reverse_reap.swe_edit import (
    EditPolicy,
    EditSession,
    EditSessionError,
    canonical_json,
    protocol_prompt,
    sha256,
    write_source_binding,
)


def _policy(path: Path) -> EditPolicy:
    return EditPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start")
    start.add_argument("--session-dir", type=Path, required=True)
    start.add_argument("--task", type=Path, required=True)
    start.add_argument("--policy", type=Path, required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--config-sha256", required=True)
    start.add_argument("--model-revision", required=True)
    start.add_argument("--tokenizer-sha256", required=True)

    step = sub.add_parser("step")
    step.add_argument("--session-dir", type=Path, required=True)
    step.add_argument("--policy", type=Path, required=True)
    step.add_argument("--expected-binding", type=Path, required=True)
    step.add_argument("--response", type=Path, required=True)
    step.add_argument("--input-tokens", type=int, required=True)
    step.add_argument("--output-tokens", type=int, required=True)

    args = parser.parse_args()
    try:
        policy = _policy(args.policy)
        if args.command == "start":
            task = json.loads(args.task.read_text(encoding="utf-8"))
            write_source_binding(args.session_dir, Path(task["repo_dir"]))
            # Create requires an otherwise empty root; preserve the binding, then move it back.
            source_binding = (args.session_dir / "source-path.txt").read_bytes()
            (args.session_dir / "source-path.txt").unlink()
            session = EditSession.create(
                args.session_dir, task, policy, run_id=args.run_id,
                config_sha256=args.config_sha256, model_revision=args.model_revision,
                tokenizer_sha256=args.tokenizer_sha256,
            )
            (args.session_dir / "source-path.txt").write_bytes(source_binding)
            prompt = protocol_prompt(task, policy)
            (args.session_dir / "initial-prompt.txt").write_text(prompt, encoding="utf-8")
            binding = session.state["binding"]
            (args.session_dir / "binding.json").write_bytes(canonical_json(binding))
            print(json.dumps({"status": "RUNNING", "binding": binding,
                              "prompt_sha256": sha256(prompt.encode())}, sort_keys=True))
            return 0
        expected = json.loads(args.expected_binding.read_text(encoding="utf-8"))
        session = EditSession.open(args.session_dir, policy, expected)
        response = args.response.read_text(encoding="utf-8")
        result = session.process_turn(
            response, input_tokens=args.input_tokens, output_tokens=args.output_tokens,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if "error" not in result else 2
    except (EditSessionError, KeyError, json.JSONDecodeError) as exc:
        print(f"fail-closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
