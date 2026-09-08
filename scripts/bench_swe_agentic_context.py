"""Model-free cumulative-context benchmark for agentic SWE sessions (CPU only).

Replays a fixed, realistic tool script per frozen v3 task against real
exact-base trees through the validated scaffold and measures every turn with
the exact donor tokenizer. No model is called: canned actions stand in for
model turns, so the measured prompt/response sizes are a lower bound on a
live session (a live model may take more turns or longer reads).

For each turn the report records prompt tokens, cumulative input/output
tokens, tool calls used, the largest tool response, the expected KV-cache
footprint, and the stop reason. Batch memory projections assume BF16 Qwen
weights (~70 GiB) plus 2*40 layers*2048 hidden*2 bytes = 327,680 bytes of KV
cache per context token per sequence, with 10% prefill headroom.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

KV_BYTES_PER_TOKEN = 2 * 40 * 2048 * 2
WEIGHTS_GIB = 70.0
VRAM_CEILING_GIB = 0.92 * 96
HEADROOM = 1.10


def expected_gib(cumulative_tokens: int, batch: int) -> float:
    kv_gib = cumulative_tokens * batch * KV_BYTES_PER_TOKEN / 1024**3
    return WEIGHTS_GIB + kv_gib * HEADROOM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True,
                        help="v4 context.json supplying real chunk paths/ranges")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, "src")
    from transformers import AutoTokenizer

    from reverse_reap.swe_agentic import (
        SessionDriver,
        count_input_tokens,
        count_output_tokens,
        render_tool_result,
    )
    from reverse_reap.swe_edit import EditPolicy, EditSession, write_source_binding

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True,
                                              trust_remote_code=False)
    policy = EditPolicy.from_dict(json.loads(args.policy.read_text()))
    tasks = json.loads(args.tasks.read_text())
    contexts = {c["sample_id"]: c for c in json.loads(args.context.read_text())["contexts"]}
    mirror_names = {"django/django": "django_django",
                    "matplotlib/matplotlib": "matplotlib_matplotlib",
                    "sympy/sympy": "sympy_sympy"}

    report: dict = {"samples": [], "batches": {}}
    worst_cumulative = 0
    for task in tasks:
        local = dict(task, repo_dir=str(
            args.mirror_root / mirror_names[task["repo"]]))
        chunks = contexts[task["sample_id"]]["chunks"]
        first, second = chunks[0], chunks[-1]
        # Fixed realistic script: list, search, read, read, trivial edit, finish.
        # The edit appends a benchmark comment to one exact line read above.
        script = [
            ("list", {"action": "list", "path": str(Path(first["path"]).parent)}),
            ("search", {"action": "search", "query": "def",
                        "path": str(Path(first["path"]).parent)}),
            ("read", {"action": "read", "path": first["path"],
                      "start_line": first["start_line"],
                      "end_line": min(first["end_line"],
                                      first["start_line"] + 39)}),
            ("read", {"action": "read", "path": second["path"],
                      "start_line": second["start_line"],
                      "end_line": min(second["end_line"],
                                      second["start_line"] + 39)}),
        ]
        root = Path(tempfile.mkdtemp(prefix="bench-agentic-"))
        write_source_binding(root, Path(local["repo_dir"]))
        binding = (root / "source-path.txt").read_bytes()
        (root / "source-path.txt").unlink()
        session = EditSession.create(
            root, local, policy, run_id="bench", config_sha256="0" * 64,
            model_revision="5" * 40, tokenizer_sha256="0" * 64,
        )
        (root / "source-path.txt").write_bytes(binding)
        driver = SessionDriver(session, policy, tokenizer, condition_id="bench",
                               run_id="bench", model_max_input_tokens=99_999_999)
        messages = driver.messages
        entry: dict = {"sample_id": task["sample_id"], "turns": []}
        cumulative_in = cumulative_out = 0
        largest: dict = {"bytes": 0, "tokens": 0, "turn": 0}
        # Benchmark edit: the full first-chunk text is unique in practice, so
        # the exact-match edit succeeds deterministically. The patch is a
        # benchmark artifact in a temp dir and is discarded afterwards.
        target = root / "workspace" / first["path"]
        span = target.read_text().splitlines(keepends=True)
        old_text = "".join(span[first["start_line"] - 1:first["end_line"]])
        assert target.read_text().count(old_text) == 1
        script.append(("edit", {"action": "edit", "path": first["path"],
                                "old_text": old_text,
                                "new_text": old_text + "# bench\n"}))
        script.append(("finish", {"action": "finish"}))
        stop = "script-end"
        for number, (kind, act) in enumerate(script, 1):
            raw = json.dumps(act)
            prompt_tokens = count_input_tokens(tokenizer, messages)
            response_tokens = count_output_tokens(tokenizer, raw)
            try:
                result = session.process_turn(
                    raw, input_tokens=prompt_tokens, output_tokens=response_tokens)
            except Exception as exc:  # noqa: BLE001 - benchmark records the stop
                stop = f"{type(exc).__name__}: {exc}"
                break
            cumulative_in += prompt_tokens
            cumulative_out += response_tokens
            rendered = render_tool_result(result)
            response_bytes = len(rendered.encode())
            response_tokens = len(tokenizer.encode(rendered))
            if response_bytes > largest["bytes"]:
                largest = {"bytes": response_bytes, "tokens": response_tokens,
                           "turn": number}
            entry["turns"].append({
                "turn": number, "kind": kind, "prompt_tokens": prompt_tokens,
                "cumulative_input_tokens": cumulative_in,
                "cumulative_output_tokens": cumulative_out,
                "tool_calls_used": number,
                "largest_tool_response_bytes": largest["bytes"],
                "largest_tool_response_tokens": largest["tokens"],
                "expected_kv_gib_b1": round(
                    cumulative_in * KV_BYTES_PER_TOKEN / 1024**3, 2),
                "stop_reason": ("COMPLETE" if result.get("status") == "COMPLETE"
                                else "continue"),
            })
            if result.get("status") == "COMPLETE":
                stop = "COMPLETE"
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": rendered})
        entry["stop_reason"] = stop
        entry["cumulative_input_tokens"] = cumulative_in
        entry["cumulative_output_tokens"] = cumulative_out
        entry["largest_tool_response"] = largest
        worst_cumulative = max(worst_cumulative, cumulative_in)
        report["samples"].append(entry)
        shutil.rmtree(root, ignore_errors=True)
    report["worst_cumulative_input_tokens"] = worst_cumulative
    for batch in (1, 2, 4, 6):
        gib = round(expected_gib(worst_cumulative, batch), 2)
        report["batches"][str(batch)] = {
            "expected_gib": gib, "ceiling_gib": round(VRAM_CEILING_GIB, 2),
            "fits": gib <= VRAM_CEILING_GIB,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"worst cumulative input tokens: {worst_cumulative}")
    for batch, row in report["batches"].items():
        print(f"B{batch}: {row['expected_gib']} GiB "
              f"({'FITS' if row['fits'] else 'EXCEEDS'} {row['ceiling_gib']})")
    print(f"stop reasons: {[s['stop_reason'] for s in report['samples']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
