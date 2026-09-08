"""Run the agentic 12-session SWE probe (CPU-launchable, GPU-executed).

3 frozen samples x 4 conditions, one donor load, fixed B1 order.

Each session is driven by ``swe_agentic.run_session`` over the validated
``swe-edit-v1`` scaffold. The donor loads once and is reused across all
twelve sessions. Every model forward runs greedy with thinking disabled;
the selected condition wraps forwards in the existing
``intervene_qwen35`` path, the no-op condition wraps with an empty mask,
and both baselines run unwrapped. Batching is fixed B1 sequential in frozen
order, identical across conditions. Model code is never rewritten: the final
patch comes mechanically from the scaffold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
    Path(handle.name).replace(path)


def _heartbeat(path: Path, payload: dict) -> None:
    payload = {"timestamp_utc": datetime.now(UTC).isoformat(), **payload}
    _atomic_write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--condition-ids", nargs="+", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heartbeat-path", type=Path, required=True)
    parser.add_argument("--fingerprint-path", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    from reverse_reap.config import load_config
    from reverse_reap.swe_agentic import plan_sessions, run_session
    from reverse_reap.swe_edit import EditPolicy
    from reverse_reap.swe_edit import sha256 as digest

    config = load_config(args.config)
    if config.runtime.batch_size != 1:
        print("agentic probe is fixed at B1 by the committed preflight plan",
              file=sys.stderr)
        return 2
    if config.budget.deadline_utc <= datetime.now(UTC):
        print("config deadline expired; resolve an approved config", file=sys.stderr)
        return 2
    if _sha256_file(args.tasks) != args.tasks_sha256:
        print("task projection differs from the approved SHA-256", file=sys.stderr)
        return 2
    if config.fingerprint()[:8] not in (args.run_id or ""):
        print("run ID must embed the config fingerprint", file=sys.stderr)
        return 2
    try:
        repo_rev = subprocess_rev_parse()
    except Exception:
        repo_rev = None
    if repo_rev != args.code_revision:
        print(f"code revision mismatch: {repo_rev} != {args.code_revision}",
              file=sys.stderr)
        return 2

    tasks = json.loads(args.tasks.read_text())
    policy = EditPolicy.from_dict(json.loads(args.policy.read_text()))
    spec = json.loads(args.conditions.read_text())
    wanted = list(args.condition_ids)
    conditions = [c for c in spec["conditions"] if c["condition_id"] in wanted]
    if len(conditions) != len(wanted) or {c["condition_id"] for c in conditions} != set(
        wanted
    ):
        print("unknown condition-ids", file=sys.stderr)
        return 2
    for cond in conditions:
        manifest = cond.get("expert_manifest")
        if manifest is not None and not Path(manifest).exists():
            print(f"frozen expert manifest missing for {cond['condition_id']}",
                  file=sys.stderr)
            return 2

    _heartbeat(args.heartbeat_path, {"run_id": args.run_id, "stage": "loading-donor"})
    import torch

    from reverse_reap.causal import load_expert_set
    from reverse_reap.instrumentation import intervene_qwen35
    from reverse_reap.qwen35 import inspect_qwen35_moe
    from reverse_reap.runtime import load_donor, validate_donor_contract

    model, tokenizer = load_donor(args.model_path, config)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    try:
        import transformers
        fingerprint = {
            "run_id": args.run_id,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())],
            "config": config.fingerprint(),
        }
    except Exception as exc:  # noqa: BLE001 - fingerprint never blocks generation
        fingerprint = {"run_id": args.run_id, "warning": str(exc)}
    args.fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    args.fingerprint_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
    print("donor loaded", flush=True)

    def make_generate_fn(masked: frozenset, instrument_noop: bool):
        def generate_fn(messages: list[dict[str, str]]) -> str:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=config.runtime.enable_thinking,
            )
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {k: v.to(model.get_input_embeddings().weight.device)
                       for k, v in encoded.items()}
            with torch.inference_mode():
                context = (intervene_qwen35(architecture, masked=masked)
                           if (masked or instrument_noop)
                           else _nullcontext())
                with context:
                    output = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=config.runtime.max_new_tokens,
                        use_cache=config.runtime.use_cache,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            generated = output[0, encoded["input_ids"].shape[1]:]
            return tokenizer.decode(generated, skip_special_tokens=True)
        return generate_fn

    config_sha = digest(config.canonical_bytes())
    records = []
    total_started = time.monotonic()
    for condition_id, task in plan_sessions(tasks, wanted):
        cond = next(c for c in conditions if c["condition_id"] == condition_id)
        manifest = cond.get("expert_manifest")
        masked = load_expert_set(Path(manifest)) if manifest else frozenset()
        session_dir = args.output_dir / condition_id / task["sample_id"]
        _heartbeat(args.heartbeat_path, {"run_id": args.run_id, "stage": condition_id,
                                         "sample_id": task["sample_id"]})
        started = time.monotonic()
        try:
            summary = run_session(
                session_dir=session_dir, task=task, policy=policy,
                tokenizer=tokenizer,
                generate_fn=make_generate_fn(masked, bool(cond.get("instrument_noop"))),
                run_id=args.run_id, condition_id=condition_id,
                config_sha256=config_sha, model_revision=config.model.revision,
                tokenizer_sha256=args.tokenizer_sha256,
                model_max_input_tokens=config.runtime.max_input_tokens,
            )
            status, error = summary.status, summary.error
        except Exception as exc:  # noqa: BLE001 - OOM ends the run terminally
            _heartbeat(args.heartbeat_path, {"run_id": args.run_id, "stage": "FAILED",
                                             "error": f"{type(exc).__name__}: {exc}"})
            raise
        records.append({
            "sample_id": task["sample_id"], "source_id": task["source_id"],
            "condition_id": condition_id, "status": status,
            "turns": summary.turns, "tool_calls": summary.tool_calls,
            "input_tokens": summary.input_tokens,
            "output_tokens": summary.output_tokens,
            "stop_reason": summary.stop_reason,
            "patch_sha256": summary.patch_sha256,
            "transcript_sha256": summary.transcript_sha256,
            "error": error, "latency_seconds": time.monotonic() - started,
        })
    manifest_out = {
        "run_id": args.run_id, "code_revision": args.code_revision,
        "config": config.fingerprint(), "conditions": wanted,
        "task_projection_sha256": args.tasks_sha256,
        "policy_sha256": _sha256_file(args.policy),
        "records": records,
        "wall_seconds": time.monotonic() - total_started,
    }
    _atomic_write_json(args.output_dir / "run-manifest.json", manifest_out)
    _heartbeat(args.heartbeat_path, {"run_id": args.run_id, "stage": "COMPLETE",
                                     "sessions": len(records)})
    print(f"probe complete: {len(records)} sessions", flush=True)
    return 0


def subprocess_rev_parse() -> str | None:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=30,
    ).strip()


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
