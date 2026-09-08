"""Generate ONLY filtered SWE-bench repair rows (probe or 208-row run).

B8 production path for the SWE-prompt repair: same frozen science as
scripts/run_causal_conditions_batched.py (same donor, BF16, greedy,
thinking-disabled, 1024/1024, use_cache WITHIN each forward, left-padded B8
chunks under the optimized intervene_qwen35 path, deterministic manifest
order, atomic per-chunk checkpoints, heartbeats, fingerprint), but scoped to
an explicit sample-ID allowlist (the 8 frozen SWE-bench validation samples)
and an explicit condition subset.

Outputs {condition_id}.swefix.jsonl (full 208 scope) or
{condition_id}.sweprobe.jsonl (bounded probe), each with the EXACT record
schema generate_condition_batched emits (so the composite builder and scorer
consume them unchanged). Refuses to overwrite destinations. Checkpoints are
atomic per chunk and re-validated on resume (fail-closed on sample-ID drift).

Examples:
  probe (2 samples x 3 conditions):
    python scripts/run_swe_repair_gen.py --config ... --model-path /models/qwen \\
      --dataset-manifest datasets/manifests/pilot-lengthmatched-swe-v2.jsonl \\
      --conditions configs/causal-pilot-conditions.json --condition-ids c0-baseline-a c0-noop-masked c2-selected \\
      --sample-ids-file /tmp/probe-ids.txt --output-dir .../probe --mode probe ...
  full 208 (8 samples x 26 conditions):
    python scripts/run_swe_repair_gen.py --config ... --model-path /models/qwen \\
      --dataset-manifest datasets/manifests/pilot-lengthmatched-swe-v2.jsonl \\
      --conditions configs/causal-pilot-conditions.json \\
      --sample-ids-file /tmp/swe8-ids.txt --output-dir .../generations-swefix --mode fix ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from reverse_reap.causal import _batched_generate
from reverse_reap.config import load_config
from reverse_reap.datasets import load_manifest
from reverse_reap.instrumentation import intervene_qwen35
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import load_donor, validate_donor_contract


def _atomic_write_jsonl(destination: Path, records: list[dict]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                     prefix=f".{destination.name}.", delete=False) as handle:
        tmp = Path(handle.name)
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
    tmp.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--condition-ids", nargs="*", default=None)
    parser.add_argument("--sample-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--heartbeat-path", type=Path, required=True)
    parser.add_argument("--fingerprint-path", type=Path, required=True)
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "unscoped"))
    parser.add_argument("--mode", choices=["probe", "fix"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    spec = json.loads(args.conditions.read_text(encoding="utf-8"))
    wanted_ids = [l.strip() for l in args.sample_ids_file.read_text().splitlines() if l.strip()]
    if not wanted_ids:
        print("empty sample-ids file", file=sys.stderr)
        return 2

    conditions = spec["conditions"]
    if args.condition_ids:
        want = set(args.condition_ids)
        conditions = [c for c in conditions if c["condition_id"] in want]
        if len(conditions) != len(want):
            print(f"unknown condition-ids: {want}", file=sys.stderr)
            return 2

    # Load and filter manifest to the allowlist, preserving manifest order.
    manifest_samples = load_manifest(args.dataset_manifest)
    by_id = {s.sample_id: s for s in manifest_samples}
    missing = [sid for sid in wanted_ids if sid not in by_id]
    if missing:
        print(f"sample-ids not in manifest: {missing}", file=sys.stderr)
        return 2
    # Preserve manifest order (not allowlist order) for determinism.
    order = [s.sample_id for s in manifest_samples]
    wanted_set = set(wanted_ids)
    samples = [by_id[sid] for sid in order if sid in wanted_set]
    # Fail closed: every filtered sample must be a validation SWE-bench row.
    for s in samples:
        if s.split != "validation" or s.scorer != "swebench":
            print(f"refusing non-validation/swebench sample: {s.sample_id} {s.split} {s.scorer}",
                  file=sys.stderr)
            return 2

    suffix = "sweprobe.jsonl" if args.mode == "probe" else "swefix.jsonl"
    print(f"loading donor once for {len(conditions)} conditions x {len(samples)} samples...",
          flush=True)
    model, tokenizer = load_donor(args.model_path, config)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    print("donor loaded", flush=True)

    # Fingerprint (same fields as production path; never blocks generation).
    try:
        import torch, transformers, platform as _plat, shutil as _sh, subprocess as _sp, hashlib as _hl
        fp = {
            "run_id": args.run_id, "mode": args.mode,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
        args.fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        args.fingerprint_path.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        print(f"fingerprint warning: {exc}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch_size = config.runtime.batch_size

    for cond in conditions:
        cid = cond["condition_id"]
        dest = args.output_dir / f"{cid}.{suffix}"
        if dest.exists():
            print(f"[skip] {cid}: {dest} already exists", flush=True)
            continue
        expert_manifest = cond.get("expert_manifest")
        if expert_manifest is not None and not Path(expert_manifest).exists():
            # On the GPU host the frozen manifests are copied alongside the repo;
            # fail closed if absent (do not silently run as baseline).
            print(f"frozen expert manifest missing for {cid}: {expert_manifest}", file=sys.stderr)
            return 2
        from reverse_reap.causal import load_expert_set
        masked = load_expert_set(Path(expert_manifest)) if expert_manifest else frozenset()
        instrument_noop = bool(cond.get("instrument_noop"))

        chunks = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        all_rows: list[dict] = []
        started_all = time.monotonic()
        for idx, chunk in enumerate(chunks):
            ckpt = args.checkpoint_dir / f"{cid}.{args.mode}.chunk-{idx:04d}.json"
            if ckpt.is_file():
                cached = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
                if [r.get("sample_id") for r in cached] != [s.sample_id for s in chunk]:
                    print(f"checkpoint drift: {ckpt}", file=sys.stderr)
                    return 2
                all_rows.extend(cached)
                continue
            if masked or instrument_noop:
                ctx = intervene_qwen35(architecture, masked=(masked if masked else frozenset()))
            else:
                ctx = contextlib.nullcontext()
            chunk_started = time.monotonic()
            with ctx:
                responses, gen_ids, _wall = _batched_generate(model, tokenizer, chunk, config)
            chunk_wall = time.monotonic() - chunk_started
            per_lat = chunk_wall / len(chunk)
            rows = []
            for sample, response, ids in zip(chunk, responses, gen_ids, strict=True):
                if not isinstance(response, str) or not response.strip():
                    print(f"{cid}/{sample.sample_id}: empty response", file=sys.stderr)
                    return 2
                rows.append({
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "source_id": sample.source_id,
                    "scorer": sample.scorer,
                    "domain": sample.domain,
                    "stratum": sample.stratum,
                    "split": sample.split,
                    "condition_id": cid,
                    "masked_experts": len(masked),
                    "response": response,
                    "generated_tokens": len(ids),
                    "truncated": len(ids) >= config.runtime.max_new_tokens,
                    "latency_seconds": per_lat,
                    "chunk": idx,
                    "batch_size": batch_size,
                })
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.checkpoint_dir,
                                             prefix=f".{ckpt.name}.", delete=False) as handle:
                tmp = Path(handle.name)
                for r in rows:
                    handle.write(json.dumps(r, sort_keys=True) + "\n")
                handle.flush()
            tmp.replace(ckpt)
            all_rows.extend(rows)
            elapsed = time.monotonic() - started_all
            hb = {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": args.run_id, "condition_id": cid, "mode": args.mode,
                "completed_items": len(all_rows), "total_items": len(samples),
                "elapsed_seconds": round(elapsed, 1),
            }
            args.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_hb = args.heartbeat_path.parent / f".{args.heartbeat_path.name}.tmp"
            tmp_hb.write_text(json.dumps(hb, sort_keys=True) + "\n")
            tmp_hb.replace(args.heartbeat_path)
            print(f"[{cid}] chunk {idx + 1}/{len(chunks)} ({len(all_rows)}/{len(samples)})",
                  flush=True)
        if [r["sample_id"] for r in all_rows] != [s.sample_id for s in samples]:
            print(f"{cid}: assembled order drift", file=sys.stderr)
            return 2
        _atomic_write_jsonl(dest, all_rows)
        print(f"[done] {cid}: {len(all_rows)} rows -> {dest}", flush=True)
    print(f"{args.mode} generation complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
