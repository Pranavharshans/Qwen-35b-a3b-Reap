"""Bounded 4x RTX 3090 (24 GB) INSTRUMENTED batching feasibility benchmark.

Directive 2026-09-06: determine whether batched INSTRUMENTED generation on a
4x RTX 3090 host amortizes the instrumentation side-path overhead enough to
host the main causal-validation run. NON-SCIENTIFIC performance probe —
writes no scientific artifacts, resumes no run, changes nothing. Reuses the
EXACT production paths (load_donor, validate_donor_contract, the pinned gen
config's runtime parameters, instrument_qwen35, load_expert_set, _generate's
decoding parameters).

Correctness bar (RELAXED vs the PRO 6000 probe, per directive): batched
outputs may differ from historical batch-1 references (batched-kernel numeric
divergence is known and expected), but the selected batch size must be
INTERNALLY deterministic: identical repeated batches must be token-identical.

Stage order (directive, fixed):
  1. instrumented B2 (4 refs -> chunks [2,2])
  2. instrumented B4 (4 refs -> chunk [4])      — VRAM-guarded
  3. instrumented B8 (8 samples -> chunk [8])   — VRAM-guarded
  4. instrumented B16 (16 samples -> chunk [16])— VRAM+time-guarded
  5. repeat the best viable batch (identical membership/order) -> must be
     token-identical per row (internal-determinism gate)
  6. empty-mask/no-op at the same batch size: plain batched (no wrapper,
     overhead reference) AND instrumented empty-mask; pairwise
     token-identity expected; instrumented-noop also compared against any
     instrumented stage with identical chunk composition
  7. frozen top-four intervention (35,239),(3,26),(7,18),(37,5) at the same
     batch size: every row must DIFFER from no-op; observer counts route
     hits proving each masked expert engaged; zeroing happens on the side
     path without rerouting or renormalization (instrumentation.py)
  8. partial-final-batch + exact-resume: 5 samples (refs + 1) chunked at the
     best N -> final partial chunk; run twice; second pass must reproduce
     token-identical outputs. Cross-process resume is structural: every
     sample record is written atomically and a re-run with --output-dir
     reuses completed records. When the partial first chunk's composition
     equals a throughput stage's chunk, its rows must match that stage too.
  optional supplementary (only if 1-8 complete and time remains): the 12
     non-ref pool samples at the best N, to widen the throughput basis.

16-sample pool: the saved 4x3090 benchmark nine (recorded order) + 7 top-up
(+2 multiple_choice controls, +1 exact_match, +1 unit_tests, +2 swebench,
manifest order) so the pool covers coding, controls, exact-match, unit-test
and SWE-bench tasks per the directive.

VRAM guard: per-GPU predicted peak = static post-load usage + measured
dynamic x (n_next/n_prev) + 2 GiB, must stay <= 0.92 x per-GPU total on
EVERY GPU. OOM is caught, cleared safely, recorded; larger batch stages are
then skipped. A first-chunk NaN/vocab check guards every output.

Hard cap: --deadline-seconds (default 5700) with a teardown reserve.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any

from reverse_reap.causal import load_expert_set
from reverse_reap.config import load_config
from reverse_reap.datasets import load_manifest
from reverse_reap.instrumentation import instrument_qwen35
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import load_donor, validate_donor_contract

PINNED_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
FROZEN_TOP4 = {(35, 239), (3, 26), (7, 18), (37, 5)}

# Saved 4x3090 benchmark selection, in its recorded manifest order
# (runs/bench-20260904/stage-sample-selection.json, archive B-final;
# restored+hash-verified 2026-09-06 from bench-20260904-results.tar.zst).
SAVED_ORDER = [
    ("4371146b066c3f9643baafd4", "unit_tests"),
    ("84ee9615bd1e09a6f3b83770", "unit_tests"),
    ("bc5e6501895b5fd99348f4bd", "unit_tests"),
    ("7db4d0460f2cb926e2cbf1a4", "exact_match"),
    ("52afb433a40c031e1b894ddd", "exact_match"),
    ("53a7e1bcf2745e04b3cd4a12", "exact_match"),
    ("2542b6cd29768cd223836536", "swebench"),
    ("8ff868702cb4a25c487f499a", "swebench"),
    ("277fda6aa9c4f5dd43ef2be7", "swebench"),
]
# Reference samples: first of each scorer type in saved order + one extra
# unit_tests — identical to the PRO 6000 probe refs (comparability).
REF_IDS = [
    "4371146b066c3f9643baafd4",  # unit_tests
    "84ee9615bd1e09a6f3b83770",  # unit_tests
    "7db4d0460f2cb926e2cbf1a4",  # exact_match
    "2542b6cd29768cd223836536",  # swebench
]
# +7 -> 16 total, covering coding, controls (multiple_choice), exact-match,
# unit-test and SWE-bench per the 2026-09-06 directive.
TOP_UP = {"multiple_choice": 2, "exact_match": 1, "unit_tests": 2, "swebench": 2}

B8_EXTRA = [
    "bc5e6501895b5fd99348f4bd",  # unit_tests
    "52afb433a40c031e1b894ddd",  # exact_match
    "53a7e1bcf2745e04b3cd4a12",  # exact_match
    "8ff868702cb4a25c487f499a",  # swebench
]
PARTIAL_EXTRA = "bc5e6501895b5fd99348f4bd"


class BenchStop(RuntimeError):
    """Raised on deadline expiry — stops further stages, keeps all records."""


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


class HostSampler:
    """Background thread: per-GPU peak VRAM + utilization + CPU load."""

    def __init__(self) -> None:
        self.peak_per_gpu: dict[int, int] = {}
        self.util_sum: dict[int, int] = {}
        self.util_n: dict[int, int] = {}
        self.util_peak: dict[int, int] = {}
        self.loadavg: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def reset(self) -> None:
        self.peak_per_gpu = {}
        self.util_sum = {}
        self.util_n = {}
        self.util_peak = {}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 3:
                        continue
                    try:
                        gpu, mem, util = int(parts[0]), int(parts[1]), int(parts[2])
                    except ValueError:
                        continue
                    if mem > self.peak_per_gpu.get(gpu, 0):
                        self.peak_per_gpu[gpu] = mem
                    self.util_sum[gpu] = self.util_sum.get(gpu, 0) + util
                    self.util_n[gpu] = self.util_n.get(gpu, 0) + 1
                    if util > self.util_peak.get(gpu, 0):
                        self.util_peak[gpu] = util
                self.loadavg.append(os.getloadavg()[0])
            except Exception:
                pass
            self._stop.wait(2.0)

    def snapshot(self) -> dict[str, Any]:
        gpus = sorted(self.peak_per_gpu)
        return {
            "peak_vram_mib_per_gpu": {str(g): self.peak_per_gpu[g] for g in gpus},
            "peak_vram_gib_per_gpu": {str(g): round(self.peak_per_gpu[g] / 1024, 2) for g in gpus},
            "gpu_util_mean_pct": {
                str(g): round(self.util_sum[g] / self.util_n[g], 1)
                for g in gpus if self.util_n.get(g)
            },
            "gpu_util_peak_pct": {str(g): self.util_peak.get(g, 0) for g in gpus},
            "loadavg_mean": round(sum(self.loadavg) / len(self.loadavg), 2) if self.loadavg else None,
            "loadavg_peak": round(max(self.loadavg), 2) if self.loadavg else None,
            "cpu_cores": os.cpu_count(),
        }


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=10)[-1]


def _gpu_health() -> dict[str, Any]:
    """Per-GPU temperature/memory snapshot (Xid surfaces as CUDA errors)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        health = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                health[parts[0]] = {"temp_c": int(parts[1]), "mem_mib": int(parts[2])}
        return health
    except Exception:
        return {}


def _encode_batch(tokenizer: Any, samples: list, config: Any, model: Any) -> dict:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.runtime.enable_thinking,
        )
        for s in samples
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    return {
        k: v.to(model.get_input_embeddings().weight.device) for k, v in encoded.items()
    }


def _batch_generate(
    tokenizer: Any, samples: list, config: Any, model: Any
) -> tuple[list[str], list[list[int]], dict]:
    """Left-padded greedy batch generation via the production parameters."""
    import torch

    encoded = _encode_batch(tokenizer, samples, config, model)
    prompt_width = encoded["input_ids"].shape[1]
    pad_id_used = tokenizer.eos_token_id  # same pad id _generate passes
    attention_sums = [int(x) for x in encoded["attention_mask"].sum(dim=1).tolist()]
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config.runtime.max_new_tokens,
                use_cache=config.runtime.use_cache,
                pad_token_id=pad_id_used,
            )
    wall = time.monotonic() - started
    kernel_warnings = sorted({str(w.message)[:160] for w in caught})
    generated = output[:, prompt_width:]
    responses, gen_ids = [], []
    for row in range(generated.shape[0]):
        ids = generated[row]
        keep = ids != pad_id_used  # strip only the padding id generate used
        gen_ids.append(ids[keep].tolist())
        responses.append(tokenizer.decode(ids, skip_special_tokens=True))
    metrics = {
        "wall_seconds": round(wall, 3),
        "prompt_tokens": int(encoded["attention_mask"].sum()),
        "generated_tokens": int(sum(len(g) for g in gen_ids)),
        "attention_mask_sums": attention_sums,
        "prompt_width": int(prompt_width),
        "pad_id_used": pad_id_used,
        "padding_side": "left",
        "kernel_warnings": kernel_warnings,
    }
    return responses, gen_ids, metrics


def _prefill_probe(tokenizer: Any, samples: list, config: Any, model: Any) -> dict:
    import torch

    encoded = _encode_batch(tokenizer, samples, config, model)
    tokens = int(encoded["attention_mask"].sum())
    started = time.monotonic()
    with torch.inference_mode():
        model(**encoded)
    wall = time.monotonic() - started
    return {
        "prefill_tokens": tokens,
        "prefill_seconds": round(wall, 3),
        "prefill_tokens_per_second": round(tokens / wall, 1) if wall else None,
    }


def _chunks(samples: list, n: int) -> list[list]:
    return [samples[i:i + n] for i in range(0, len(samples), n)]


def _select_samples(manifest_path: Path) -> list:
    """Saved 4x3090 nine (recorded order) + TOP_UP extras, manifest order."""
    saved_ids = {sid for sid, _ in SAVED_ORDER}
    counts = dict(TOP_UP)
    selected: list = []
    seen: set[str] = set()
    for sample in load_manifest(manifest_path):
        if sample.split != "validation":
            continue
        if sample.sample_id in saved_ids:
            selected.append(sample)
            seen.add(sample.sample_id)
        elif sample.scorer in counts and counts[sample.scorer] > 0:
            counts[sample.scorer] -= 1
            selected.append(sample)
            seen.add(sample.sample_id)
    if len(selected) != len(SAVED_ORDER) + sum(TOP_UP.values()):
        missing = {sid for sid, _ in SAVED_ORDER} - seen
        raise BenchStop(f"sample selection incomplete ({len(selected)}); missing: {missing}")
    order = {sid: i for i, (sid, _) in enumerate(SAVED_ORDER)}
    selected.sort(key=lambda s: (0, order[s.sample_id]) if s.sample_id in order
                  else (1, 0))
    return selected


def _observer_counter(masked_set: frozenset):
    """Count route hits per (layer, expert) during instrumented forwards."""
    hits: dict[tuple[int, int], int] = {}

    def observer(layer: int, batch: Any, _norms: Any) -> None:
        import numpy as np

        experts_in_routes = np.unique(batch.indices.flatten())
        for expert in experts_in_routes.tolist():
            key = (layer, int(expert))
            hits[key] = hits.get(key, 0) + int((batch.indices == expert).sum())

    return hits, observer


def _token_identical(rows_a: dict, rows_b: dict, ids: list[str]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for rid in ids:
        if rid not in rows_a or rid not in rows_b:
            bad.append(f"{rid}:missing")
        elif rows_a[rid]["generated_ids"] != rows_b[rid]["generated_ids"]:
            bad.append(f"{rid}:ids-differ")
        elif rows_a[rid]["response"] != rows_b[rid]["response"]:
            bad.append(f"{rid}:text-differ")
    return (not bad), bad


def _repo_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=5700)
    parser.add_argument("--rate-usd-per-hour", type=float, default=0.4814,
                        help="all-inclusive host rate used for projections")
    parser.add_argument("--setup-minutes", type=float, default=25.0,
                        help="one-time setup burn already elapsed before this run")
    args = parser.parse_args()

    deadline = time.monotonic() + args.deadline_seconds
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark": "rtx3090-instrumented-batching-feasibility",
        "scientific": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deadline_seconds": args.deadline_seconds,
        "repo_commit": _repo_commit(),
        "pinned_revision": PINNED_REVISION,
        "rate_usd_per_hour": args.rate_usd_per_hour,
        "stages": [],
        "validations": {},
        "stop_reason": "completed",
    }

    def remaining() -> float:
        return deadline - time.monotonic()

    def check_deadline(stage: str) -> None:
        if remaining() < 240:  # keep 4 min for teardown/report
            raise BenchStop(f"deadline reached before stage {stage}")

    def record(stage: str, payload: dict) -> None:
        entry = {"stage": stage, **payload}
        report["stages"].append(entry)
        _atomic_write_json(output_dir / f"stage-{stage}.json", entry)
        print(f"[stage] {stage}: {json.dumps(payload, default=str)[:240]}", flush=True)

    def load_stage_rows(stage: str, samples: list) -> dict:
        rows = {}
        for s in samples:
            p = output_dir / stage / f"{s.sample_id}.json"
            if p.exists():
                rows[s.sample_id] = json.loads(p.read_text())
        return rows

    config = load_config(args.config)
    sampler = HostSampler()
    sampler.start()

    # ---------- model load (once per process) ----------
    if (output_dir / "stage-model-load.json").exists():
        entry = json.loads((output_dir / "stage-model-load.json").read_text())
        report["stages"].append({"stage": "model-load", "resumed": True, **entry})
    started = time.monotonic()
    model, tokenizer = load_donor(args.model_path, config)
    load_seconds = round(time.monotonic() - started, 1)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    n_gpus = torch.cuda.device_count()
    per_gpu_total = {
        i: int(torch.cuda.get_device_properties(i).total_memory / 2**20)
        for i in range(n_gpus)
    }
    props = {
        "seconds": load_seconds,
        "contract": "valid",
        "gpu_count": n_gpus,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(n_gpus)],
        "device_capabilities": [
            ".".join(map(str, torch.cuda.get_device_capability(i))) for i in range(n_gpus)
        ],
        "vram_total_mib_per_gpu": per_gpu_total,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    if not (output_dir / "stage-model-load.json").exists():
        record("model-load", props)
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        env = {
            "gpu_count": n_gpus,
            "gpu_names": props["gpu_names"],
            "compute_capabilities": props["device_capabilities"],
            "vram_total_mib_per_gpu": per_gpu_total,
            "driver_version": driver[0].strip() if driver else None,
            "torch": props["torch_version"],
            "cuda": props["cuda_version"],
            "transformers": props["transformers_version"],
            "python": props["python"],
            "pinned_revision": PINNED_REVISION,
            "max_new_tokens": config.runtime.max_new_tokens,
            "use_cache": config.runtime.use_cache,
            "enable_thinking": config.runtime.enable_thinking,
            "repo_commit": report["repo_commit"],
        }
        _atomic_write_json(output_dir / "environment-report.json", env)

    # Static per-GPU usage (weights + CUDA context) — baseline for VRAM guards.
    sampler.reset()
    time.sleep(5.0)  # let the sampler observe the post-load steady state
    static_per_gpu = dict(sampler.peak_per_gpu)

    samples = _select_samples(args.manifest)
    by_id = {s.sample_id: s for s in samples}
    refs = [by_id[i] for i in REF_IDS]
    b8_set = refs + [by_id[i] for i in B8_EXTRA]
    partial5 = refs + [by_id[PARTIAL_EXTRA]]
    pool_minus_refs = [s for s in samples if s.sample_id not in set(REF_IDS)]
    record("sample-selection", {
        "count": len(samples),
        "order": [s.sample_id for s in samples],
        "scorers": [s.scorer for s in samples],
        "references": REF_IDS,
        "b8_set": [s.sample_id for s in b8_set],
        "partial5": [s.sample_id for s in partial5],
    })

    vocab_size = len(tokenizer)
    stage_status: dict[str, str] = {}  # "ok" | "oom" | "error" | "skipped"
    throughput_peaks: dict[str, tuple[int, dict[int, int]]] = {}  # stage -> (n, peaks)

    def run_stage(
        stage: str, stage_samples: list, n: int,
        masked: frozenset | None = None, observer=None,
        instrumented: bool = True, min_remaining: float = 240.0,
    ) -> dict:
        """Process stage_samples in chunks of <=n. Returns rows by sample_id."""
        stage_path = output_dir / f"stage-{stage}.json"
        if stage_path.exists():
            entry = json.loads(stage_path.read_text())
            report["stages"].append({"stage": stage, "resumed": True, **entry})
            stage_status[stage] = entry.get("status", "ok")
            if entry.get("status") == "ok" and instrumented:
                peaks = {int(k): v for k, v in (entry.get("peak_vram_mib_per_gpu") or {}).items()}
                throughput_peaks[stage] = (entry.get("batch_size", n), peaks)
            return load_stage_rows(stage, stage_samples)
        if remaining() < min_remaining:
            stage_status[stage] = "skipped"
            record(f"{stage}-skipped", {"reason": f"remaining {remaining():.0f}s < {min_remaining}s"})
            return {}
        sampler.reset()
        prefill = None
        if instrumented:
            with instrument_qwen35(architecture, masked=masked or frozenset(), observer=observer):
                prefill = _prefill_probe(tokenizer, stage_samples, config, model)
        else:
            prefill = _prefill_probe(tokenizer, stage_samples, config, model)

        chunk_manifest: list[dict] = []
        rows: dict[str, dict] = {}
        walls: list[float] = []
        latencies: list[float] = []
        total_gen = 0
        kernel_warnings: set[str] = set()
        nan_failures: list[str] = []
        status = "ok"
        for ci, chunk in enumerate(_chunks(stage_samples, n)):
            pending = [s for s in chunk
                       if not (output_dir / stage / f"{s.sample_id}.json").exists()]
            chunk_plan = {
                "chunk": ci,
                "sample_ids": [s.sample_id for s in chunk],
                "ran_sample_ids": [s.sample_id for s in pending],
            }
            if not pending:
                chunk_manifest.append(chunk_plan)
                continue
            check_deadline(stage)
            try:
                if instrumented:
                    with instrument_qwen35(
                        architecture, masked=masked or frozenset(), observer=observer,
                    ):
                        responses, gen_ids, metrics = _batch_generate(
                            tokenizer, pending, config, model)
                else:
                    responses, gen_ids, metrics = _batch_generate(
                        tokenizer, pending, config, model)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                status = "oom"
                chunk_plan["error"] = "CUDA OOM — cleared safely"
                chunk_manifest.append(chunk_plan)
                record(stage, {
                    "status": status, "batch_size": n, "chunk": ci,
                    "error": "CUDA OOM", "cleared_safely": True,
                    "chunk_manifest": chunk_manifest, **sampler.snapshot(),
                })
                break
            except RuntimeError as exc:  # NaN / kernel failure — fail closed
                status = "error"
                chunk_plan["error"] = str(exc)[:200]
                chunk_manifest.append(chunk_plan)
                record(stage, {
                    "status": status, "batch_size": n, "chunk": ci,
                    "error": str(exc)[:200],
                    "chunk_manifest": chunk_manifest, **sampler.snapshot(),
                })
                break
            wall = metrics["wall_seconds"]
            walls.append(wall)
            latencies.extend([wall / len(pending)] * len(pending))
            total_gen += metrics["generated_tokens"]
            kernel_warnings.update(metrics["kernel_warnings"])
            chunk_plan.update({
                "attention_mask_sums": metrics["attention_mask_sums"],
                "prompt_tokens": metrics["prompt_tokens"],
                "prompt_width": metrics["prompt_width"],
                "pad_id_used": metrics["pad_id_used"],
                "wall_seconds": wall,
            })
            chunk_manifest.append(chunk_plan)
            for s, text, ids in zip(pending, responses, gen_ids, strict=True):
                in_vocab = all(0 <= i < vocab_size for i in ids)
                if not in_vocab:
                    nan_failures.append(s.sample_id)
                rec = {
                    "sample_id": s.sample_id, "scorer": s.scorer,
                    "response": text, "generated_ids": ids,
                    "generated_tokens": len(ids),
                    "truncated": len(ids) >= config.runtime.max_new_tokens,
                    "nan_vocab_check": "pass" if in_vocab else "FAIL",
                    "chunk": ci, "batch_size": n,
                }
                _atomic_write_json(output_dir / stage / f"{s.sample_id}.json", rec)
                rows[s.sample_id] = rec
        # merge pre-existing records (chunks completed in an earlier pass)
        for s in stage_samples:
            if s.sample_id not in rows:
                p = output_dir / stage / f"{s.sample_id}.json"
                if p.exists():
                    rows[s.sample_id] = json.loads(p.read_text())
        total_wall = sum(walls)
        expected = len(stage_samples)
        complete = status == "ok" and len(rows) == expected
        stage_status[stage] = status if status != "ok" else ("ok" if complete else "partial")
        entry = {
            "status": stage_status[stage],
            "batch_size": n,
            "instrumented": instrumented,
            "masked_experts": sorted(masked) if masked else [],
            "samples": len(rows),
            "expected_samples": expected,
            "chunks": len(_chunks(stage_samples, n)),
            "wall_seconds": round(total_wall, 1),
            "samples_per_minute": round(len(rows) / total_wall * 60, 3) if total_wall else None,
            "generated_tokens": total_gen,
            "output_lengths": {
                "mean": round(total_gen / len(rows), 1) if rows else None,
                "min": min((r["generated_tokens"] for r in rows.values()), default=None),
                "max": max((r["generated_tokens"] for r in rows.values()), default=None),
            },
            "decode_tokens_per_second": round(total_gen / total_wall, 2) if total_wall else None,
            "mean_latency_seconds": round(total_wall / len(rows), 3) if rows else None,
            "p90_latency_seconds": round(_p90(latencies), 3) if latencies else None,
            "prefill": prefill,
            "kernel_warnings": sorted(kernel_warnings),
            "nan_vocab_check_failures": nan_failures,
            "chunk_manifest": chunk_manifest,
            **sampler.snapshot(),
        }
        record(stage, entry)
        if complete and instrumented:
            peaks = {int(k): v for k, v in entry["peak_vram_mib_per_gpu"].items()}
            throughput_peaks[stage] = (n, peaks)
        return rows

    def vram_guard(stage: str, n_next: int, need_remaining: float) -> tuple[bool, str]:
        """Predict per-GPU peak for n_next from the largest completed smaller n."""
        if remaining() < need_remaining:
            return False, f"remaining {remaining():.0f}s < {need_remaining}s needed"
        ceiling = min(per_gpu_total.values())
        prior = [(n, peaks) for (st, (n, peaks)) in throughput_peaks.items()
                 if stage_status.get(st) == "ok" and n < n_next]
        if not prior:
            return False, "no completed smaller-batch stage to extrapolate from"
        n_prev, prev_peaks = max(prior, key=lambda x: x[0])
        preds = {}
        for g in range(n_gpus):
            stat = static_per_gpu.get(g, 0)
            prev = prev_peaks.get(g, stat)
            dyn = max(prev - stat, 0)
            preds[g] = stat + dyn * (n_next / n_prev) + 2048  # 2 GiB guard
        worst_gpu = max(preds, key=lambda g: preds[g])
        ok = preds[worst_gpu] <= int(ceiling * 0.92)
        return ok, (
            f"predicted GPU{worst_gpu} {preds[worst_gpu]:.0f} MiB vs ceiling "
            f"{int(ceiling * 0.92)} (static {static_per_gpu.get(worst_gpu)}, "
            f"prev peak {prev_peaks.get(worst_gpu)} @ n={n_prev}, "
            f"scale x{n_next / n_prev:.1f})")

    def prior_oom(n_next: int) -> str | None:
        for st, (n, _peaks) in throughput_peaks.items():
            if stage_status.get(st) == "oom" and n < n_next:
                return f"OOM already occurred at n={n} ({st})"
        return None

    try:
        # ---------- Stages 1-4: instrumented throughput ladder ----------
        run_stage("instr-B2", refs, 2, masked=frozenset())
        safe, reason = vram_guard("B4", 4, 420)
        record("B4-guard", {"safe": safe, "reason": reason})
        if safe and prior_oom(4) is None:
            run_stage("instr-B4", refs, 4, masked=frozenset(), min_remaining=420)
        else:
            stage_status["instr-B4"] = "skipped"
        safe, reason = vram_guard("B8", 8, 720)
        oom = prior_oom(8)
        record("B8-guard", {"safe": safe and oom is None,
                            "reason": reason if oom is None else f"{reason}; {oom}"})
        if safe and oom is None:
            run_stage("instr-B8", b8_set, 8, masked=frozenset(), min_remaining=720)
        else:
            stage_status["instr-B8"] = "skipped"
        safe, reason = vram_guard("B16", 16, 1500)
        oom = prior_oom(16)
        record("B16-guard", {"safe": safe and oom is None,
                             "reason": reason if oom is None else f"{reason}; {oom}"})
        if safe and oom is None:
            run_stage("instr-B16", samples, 16, masked=frozenset(), min_remaining=1500)
        else:
            stage_status["instr-B16"] = "skipped"

        # ---------- best viable instrumented batch ----------
        ladder = [("instr-B2", refs, 2), ("instr-B4", refs, 4),
                  ("instr-B8", b8_set, 8), ("instr-B16", samples, 16)]
        best_stage, best_samples, best_n = None, None, None
        for st, st_samples, n in ladder:
            if stage_status.get(st) == "ok":
                best_stage, best_samples, best_n = st, st_samples, n
        if best_stage is None:
            # Fallback floor: instrumented batch-1 on the 4 refs (NOT the full
            # historical B1 benchmark — a 4-sample instrumented floor only).
            print("[fallback] no instrumented batched stage completed; "
                  "running instrumented B1 floor on 4 refs", flush=True)
            run_stage("instr-B1-floor", refs, 1, masked=frozenset())
            if stage_status.get("instr-B1-floor") == "ok":
                best_stage, best_samples, best_n = "instr-B1-floor", refs, 1
        if best_stage is None:
            report["validations"]["viable_configuration"] = (
                "FAIL: no instrumented configuration ran (OOM at every batch size)")
            record("best-batch", {"selected": None})
            raise BenchStop("no viable instrumented configuration")
        record("best-batch", {"selected": best_stage, "batch_size": best_n,
                              "samples": len(best_samples)})

        best_rows = load_stage_rows(best_stage, best_samples)

        # ---------- Stage 5: repeat the best viable batch ----------
        repeat_rows = run_stage("repeat-best", best_samples, best_n,
                                masked=frozenset(), min_remaining=420)
        if repeat_rows and stage_status.get("repeat-best") != "skipped":
            ok, bad = _token_identical(repeat_rows, best_rows,
                                       [s.sample_id for s in best_samples])
            report["validations"]["internal_determinism_repeat"] = (
                "PASS token-identical" if ok else f"FAIL: {bad[:6]}")
            _chunk_frozen(best_stage, "repeat-best", report, output_dir)
        else:
            report["validations"]["internal_determinism_repeat"] = "NOT RUN (stage skipped)"

        # ---------- Stage 6: no-op at the same batch size ----------
        plain_rows = run_stage("plain-batch", refs, best_n, instrumented=False,
                               min_remaining=420)
        noop_rows = run_stage("noop", refs, best_n, masked=frozenset(),
                              min_remaining=420)
        if noop_rows and plain_rows and \
                stage_status.get("noop") != "skipped" and \
                stage_status.get("plain-batch") != "skipped":
            ok, bad = _token_identical(noop_rows, plain_rows, REF_IDS)
            report["validations"]["noop_equals_plain_batched"] = (
                "PASS token-identical" if ok else f"FAIL: {bad[:6]}")
        else:
            report["validations"]["noop_equals_plain_batched"] = "NOT RUN (stage skipped)"
        # noop must also equal any instrumented stage with identical chunking
        for st, st_samples, n in ladder[:2]:  # B2 [2,2] and B4 [4] over refs
            if stage_status.get(st) == "ok" and n == best_n:
                st_rows = load_stage_rows(st, st_samples)
                ok, bad = _token_identical(noop_rows, st_rows, REF_IDS)
                report["validations"][f"noop_equals_{st}"] = (
                    "PASS token-identical" if ok else f"FAIL: {bad[:6]}")

        # ---------- Stage 7: frozen top-four intervention ----------
        masked_set = load_expert_set(args.selected_manifest)
        if masked_set != FROZEN_TOP4:
            raise BenchStop(f"selected manifest != frozen top-4: {sorted(masked_set)}")
        hits, observer = _observer_counter(masked_set)
        sel_rows = run_stage("selected", refs, best_n, masked=masked_set,
                             observer=observer, min_remaining=420)
        missing_sel = [rid for rid in REF_IDS
                       if rid not in sel_rows or rid not in noop_rows]
        unchanged = [rid for rid in REF_IDS
                     if rid in sel_rows and rid in noop_rows
                     and sel_rows[rid]["response"] == noop_rows[rid]["response"]]
        masked_hits = {f"({l},{e})": hits.get((l, e), 0) for l, e in sorted(masked_set)}
        all_hit = all(v > 0 for v in masked_hits.values())
        report["validations"]["selected_intervention"] = (
            "PASS: all refs differ, frozen top-4 masked, zeroed side-path "
            "without rerouting or renormalization"
            if not unchanged and all_hit and not missing_sel
            and stage_status.get("selected") not in ("skipped", None)
            else f"FAIL: unchanged={unchanged} missing={missing_sel} "
                 f"route_hits={masked_hits}")
        record("selected-gate", {
            "masked_experts": sorted(masked_set),
            "masked_expert_route_hits": masked_hits,
            "unchanged_refs": unchanged,
        })

        # ---------- Stage 8: partial-final-batch + exact-resume ----------
        part_a = run_stage("partial-a", partial5, best_n, masked=frozenset(),
                           min_remaining=480)
        part_b = run_stage("partial-b", partial5, best_n, masked=frozenset(),
                           min_remaining=480)
        if part_a and part_b and stage_status.get("partial-a") != "skipped" \
                and stage_status.get("partial-b") != "skipped":
            ok, bad = _token_identical(part_a, part_b, [s.sample_id for s in partial5])
            report["validations"]["partial_resume_identical"] = (
                "PASS token-identical" if ok else f"FAIL: {bad[:6]}")
        else:
            report["validations"]["partial_resume_identical"] = "NOT RUN (stage skipped)"
        # When the partial first chunk's composition equals a throughput
        # stage's chunk, those rows must match that stage as well.
        if best_n == 2 and stage_status.get("instr-B2") == "ok":
            b2_rows = load_stage_rows("instr-B2", refs)
            ok, bad = _token_identical(part_a, b2_rows, REF_IDS)
            report["validations"]["partial_first_chunks_equal_instr_B2"] = (
                "PASS token-identical" if ok else f"FAIL: {bad[:6]}")
        elif best_n == 4 and stage_status.get("instr-B4") == "ok":
            b4_rows = load_stage_rows("instr-B4", refs)
            ok, bad = _token_identical(part_a, b4_rows, REF_IDS)
            report["validations"]["partial_first_chunk_equals_instr_B4"] = (
                "PASS token-identical" if ok else f"FAIL: {bad[:6]}")

        # ---------- Supplementary: widen the throughput basis ----------
        if remaining() > 900 and best_n and best_stage != "instr-B1-floor":
            ext_stage = f"instr-B{best_n}-extended"
            ext_rows = run_stage(ext_stage, pool_minus_refs, best_n,
                                 masked=frozenset(), min_remaining=720)
            if stage_status.get(ext_stage) == "ok":
                best_entry = json.loads((output_dir / f"stage-{best_stage}.json").read_text())
                pooled_n = best_entry["samples"] + len(ext_rows)
                pooled_wall = best_entry["wall_seconds"] + \
                    json.loads((output_dir / f"stage-{ext_stage}.json").read_text())["wall_seconds"]
                record("pooled-throughput", {
                    "basis": "refs+pool_minus_refs",
                    "samples": pooled_n,
                    "wall_seconds": round(pooled_wall, 1),
                    "samples_per_minute": round(pooled_n / pooled_wall * 60, 3),
                })
    except BenchStop as stop:
        report["stop_reason"] = str(stop)

    # ---------- projections ----------
    try:
        projection: dict[str, Any] = {"gens": 1300, "rate_usd_per_hour": args.rate_usd_per_hour}
        basis_rows = None
        for cand in ("pooled-throughput", "instr-B2", "instr-B4",
                     "instr-B8", "instr-B16", "instr-B1-floor"):
            p = output_dir / f"stage-{cand}.json"
            if p.exists():
                entry = json.loads(p.read_text())
                if entry.get("samples_per_minute"):
                    projection["basis_stage"] = cand
                    projection["instrumented_samples_per_minute"] = entry["samples_per_minute"]
                    projection["p90_sample_latency_seconds"] = entry.get("p90_latency_seconds")
                    basis_rows = entry
                    break
        if basis_rows:
            spm = projection["instrumented_samples_per_minute"]
            p90 = projection["p90_sample_latency_seconds"]
            if p90 is None:  # pooled basis has no latency of its own
                for alt in ("repeat-best", "instr-B2", "instr-B4",
                            "instr-B8", "instr-B16", "instr-B1-floor"):
                    ap = output_dir / f"stage-{alt}.json"
                    if ap.exists():
                        e = json.loads(ap.read_text())
                        if e.get("p90_latency_seconds"):
                            p90 = e["p90_latency_seconds"]
                            break
            p90 = p90 or (3600.0 / (spm * 60))
            expected_h = 1300 / (spm * 60)
            conserv_h = 1300 * p90 / 3600
            setup_cost = args.rate_usd_per_hour * args.setup_minutes / 60
            projection.update({
                "expected_wall_hours": round(expected_h, 2),
                "conservative_wall_hours": round(conserv_h, 2),
                "setup_burn_usd": round(setup_cost, 2),
                "expected_cost_usd": round(args.rate_usd_per_hour * expected_h + setup_cost, 2),
                "conservative_cost_usd": round(args.rate_usd_per_hour * conserv_h + setup_cost, 2),
            })
            peaks: dict = {}
            for src in (projection["basis_stage"], "repeat-best", "instr-B2",
                        "instr-B4", "instr-B8", "instr-B16", "instr-B1-floor"):
                sp_ = output_dir / f"stage-{src}.json"
                if sp_.exists():
                    e = json.loads(sp_.read_text())
                    pk = e.get("peak_vram_mib_per_gpu") or {}
                    if pk:
                        peaks = pk
                        break
            ceiling = min(per_gpu_total.values())
            headroom_ok = all(
                v <= int(ceiling * 0.92) for v in peaks.values()) if peaks else False
            projection["vram_headroom"] = "safe" if headroom_ok else "tight/unknown"
            projection["qualification_rule"] = (
                "conservative < 6.4 host-hours AND conservative < $10 AND safe VRAM headroom")
            projection["qualifies"] = bool(
                projection["conservative_wall_hours"] < 6.4
                and projection["conservative_cost_usd"] < 10
                and headroom_ok)
            # instrumentation overhead ratio (plain vs instrumented, same n)
            plain_path = output_dir / "stage-plain-batch.json"
            if plain_path.exists():
                plain = json.loads(plain_path.read_text())
                if plain.get("samples_per_minute"):
                    projection["plain_batched_samples_per_minute"] = plain["samples_per_minute"]
                    projection["instrumentation_overhead_ratio"] = round(
                        plain["samples_per_minute"] / spm, 2) if spm else None
        else:
            projection["error"] = "no completed throughput stage to project from"
        report["projection"] = projection
        record("projection", projection)
    except Exception as exc:  # projection must never lose the stage data
        report["projection"] = {"error": str(exc)[:200]}

    # ---------- aggregate health gates ----------
    stage_files = sorted(output_dir.glob("stage-*.json"))
    ooms = [p.name for p in stage_files
            if json.loads(p.read_text()).get("status") == "oom"]
    errors = [p.name for p in stage_files
              if json.loads(p.read_text()).get("status") == "error"]
    nan_fails = []
    kernel_warns = []
    for rec_path in output_dir.glob("*/**/*.json"):
        if rec_path.name.startswith("stage-") or rec_path.name in (
                "benchmark-report.json", "environment-report.json"):
            continue
        try:
            rec = json.loads(rec_path.read_text())
        except Exception:
            continue
        if rec.get("nan_vocab_check") == "FAIL":
            nan_fails.append(rec_path.stem)
        if rec.get("kernel_warnings"):
            kernel_warns.extend(rec["kernel_warnings"])
    report["validations"]["no_oom_nan_kernel_failure"] = (
        "PASS" if not ooms and not errors and not nan_fails
        else f"FAIL: oom={ooms} errors={errors} nan={nan_fails[:6]}")
    if kernel_warns:
        report["validations"]["kernel_warnings_seen"] = sorted(set(kernel_warns))[:8]

    sampler.stop()
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["elapsed_seconds"] = round(args.deadline_seconds - remaining(), 1)
    report["gpu_health_at_end"] = _gpu_health()
    _atomic_write_json(output_dir / "benchmark-report.json", report)
    print(json.dumps({"stop_reason": report["stop_reason"],
                      "validations": report["validations"],
                      "projection": report.get("projection")}, indent=1, default=str),
          flush=True)
    return 0


def _chunk_frozen(stage_a: str, stage_b: str, report: dict, output_dir: Path) -> None:
    """Gate: identical membership/order/padding/attention masks across stages."""
    try:
        a = json.loads((output_dir / f"stage-{stage_a}.json").read_text()).get("chunk_manifest", [])
        b = json.loads((output_dir / f"stage-{stage_b}.json").read_text()).get("chunk_manifest", [])
    except Exception as exc:
        report["validations"]["batching_frozen"] = f"UNKNOWN: {exc}"
        return
    problems: list[str] = []
    for ca, cb in zip(a, b, strict=False):
        if ca.get("sample_ids") != cb.get("sample_ids"):
            problems.append(f"chunk {ca.get('chunk')}: membership/order differs")
            continue
        if ca.get("ran_sample_ids") and cb.get("ran_sample_ids") and \
                ca.get("ran_sample_ids") != cb.get("ran_sample_ids"):
            problems.append(f"chunk {ca.get('chunk')}: ran-subset differs")
            continue
        if ca.get("ran_sample_ids") and cb.get("ran_sample_ids"):
            if ca.get("attention_mask_sums") != cb.get("attention_mask_sums"):
                problems.append(f"chunk {ca.get('chunk')}: attention masks differ")
            if ca.get("pad_id_used") != cb.get("pad_id_used"):
                problems.append(f"chunk {ca.get('chunk')}: pad id differs")
            if ca.get("prompt_width") != cb.get("prompt_width"):
                problems.append(f"chunk {ca.get('chunk')}: prompt width differs")
    report["validations"]["batching_frozen"] = (
        "PASS: membership, ordering, padding and attention masks frozen"
        if not problems else f"FAIL: {problems[:4]}")


if __name__ == "__main__":
    sys.exit(main())
