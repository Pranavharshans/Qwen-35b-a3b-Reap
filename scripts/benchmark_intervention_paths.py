"""Bounded intervention-path feasibility benchmark (old vs optimized).

Directive 2026-09-07: determine whether the optimized intervention-only path
(``intervene_qwen35`` — native fused forward with zeroed router weights, no
telemetry side path) carries the main causal-validation run within budget.
NON-SCIENTIFIC performance probe — writes no scientific artifacts, resumes no
run, changes nothing frozen. Reuses the EXACT production paths (load_donor,
validate_donor_contract, the pinned gen config's runtime parameters,
load_expert_set, _generate's decoding parameters).

Three arms, same samples and batching within each batch size:
  A. native      — no wrapper (uninstrumented baseline).
  B. noop-int    — ``intervene_qwen35`` with an empty mask (must equal A).
  D. sel-int     — ``intervene_qwen35`` with the frozen top-four (new path).

The slow telemetry/replay oracle (``instrument_qwen35``) is restricted to a
BOUNDED probe — prefill logits on two reference samples plus a 32-token
short generation on one — reporting max absolute/relative error against the
optimized path. Full slow generations are never run: old-implementation
throughput is already on record in the bench-20260904 /
bench-3090-instr-20260906 / pro6000 archives, which are PRESERVED as the
old-implementation evidence and never overwritten.

Stage priority (largest batch first — B8 decides, B1 runs only if the window
remains):
  1. optimized native/no-op/selected at the largest batch
  2. bounded oracle probe + repeat determinism + partial-resume checks
  3. smaller batches in descending order (B1 requires 1500 s remaining)

Phase plan (separate launches, same script):
  Phase 1: 4x RTX 3090, --batches 1 4 8 (decisive).
  Phase 2 (ONLY as needed): RTX PRO 6000, --batches 8 16 — run only if the
  Phase 1 sel-int projection exceeds 6.4 host-hours or $10, or B8 OOMs.

Gates: per-batch no-op equivalence (B == A token-identical), repeat
determinism of the largest-batch D stage, partial-final-batch exact resume,
D-engagement (every ref differs from B, frozen manifest verified), C-vs-D
agreement reported (informational: grouped_mm-vs-eager numerics may diverge;
semantic equivalence is proven by unit tests within BF16_TOLERANCE).

VRAM guard (fixed 2026-09-07, pure logic in src/reverse_reap/bench_guard.py):
pre-attempt prediction is direction-aware (nearest completed batch in EITHER
direction; ties resolve to the larger n so scaling down overpredicts) and
predicts only the dynamic component against free memory minus one explicit
1 GiB margin. The 92% ceiling on MEASURED peaks is UNCHANGED and governs the
batch-selection verdict below. OOM is caught, cleared safely, recorded; an
OOM at batch n skips larger batches only (smaller batches may still be
attempted). A first-chunk NaN/vocab check guards every output.

Batch-selection verdict: among ok sel-int batches (descending), the largest
with measured peak <= 0.92 x physical on every GPU, conservative (p90-based)
rate >= ~3.4 spm, no-op equivalence PASS, repeat determinism PASS, and a
passing --layer-diagnostic report (scripts/validate_layer_local_deltas.py
output) is SELECTed; otherwise the report records NO BATCH QUALIFIES.
--no-oracle-probe skips the bounded slow-oracle probe (the full-model oracle
result and its 0.0625 threshold are preserved as-is; the layer-local
diagnostic supersedes it as the equivalence evidence).

Hard cap: --deadline-seconds (default 5400) with a teardown reserve.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any

from reverse_reap.bench_guard import blocking_oom, predict_batch_safe
from reverse_reap.causal import load_expert_set
from reverse_reap.config import load_config
from reverse_reap.datasets import load_manifest
from reverse_reap.instrumentation import (
    FROZEN_SELECTED_TOP4,
    instrument_qwen35,
    intervene_qwen35,
)
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import load_donor, validate_donor_contract

PINNED_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"

# Bound for optimized-vs-slow-oracle prefill-logit agreement. BF16 epsilon is
# 2**-8 (~0.0039); 0.0625 (16x eps) absorbs fp32 reduction-ordering drift of
# the grouped_mm-vs-eager accumulation across 40 layers. Same bound as the
# GPU unit tests (tests/test_intervention.py::BF16_TOLERANCE).
ORACLE_BF16_TOLERANCE = 0.0625

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

ARMS = ("native", "noop-int", "sel-int")


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
            "loadavg_mean": (
                round(sum(self.loadavg) / len(self.loadavg), 2) if self.loadavg else None
            ),
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
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--deadline-seconds", type=int, default=5400)
    parser.add_argument("--rate-usd-per-hour", type=float, required=True,
                        help="all-inclusive host rate used for projections")
    parser.add_argument("--setup-minutes", type=float, default=25.0,
                        help="one-time setup burn already elapsed before this run")
    parser.add_argument("--expect-gpu-count", type=int, default=4)
    parser.add_argument("--expect-gpu-name", type=str, default="3090")
    parser.add_argument("--min-disk-gib", type=float, default=240.0)
    parser.add_argument("--oracle-probe", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run the bounded slow-oracle probe (default on; "
                             "--no-oracle-probe skips it, preserving the old "
                             "result and threshold untouched)")
    parser.add_argument("--layer-diagnostic", type=Path, default=None,
                        help="layer-delta-probe.json from "
                             "validate_layer_local_deltas.py; incorporated "
                             "into the batch-selection verdict when present")
    args = parser.parse_args()

    batches = sorted(set(args.batches))
    deadline = time.monotonic() + args.deadline_seconds
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark": "intervention-paths-feasibility",
        "scientific": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deadline_seconds": args.deadline_seconds,
        "repo_commit": _repo_commit(),
        "pinned_revision": PINNED_REVISION,
        "rate_usd_per_hour": args.rate_usd_per_hour,
        "batches": batches,
        "arms": args.arms,
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
    # ---------- host preflight: GPU identity + usable storage (fail closed) ----
    if n_gpus != args.expect_gpu_count or not all(
        args.expect_gpu_name in name for name in props["gpu_names"]
    ):
        raise BenchStop(
            f"host is not {args.expect_gpu_count}x{args.expect_gpu_name}: "
            f"count={n_gpus} names={props['gpu_names']}"
        )
    disk = shutil.disk_usage(output_dir)
    record("host-preflight", {
        "gpu_count": n_gpus, "gpu_names": props["gpu_names"],
        "vram_total_mib_per_gpu": per_gpu_total,
        "disk_total_gib": round(disk.total / 2**30, 1),
        "disk_free_gib": round(disk.free / 2**30, 1),
    })
    if disk.total < args.min_disk_gib * 2**30:
        raise BenchStop(
            f"usable storage below {args.min_disk_gib} GiB: "
            f"{round(disk.total / 2**30, 1)} GiB"
        )

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
    record("sample-selection", {
        "count": len(samples),
        "order": [s.sample_id for s in samples],
        "scorers": [s.scorer for s in samples],
        "references": REF_IDS,
        "b8_set": [s.sample_id for s in b8_set],
        "partial5": [s.sample_id for s in partial5],
    })

    def samples_for(n: int) -> list:
        if n <= 4:
            return refs
        if n <= 8:
            return b8_set
        return samples

    masked_set = load_expert_set(args.selected_manifest)
    if masked_set != FROZEN_SELECTED_TOP4:
        raise BenchStop(f"selected manifest != frozen top-4: {sorted(masked_set)}")
    record("selected-manifest-check", {
        "masked_experts": sorted(masked_set), "match": "frozen-top-4",
    })

    vocab_size = len(tokenizer)
    stage_status: dict[str, str] = {}  # "ok" | "oom" | "error" | "skipped"
    throughput_peaks: dict[str, tuple[int, dict[int, int]]] = {}  # stage -> (n, peaks)

    def arm_context(arm: str, observer: Any = None) -> Any:
        """Wrapper for one arm: nullcontext (native) or the path under test."""
        if arm == "native":
            return contextlib.nullcontext()
        if arm == "noop-int":
            return intervene_qwen35(architecture, masked=frozenset())
        if arm == "sel-int":
            return intervene_qwen35(architecture, masked=masked_set)
        if arm == "sel-slow":
            return instrument_qwen35(architecture, masked=masked_set, observer=observer)
        raise BenchStop(f"unknown arm: {arm}")

    def run_stage(
        stage: str, stage_samples: list, n: int, arm: str,
        observer: Any = None, min_remaining: float = 240.0,
    ) -> dict:
        """Process stage_samples in chunks of <=n under one arm. Returns rows."""
        stage_path = output_dir / f"stage-{stage}.json"
        if stage_path.exists():
            entry = json.loads(stage_path.read_text())
            report["stages"].append({"stage": stage, "resumed": True, **entry})
            stage_status[stage] = entry.get("status", "ok")
            if entry.get("status") == "ok":
                peaks = {int(k): v for k, v in (entry.get("peak_vram_mib_per_gpu") or {}).items()}
                throughput_peaks[stage] = (entry.get("batch_size", n), peaks)
            return load_stage_rows(stage, stage_samples)
        if remaining() < min_remaining:
            stage_status[stage] = "skipped"
            record(f"{stage}-skipped", {
                "reason": f"remaining {remaining():.0f}s < {min_remaining}s",
            })
            return {}
        sampler.reset()
        prefill = None
        try:
            with arm_context(arm, observer):
                prefill = _prefill_probe(tokenizer, stage_samples, config, model)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            prefill = {"error": "prefill probe OOM — skipped, cleared; chunks unaffected"}

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
                with arm_context(arm, observer):
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
            "arm": arm,
            "masked_experts": sorted(masked_set) if arm.startswith("sel") else [],
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
        if complete:
            peaks = {int(k): v for k, v in entry["peak_vram_mib_per_gpu"].items()}
            throughput_peaks[stage] = (n, peaks)
        return rows

    def vram_guard(n_next: int, need_remaining: float) -> tuple[bool, str]:
        """Delegate to the direction-aware guard (bench_guard, unit-tested)."""
        completed = [(n, stage_status.get(st, ""), peaks)
                     for st, (n, peaks) in throughput_peaks.items()]
        return predict_batch_safe(
            n_next, completed=completed, static_per_gpu=static_per_gpu,
            total_per_gpu=per_gpu_total, remaining_seconds=remaining(),
            need_remaining_seconds=need_remaining)

    def prior_oom(n_next: int) -> str | None:
        completed = [(n, stage_status.get(st, ""), peaks)
                     for st, (n, peaks) in throughput_peaks.items()]
        hit = blocking_oom(n_next, completed=completed)
        if hit is None:
            return None
        stages = sorted(st for st, (n, _p) in throughput_peaks.items()
                        if stage_status.get(st) == "oom")
        return f"{hit} ({stages[0] if stages else 'unknown stage'})"

    def stage_name(arm: str, n: int) -> str:
        return f"{arm}-B{n}"

    def run_oracle_probe(probe_samples: list, short_tokens: int = 32) -> dict:
        """Bounded slow-oracle comparison: prefill logits + short generation.

        One prefill forward per probe sample under each path plus a single
        short generation — never full 1,024-token slow generations. Reports
        max absolute/relative prefill-logit error, short-gen token agreement,
        and masked-expert route hits from the slow path's observer.
        """
        import numpy as np

        stage = "oracle-probe"
        stage_path = output_dir / f"stage-{stage}.json"
        if stage_path.exists():
            entry = json.loads(stage_path.read_text())
            report["stages"].append({"stage": stage, "resumed": True, **entry})
            return entry
        check_deadline(stage)
        hits: dict[tuple[int, int], int] = {}

        def observer(layer: int, batch: Any, _norms: Any) -> None:
            for expert in np.unique(batch.indices.flatten()).tolist():
                key = (layer, int(expert))
                hits[key] = hits.get(key, 0) + int((batch.indices == expert).sum())

        sampler.reset()
        entry: dict[str, Any] = {"status": "ok"}
        try:
            max_abs = 0.0
            max_rel = 0.0
            denom_at_max = 0.0
            with torch.inference_mode():
                for sample in probe_samples:
                    encoded = _encode_batch(tokenizer, [sample], config, model)
                    ids, mask = encoded["input_ids"], encoded["attention_mask"]
                    with intervene_qwen35(architecture, masked=masked_set):
                        fast = model(input_ids=ids, attention_mask=mask,
                                     use_cache=False).logits.float()
                    with instrument_qwen35(architecture, masked=masked_set,
                                          observer=observer):
                        slow = model(input_ids=ids, attention_mask=mask,
                                     use_cache=False).logits.float()
                    sample_max = float((fast - slow).abs().amax().item())
                    denom = float(slow.abs().amax().item())
                    if sample_max > max_abs:
                        max_abs = sample_max
                        denom_at_max = denom
                    if denom:
                        max_rel = max(max_rel, sample_max / denom)
                    del fast, slow
            torch.cuda.empty_cache()
            short_config = config.model_copy(update={
                "runtime": config.runtime.model_copy(update={"max_new_tokens": short_tokens}),
            })
            with intervene_qwen35(architecture, masked=masked_set):
                _, fast_ids, _ = _batch_generate(
                    tokenizer, probe_samples[:1], short_config, model)
            with instrument_qwen35(architecture, masked=masked_set):
                _, slow_ids, _ = _batch_generate(
                    tokenizer, probe_samples[:1], short_config, model)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            entry = {"status": "oom", "passed": False,
                     "error": "CUDA OOM in bounded probe — cleared safely"}
            record(stage, entry)
            return entry
        masked_hits = {f"({layer},{expert})": hits.get((layer, expert), 0)
                       for layer, expert in sorted(masked_set)}
        entry.update({
            "probe_samples": [s.sample_id for s in probe_samples],
            "short_gen_tokens": short_tokens,
            "max_abs_logit_diff": max_abs,
            "denominator_abs_max": denom_at_max,
            "max_rel_logit_diff": max_rel,
            "tolerance": ORACLE_BF16_TOLERANCE,
            "short_gen_token_identical": fast_ids == slow_ids,
            "masked_expert_route_hits": masked_hits,
            "passed": bool(max_abs <= ORACLE_BF16_TOLERANCE),
            **sampler.snapshot(),
        })
        record(stage, entry)
        return entry

    try:
        # ---------- Ladder largest-first: B8 decides; B1 needs 1500 s left -----
        ordered_batches = sorted(batches, reverse=True)
        for n in ordered_batches:
            floor = 1500.0 if n == 1 else 420.0
            # The guard itself approves the first batch (nothing to extrapolate
            # from; the prefill-probe + per-chunk OOM catches apply instead),
            # so every position goes through the same call.
            safe, reason = vram_guard(n, floor)
            oom = prior_oom(n)
            record(f"B{n}-guard", {"safe": safe and oom is None,
                                    "reason": reason if oom is None else f"{reason}; {oom}"})
            if not safe or oom is not None:
                for arm in ("native", "noop-int", "sel-int"):
                    stage_status[stage_name(arm, n)] = "skipped"
                continue
            for arm in ("native", "noop-int", "sel-int"):
                if arm not in args.arms:
                    continue
                run_stage(stage_name(arm, n), samples_for(n), n, arm, min_remaining=floor)

        # ---------- Bounded oracle probe (never full slow generations) ----------
        probe_entry: dict = {}
        sel_ok = [n for n in ordered_batches
                  if stage_status.get(stage_name("sel-int", n)) == "ok"]
        if not args.oracle_probe:
            report["validations"]["bounded_oracle_agreement"] = (
                "NOT RUN (skipped via --no-oracle-probe; prior result and "
                f"{ORACLE_BF16_TOLERANCE} threshold preserved)")
        elif sel_ok and remaining() > 600:
            probe_entry = run_oracle_probe([refs[0], refs[2]])
            if probe_entry.get("status") == "ok":
                report["validations"]["bounded_oracle_agreement"] = (
                    f"PASS: max_abs={probe_entry['max_abs_logit_diff']:.6f} "
                    f"max_rel={probe_entry['max_rel_logit_diff']:.6f} "
                    f"(tol {ORACLE_BF16_TOLERANCE}), short-gen identical="
                    f"{probe_entry['short_gen_token_identical']}"
                    if probe_entry.get("passed") else
                    f"FAIL: max_abs={probe_entry['max_abs_logit_diff']:.6f} "
                    f"exceeds tol {ORACLE_BF16_TOLERANCE}")
            else:
                report["validations"]["bounded_oracle_agreement"] = (
                    f"FAIL: probe status={probe_entry.get('status')}")
        else:
            report["validations"]["bounded_oracle_agreement"] = "NOT RUN (no ok sel-int stage)"

        # ---------- Per-batch determinism repeats (sel-int, every ok n) ----------
        repeat_ns = [n for n in sorted(batches, reverse=True)
                     if stage_status.get(stage_name("sel-int", n)) == "ok"]
        repeat_arm = "sel-int"
        if not repeat_ns:
            for n in sorted(batches, reverse=True):
                if stage_status.get(stage_name("noop-int", n)) == "ok":
                    repeat_ns = [n]
                    repeat_arm = "noop-int"
                    break
        for n in repeat_ns:
            base_rows = load_stage_rows(stage_name(repeat_arm, n), samples_for(n))
            repeat_rows = run_stage(f"repeat-{repeat_arm}-B{n}",
                                    samples_for(n), n, repeat_arm,
                                    min_remaining=420)
            key = f"determinism_repeat_B{n}"
            if repeat_rows and stage_status.get(f"repeat-{repeat_arm}-B{n}") != "skipped":
                ok, bad = _token_identical(
                    repeat_rows, base_rows, [s.sample_id for s in samples_for(n)])
                report["validations"][key] = (
                    "PASS token-identical" if ok else f"FAIL: {bad[:6]}")
                # Frozen-batching check per repeated n; the largest keeps the
                # Phase-1 key for report comparability.
                _chunk_frozen(stage_name(repeat_arm, n),
                              f"repeat-{repeat_arm}-B{n}", report, output_dir,
                              key=None if n != repeat_ns[0]
                              else "batching_frozen",
                              extra_key=f"batching_frozen_B{n}")
            else:
                report["validations"][key] = "NOT RUN (stage skipped)"
        if repeat_ns:
            report["validations"]["internal_determinism_repeat"] = report["validations"][
                f"determinism_repeat_B{repeat_ns[0]}"]
        else:
            report["validations"]["internal_determinism_repeat"] = "NOT RUN (no ok stage)"

        # ---------- No-op equivalence per batch (noop-int == native) ----------
        for n in batches:
            a_rows = load_stage_rows(stage_name("native", n), samples_for(n))
            b_rows = load_stage_rows(stage_name("noop-int", n), samples_for(n))
            key = f"noop_equals_native_B{n}"
            if a_rows and b_rows and stage_status.get(stage_name("native", n)) == "ok" \
                    and stage_status.get(stage_name("noop-int", n)) == "ok":
                ok, bad = _token_identical(b_rows, a_rows, [s.sample_id for s in samples_for(n)])
                report["validations"][key] = (
                    "PASS token-identical" if ok else f"FAIL: {bad[:6]}")
            else:
                report["validations"][key] = "NOT RUN (stage skipped)"

        # ---------- Batch-selection verdict (bounded-check decision rule) ----------
        layer_diag: dict = {}
        if args.layer_diagnostic is not None:
            if args.layer_diagnostic.exists():
                try:
                    layer_diag = json.loads(args.layer_diagnostic.read_text())
                except Exception as exc:
                    layer_diag = {"load_error": str(exc)[:160]}
            else:
                layer_diag = {"load_error": f"missing: {args.layer_diagnostic}"}
        layer_ok = (layer_diag.get("status") == "pass"
                    and layer_diag.get("passed") is True)
        ceiling = min(per_gpu_total.values())
        vram_cell = int(ceiling * 0.92)
        selection: dict[str, Any] = {}
        for n in sorted(batches, reverse=True):
            stage = stage_name("sel-int", n)
            entry_path = output_dir / f"stage-{stage}.json"
            checks: dict[str, bool] = {}
            detail: dict[str, Any] = {}
            if stage_status.get(stage) == "ok" and entry_path.exists():
                entry = json.loads(entry_path.read_text())
                peaks = {int(k): v for k, v in
                         (entry.get("peak_vram_mib_per_gpu") or {}).items()}
                peak_max = max(peaks.values()) if peaks else None
                p90 = entry.get("p90_latency_seconds")
                conserv_spm = (60.0 / p90) if p90 else None
                checks = {
                    "no_oom_or_error": True,
                    "vram_peak_within_92pct": peak_max is not None
                    and peak_max <= vram_cell,
                    "conservative_spm_at_least_3_4": conserv_spm is not None
                    and conserv_spm >= 3.4,
                    "noop_equiv": report["validations"].get(
                        f"noop_equals_native_B{n}", "").startswith("PASS"),
                    "determinism": report["validations"].get(
                        f"determinism_repeat_B{n}", "").startswith("PASS"),
                    "layer_diagnostic": layer_ok,
                }
                detail = {"peak_max_mib": peak_max,
                          "conservative_spm": round(conserv_spm, 3)
                          if conserv_spm is not None else None}
            qualifies = bool(checks) and all(checks.values())
            selection[f"B{n}"] = {"checks": checks, "detail": detail,
                                  "qualifies": qualifies}
        selected = max(
            (int(name[1:]) for name, info in selection.items() if info["qualifies"]),
            default=None)
        if selected is not None:
            info = selection[f"B{selected}"]
            report["validations"]["batch_selection"] = (
                f"SELECT B{selected}: peak {info['detail']['peak_max_mib']} MiB "
                f"<= {vram_cell}, conserv {info['detail']['conservative_spm']} spm, "
                "noop-equiv/determinism/layer-diagnostic PASS")
        else:
            failed = {name: [c for c, ok in info["checks"].items() if not ok]
                      for name, info in selection.items() if info["checks"]}
            report["validations"]["batch_selection"] = (
                "NO BATCH QUALIFIES"
                + (f": {failed}" if failed else " (no ok sel-int stage)")
                + ("; layer-diagnostic NOT incorporated "
                   f"({layer_diag.get('load_error', 'not provided')})"
                   if not layer_ok and args.layer_diagnostic is None
                   else ""))
        record("batch-selection", {
            "selected_batch": selected,
            "ceiling_92pct_mib": vram_cell,
            "conservative_spm_floor": 3.4,
            "layer_diagnostic": {
                "path": str(args.layer_diagnostic)
                if args.layer_diagnostic is not None else None,
                "status": layer_diag.get("status"),
                "passed": layer_diag.get("passed"),
                "load_error": layer_diag.get("load_error"),
            },
            "batches": selection,
        })

        # ---------- Decision batch: selected, else largest ok (fallback) ----------
        decision_n = selected
        if decision_n is None:
            for n in sorted(batches, reverse=True):
                if stage_status.get(stage_name("sel-int", n)) == "ok":
                    decision_n = n
                    break
        if decision_n is None:
            for n in sorted(batches, reverse=True):
                if stage_status.get(stage_name("noop-int", n)) == "ok":
                    decision_n = n
                    break
        decision_has_sel = (
            decision_n is not None
            and stage_status.get(stage_name("sel-int", decision_n)) == "ok")

        # ---------- D-engagement: sel-int differs from noop-int ----------
        if decision_has_sel:
            assert decision_n is not None
            d_rows = load_stage_rows(stage_name("sel-int", decision_n),
                                     samples_for(decision_n))
            b_rows = load_stage_rows(stage_name("noop-int", decision_n),
                                     samples_for(decision_n))
            ref_ids = [s.sample_id for s in samples_for(decision_n)
                       if s.sample_id in set(REF_IDS)]
            unchanged = [rid for rid in ref_ids
                         if rid in d_rows and rid in b_rows
                         and d_rows[rid]["response"] == b_rows[rid]["response"]]
            missing = [rid for rid in ref_ids if rid not in d_rows or rid not in b_rows]
            report["validations"]["selected_engagement"] = (
                "PASS: all refs differ under the frozen top-four, zeroed side-path "
                "without rerouting or renormalization"
                if not unchanged and not missing and d_rows
                else f"FAIL: unchanged={unchanged} missing={missing}")
        else:
            report["validations"]["selected_engagement"] = "NOT RUN (no ok stage)"

        # ---------- Partial-final-batch + exact-resume ----------
        if decision_n is not None:
            part_a = run_stage("partial-a", partial5, decision_n, "noop-int",
                               min_remaining=480)
            part_b = run_stage("partial-b", partial5, decision_n, "noop-int",
                               min_remaining=480)
            if part_a and part_b and stage_status.get("partial-a") != "skipped" \
                    and stage_status.get("partial-b") != "skipped":
                ok, bad = _token_identical(part_a, part_b, [s.sample_id for s in partial5])
                report["validations"]["partial_resume_identical"] = (
                    "PASS token-identical" if ok else f"FAIL: {bad[:6]}")
            else:
                report["validations"]["partial_resume_identical"] = "NOT RUN (stage skipped)"
    except BenchStop as stop:
        report["stop_reason"] = str(stop)
    except torch.OutOfMemoryError:
        # Last-resort catch so stage records survive and the report is written.
        torch.cuda.empty_cache()
        report["stop_reason"] = "OOM escaped a stage (records preserved)"

    # ---------- per-arm projections ----------
    try:
        projections: dict[str, Any] = {"gens": 1300, "rate_usd_per_hour": args.rate_usd_per_hour}
        for arm in args.arms:
            basis = None
            for n in sorted(batches, reverse=True):
                p = output_dir / f"stage-{stage_name(arm, n)}.json"
                if p.exists():
                    entry = json.loads(p.read_text())
                    if entry.get("status") == "ok" and entry.get("samples_per_minute"):
                        basis = (n, entry)
                        break
            if basis is None:
                projections[arm] = {"error": "no completed stage to project from"}
                continue
            n, entry = basis
            spm = entry["samples_per_minute"]
            p90 = entry.get("p90_latency_seconds") or (3600.0 / (spm * 60))
            expected_h = 1300 / (spm * 60)
            conserv_h = 1300 * p90 / 3600
            setup_cost = args.rate_usd_per_hour * args.setup_minutes / 60
            ceiling = min(per_gpu_total.values())
            peaks = {int(k): v for k, v in (entry.get("peak_vram_mib_per_gpu") or {}).items()}
            headroom_ok = bool(peaks) and all(v <= int(ceiling * 0.92) for v in peaks.values())
            projections[arm] = {
                "basis_stage": f"{arm}-B{n}",
                "samples_per_minute": spm,
                "p90_sample_latency_seconds": p90,
                "expected_wall_hours": round(expected_h, 2),
                "conservative_wall_hours": round(conserv_h, 2),
                "setup_burn_usd": round(setup_cost, 2),
                "expected_cost_usd": round(args.rate_usd_per_hour * expected_h + setup_cost, 2),
                "conservative_cost_usd": round(args.rate_usd_per_hour * conserv_h + setup_cost, 2),
                "vram_headroom": "safe" if headroom_ok else "tight/unknown",
                "qualifies": bool(conserv_h < 6.4
                                  and args.rate_usd_per_hour * conserv_h + setup_cost < 10
                                  and headroom_ok),
            }
        projections["qualification_rule"] = (
            "conservative < 6.4 host-hours AND conservative < $10 AND safe VRAM headroom")
        report["projection"] = projections
        record("projection", projections)
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


def _chunk_frozen(stage_a: str, stage_b: str, report: dict, output_dir: Path,
                  key: str | None = "batching_frozen",
                  extra_key: str | None = None) -> None:
    """Gate: identical membership/order/padding/attention masks across stages."""
    target_keys = [k for k in (key, extra_key) if k is not None] or ["batching_frozen"]
    try:
        a = json.loads((output_dir / f"stage-{stage_a}.json").read_text()).get("chunk_manifest", [])
        b = json.loads((output_dir / f"stage-{stage_b}.json").read_text()).get("chunk_manifest", [])
    except Exception as exc:
        for target in target_keys:
            report["validations"][target] = f"UNKNOWN: {exc}"
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
    verdict = ("PASS: membership, ordering, padding and attention masks frozen"
               if not problems else f"FAIL: {problems[:4]}")
    for target in target_keys:
        report["validations"][target] = verdict


if __name__ == "__main__":
    sys.exit(main())
