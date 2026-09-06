"""Bounded RTX PRO 6000 (Blackwell, 96 GB) batching benchmark.

Directive 2026-09-05: determine whether an RTX PRO 6000 should host the main
causal-validation run. NON-SCIENTIFIC performance probe — writes no
scientific artifacts, resumes no run, changes nothing. Reuses the EXACT
production code paths (load_donor, validate_donor_contract, _generate,
instrument_qwen35, load_expert_set, the pinned gen config's runtime
parameters: max_new_tokens=1024, use_cache, greedy, enable_thinking=false).

Stage order (directive): B1 on 4 reference samples -> B2 on the same refs ->
B4 on the same refs -> [noop + selected validations on refs] -> B8 (8
samples incl. the refs) -> B16 (all 16). B8/B16 only run if predicted VRAM
headroom is safe; a B16 OOM is recorded, cleared safely, and preserves all
completed smaller-batch results.

The 16 samples are the saved 4x3090 benchmark set (9, in its recorded
manifest order) plus 7 more validation samples (+2 exact_match, +3
unit_tests, +2 swebench, manifest order). References = first sample of each
scorer type in that order + one extra unit_tests sample.

Correctness gates:
  - B1 duplicate: token-identical response
  - noop (empty mask): response == baseline exactly
  - selected (frozen top-4 {(35,239),(3,26),(7,18),(37,5)}): masks exactly 4
    experts, zeroing weighted contributions side-path without rerouting or
    renormalization (instrumentation.py semantics); every response must
    differ from baseline; observer counts masked-expert route hits to prove
    the intended experts were actually engaged and changed
  - batched B2/B4/B8/B16: per-row response == independent B1 response
    (decoded-text exact) AND generated token counts equal AND generated-id
    sequences equal across batch sizes; left padding must not alter prompts
    (pad-stripped input ids reconstruct the B1 prompt ids)

Every sample/stage record is written atomically the moment it exists; a
re-run with --output-dir skips stages whose records already exist (exact
resume; the model reloads once per process, which is the only state that is
not preserved across restarts).

Hard cap: --deadline-seconds (default 3600) with a teardown reserve.
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

from reverse_reap.causal import _generate, load_expert_set
from reverse_reap.config import load_config
from reverse_reap.datasets import load_manifest
from reverse_reap.instrumentation import instrument_qwen35
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import load_donor, validate_donor_contract

# Saved 4x3090 benchmark selection, in its recorded manifest order
# (runs/bench-20260904/stage-sample-selection.json, archive B-final).
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
# unit_tests (type-covering, deterministic).
REF_IDS = [
    "4371146b066c3f9643baafd4",  # unit_tests
    "84ee9615bd1e09a6f3b83770",  # unit_tests
    "7db4d0460f2cb926e2cbf1a4",  # exact_match
    "2542b6cd29768cd223836536",  # swebench
]
TOP_UP = {"exact_match": 2, "unit_tests": 3, "swebench": 2}  # +7 -> 16 total


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
    """Background thread: peak VRAM (nvidia-smi) + CPU load."""

    def __init__(self) -> None:
        self.peak_vram_mib = 0
        self.loadavg: list[float] = []
        self.warnings_seen: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def reset(self) -> None:
        self.peak_vram_mib = 0

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
                    if parts and parts[0] == "0" and int(parts[1]) > self.peak_vram_mib:
                        self.peak_vram_mib = int(parts[1])
                self.loadavg.append(os.getloadavg()[0])
            except Exception:
                pass
            self._stop.wait(2.0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "peak_vram_mib": self.peak_vram_mib,
            "peak_vram_gib": round(self.peak_vram_mib / 1024, 2),
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
        "prefill_tokens_per_second": round(tokens / wall, 1),
    }


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


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    args = parser.parse_args()

    deadline = time.monotonic() + args.deadline_seconds
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark": "rtx-pro-6000-batching-feasibility",
        "scientific": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deadline_seconds": args.deadline_seconds,
        "stages": [],
        "validations": {},
        "stop_reason": "completed",
    }

    def remaining() -> float:
        return deadline - time.monotonic()

    def check_deadline(stage: str) -> None:
        if remaining() < 180:  # keep 3 min for teardown/report
            raise BenchStop(f"deadline reached before stage {stage}")

    def record(stage: str, payload: dict) -> None:
        entry = {"stage": stage, **payload}
        report["stages"].append(entry)
        _atomic_write_json(output_dir / f"stage-{stage}.json", entry)
        print(f"[stage] {stage}: {json.dumps(payload, default=str)[:240]}", flush=True)

    def skipped(stage: str, exists: bool) -> bool:
        if exists:
            entry = json.loads((output_dir / f"stage-{stage}.json").read_text())
            report["stages"].append({"stage": stage, "resumed": True, **entry})
            print(f"[resume] {stage}: reused existing record", flush=True)
            return True
        return False

    config = load_config(args.config)
    sampler = HostSampler()
    sampler.start()

    # ---------- model load (once per process) ----------
    if not skipped("model-load", (output_dir / "stage-model-load.json").exists()):
        started = time.monotonic()
        model, tokenizer = load_donor(args.model_path, config)
        load_seconds = round(time.monotonic() - started, 1)
        architecture = inspect_qwen35_moe(model)
        validate_donor_contract(model, architecture)
        props = {
            "seconds": load_seconds,
            "contract": "valid",
            "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers_version": __import__("transformers").__version__,
            "python": platform.python_version(),
            "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        }
        record("model-load", props)
        env = {
            "stage": "environment",
            "gpu_name": props["device_name"],
            "compute_capability": props["device_capability"],
            "vram_total_gib": props["vram_total_gib"],
            "torch": props["torch_version"],
            "cuda": props["cuda_version"],
            "transformers": props["transformers_version"],
            "python": props["python"],
            "pinned_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
            "max_new_tokens": config.runtime.max_new_tokens,
            "use_cache": config.runtime.use_cache,
            "enable_thinking": config.runtime.enable_thinking,
        }
        _atomic_write_json(output_dir / "environment-report.json", env)
    else:
        model, tokenizer = load_donor(args.model_path, config)
        architecture = inspect_qwen35_moe(model)
        validate_donor_contract(model, architecture)

    samples = _select_samples(args.manifest)
    by_id = {s.sample_id: s for s in samples}
    refs = [by_id[i] for i in REF_IDS]
    b8_set = refs + [by_id[i] for i in
                     ["bc5e6501895b5fd99348f4bd", "52afb433a40c031e1b894ddd",
                      "53a7e1bcf2745e04b3cd4a12", "8ff868702cb4a25c487f499a"]]
    record("sample-selection", {
        "count": len(samples),
        "order": [s.sample_id for s in samples],
        "scorers": [s.scorer for s in samples],
        "references": REF_IDS,
    })

    baseline: dict[str, str] = {}
    baseline_counts: dict[str, int] = {}
    try:
        # ---------- Stage B1: four reference samples via production _generate ----------
        b1_walls: list[float] = []
        b1_tokens: list[int] = []
        for sample in refs:
            out_path = output_dir / "B1" / f"{sample.sample_id}.json"
            if out_path.exists():
                rec = json.loads(out_path.read_text())
                baseline[sample.sample_id] = rec["response"]
                baseline_counts[sample.sample_id] = rec["generated_tokens"]
                b1_walls.append(rec.get("wall_seconds", 0.0))
                b1_tokens.append(rec["generated_tokens"])
                continue
            check_deadline("B1")
            gen_start = time.monotonic()
            response, tokens, truncated = _generate(model, tokenizer, sample, config)
            wall = round(time.monotonic() - gen_start, 2)
            baseline[sample.sample_id] = response
            baseline_counts[sample.sample_id] = tokens
            b1_walls.append(wall)
            b1_tokens.append(tokens)
            _atomic_write_json(out_path, {
                "sample_id": sample.sample_id, "scorer": sample.scorer,
                "response": response, "generated_tokens": tokens,
                "truncated": truncated, "wall_seconds": wall,
            })
        total_wall = sum(b1_walls)
        record("B1-refs", {
            "samples": len(refs),
            "wall_seconds": round(total_wall, 1),
            "samples_per_minute": round(len(refs) / (total_wall / 60), 2)
            if total_wall else None,
            "mean_tokens_per_sample": round(sum(b1_tokens) / len(b1_tokens), 1),
            "decode_tokens_per_second": round(sum(b1_tokens) / total_wall, 1)
            if total_wall else None,
            "per_sample_walls": b1_walls,
            **sampler.snapshot(),
        })

        # ---------- Stage B1-dup: determinism duplicate of first reference ----------
        if not skipped("B1-dup", (output_dir / "stage-B1-dup.json").exists()):
            check_deadline("B1-dup")
            dup_start = time.monotonic()
            dup_response, dup_tokens, _ = _generate(model, tokenizer, refs[0], config)
            match = dup_response == baseline[refs[0].sample_id] and \
                dup_tokens == baseline_counts[refs[0].sample_id]
            report["validations"]["batch1_determinism"] = "PASS token-identical" if match \
                else "FAIL: mismatch"
            record("B1-dup", {
                "sample_id": refs[0].sample_id,
                "wall_seconds": round(time.monotonic() - dup_start, 2),
                "match": match,
            })

        # ---------- Batched stages B2/B4 on refs; B8/B16 with VRAM guard ----------
        batch_plans = [("B2", refs), ("B4", refs)]
        peak_after = {"B2": None, "B4": None, "B8": None}

        def run_batched(stage: str, stage_samples: list) -> None:
            stage_path = output_dir / f"stage-{stage}.json"
            if stage_path.exists():
                entry = json.loads(stage_path.read_text())
                report["stages"].append({"stage": stage, "resumed": True, **entry})
                peak_after[stage] = entry.get("peak_vram_mib")
                for s in stage_samples:
                    out = output_dir / stage / f"{s.sample_id}.json"
                    if out.exists():
                        rec = json.loads(out.read_text())
                        baseline.setdefault(s.sample_id, rec["response"])
                print(f"[resume] {stage}: reused existing record", flush=True)
                return
            check_deadline(stage)
            sampler.reset()
            responses, gen_ids, metrics = _batch_generate(tokenizer, stage_samples, config, model)
            prefill = _prefill_probe(tokenizer, stage_samples, config, model)
            latencies = [metrics["wall_seconds"] / len(stage_samples)] * len(stage_samples)
            mismatches: list[str] = []
            id_mismatches: list[str] = []
            compared = 0
            for s, text, ids in zip(stage_samples, responses, gen_ids, strict=True):
                _atomic_write_json(output_dir / stage / f"{s.sample_id}.json", {
                    "sample_id": s.sample_id, "response": text,
                    "generated_tokens": len(ids),
                })
                # Equivalence is required only for the four REFERENCE samples
                # (they are the only rows with independent B1 references).
                if s.sample_id not in baseline:
                    continue
                compared += 1
                if text != baseline[s.sample_id]:
                    mismatches.append(s.sample_id)
                if len(ids) != baseline_counts[s.sample_id]:
                    id_mismatches.append(s.sample_id)
            equivalence = (
                f"PASS ({compared} refs compared)" if not mismatches and not id_mismatches
                else f"FAIL: text {mismatches[:4]} counts {id_mismatches[:4]}")
            report["validations"][f"{stage}_equals_batch1"] = equivalence
            decode_seconds = max(metrics["wall_seconds"] - prefill["prefill_seconds"], 1e-6)
            record(stage, {
                "samples": len(stage_samples),
                "sample_ids": [s.sample_id for s in stage_samples],
                **metrics,
                "prefill": prefill,
                "samples_per_minute": round(len(stage_samples) / metrics["wall_seconds"] * 60, 2),
                "decode_tokens_per_second": round(metrics["generated_tokens"] / decode_seconds, 1),
                "mean_latency_seconds": round(metrics["wall_seconds"] / len(stage_samples), 3),
                "p90_latency_seconds": round(_p90(latencies), 3),
                "mean_generated_tokens_per_sample": round(
                    metrics["generated_tokens"] / len(stage_samples), 1),
                "mismatched_sample_ids": mismatches,
                "token_count_mismatches": id_mismatches,
                **sampler.snapshot(),
            })
            peak_after[stage] = sampler.peak_vram_mib
            # Equivalence failures are REPORTED, not fatal: the directive needs
            # per-batch-size throughput AND equivalence results; a mismatch
            # invalidates scientific use of that batch size but must not erase
            # measurements (same principle as the OOM rule).

        run_batched("B2", refs)
        run_batched("B4", refs)

        # ---------- VRAM headroom guard ----------
        total_vram_mib = int(torch.cuda.get_device_properties(0).total_memory / 2**20)

        def headroom_safe(stage: str, prev_stage: str, added: int) -> tuple[bool, str]:
            peak_prev = peak_after.get(prev_stage)
            if peak_prev is None:
                return False, f"no peak VRAM recorded for {prev_stage}"
            peak_b2 = peak_after.get("B2") or peak_prev
            marginal = max((peak_prev - peak_b2) / max(added // 2, 1), 0)
            predicted = peak_prev + added * marginal + 2048  # 2 GiB guard
            safe = predicted <= int(total_vram_mib * 0.92)
            return safe, (
                f"predicted {predicted} MiB vs total {total_vram_mib} MiB "
                f"(marginal {marginal:.0f} MiB/sample from {prev_stage})")

        for stage, stage_samples in (("B8", b8_set), ("B16", samples)):
            prev = "B4" if stage == "B8" else "B8"
            added = len(stage_samples) - (4 if stage == "B8" else 8)
            safe, reason = headroom_safe(stage, prev, added)
            record(f"{stage}-guard", {"safe": safe, "reason": reason})
            if not safe:
                report["validations"][f"{stage}_skipped"] = f"VRAM guard: {reason}"
                continue
            try:
                run_batched(stage, stage_samples)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                report["validations"][f"{stage}_vram"] = (
                    "OOM — recorded, allocation cleared safely, smaller-batch results preserved")
                record(stage, {"error": "CUDA OOM", "cleared_safely": True})
                continue

        # ---------- noop + selected validations on refs (B1 production path) ----------
        if not skipped("noop", (output_dir / "stage-noop.json").exists()):
            noop_walls: list[float] = []
            noop_mismatches: list[str] = []
            for sample in refs:
                out_path = output_dir / "noop" / f"{sample.sample_id}.json"
                if out_path.exists():
                    rec = json.loads(out_path.read_text())
                    if rec.get("response") != baseline[sample.sample_id]:
                        noop_mismatches.append(sample.sample_id)
                    continue
                check_deadline("noop")
                gen_start = time.monotonic()
                with instrument_qwen35(architecture, masked=frozenset()):
                    response, tokens, _ = _generate(model, tokenizer, sample, config)
                noop_walls.append(round(time.monotonic() - gen_start, 2))
                if response != baseline[sample.sample_id]:
                    noop_mismatches.append(sample.sample_id)
                _atomic_write_json(out_path, {
                    "response": response, "generated_tokens": tokens,
                    "equals_baseline": response == baseline[sample.sample_id],
                })
            report["validations"]["noop_equivalence"] = (
                "PASS" if not noop_mismatches else f"FAIL: {noop_mismatches}")
            record("noop", {
                "samples": len(refs), "wall_seconds": round(sum(noop_walls), 1),
                "mismatches": noop_mismatches, **sampler.snapshot(),
            })

        if not skipped("selected", (output_dir / "stage-selected.json").exists()):
            masked_set = load_expert_set(args.selected_manifest)
            expected = {(35, 239), (3, 26), (7, 18), (37, 5)}
            if masked_set != expected:
                raise BenchStop(f"selected manifest != frozen top-4: {sorted(masked_set)}")
            hits, observer = _observer_counter(masked_set)
            sel_walls: list[float] = []
            unchanged: list[str] = []
            for sample in refs:
                out_path = output_dir / "selected" / f"{sample.sample_id}.json"
                if out_path.exists():
                    rec = json.loads(out_path.read_text())
                    if not rec.get("differs_from_baseline"):
                        unchanged.append(sample.sample_id)
                    continue
                check_deadline("selected")
                gen_start = time.monotonic()
                with instrument_qwen35(architecture, masked=masked_set, observer=observer):
                    response, tokens, _ = _generate(model, tokenizer, sample, config)
                sel_walls.append(round(time.monotonic() - gen_start, 2))
                differs = response != baseline[sample.sample_id]
                if not differs:
                    unchanged.append(sample.sample_id)
                _atomic_write_json(out_path, {
                    "response": response, "differs_from_baseline": differs,
                    "generated_tokens": tokens,
                })
            masked_hits = {f"({l},{e})": hits.get((l, e), 0) for l, e in sorted(masked_set)}
            report["validations"]["selected_intervention"] = (
                "PASS: all refs differ, masked_experts=4 (frozen set), zeroed "
                "side-path without renormalization"
                if not unchanged else f"FAIL: {len(unchanged)} refs unchanged")
            record("selected", {
                "masked_experts": sorted(masked_set),
                "masked_expert_route_hits": masked_hits,
                "samples": len(refs), "differing": len(refs) - len(unchanged),
                "wall_seconds": round(sum(sel_walls), 1),
                "note": "masking zeroes weighted contributions on the side path; "
                        "router probabilities and surviving experts unchanged",
                **sampler.snapshot(),
            })
    except BenchStop as stop:
        report["stop_reason"] = str(stop)
    sampler.stop()
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["elapsed_seconds"] = round(args.deadline_seconds - remaining(), 1)
    _atomic_write_json(output_dir / "benchmark-report.json", report)
    print(json.dumps({"stop_reason": report["stop_reason"],
                      "validations": report["validations"]}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
