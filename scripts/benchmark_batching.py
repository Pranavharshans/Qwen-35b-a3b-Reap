"""NON-SCIENTIFIC batching benchmark for the four-RTX-3090 host.

Directive 2026-09-04: measure whether batched greedy generation can make the
1,300-generation causal validation stage fit the authorized budget. This
script is a performance probe ONLY — it writes no scientific artifacts,
resumes nothing, and changes nothing. It reuses the EXACT production code
paths (load_donor, validate_donor_contract, instrument_qwen35, the gen
config's runtime parameters) so its batch-1 reference is the production
reference behavior.

Stages (batch sizes 1, 2, 4, 8 sequentially; stop on OOM/mismatch/cap):
  - B1 baseline over representative frozen validation samples
    (3 exact_match + 3 unit_tests + 3 swebench + 1 control, manifest order)
  - B1 determinism duplicate of the first sample
  - B1 no-op intervention (empty mask) — must equal baseline
  - B1 selected intervention (frozen top-4) — must differ, masked==4
  - Bn baseline batches — every response must equal its batch-1 response
Metrics per stage: wall time, prefill tok/s, decode tok/s, samples/min,
generated tokens/sample, peak VRAM per GPU (nvidia-smi sampler), CPU load.
Every sample record is written atomically the moment it exists, so a
timeout discards no completed measurement.

Hard caps: --deadline-seconds (default 2700 = 45 min) plus OOM, output
mismatch, or intervention mismatch — each stops escalation immediately.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from reverse_reap.causal import _generate, load_expert_set
from reverse_reap.config import ExperimentConfig, load_config
from reverse_reap.datasets import load_manifest
from reverse_reap.instrumentation import instrument_qwen35
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import load_donor, validate_donor_contract


class BenchStop(RuntimeError):
    """Raised on OOM, mismatch, or deadline — stops batch-size escalation."""


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
    """Background thread: peak VRAM per GPU (nvidia-smi) + CPU load."""

    def __init__(self) -> None:
        self.peak_vram_mib: dict[int, int] = {}
        self.loadavg: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                for line in out.strip().splitlines():
                    index_text, mib_text = (part.strip() for part in line.split(","))
                    gpu = int(index_text)
                    used = int(mib_text)
                    if used > self.peak_vram_mib.get(gpu, 0):
                        self.peak_vram_mib[gpu] = used
                self.loadavg.append(os.getloadavg()[0])
            except Exception:
                pass
            self._stop.wait(2.0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "peak_vram_mib_per_gpu": {str(k): v for k, v in sorted(self.peak_vram_mib.items())},
            "loadavg_mean": round(sum(self.loadavg) / len(self.loadavg), 2)
            if self.loadavg
            else None,
            "loadavg_peak": round(max(self.loadavg), 2) if self.loadavg else None,
        }


def _encode_batch(tokenizer: Any, samples: list, config: ExperimentConfig, model: Any) -> dict:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": sample.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.runtime.enable_thinking,
        )
        for sample in samples
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    return {
        key: value.to(model.get_input_embeddings().weight.device) for key, value in encoded.items()
    }


def _prefill_probe(tokenizer: Any, samples: list, config: ExperimentConfig, model: Any) -> dict:
    """One prefill-only forward over the batch's prompts (no decode)."""
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


def _batch_generate(
    tokenizer: Any, samples: list, config: ExperimentConfig, model: Any
) -> tuple[list[str], list[int], dict]:
    """Left-padded greedy batch generation via the production parameters."""
    import torch

    encoded = _encode_batch(tokenizer, samples, config, model)
    prompt_width = encoded["input_ids"].shape[1]
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=config.runtime.max_new_tokens,
            use_cache=config.runtime.use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    wall = time.monotonic() - started
    generated = output[:, prompt_width:]
    responses = [
        tokenizer.decode(generated[row], skip_special_tokens=True)
        for row in range(generated.shape[0])
    ]
    counts = [
        int((generated[row] != tokenizer.pad_token_id).sum()) for row in range(generated.shape[0])
    ]
    metrics = {
        "wall_seconds": round(wall, 2),
        "prompt_tokens": int(encoded["attention_mask"].sum()),
        "generated_tokens": int(sum(counts)),
        "tokens_per_second_overall": round(sum(counts) / wall, 1),
    }
    return responses, counts, metrics


def _select_samples(manifest_path: Path) -> list:
    """3 exact_match + 3 unit_tests + 3 swebench + 1 control, manifest order."""
    wanted = {"exact_match": 3, "unit_tests": 3, "swebench": 3, "control": 1}
    counts = dict.fromkeys(wanted, 0)
    selected: list = []
    for sample in load_manifest(manifest_path):
        if sample.split != "validation":
            continue
        scorer = "control" if sample.scorer == "control" else sample.scorer
        if scorer in wanted and counts[scorer] < wanted[scorer]:
            counts[scorer] += 1
            selected.append(sample)
        if sum(counts.values()) == sum(wanted.values()):
            break
    if len(selected) < 7:
        raise BenchStop(f"manifest yielded only {len(selected)} representative samples")
    return selected


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--deadline-seconds", type=int, default=2700)
    args = parser.parse_args()

    deadline = time.monotonic() + args.deadline_seconds
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark": "batching-feasibility",
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
        if remaining() < 120:  # keep 2 min for teardown/report
            raise BenchStop(f"deadline reached before stage {stage}")

    def record(stage: str, payload: dict) -> None:
        entry = {"stage": stage, **payload}
        report["stages"].append(entry)
        _atomic_write_json(output_dir / f"stage-{stage}.json", entry)
        print(f"[stage] {stage}: {json.dumps(payload)[:220]}", flush=True)

    config = load_config(args.config)
    sampler = HostSampler()
    sampler.start()

    started = time.monotonic()
    model, tokenizer = load_donor(args.model_path, config)
    load_seconds = round(time.monotonic() - started, 1)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    record("model-load", {"seconds": load_seconds, "contract": "valid"})

    samples = _select_samples(args.manifest)
    record(
        "sample-selection",
        {
            "count": len(samples),
            "order": [s.sample_id for s in samples],
            "scorers": [s.scorer for s in samples],
        },
    )

    baseline: dict[str, str] = {}
    try:
        # ---- Stage B1: batch-1 production reference baseline ----
        b1_walls: list[float] = []
        for sample in samples:
            check_deadline("B1")
            gen_start = time.monotonic()
            response, tokens, truncated = _generate(model, tokenizer, sample, config)
            wall = round(time.monotonic() - gen_start, 2)
            baseline[sample.sample_id] = response
            b1_walls.append(wall)
            _atomic_write_json(
                output_dir / "B1" / f"{sample.sample_id}.json",
                {
                    "sample_id": sample.sample_id,
                    "scorer": sample.scorer,
                    "response": response,
                    "generated_tokens": tokens,
                    "truncated": truncated,
                    "wall_seconds": wall,
                },
            )
        total_wall = sum(b1_walls)
        total_tokens = sum(
            json.loads((output_dir / "B1" / f"{s.sample_id}.json").read_text())["generated_tokens"]
            for s in samples
        )
        record(
            "B1-baseline",
            {
                "samples": len(samples),
                "wall_seconds": round(total_wall, 1),
                "samples_per_minute": round(len(samples) / (total_wall / 60), 2),
                "mean_tokens_per_sample": round(total_tokens / len(samples), 1),
                "decode_tokens_per_second": round(total_tokens / total_wall, 1),
                **sampler.snapshot(),
            },
        )
        report["validations"]["batch1_reference"] = (
            "production _generate path; per-sample records written"
        )

        # ---- Stage B1-dup: determinism duplicate of the first sample ----
        check_deadline("B1-dup")
        dup_start = time.monotonic()
        dup_response, _, _ = _generate(model, tokenizer, samples[0], config)
        first_id = samples[0].sample_id
        match = dup_response == baseline[first_id]
        report["validations"]["batch1_determinism"] = "PASS" if match else "FAIL: mismatch"
        record(
            "B1-dup",
            {
                "sample_id": first_id,
                "wall_seconds": round(time.monotonic() - dup_start, 2),
                "match": match,
            },
        )

        # ---- Stage noop: empty-mask no-op must equal baseline ----
        check_deadline("noop")
        noop_walls: list[float] = []
        noop_mismatches: list[str] = []
        for sample in samples:
            check_deadline("noop")
            gen_start = time.monotonic()
            with instrument_qwen35(architecture, masked=frozenset()):
                response, tokens, _ = _generate(model, tokenizer, sample, config)
            noop_walls.append(round(time.monotonic() - gen_start, 2))
            if response != baseline[sample.sample_id]:
                noop_mismatches.append(sample.sample_id)
            _atomic_write_json(
                output_dir / "noop" / f"{sample.sample_id}.json",
                {"response": response, "generated_tokens": tokens},
            )
        noop_ok = not noop_mismatches
        report["validations"]["noop_equivalence"] = (
            "PASS" if noop_ok else f"FAIL: {noop_mismatches}"
        )
        record(
            "noop",
            {
                "samples": len(samples),
                "wall_seconds": round(sum(noop_walls), 1),
                "samples_per_minute": round(len(samples) / (sum(noop_walls) / 60), 2),
                "mismatches": noop_mismatches,
            },
        )

        # ---- Stage selected: frozen top-4 must change output, masked==4 ----
        check_deadline("selected")
        masked_set = load_expert_set(args.selected_manifest)
        if len(masked_set) != 4:
            raise BenchStop(f"selected manifest does not carry 4 experts: {len(masked_set)}")
        sel_walls: list[float] = []
        differing = 0
        for sample in samples:
            check_deadline("selected")
            gen_start = time.monotonic()
            with instrument_qwen35(architecture, masked=masked_set):
                response, tokens, _ = _generate(model, tokenizer, sample, config)
            sel_walls.append(round(time.monotonic() - gen_start, 2))
            changed = response != baseline[sample.sample_id]
            differing += int(changed)
            _atomic_write_json(
                output_dir / "selected" / f"{sample.sample_id}.json",
                {
                    "response": response,
                    "differs_from_baseline": changed,
                    "generated_tokens": tokens,
                },
            )
        selected_ok = differing == len(samples)
        report["validations"]["selected_intervention"] = (
            "PASS: all samples differ, masked_experts=4 (frozen set)"
            if selected_ok
            else f"FAIL: {len(samples) - differing} samples unchanged"
        )
        record(
            "selected",
            {
                "masked_experts": sorted(masked_set),
                "samples": len(samples),
                "differing": differing,
                "wall_seconds": round(sum(sel_walls), 1),
                **sampler.snapshot(),
            },
        )

        # ---- Stages B2/B4/B8: batched baseline must equal batch-1 ----
        for batch_size in (int(b) for b in args.batch_sizes.split(",") if b.strip()):
            if batch_size == 1:
                continue
            check_deadline(f"B{batch_size}")
            sampler.peak_vram_mib.clear()
            batch_samples = list(samples)
            try:
                responses, counts, metrics = _batch_generate(
                    tokenizer, batch_samples, config, model
                )
                prefill = _prefill_probe(tokenizer, batch_samples, config, model)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                report["validations"][f"B{batch_size}_vram"] = "OOM — stopped escalation"
                record(f"B{batch_size}", {"error": "CUDA OOM"})
                break
            batch_responses: dict[str, str] = {}
            for sample_obj, response, count in zip(batch_samples, responses, counts, strict=True):
                batch_responses[sample_obj.sample_id] = response
                _atomic_write_json(
                    output_dir / f"B{batch_size}" / f"{sample_obj.sample_id}.json",
                    {"response": response, "generated_tokens": count},
                )
            mismatches = [
                s.sample_id
                for s in batch_samples
                if batch_responses[s.sample_id] != baseline[s.sample_id]
            ]
            report["validations"][f"B{batch_size}_equals_batch1"] = (
                "PASS"
                if not mismatches
                else f"FAIL: {len(mismatches)} mismatch(es): {mismatches[:4]}"
            )
            decode_seconds = max(metrics["wall_seconds"] - prefill["prefill_seconds"], 1e-6)
            record(
                f"B{batch_size}",
                {
                    **metrics,
                    "prefill": prefill,
                    "samples_per_minute": round(
                        len(batch_samples) / metrics["wall_seconds"] * 60, 2
                    ),
                    "decode_tokens_per_second": round(
                        metrics["generated_tokens"] / decode_seconds, 1
                    ),
                    "mismatched_sample_ids": mismatches,
                    **sampler.snapshot(),
                },
            )
            if mismatches:
                report["stop_reason"] = f"output mismatch at B={batch_size}"
                break
    except BenchStop as stop:
        report["stop_reason"] = str(stop)
    sampler.stop()
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["elapsed_seconds"] = round(args.deadline_seconds - remaining(), 1)
    _atomic_write_json(output_dir / "benchmark-report.json", report)
    print(
        json.dumps(
            {"stop_reason": report["stop_reason"], "validations": report["validations"]}, indent=1
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
