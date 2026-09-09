#!/usr/bin/env python3
"""Probe target-capture batch sizes on the exact donor before bulk capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reverse_reap.bridge_capture import load_bridge_manifest
from reverse_reap.config import load_config
from reverse_reap.instrumentation import instrument_qwen35_targeted
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import _render_ids, load_donor, pad_token_batch


def _forward_logits(model, batch_ids, attention_mask):
    """Return only the final-token logits when the pinned model supports it."""
    kwargs = {
        "input_ids": batch_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    try:
        return model(**kwargs, logits_to_keep=1).logits
    except TypeError:
        return model(**kwargs).logits[:, -1:, :]


def _run_batch(model, architecture, batch_ids, attention_mask, targets):
    import torch

    device = model.get_input_embeddings().weight.device
    batch_ids = batch_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        baseline = _forward_logits(model, batch_ids, attention_mask)
        with instrument_qwen35_targeted(
            architecture, targets, observer=lambda _observation: None
        ):
            captured = _forward_logits(model, batch_ids, attention_mask)
    return torch.equal(baseline, captured), baseline.shape


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("capture_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    try:
        import torch

        config = load_config(args.config)
        manifest = load_bridge_manifest(args.capture_manifest)
        if config.run_id is not None and config.run_id != manifest.run_id:
            raise ValueError("config and capture manifest run IDs differ")
        model, tokenizer = load_donor(args.model_path, config)
        architecture = inspect_qwen35_moe(model)
        targets = frozenset((item.layer, item.expert) for item in manifest.experts)
        rendered = [
            _render_ids(tokenizer, item.sample, manifest.enable_thinking)[1][0]
            for item in manifest.samples
        ]
        rendered.sort(key=lambda ids: int(ids.numel()), reverse=True)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        results = []
        for batch_size in args.batches:
            if batch_size <= 0:
                raise ValueError("batch sizes must be positive")
            if not rendered:
                raise ValueError("capture manifest has no samples")
            sequences = [rendered[index % len(rendered)] for index in range(batch_size)]
            batch_ids, attention_mask, _mapping = pad_token_batch(
                sequences, int(pad_token_id), left=False
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                identical, logits_shape = _run_batch(
                    model, architecture, batch_ids, attention_mask, targets
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                per_device = []
                if torch.cuda.is_available():
                    for index in range(torch.cuda.device_count()):
                        total = torch.cuda.get_device_properties(index).total_memory
                        peak = torch.cuda.max_memory_reserved(index)
                        per_device.append(
                            {
                                "device": index,
                                "peak_reserved_bytes": peak,
                                "total_memory_bytes": total,
                                "peak_fraction": peak / total,
                            }
                        )
                memory_safe = all(item["peak_fraction"] <= 0.92 for item in per_device)
                results.append(
                    {
                        "batch_size": batch_size,
                        "passed": identical and memory_safe,
                        "logits_shape": list(logits_shape),
                        "per_device_memory": per_device,
                        "memory_limit_fraction": 0.92,
                        "error": (
                            None
                            if identical and memory_safe
                            else "target hook changed logits"
                            if not identical
                            else "peak reserved memory exceeded 92%"
                        ),
                    }
                )
            except RuntimeError as error:
                is_oom = "out of memory" in str(error).lower()
                results.append(
                    {
                        "batch_size": batch_size,
                        "passed": False,
                        "logits_shape": None,
                        "peak_memory_bytes": None,
                        "oom": is_oom,
                        "error": str(error),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # Select the largest contiguous passing size. A larger batch cannot
        # leapfrog a failed smaller batch because that would hide an unstable
        # or resource-sensitive operating point.
        passing = []
        for item in results:
            if not item["passed"]:
                break
            passing.append(item)
        report = {
            "passed": bool(passing),
            "selected_batch_size": max(item["batch_size"] for item in passing)
            if passing
            else None,
            "batches": results,
            "run_id": manifest.run_id,
            "manifest_sha256": manifest.manifest_sha256,
            "target_expert_count": len(targets),
        }
    except Exception as error:
        report = {"passed": False, "error": str(error), "batches": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
