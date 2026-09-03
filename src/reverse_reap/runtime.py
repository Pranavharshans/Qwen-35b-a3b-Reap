"""Pinned Qwen3.5 loading, architecture preflight, and teacher-forced telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from reverse_reap.config import ExperimentConfig
from reverse_reap.datasets import NormalizedSample, balanced_subset, load_manifest
from reverse_reap.instrumentation import CaptureState, instrument_qwen35
from reverse_reap.qwen35 import ArchitectureError, Qwen35Architecture, inspect_qwen35_moe


class RuntimeCompatibilityError(RuntimeError):
    """Raised when the pinned runtime cannot satisfy the donor contract."""


def load_donor(model_path: Path, config: ExperimentConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    dtype = torch.bfloat16 if config.model.execution_precision == "bf16" else torch.float16
    common = {
        "local_files_only": True,
        "revision": config.model.revision,
        "dtype": dtype,
        "device_map": "balanced",
        "trust_remote_code": False,
    }
    errors = []
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = loader.from_pretrained(str(model_path), **common)
            break
        except (ValueError, OSError) as error:
            errors.append(f"{loader.__name__}: {error}")
    else:
        raise RuntimeCompatibilityError("; ".join(errors))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        revision=config.model.revision,
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


def validate_donor_contract(model: Any, architecture: Qwen35Architecture) -> dict[str, Any]:
    model_config = model.config
    text_config = getattr(model_config, "text_config", model_config)
    actual = {
        "model_type": getattr(model_config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "num_layers": architecture.num_layers,
        "hidden_size": architecture.hidden_size,
        "num_experts": architecture.num_experts,
        "experts_per_token": architecture.experts_per_token,
        "expert_intermediate_size": architecture.expert_intermediate_size,
        "shared_expert_present": all(
            hasattr(layer.mlp, "shared_expert") for layer in architecture.layers
        ),
    }
    expected = {
        "num_layers": 40,
        "hidden_size": 2048,
        "num_experts": 256,
        "experts_per_token": 8,
        "expert_intermediate_size": 512,
        "shared_expert_present": True,
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    accepted_types = {"qwen3_5_moe", "qwen3_5_moe_text"}
    if (
        actual["model_type"] not in accepted_types
        and actual["text_model_type"] not in accepted_types
    ):
        mismatches["model_type"] = {
            "expected": sorted(accepted_types),
            "actual": actual["model_type"],
        }
    if mismatches:
        raise RuntimeCompatibilityError(f"donor architecture mismatch: {mismatches}")
    return {"compatible": True, "actual": actual, "expected": expected}


def environment_report() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> Any:
    """Render a chat as a [1, T] long tensor across transformers return-type changes.

    transformers >= 5 returns a BatchEncoding from apply_chat_template with
    return_tensors="pt"; older versions returned a bare tensor. Normalize to a
    tensor here so downstream shape/prefix logic is version-independent.
    """
    import torch

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered, torch.Tensor):
        rendered = rendered["input_ids"]
    return rendered.to(dtype=torch.long)


def _render_ids(tokenizer: Any, sample: NormalizedSample, enable_thinking: bool) -> tuple[Any, Any]:
    import torch

    messages = [{"role": "user", "content": sample.prompt}]
    prompt = _chat_ids(tokenizer, messages, enable_thinking=enable_thinking)
    if sample.reference is None:
        raise RuntimeCompatibilityError(
            f"sample {sample.sample_id} has no teacher-forced reference"
        )
    rendered_full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": sample.reference}],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered_full, torch.Tensor):
        rendered_full = rendered_full["input_ids"]
    full = rendered_full.to(dtype=torch.long)
    if full.shape[1] < prompt.shape[1]:
        raise RuntimeCompatibilityError("full teacher-forced sequence is shorter than prompt")
    if not torch.equal(full[:, : prompt.shape[1]], prompt):
        raise RuntimeCompatibilityError(
            "teacher-forced sequence does not preserve the prompt prefix"
        )
    return prompt, full


def _run_capture(
    model: Any,
    architecture: Qwen35Architecture,
    input_ids: Any,
    *,
    observer: Any | None = None,
) -> CaptureState:
    import torch

    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    with torch.inference_mode(), instrument_qwen35(architecture, observer=observer) as capture:
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    return capture


def _segment_rows(
    full: CaptureState,
    prompt: CaptureState,
    sample: NormalizedSample,
) -> list[dict[str, Any]]:
    rows = []
    fields = ("count", "router_mass", "output_norm_sum", "weighted_norm_sum")
    for layer in range(full.num_layers):
        full_acc, prompt_acc = full.accumulators[layer], prompt.accumulators[layer]
        for segment in ("prompt", "completion", "joint"):
            arrays = {}
            for field in fields:
                if segment == "prompt":
                    values = getattr(prompt_acc, field)
                elif segment == "joint":
                    values = getattr(full_acc, field)
                else:
                    values = getattr(full_acc, field) - getattr(prompt_acc, field)
                if np.any(values < -1e-7):
                    raise RuntimeCompatibilityError(
                        f"non-causal telemetry subtraction at layer {layer}, field {field}"
                    )
                arrays[field] = np.maximum(values, 0)
            for expert in np.flatnonzero(arrays["count"]):
                count = int(arrays["count"][expert])
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "domain": sample.domain,
                        "stratum": sample.stratum,
                        "language": sample.language,
                        "split": sample.split,
                        "segment": segment,
                        "layer": layer,
                        "expert": int(expert),
                        "routed_count": count,
                        "router_mass": float(arrays["router_mass"][expert]),
                        "reap_saliency": float(arrays["weighted_norm_sum"][expert] / count),
                    }
                )
    return rows


def capture_manifest(
    model_path: Path,
    manifest_path: Path,
    destination: Path,
    config: ExperimentConfig,
    *,
    split: str,
    limit: int | None = None,
) -> dict[str, Any]:
    model, tokenizer = load_donor(model_path, config)
    try:
        architecture = inspect_qwen35_moe(model)
    except ArchitectureError as error:
        raise RuntimeCompatibilityError(str(error)) from error
    architecture_report = validate_donor_contract(model, architecture)
    samples = [
        sample for sample in load_manifest(manifest_path) if split == "all" or sample.split == split
    ]
    samples = balanced_subset(samples, limit)
    if not samples:
        raise RuntimeCompatibilityError(f"manifest has no samples for split {split}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeCompatibilityError(f"refusing to overwrite telemetry: {destination}")
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    count = 0
    analysed_tokens = 0
    condition_id = "C1" if config.runtime.enable_thinking else "C0"
    run_id = config.run_id or f"unresolved-{config.fingerprint()[:16]}"
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for sample_number, sample in enumerate(samples):
                prompt_ids, full_ids = _render_ids(
                    tokenizer, sample, config.runtime.enable_thinking
                )
                token_ids = full_ids[0].tolist()
                prompt_tokens = prompt_ids.shape[1]

                def observe(
                    layer: int,
                    batch: Any,
                    norms: np.ndarray,
                    *,
                    _token_ids: list[int] = token_ids,
                    _prompt_tokens: int = prompt_tokens,
                    _sample: NormalizedSample = sample,
                    _sample_number: int = sample_number,
                ) -> None:
                    nonlocal count
                    if batch.tokens != len(_token_ids):
                        raise RuntimeCompatibilityError(
                            f"layer {layer} routed {batch.tokens} tokens, "
                            f"expected {len(_token_ids)}"
                        )
                    for token_index in range(batch.tokens):
                        segment = "prompt" if token_index < _prompt_tokens else "reference"
                        for rank in range(batch.top_k):
                            row = {
                                "schema_version": 1,
                                "run_id": run_id,
                                "sample_id": _sample.sample_id,
                                "condition_id": condition_id,
                                "segment": segment,
                                "token_index": token_index,
                                "token_id": int(_token_ids[token_index]),
                                "layer_index": layer,
                                "expert_index": int(batch.indices[token_index, rank]),
                                "route_rank": rank,
                                "router_weight": float(batch.weights[token_index, rank]),
                                "expert_output_l2": float(norms[token_index, rank]),
                                "chunk_id": f"sample-{_sample_number:06d}",
                                "domain": _sample.domain,
                                "stratum": _sample.stratum,
                                "language": _sample.language,
                                "split": _sample.split,
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            count += 1

                _run_capture(model, architecture, full_ids, observer=observe)
                analysed_tokens += len(token_ids)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "routing_rows": count,
        "analysed_tokens": analysed_tokens,
        "expected_routing_rows": (
            analysed_tokens * architecture.num_layers * architecture.experts_per_token
        ),
        "row_count_valid": count
        == analysed_tokens * architecture.num_layers * architecture.experts_per_token,
        "samples": len(samples),
        "telemetry_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "architecture": architecture_report,
        "environment": environment_report(),
    }


def probe_instrumentation(
    model_path: Path, config: ExperimentConfig, prompt: str
) -> dict[str, Any]:
    import torch

    model, tokenizer = load_donor(model_path, config)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = model.get_input_embeddings().weight.device
    ids = ids.to(device)
    with torch.inference_mode():
        baseline = model(input_ids=ids, use_cache=False).logits
        with instrument_qwen35(architecture) as capture:
            instrumented = model(input_ids=ids, use_cache=False).logits
    exact = torch.equal(baseline, instrumented)
    maximum_difference = float((baseline.float() - instrumented.float()).abs().max().item())
    routed = sum(int(acc.count.sum()) for acc in capture.accumulators)
    return {
        "exact_logits": exact,
        "maximum_logit_difference": maximum_difference,
        "routed_records": routed,
        "passed": exact
        and routed
        == ids.numel() * architecture.num_layers * architecture.experts_per_token,
    }


def probe_single_expert_intervention(
    model_path: Path,
    manifest_path: Path,
    candidate_manifest_path: Path,
    config: ExperimentConfig,
    *,
    split: str = "selection",
    limit: int = 20,
) -> dict[str, Any]:
    """Prove one real routed expert can be zeroed without changing no-op execution."""
    import torch

    candidates = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    experts = candidates.get("experts")
    if not isinstance(experts, list) or not experts:
        raise RuntimeCompatibilityError("candidate manifest contains no experts")
    target = (int(experts[0]["layer"]), int(experts[0]["expert"]))
    model, tokenizer = load_donor(model_path, config)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    samples = balanced_subset(
        [sample for sample in load_manifest(manifest_path) if sample.split == split], limit
    )
    for sample in samples:
        _, ids = _render_ids(tokenizer, sample, config.runtime.enable_thinking)
        ids = ids[:, : config.runtime.max_input_tokens]
        device = model.get_input_embeddings().weight.device
        ids = ids.to(device)
        attention_mask = torch.ones_like(ids)
        with torch.inference_mode():
            baseline = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
            with instrument_qwen35(architecture) as capture:
                noop = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
        if not torch.equal(baseline, noop):
            return {
                "passed": False,
                "reason": "no-op instrumentation changed logits",
                "sample_id": sample.sample_id,
                "target": {"layer": target[0], "expert": target[1]},
            }
        routed_count = int(capture.accumulators[target[0]].count[target[1]])
        if not routed_count:
            continue
        with torch.inference_mode(), instrument_qwen35(
            architecture, masked=frozenset({target})
        ):
            masked = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
        maximum_difference = float((baseline.float() - masked.float()).abs().max().item())
        return {
            "passed": maximum_difference > 0,
            "sample_id": sample.sample_id,
            "target": {"layer": target[0], "expert": target[1]},
            "routed_count": routed_count,
            "noop_logits_exact": True,
            "masked_logits_changed": maximum_difference > 0,
            "maximum_masked_logit_difference": maximum_difference,
            "semantics": "zero-weighted-contribution-without-router-renormalization",
        }
    return {
        "passed": False,
        "reason": f"target expert was not routed in {len(samples)} bounded samples",
        "target": {"layer": target[0], "expert": target[1]},
    }
