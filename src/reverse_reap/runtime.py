"""Pinned Qwen donor loading, architecture preflight, and teacher-forced telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from reverse_reap.bridge_capture import (
    AtomicTargetShardWriter,
    BridgeCaptureError,
    CoverageTracker,
    ceiling_batch_decision,
    load_bridge_manifest,
    load_target_capture_state,
    target_event_id,
    validate_target_shard,
    write_target_capture_state,
)
from reverse_reap.config import ExperimentConfig
from reverse_reap.datasets import NormalizedSample, balanced_subset, load_manifest
from reverse_reap.donors import DONOR_CONTRACTS, donor_contract
from reverse_reap.instrumentation import (
    CaptureState,
    TargetedRouteObservation,
    instrument_qwen35,
    instrument_qwen35_targeted,
)
from reverse_reap.qwen35 import ArchitectureError, Qwen35Architecture, inspect_qwen35_moe


class RuntimeCompatibilityError(RuntimeError):
    """Raised when the pinned runtime cannot satisfy the donor contract."""


def _loader_max_memory(gpu_count: int) -> dict[int | str, str] | None:
    """Build an opt-in Accelerate memory map with deterministic GPU headroom."""
    raw_gpu = os.environ.get("REVERSE_REAP_GPU_MAX_MEMORY_GIB")
    if raw_gpu is None:
        return None
    try:
        gpu_gib = int(raw_gpu)
        cpu_gib = int(os.environ.get("REVERSE_REAP_CPU_MAX_MEMORY_GIB", "512"))
    except ValueError as error:
        raise RuntimeCompatibilityError(
            "loader memory limits must be integer GiB values"
        ) from error
    if gpu_count < 1 or gpu_gib < 1 or cpu_gib < 1:
        raise RuntimeCompatibilityError("loader memory limits and GPU count must be positive")
    return {
        **{index: f"{gpu_gib}GiB" for index in range(gpu_count)},
        "cpu": f"{cpu_gib}GiB",
    }


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
    max_memory = _loader_max_memory(torch.cuda.device_count())
    if max_memory is not None:
        common["max_memory"] = max_memory
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
    model_id = getattr(model, "name_or_path", None)
    configured_id = getattr(model_config := model.config, "_name_or_path", None)
    if configured_id in (None, ""):
        configured_id = model_id
    try:
        contract = donor_contract(str(configured_id))
    except ValueError as error:
        root_type = getattr(model_config, "model_type", None)
        candidates = [
            item for item in DONOR_CONTRACTS.values() if item.root_model_type == root_type
        ]
        if len(candidates) > 1:
            quantization = getattr(model_config, "quantization_config", None)
            is_fp8 = isinstance(quantization, dict) and quantization.get("quant_method") == "fp8"
            candidates = [
                item
                for item in candidates
                if (item.source_precision == "fp8") == is_fp8
            ]
        if len(candidates) != 1:
            raise RuntimeCompatibilityError(
                f"unsupported donor model/type: {configured_id!r}/{root_type!r}"
            ) from error
        contract = candidates[0]
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
        "num_layers": contract.num_hidden_layers,
        "hidden_size": contract.hidden_size,
        "num_experts": contract.num_experts,
        "experts_per_token": contract.num_experts_per_tok,
        "expert_intermediate_size": contract.moe_intermediate_size,
        "shared_expert_present": True,
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    accepted_types = {contract.root_model_type, contract.text_model_type}
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


def pad_token_batch(
    sequences: list[Any], pad_token_id: int, *, left: bool = False
) -> tuple[Any, Any, list[tuple[int, int] | None]]:
    """Pad one-dimensional token sequences and return a valid-token index map.

    Qwen's expert kernel receives a flattened ``batch * sequence`` dimension.
    The returned map converts that flattened index back to ``(sample_index,
    unpadded_token_position)`` and marks padding positions as ``None``.
    """
    import torch

    if not sequences:
        raise RuntimeCompatibilityError("cannot pad an empty token batch")
    normalized = []
    for sequence in sequences:
        value = sequence
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.long)
        if value.ndim == 2 and tuple(value.shape[:1]) == (1,):
            value = value[0]
        if value.ndim != 1 or value.numel() == 0:
            raise RuntimeCompatibilityError("each token sequence must be a non-empty 1-D tensor")
        normalized.append(value.to(dtype=torch.long))
    width = max(int(value.numel()) for value in normalized)
    batch = torch.full(
        (len(normalized), width), int(pad_token_id), dtype=torch.long, device=normalized[0].device
    )
    attention = torch.zeros_like(batch)
    mapping: list[tuple[int, int] | None] = [None] * (len(normalized) * width)
    for sample_index, value in enumerate(normalized):
        length = int(value.numel())
        start = width - length if left else 0
        batch[sample_index, start : start + length] = value
        attention[sample_index, start : start + length] = 1
        for position in range(length):
            mapping[sample_index * width + start + position] = (sample_index, position)
    return batch, attention, mapping


def _run_targeted_batch(
    model: Any,
    architecture: Qwen35Architecture,
    input_ids: Any,
    attention_mask: Any,
    targets: frozenset[tuple[int, int]],
    observer: Any,
) -> None:
    """Run one padded, teacher-forced batch through the native donor path."""
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    import torch

    with (
        torch.inference_mode(),
        instrument_qwen35_targeted(architecture, targets, observer=observer),
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )


def capture_targeted_manifest(
    model_path: Path,
    capture_manifest_path: Path,
    destination: Path,
    config: ExperimentConfig,
    *,
    batch_size: int | None = None,
    left_padding: bool = False,
    shard_max_records: int = 4096,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Capture selected donor expert vectors into resumable BF16 shards.

    This path is deliberately separate from :func:`capture_manifest`: it saves
    only target routes and never changes the returned native model output.
    ``run_id`` is supplied by the lead controller so a CLI invocation resolves
    the identity once and all records share it.
    """
    manifest = load_bridge_manifest(capture_manifest_path)
    if config.model.id != manifest.model_id:
        raise RuntimeCompatibilityError("capture manifest and config donor model IDs differ")
    if config.model.revision != manifest.model_revision:
        raise RuntimeCompatibilityError("capture manifest and config donor revisions differ")
    if config.runtime.enable_thinking or manifest.enable_thinking:
        raise RuntimeCompatibilityError(
            "bridge target capture is currently C0 thinking-disabled only"
        )
    selected_batch_size = batch_size or config.runtime.batch_size
    if selected_batch_size not in (1, 2, 4, 8):
        raise RuntimeCompatibilityError("target capture batch_size must be one of 1, 2, 4, or 8")
    effective_run_id = run_id or config.run_id
    if not effective_run_id or effective_run_id == "unresolved":
        raise RuntimeCompatibilityError("target capture requires a resolved run_id")
    if manifest.run_id != effective_run_id:
        raise RuntimeCompatibilityError("capture manifest and runtime run IDs differ")
    if manifest.config_sha256 != config.fingerprint():
        raise RuntimeCompatibilityError("capture manifest and config hashes differ")
    targets = frozenset((item.layer, item.expert) for item in manifest.experts)
    sample_ordinals = {item.sample.sample_id: item.sample_ordinal for item in manifest.samples}
    model, tokenizer = load_donor(model_path, config)
    try:
        architecture = inspect_qwen35_moe(model)
    except ArchitectureError as error:
        raise RuntimeCompatibilityError(str(error)) from error
    validate_donor_contract(model, architecture)
    if any(
        layer < 0
        or layer >= architecture.num_layers
        or expert < 0
        or expert >= architecture.num_experts
        for layer, expert in targets
    ):
        raise RuntimeCompatibilityError("capture manifest contains an out-of-range target expert")

    rendered: list[tuple[NormalizedSample, list[int]]] = []
    for selected in manifest.samples:
        _, full_ids = _render_ids(tokenizer, selected.sample, manifest.enable_thinking)
        token_ids = [int(value) for value in full_ids[0].tolist()]
        if len(token_ids) != selected.token_count:
            raise RuntimeCompatibilityError(
                f"token count changed for {selected.sample.sample_id}: "
                f"manifest={selected.token_count}, runtime={len(token_ids)}"
            )
        if len(token_ids) > manifest.max_input_tokens:
            raise RuntimeCompatibilityError(
                f"sample {selected.sample.sample_id} exceeds max_input_tokens without truncation"
            )
        rendered.append((selected.sample, token_ids))

    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "capture-state.json"
    existing_shards = sorted(path for path in destination.glob("shard-*") if path.is_dir())
    checkpoint = load_target_capture_state(
        checkpoint_path,
        run_id=effective_run_id,
        capture_manifest_sha256=manifest.manifest_sha256,
    )
    if existing_shards and checkpoint is None:
        raise BridgeCaptureError(
            "completed target shards exist without a resumable capture checkpoint"
        )
    for shard_path in existing_shards:
        validate_target_shard(shard_path, hidden_size=architecture.hidden_size)
        metadata = json.loads((shard_path / "shard.json").read_text(encoding="utf-8"))
        if (
            metadata.get("run_id") != effective_run_id
            or metadata.get("source_manifest_hash") != manifest.source_manifest_sha256
            or metadata.get("model_revision") != manifest.model_revision
            or metadata.get("tokenizer_fingerprint") != manifest.tokenizer_fingerprint
            or metadata.get("config_sha256") != manifest.config_sha256
            or metadata.get("candidate_manifest_sha256") != manifest.candidate_manifest_sha256
        ):
            raise BridgeCaptureError(
                f"existing target shard does not match this capture: {shard_path}"
            )
    if checkpoint is not None:
        existing_names = {path.name for path in existing_shards}
        if set(checkpoint.completed_shards) != existing_names:
            raise BridgeCaptureError("capture checkpoint and completed target shards disagree")
    coverage = CoverageTracker(
        targets,
        min_coding_events=manifest.min_coding_events_per_expert,
        min_control_events=manifest.min_control_events_per_expert,
        target_tokens=manifest.target_analyzed_tokens,
        hard_token_ceiling=manifest.hard_token_ceiling,
    )
    if checkpoint is not None:
        coverage.restore(checkpoint.coverage)
    shard_ids = [path.name.removeprefix("shard-") for path in existing_shards]
    try:
        shard_index = max((int(value) for value in shard_ids), default=-1) + 1
    except ValueError as error:
        raise BridgeCaptureError("target shard directory has a non-numeric id") from error
    shard_dirs: list[Path] = list(existing_shards)
    writer: AtomicTargetShardWriter | None = None

    def persist_checkpoint(
        *, next_sample_index: int, next_batch_number: int, complete: bool
    ) -> None:
        write_target_capture_state(
            checkpoint_path,
            {
                "run_id": effective_run_id,
                "capture_manifest_sha256": manifest.manifest_sha256,
                "next_sample_index": next_sample_index,
                "next_batch_number": next_batch_number,
                "completed_shards": [path.name for path in shard_dirs],
                "coverage": coverage.report(),
                "complete": complete,
            },
        )

    if checkpoint is None:
        persist_checkpoint(next_sample_index=0, next_batch_number=0, complete=False)
    elif checkpoint.complete and not shard_dirs:
        raise BridgeCaptureError("completed target capture checkpoint has no completed shards")
    elif checkpoint.complete:
        return {
            "run_id": effective_run_id,
            "manifest_sha256": manifest.manifest_sha256,
            "selected_batch_size": selected_batch_size,
            "left_padding": left_padding,
            "shards": [str(path) for path in shard_dirs],
            "coverage": coverage.report(),
            "analyzed_tokens": coverage.analyzed_tokens,
            "target_reached": coverage.analyzed_tokens >= coverage.target_tokens,
            "coverage_sufficient": coverage.capture_success,
            "hard_ceiling_reached": coverage.hard_ceiling_reached,
            "valid": coverage.capture_success,
            "resumed": True,
        }

    def new_writer() -> AtomicTargetShardWriter:
        nonlocal shard_index
        candidate = AtomicTargetShardWriter(
            destination,
            run_id=effective_run_id,
            shard_id=f"{shard_index:06d}",
            source_manifest_hash=manifest.source_manifest_sha256,
            model_revision=manifest.model_revision,
            tokenizer_fingerprint_value=manifest.tokenizer_fingerprint,
            config_sha256=manifest.config_sha256,
            candidate_manifest_sha256=manifest.candidate_manifest_sha256,
            target_experts=targets,
            hidden_size=architecture.hidden_size,
            max_records=shard_max_records,
        )
        shard_index += 1
        return candidate

    def observe(observation: TargetedRouteObservation) -> None:
        nonlocal writer
        if writer is None:
            writer = new_writer()
        token_indices = observation.token_indices.detach().cpu().tolist()
        route_ranks = observation.route_ranks.detach().cpu().tolist()
        weights = observation.router_weights.detach().cpu().tolist()
        for offset, flat_index in enumerate(token_indices):
            if not 0 <= int(flat_index) < len(batch_mapping):
                raise RuntimeCompatibilityError(
                    "target hook returned an invalid flattened token index"
                )
            metadata = batch_mapping[int(flat_index)]
            if metadata is None:
                continue
            sample_index, token_position = metadata
            sample, token_ids = batch_samples[sample_index]
            if writer.full:
                shard_dirs.append(writer.finalize())
                writer = new_writer()
            record = {
                "schema_version": 1,
                "run_id": effective_run_id,
                "row_index": 0,
                "event_id": target_event_id(
                    sample.sample_id,
                    token_position,
                    observation.layer_index,
                    observation.expert_index,
                    int(route_ranks[offset]),
                ),
                "sample_ordinal": sample_ordinals[sample.sample_id],
                "sample_id": sample.sample_id,
                "sample_content_sha256": sample.content_sha256,
                "domain": sample.domain,
                "split": sample.split,
                "token_position": token_position,
                "token_id": token_ids[token_position],
                "donor_layer": observation.layer_index,
                "expert_id": observation.expert_index,
                "route_rank": int(route_ranks[offset]),
                "router_weight": float(weights[offset]),
                "source_manifest_hash": manifest.source_manifest_sha256,
                "model_revision": manifest.model_revision,
                "tokenizer_fingerprint": manifest.tokenizer_fingerprint,
                "config_sha256": manifest.config_sha256,
                "candidate_manifest_sha256": manifest.candidate_manifest_sha256,
                "condition_id": manifest.condition_id,
                "chunk_id": f"batch-{batch_number:06d}",
            }
            writer.append(
                record,
                observation.expert_inputs[offset],
                observation.replayed_expert_output[offset],
                observation.weighted_replayed_expert_output[offset],
            )
            coverage.add_event(sample.domain, observation.layer_index, observation.expert_index)

    sample_offset = checkpoint.next_sample_index if checkpoint is not None else 0
    batch_number = checkpoint.next_batch_number if checkpoint is not None else 0
    while sample_offset < len(rendered) and not coverage.should_stop:
        batch_samples: list[tuple[NormalizedSample, list[int]]] = []
        token_total = 0
        while sample_offset < len(rendered) and len(batch_samples) < selected_batch_size:
            candidate = rendered[sample_offset]
            candidate_tokens = len(candidate[1])
            decision = ceiling_batch_decision(
                analyzed_tokens=coverage.analyzed_tokens,
                batch_tokens=token_total,
                sample_tokens=candidate_tokens,
                hard_token_ceiling=coverage.hard_token_ceiling,
            )
            if decision == "seal_batch":
                break
            if decision == "stop_cleanly":
                # Checkpoint and stop cleanly with coverage-incomplete: the next
                # complete sample cannot fit under the hard ceiling. The sample
                # is never partially processed or truncated.
                break
            if decision == "raise_infeasible":
                raise RuntimeCompatibilityError(
                    f"sample {candidate[0].sample_id} would exceed hard token ceiling"
                )
            batch_samples.append(candidate)
            token_total += candidate_tokens
            sample_offset += 1
        if not batch_samples:
            break
        sequences = [token_ids for _, token_ids in batch_samples]
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeCompatibilityError("tokenizer has neither pad_token_id nor eos_token_id")
        batch_ids, attention_mask, batch_mapping = pad_token_batch(
            sequences, int(pad_token_id), left=left_padding
        )
        if batch_ids.device.type != "cpu":
            batch_ids = batch_ids.cpu()
            attention_mask = attention_mask.cpu()
        _run_targeted_batch(
            model,
            architecture,
            batch_ids,
            attention_mask,
            targets,
            observe,
        )
        coverage.add_tokens(token_total)
        batch_number += 1
        if writer is not None and writer.record_count:
            shard_dirs.append(writer.finalize())
            writer = None
        persist_checkpoint(
            next_sample_index=sample_offset,
            next_batch_number=batch_number,
            complete=coverage.should_stop or sample_offset >= len(rendered),
        )
    if writer is not None and writer.record_count:
        shard_dirs.append(writer.finalize())
        writer = None
    persist_checkpoint(
        next_sample_index=sample_offset,
        next_batch_number=batch_number,
        complete=coverage.should_stop or sample_offset >= len(rendered),
    )
    if not shard_dirs:
        raise BridgeCaptureError("targeted capture produced no routed target records")
    return {
        "valid": coverage.capture_success,
        "run_id": effective_run_id,
        "manifest_sha256": manifest.manifest_sha256,
        "selected_batch_size": selected_batch_size,
        "left_padding": left_padding,
        "shards": [str(path) for path in shard_dirs],
        "coverage": coverage.report(),
        "analyzed_tokens": coverage.analyzed_tokens,
        "target_reached": coverage.analyzed_tokens >= coverage.target_tokens,
        "coverage_sufficient": coverage.capture_success,
        "hard_ceiling_reached": coverage.hard_ceiling_reached,
    }


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
        and routed == ids.numel() * architecture.num_layers * architecture.experts_per_token,
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
        with torch.inference_mode(), instrument_qwen35(architecture, masked=frozenset({target})):
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
