"""Governed donor-target capture artifacts for later representation-bridge work.

The v0 telemetry path intentionally keeps only streaming norms.  This module is a
separate, opt-in hand-off path: it stores only the selected donor experts' route
vectors, in BF16 safetensors shards, together with a typed JSONL index.  Nothing
here changes the v0 causal or extraction semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from reverse_reap.config import StrictModel
from reverse_reap.datasets import NormalizedSample, canonical_json, load_manifest, sha256
from reverse_reap.donors import QWEN35_MODEL_ID, QWEN38_MODEL_ID, donor_contract


class BridgeCaptureError(ValueError):
    """Raised when a bridge-capture manifest or shard is unsafe to consume."""


class BridgeCaptureExpert(StrictModel):
    """Stable donor layer/expert identity."""

    layer: int = Field(ge=0)
    expert: int = Field(ge=0)


class BridgeCaptureSample(StrictModel):
    """A self-contained, length-audited sample selected for donor capture."""

    sample_ordinal: int = Field(ge=0)
    sample: NormalizedSample
    prompt_token_count: int = Field(gt=0)
    token_count: int = Field(gt=0)


class BridgeCaptureManifest(StrictModel):
    """Immutable input contract for a targeted donor-target capture."""

    schema_version: Literal[1]
    kind: Literal["bridge-target-capture"]
    run_id: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: Literal[QWEN35_MODEL_ID, QWEN38_MODEL_ID]
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    tokenizer_fingerprint: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition_id: Literal["C0"]
    enable_thinking: Literal[False] = False
    max_input_tokens: int = Field(ge=64)
    target_analyzed_tokens: int = Field(gt=0)
    hard_token_ceiling: int = Field(gt=0)
    eligible_analyzed_tokens: int = Field(gt=0)
    min_coding_events_per_expert: int = Field(gt=0)
    min_control_events_per_expert: int = Field(gt=0)
    experts: list[BridgeCaptureExpert] = Field(min_length=1)
    allowed_splits: list[Literal["calibration", "selection", "validation", "replication"]] = Field(
        min_length=1
    )
    samples: list[BridgeCaptureSample] = Field(min_length=1)
    analyzed_tokens: int = Field(gt=0)
    coding_tokens: int = Field(gt=0)
    control_tokens: int = Field(gt=0)
    target_reached: bool
    exclusions: list[dict[str, str]] = Field(default_factory=list)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("experts")
    @classmethod
    def unique_experts(cls, value: list[BridgeCaptureExpert]) -> list[BridgeCaptureExpert]:
        identities = [(item.layer, item.expert) for item in value]
        if len(identities) != len(set(identities)):
            raise ValueError("bridge capture experts must be unique")
        return value

    @field_validator("samples")
    @classmethod
    def unique_samples(cls, value: list[BridgeCaptureSample]) -> list[BridgeCaptureSample]:
        identities = [item.sample.sample_id for item in value]
        if len(identities) != len(set(identities)):
            raise ValueError("bridge capture samples must be unique")
        return value

    @field_validator("run_id")
    @classmethod
    def resolved_run_id(cls, value: str) -> str:
        if value == "unresolved":
            raise ValueError("bridge capture manifest requires a resolved run_id")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> BridgeCaptureManifest:
        if self.hard_token_ceiling < self.target_analyzed_tokens:
            raise ValueError("hard_token_ceiling must be at least target_analyzed_tokens")
        if self.analyzed_tokens < self.hard_token_ceiling:
            raise ValueError("capture manifest must include samples through hard_token_ceiling")
        if self.eligible_analyzed_tokens < self.hard_token_ceiling:
            raise ValueError("eligible source pool cannot reach hard_token_ceiling")
        calculated = sum(item.token_count for item in self.samples)
        if calculated != self.analyzed_tokens:
            raise ValueError(
                "analyzed_tokens="
                f"{self.analyzed_tokens} does not match selected sample total {calculated}"
            )
        coding = sum(item.token_count for item in self.samples if item.sample.domain == "coding")
        control = sum(item.token_count for item in self.samples if item.sample.domain == "control")
        if (coding, control) != (self.coding_tokens, self.control_tokens):
            raise ValueError("coding/control token counts do not match selected samples")
        ordinals = sorted(item.sample_ordinal for item in self.samples)
        if ordinals != list(range(len(self.samples))):
            raise ValueError("capture sample ordinals must be a contiguous prefix")
        return self


class BridgeTargetRecord(StrictModel):
    """Metadata row pointing to one row in stacked BF16 safetensors tensors."""

    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_ordinal: int = Field(ge=0)
    sample_id: str = Field(min_length=1)
    sample_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain: Literal["coding", "control"]
    split: Literal["calibration", "selection", "validation", "replication"]
    token_position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    donor_layer: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    route_rank: int = Field(ge=0)
    router_weight: float = Field(ge=0, allow_inf_nan=False)
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    tokenizer_fingerprint: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition_id: Literal["C0"]
    chunk_id: str = Field(min_length=1)


class BridgeTargetShard(StrictModel):
    """Completion marker and checksums for one atomic target shard."""

    schema_version: Literal[1]
    kind: Literal["bridge-target-shard"]
    run_id: str = Field(min_length=1)
    shard_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    tokenizer_fingerprint: str = Field(min_length=1)
    record_count: int = Field(gt=0)
    analyzed_tokens: int = Field(gt=0)
    event_counts: dict[str, int]
    records_file: str = Field(min_length=1)
    tensors_file: str = Field(min_length=1)
    checksums_file: str = Field(min_length=1)
    checksums: dict[str, str]
    tensor_keys: list[Literal[
        "expert_input", "replayed_expert_output", "weighted_replayed_expert_output"
    ]] = Field(min_length=3, max_length=3)
    target_experts: list[BridgeCaptureExpert] = Field(min_length=1)
    shard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("records_file", "tensors_file", "checksums_file")
    @classmethod
    def local_filename_only(cls, value: str) -> str:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("target shard artifact names must be local filenames")
        return value

    @field_validator("checksums")
    @classmethod
    def checksum_keys_are_local(cls, value: dict[str, str]) -> dict[str, str]:
        for filename, digest in value.items():
            if Path(filename).name != filename or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("target shard checksums contain an unsafe filename or digest")
        return value

    @field_validator("tensor_keys")
    @classmethod
    def exact_tensor_keys(cls, value: list[str]) -> list[str]:
        expected = {
            "expert_input",
            "replayed_expert_output",
            "weighted_replayed_expert_output",
        }
        if set(value) != expected:
            raise ValueError("target shard must contain the three stacked target tensors")
        return value


class BridgeTargetBundle(StrictModel):
    """Hash inventory crossing from the donor VM to later bridge training."""

    schema_version: Literal[1]
    kind: Literal["bridge-target-handoff"]
    run_id: str = Field(min_length=1)
    capture_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_experts: list[BridgeCaptureExpert] = Field(min_length=1)
    artifacts: list[dict[str, Any]] = Field(min_length=1)
    record_count: int = Field(ge=0)
    analyzed_tokens: int = Field(ge=0)
    capture_outcome: Literal["coverage-complete", "coverage-incomplete"] = (
        "coverage-complete"
    )
    coverage: dict[str, Any] = Field(default_factory=dict)
    extraction_artifact: str | None = None
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BridgeTargetCaptureState(StrictModel):
    """Atomic progress checkpoint used to resume a targeted capture safely."""

    schema_version: Literal[1]
    kind: Literal["bridge-target-capture-state"]
    run_id: str = Field(min_length=1)
    capture_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    next_sample_index: int = Field(ge=0)
    next_batch_number: int = Field(ge=0)
    completed_shards: list[str] = Field(default_factory=list)
    coverage: dict[str, Any]
    complete: bool = False
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("completed_shards")
    @classmethod
    def unique_shards(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("capture checkpoint contains duplicate shard IDs")
        return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_without_hash(payload: dict[str, Any], field: str) -> bytes:
    body = dict(payload)
    body.pop(field, None)
    return canonical_json(body)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise BridgeCaptureError(f"refusing to overwrite existing artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_replace(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a mutable checkpoint after fsyncing its contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_target_capture_state(path: Path, payload: dict[str, Any]) -> BridgeTargetCaptureState:
    """Write a hash-bound, replaceable capture checkpoint."""
    body = dict(payload)
    body["schema_version"] = 1
    body["kind"] = "bridge-target-capture-state"
    body.pop("state_sha256", None)
    body["state_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    state = BridgeTargetCaptureState.model_validate(body)
    _write_json_replace(path, state.model_dump(mode="json"))
    return state


def load_target_capture_state(
    path: Path,
    *,
    run_id: str,
    capture_manifest_sha256: str,
) -> BridgeTargetCaptureState | None:
    """Load and validate the latest checkpoint, if one exists."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = BridgeTargetCaptureState.model_validate(payload)
    except Exception as error:
        raise BridgeCaptureError(f"invalid target capture checkpoint: {path}") from error
    expected = hashlib.sha256(
        canonical_json({key: value for key, value in payload.items() if key != "state_sha256"})
    ).hexdigest()
    if expected != state.state_sha256:
        raise BridgeCaptureError(f"target capture checkpoint hash mismatch: {path}")
    if state.run_id != run_id or state.capture_manifest_sha256 != capture_manifest_sha256:
        raise BridgeCaptureError("target capture checkpoint does not match this run")
    return state


def _tokenizer_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool
) -> list[int]:
    """Normalize tokenizer return types without importing torch on CPU preparation."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=len(messages) == 1,
        return_tensors=None,
        enable_thinking=enable_thinking,
    )
    # transformers>=5 slow-tokenizer path returns a BatchEncoding (a UserDict,
    # not a dict) here; Mapping covers both without touching list/ndarray/tensor.
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    while rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    if not isinstance(rendered, list) or not all(
        isinstance(item, (int, np.integer)) for item in rendered
    ):
        raise BridgeCaptureError("tokenizer did not return a one-dimensional token-id sequence")
    return [int(item) for item in rendered]


def tokenizer_fingerprint(tokenizer_path: Path) -> str:
    """Hash tokenizer files deterministically without loading model weights."""
    if not tokenizer_path.exists():
        raise BridgeCaptureError(f"tokenizer path does not exist: {tokenizer_path}")
    files = [path for path in tokenizer_path.rglob("*") if path.is_file()]
    if not files:
        raise BridgeCaptureError(f"tokenizer path contains no files: {tokenizer_path}")
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(tokenizer_path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _sample_order(sample: NormalizedSample, seed: int) -> str:
    return sha256(f"bridge-capture-v1\0{seed}\0{sample.sample_id}".encode())


def target_event_id(
    sample_id: str, token_position: int, layer: int, expert: int, route_rank: int
) -> str:
    """Return the stable identity for one routed target event."""
    return sha256(
        f"bridge-target-v1\0{sample_id}\0{token_position}\0{layer}\0{expert}\0{route_rank}".encode()
    )


def _candidate_manifest_metadata(path: Path) -> tuple[str, list[tuple[int, int]]]:
    if not path.is_file():
        raise BridgeCaptureError(f"candidate manifest is unavailable: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        experts_payload = payload["experts"]
        if payload.get("gate_passed") is not True or not isinstance(experts_payload, list):
            raise ValueError("candidate manifest is not a passed Gate C artifact")
        selected = [(int(item["layer"]), int(item["expert"])) for item in experts_payload]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BridgeCaptureError(f"invalid candidate manifest: {path}") from error
    if not selected or len(set(selected)) != len(selected):
        raise BridgeCaptureError("candidate manifest has no unique selected experts")
    return _file_sha256(path), sorted(set(selected))


def freeze_bridge_manifest(
    full_manifest: Path,
    tokenizer: Any,
    destination: Path,
    *,
    model_id: str = QWEN35_MODEL_ID,
    model_revision: str,
    tokenizer_fingerprint_value: str,
    config_sha256: str,
    candidate_manifest: Path,
    run_id: str,
    target_tokens: int = 200_000,
    hard_token_ceiling: int = 500_000,
    max_input_tokens: int = 1024,
    coding_fraction: float = 0.70,
    seed: int = 20260903,
    min_coding_events_per_expert: int = 5_000,
    min_control_events_per_expert: int = 1_000,
    experts: list[tuple[int, int]] | None = None,
    allowed_splits: tuple[str, ...] = ("calibration", "selection"),
) -> dict[str, Any]:
    """Freeze a deterministic, self-contained target-capture manifest.

    Samples are selected before any donor activation result is observed.  The
    manifest includes a deterministic sample prefix reaching the hard ceiling;
    an insufficient eligible source pool fails before any donor is loaded.
    """
    if not 0 < coding_fraction < 1:
        raise BridgeCaptureError("coding_fraction must be between zero and one")
    if target_tokens <= 0 or hard_token_ceiling < target_tokens:
        raise BridgeCaptureError("invalid target/hard token ceiling")
    if not run_id or run_id == "unresolved":
        raise BridgeCaptureError("bridge capture requires a resolved run_id")
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha256):
        raise BridgeCaptureError("config_sha256 must be a 64-character hex digest")
    candidate_hash, selected_experts = _candidate_manifest_metadata(candidate_manifest)
    try:
        contract = donor_contract(model_id)
    except ValueError as error:
        raise BridgeCaptureError(str(error)) from error
    invalid_experts = [
        (layer, expert)
        for layer, expert in selected_experts
        if not (0 <= layer < contract.num_hidden_layers and 0 <= expert < contract.num_experts)
    ]
    if invalid_experts:
        raise BridgeCaptureError(
            f"candidate manifest contains experts outside {model_id}: {invalid_experts}"
        )
    source_bytes = full_manifest.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    samples = load_manifest(full_manifest)
    allowed = set(allowed_splits)
    exclusions: list[dict[str, str]] = []
    candidates: dict[str, list[tuple[NormalizedSample, int, int]]] = {"coding": [], "control": []}
    for sample in samples:
        if sample.split not in allowed:
            exclusions.append({"sample_id": sample.sample_id, "reason": "split-not-allowed"})
            continue
        if sample.reference is None:
            exclusions.append({"sample_id": sample.sample_id, "reason": "missing-reference"})
            continue
        prompt_ids = _tokenizer_ids(
            tokenizer, [{"role": "user", "content": sample.prompt}], enable_thinking=False
        )
        full_ids = _tokenizer_ids(
            tokenizer,
            [
                {"role": "user", "content": sample.prompt},
                {"role": "assistant", "content": sample.reference},
            ],
            enable_thinking=False,
        )
        if len(full_ids) > max_input_tokens:
            exclusions.append({"sample_id": sample.sample_id, "reason": "over-max-input-tokens"})
            continue
        candidates[sample.domain].append((sample, len(prompt_ids), len(full_ids)))

    if not candidates["coding"] or not candidates["control"]:
        raise BridgeCaptureError(
            "bridge capture manifest requires at least one eligible coding and control sample"
        )

    for domain in candidates:
        candidates[domain].sort(key=lambda item: _sample_order(item[0], seed))

    eligible_token_capacity = sum(item[2] for values in candidates.values() for item in values)
    if eligible_token_capacity < hard_token_ceiling:
        raise BridgeCaptureError(
            "eligible source pool cannot reach hard_token_ceiling: "
            f"capacity={eligible_token_capacity}, ceiling={hard_token_ceiling}"
        )

    selected: list[BridgeCaptureSample] = []
    positions = {"coding": 0, "control": 0}
    token_counts = Counter()

    # Keep both domains represented even for a tiny smoke target.  A capture
    # manifest with no control observations cannot support the later bridge
    # retention check and should fail at freeze time instead of much later.
    for domain in ("coding", "control"):
        if not candidates[domain]:
            continue
        sample, prompt_count, total_count = candidates[domain][positions[domain]]
        positions[domain] += 1
        selected.append(
            BridgeCaptureSample(
                sample_ordinal=len(selected),
                sample=sample,
                prompt_token_count=prompt_count,
                token_count=total_count,
            )
        )
        token_counts[domain] += total_count

    while positions["coding"] < len(candidates["coding"]) or positions["control"] < len(
        candidates["control"]
    ):
        total = sum(token_counts.values())
        if total >= hard_token_ceiling:
            break
        coding_ratio = token_counts["coding"] / max(1, total)
        preferred = "coding" if coding_ratio < coding_fraction else "control"
        if positions[preferred] >= len(candidates[preferred]):
            preferred = "control" if preferred == "coding" else "coding"
        if positions[preferred] >= len(candidates[preferred]):
            break
        sample, prompt_count, total_count = candidates[preferred][positions[preferred]]
        positions[preferred] += 1
        selected.append(
            BridgeCaptureSample(
                sample_ordinal=len(selected),
                sample=sample,
                prompt_token_count=prompt_count,
                token_count=total_count,
            )
        )
        token_counts[preferred] += total_count

    if not selected:
        raise BridgeCaptureError("no eligible teacher-forced samples remain after manifest audit")
    selected.sort(key=lambda item: _sample_order(item.sample, seed))
    selected = [
        item.model_copy(update={"sample_ordinal": index}) for index, item in enumerate(selected)
    ]
    selected_tokens = sum(item.token_count for item in selected)
    if selected_tokens < hard_token_ceiling:
        # The capacity check above is intentionally conservative; retain an
        # explicit post-selection guard so a future selector change cannot
        # publish a manifest that silently stops at the target instead.
        raise BridgeCaptureError(
            "deterministic eligible sample prefix cannot reach hard_token_ceiling: "
            f"selected={selected_tokens}, ceiling={hard_token_ceiling}"
        )
    coding_tokens = sum(item.token_count for item in selected if item.sample.domain == "coding")
    control_tokens = selected_tokens - coding_tokens
    if experts is not None and sorted(set(experts)) != selected_experts:
        raise BridgeCaptureError("explicit experts differ from the candidate manifest")
    allowed_splits_typed = tuple(allowed)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bridge-target-capture",
        "run_id": run_id,
        "source_manifest_sha256": source_hash,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint_value,
        "config_sha256": config_sha256,
        "candidate_manifest_sha256": candidate_hash,
        "condition_id": "C0",
        "enable_thinking": False,
        "max_input_tokens": max_input_tokens,
        "target_analyzed_tokens": target_tokens,
        "hard_token_ceiling": hard_token_ceiling,
        "eligible_analyzed_tokens": eligible_token_capacity,
        "min_coding_events_per_expert": min_coding_events_per_expert,
        "min_control_events_per_expert": min_control_events_per_expert,
        "experts": [
            {"layer": int(layer), "expert": int(expert)} for layer, expert in selected_experts
        ],
        "allowed_splits": sorted(allowed_splits_typed),
        "samples": [item.model_dump(mode="json") for item in selected],
        "analyzed_tokens": selected_tokens,
        "coding_tokens": coding_tokens,
        "control_tokens": control_tokens,
        "target_reached": selected_tokens >= target_tokens,
        "exclusions": exclusions,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        _canonical_without_hash(payload, "manifest_sha256")
    ).hexdigest()
    manifest = BridgeCaptureManifest.model_validate(payload)
    _write_json_atomic(destination, manifest.model_dump(mode="json"))
    return manifest.model_dump(mode="json")


def load_bridge_manifest(path: Path) -> BridgeCaptureManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = BridgeCaptureManifest.model_validate(payload)
    expected = hashlib.sha256(_canonical_without_hash(payload, "manifest_sha256")).hexdigest()
    if expected != manifest.manifest_sha256:
        raise BridgeCaptureError(f"bridge manifest hash mismatch: {path}")
    return manifest


def ceiling_batch_decision(
    *,
    analyzed_tokens: int,
    batch_tokens: int,
    sample_tokens: int,
    hard_token_ceiling: int,
) -> str:
    """Decide how one teacher-forced sample interacts with the hard token ceiling.

    Returns one of ``"append"``, ``"seal_batch"``, ``"stop_cleanly"`` or
    ``"raise_infeasible"``, mirroring the governed capture loop exactly.
    Samples are never partially processed: a sample that does not fit is
    either deferred to the next batch or ends the run with coverage-incomplete.
    ``"stop_cleanly"`` applies only after forward progress exists; with zero
    analyzed tokens the run is infeasible and must fail loudly instead.
    """
    if analyzed_tokens < 0 or batch_tokens < 0 or sample_tokens < 0:
        raise BridgeCaptureError("token counts cannot be negative")
    if (
        batch_tokens
        and analyzed_tokens + batch_tokens + sample_tokens > hard_token_ceiling
    ):
        return "seal_batch"
    if not batch_tokens and analyzed_tokens + sample_tokens > hard_token_ceiling:
        return "stop_cleanly" if analyzed_tokens > 0 else "raise_infeasible"
    return "append"


@dataclass
class CoverageTracker:
    """Explicit adaptive stopping state for selected-expert capture."""

    experts: frozenset[tuple[int, int]]
    min_coding_events: int
    min_control_events: int
    target_tokens: int
    hard_token_ceiling: int
    analyzed_tokens: int = 0
    coding_events: dict[tuple[int, int], int] = field(default_factory=dict)
    control_events: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_tokens <= 0 or self.target_tokens > self.hard_token_ceiling:
            raise BridgeCaptureError("coverage target must be within the hard ceiling")
        self.coding_events = {expert: 0 for expert in self.experts}
        self.control_events = {expert: 0 for expert in self.experts}

    def add_tokens(self, count: int) -> None:
        if count < 0:
            raise BridgeCaptureError("analyzed token increment cannot be negative")
        self.analyzed_tokens += count

    def add_event(self, domain: str, layer: int, expert: int, count: int = 1) -> None:
        identity = (int(layer), int(expert))
        if identity not in self.experts:
            raise BridgeCaptureError(f"event is outside targeted expert set: {identity}")
        if domain not in ("coding", "control"):
            raise BridgeCaptureError(f"unsupported event domain: {domain}")
        if count < 0:
            raise BridgeCaptureError("event increment cannot be negative")
        target = self.coding_events if domain == "coding" else self.control_events
        target[identity] += count

    @property
    def minimum_coverage_reached(self) -> bool:
        return all(
            self.coding_events[item] >= self.min_coding_events
            and self.control_events[item] >= self.min_control_events
            for item in self.experts
        )

    @property
    def hard_ceiling_reached(self) -> bool:
        return self.analyzed_tokens >= self.hard_token_ceiling

    @property
    def should_stop(self) -> bool:
        return self.capture_success or self.hard_ceiling_reached

    @property
    def capture_success(self) -> bool:
        return self.analyzed_tokens >= self.target_tokens and self.minimum_coverage_reached

    @property
    def stop_reason(self) -> str | None:
        if self.capture_success:
            return "target-and-minimum-coverage"
        if self.hard_ceiling_reached:
            return "hard-token-ceiling"
        if self.minimum_coverage_reached:
            return "minimum-coverage-before-target"
        return None

    def report(self) -> dict[str, Any]:
        return {
            "analyzed_tokens": self.analyzed_tokens,
            "target_tokens": self.target_tokens,
            "hard_token_ceiling": self.hard_token_ceiling,
            "minimum_coverage_reached": self.minimum_coverage_reached,
            "capture_success": self.capture_success,
            "hard_ceiling_reached": self.hard_ceiling_reached,
            "stop_reason": self.stop_reason,
            "experts": {
                f"{layer}:{expert}": {
                    "coding_events": self.coding_events[(layer, expert)],
                    "control_events": self.control_events[(layer, expert)],
                    "coding_minimum": self.min_coding_events,
                    "control_minimum": self.min_control_events,
                }
                for layer, expert in sorted(self.experts)
            },
        }

    def restore(self, report: dict[str, Any]) -> None:
        """Restore counters from a previously validated checkpoint report."""
        if int(report.get("target_tokens", -1)) != self.target_tokens:
            raise BridgeCaptureError("checkpoint target token count differs from capture")
        if int(report.get("hard_token_ceiling", -1)) != self.hard_token_ceiling:
            raise BridgeCaptureError("checkpoint hard token ceiling differs from capture")
        analyzed_tokens = int(report.get("analyzed_tokens", -1))
        if not 0 <= analyzed_tokens <= self.hard_token_ceiling:
            raise BridgeCaptureError("checkpoint analyzed token count is invalid")
        checkpoint_experts = report.get("experts")
        if not isinstance(checkpoint_experts, dict):
            raise BridgeCaptureError("checkpoint has no expert coverage map")
        expected_experts = {f"{layer}:{expert}" for layer, expert in self.experts}
        if set(checkpoint_experts) != expected_experts:
            raise BridgeCaptureError("checkpoint expert set differs from capture")
        for layer, expert in self.experts:
            values = checkpoint_experts[f"{layer}:{expert}"]
            coding = int(values.get("coding_events", -1))
            control = int(values.get("control_events", -1))
            if coding < 0 or control < 0:
                raise BridgeCaptureError("checkpoint contains negative coverage")
            self.coding_events[(layer, expert)] = coding
            self.control_events[(layer, expert)] = control
        self.analyzed_tokens = analyzed_tokens


def _validate_bf16_vector(value: Any, *, hidden_size: int) -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised on CPU-only installs
        raise BridgeCaptureError("BF16 target shards require torch") from error
    if not isinstance(value, torch.Tensor):
        raise BridgeCaptureError("target vectors must be torch tensors")
    if value.ndim != 1 or value.shape[0] != hidden_size:
        raise BridgeCaptureError(
            f"target vectors must have shape [{hidden_size}], got {tuple(value.shape)}"
        )
    if value.dtype != torch.bfloat16:
        raise BridgeCaptureError(f"target vectors must remain BF16, got {value.dtype}")
    if not torch.isfinite(value.float()).all():
        raise BridgeCaptureError("target vectors contain NaN or infinity")
    return value.detach().to(device="cpu").contiguous()


def _relative_safe(root: Path, child: Path) -> Path:
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    try:
        return resolved_child.relative_to(resolved_root)
    except ValueError as error:
        raise BridgeCaptureError(f"artifact escapes bundle root: {child}") from error


class AtomicTargetShardWriter:
    """Collect and atomically publish one BF16 target shard."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        shard_id: str,
        source_manifest_hash: str,
        model_revision: str,
        tokenizer_fingerprint_value: str,
        config_sha256: str,
        candidate_manifest_sha256: str,
        target_experts: frozenset[tuple[int, int]],
        hidden_size: int = 2048,
        max_records: int = 4096,
    ) -> None:
        if not target_experts:
            raise BridgeCaptureError("target shard requires at least one expert")
        if max_records <= 0:
            raise BridgeCaptureError("max_records must be positive")
        self.root = root
        self.run_id = run_id
        self.shard_id = shard_id
        self.source_manifest_hash = source_manifest_hash
        self.model_revision = model_revision
        self.tokenizer_fingerprint = tokenizer_fingerprint_value
        self.config_sha256 = config_sha256
        self.candidate_manifest_sha256 = candidate_manifest_sha256
        self.target_experts = target_experts
        self.hidden_size = hidden_size
        self.max_records = max_records
        self._rows: list[tuple[BridgeTargetRecord, Any, Any, Any]] = []
        self._event_ids: set[str] = set()
        self.root.mkdir(parents=True, exist_ok=True)
        self.final_dir = self.root / f"shard-{shard_id}"
        if self.final_dir.exists():
            raise BridgeCaptureError(f"refusing to overwrite target shard: {self.final_dir}")

    @property
    def record_count(self) -> int:
        return len(self._rows)

    @property
    def full(self) -> bool:
        return self.record_count >= self.max_records

    def append(
        self,
        record: BridgeTargetRecord | dict[str, Any],
        expert_input: Any,
        replayed_expert_output: Any,
        weighted_replayed_expert_output: Any,
    ) -> None:
        if self.full:
            raise BridgeCaptureError("target shard is full")
        parsed = BridgeTargetRecord.model_validate(record)
        if parsed.run_id != self.run_id:
            raise BridgeCaptureError("target record run_id differs from shard")
        if (
            parsed.source_manifest_hash != self.source_manifest_hash
            or parsed.model_revision != self.model_revision
            or parsed.tokenizer_fingerprint != self.tokenizer_fingerprint
            or parsed.config_sha256 != self.config_sha256
            or parsed.candidate_manifest_sha256 != self.candidate_manifest_sha256
        ):
            raise BridgeCaptureError("target record provenance differs from shard")
        if (parsed.donor_layer, parsed.expert_id) not in self.target_experts:
            raise BridgeCaptureError("target record is outside shard expert set")
        expected_event_id = target_event_id(
            parsed.sample_id,
            parsed.token_position,
            parsed.donor_layer,
            parsed.expert_id,
            parsed.route_rank,
        )
        if parsed.event_id != expected_event_id:
            raise BridgeCaptureError("target record event_id does not match its route identity")
        if parsed.event_id in self._event_ids:
            raise BridgeCaptureError(f"duplicate target event: {parsed.event_id}")
        self._event_ids.add(parsed.event_id)
        self._rows.append(
            (
                parsed,
                _validate_bf16_vector(expert_input, hidden_size=self.hidden_size),
                _validate_bf16_vector(
                    replayed_expert_output, hidden_size=self.hidden_size
                ),
                _validate_bf16_vector(
                    weighted_replayed_expert_output, hidden_size=self.hidden_size
                ),
            )
        )

    def finalize(self) -> Path:
        if not self._rows:
            raise BridgeCaptureError("cannot publish an empty target shard")
        temporary = Path(tempfile.mkdtemp(prefix=f".shard-{self.shard_id}.", dir=self.root))
        try:
            records_path = temporary / "records.jsonl"
            tensors_path = temporary / "targets.safetensors"
            checksums_path = temporary / "checksums.sha256"
            metadata_path = temporary / "shard.json"
            try:
                import torch
                from safetensors.torch import save_file
            except ImportError as error:  # pragma: no cover - dependency is declared
                raise BridgeCaptureError(
                    "safetensors.torch is required for target shards"
                ) from error
            ordered = sorted(
                self._rows,
                key=lambda row: (
                    row[0].sample_ordinal,
                    row[0].token_position,
                    row[0].donor_layer,
                    row[0].expert_id,
                    row[0].route_rank,
                    row[0].event_id,
                ),
            )
            records = [
                record.model_copy(update={"row_index": row_index})
                for row_index, (record, _, _, _) in enumerate(ordered)
            ]
            tensors = {
                "expert_input": torch.stack([row[1] for row in ordered]),
                "replayed_expert_output": torch.stack([row[2] for row in ordered]),
                "weighted_replayed_expert_output": torch.stack(
                    [row[3] for row in ordered]
                ),
            }
            save_file(
                tensors,
                str(tensors_path),
                metadata={"run_id": self.run_id, "dtype": "BF16", "schema_version": "1"},
            )
            with records_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            checksums = {
                records_path.name: _file_sha256(records_path),
                tensors_path.name: _file_sha256(tensors_path),
            }
            checksums_path.write_text(
                "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
                encoding="utf-8",
            )
            event_counts = Counter(
                f"{record.domain}:{record.donor_layer}:{record.expert_id}"
                for record in records
            )
            analyzed_tokens = len(
                {(record.sample_id, record.token_position) for record in records}
            )
            base = {
                "schema_version": 1,
                "kind": "bridge-target-shard",
                "run_id": self.run_id,
                "shard_id": self.shard_id,
                "source_manifest_hash": self.source_manifest_hash,
                "model_revision": self.model_revision,
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "record_count": len(records),
                "analyzed_tokens": analyzed_tokens,
                "event_counts": dict(sorted(event_counts.items())),
                "records_file": records_path.name,
                "tensors_file": tensors_path.name,
                "checksums_file": checksums_path.name,
                "checksums": checksums,
                "tensor_keys": [
                    "expert_input",
                    "replayed_expert_output",
                    "weighted_replayed_expert_output",
                ],
                "config_sha256": self.config_sha256,
                "candidate_manifest_sha256": self.candidate_manifest_sha256,
                "target_experts": [
                    {"layer": layer, "expert": expert}
                    for layer, expert in sorted(self.target_experts)
                ],
            }
            base["shard_sha256"] = hashlib.sha256(canonical_json(base)).hexdigest()
            shard = BridgeTargetShard.model_validate({**base})
            _write_json_atomic(metadata_path, shard.model_dump(mode="json"))
            validate_target_shard(temporary, hidden_size=self.hidden_size)
            os.replace(temporary, self.final_dir)
            return self.final_dir
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def validate_target_shard(path: Path, *, hidden_size: int = 2048) -> dict[str, Any]:
    """Validate one completed shard before it can be resumed or transferred."""
    metadata_path = path / "shard.json"
    if not metadata_path.exists():
        raise BridgeCaptureError(f"target shard has no completion marker: {path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    shard = BridgeTargetShard.model_validate(payload)
    expected_shard_hash = hashlib.sha256(
        canonical_json({k: v for k, v in payload.items() if k != "shard_sha256"})
    ).hexdigest()
    if expected_shard_hash != shard.shard_sha256:
        raise BridgeCaptureError(f"target shard metadata hash mismatch: {path}")
    records_path = path / shard.records_file
    tensors_path = path / shard.tensors_file
    checksums_path = path / shard.checksums_file
    expected_checksum_names = {shard.records_file, shard.tensors_file}
    if set(shard.checksums) != expected_checksum_names:
        raise BridgeCaptureError("target shard checksum inventory is incomplete")
    for filename, digest in shard.checksums.items():
        candidate = path / filename
        if not candidate.is_file() or _file_sha256(candidate) != digest:
            raise BridgeCaptureError(f"target shard checksum mismatch: {candidate}")
    if not checksums_path.is_file():
        raise BridgeCaptureError("target shard checksums file is missing")
    checksum_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    parsed_checksums: dict[str, str] = {}
    for line in checksum_lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise BridgeCaptureError("target shard checksums file is malformed")
        if parts[1] in parsed_checksums:
            raise BridgeCaptureError("target shard checksums file contains duplicates")
        parsed_checksums[parts[1]] = parts[0]
    if parsed_checksums != shard.checksums:
        raise BridgeCaptureError("target shard checksums file disagrees with metadata")
    records: list[BridgeTargetRecord] = []
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(BridgeTargetRecord.model_validate_json(line))
            except Exception as error:
                raise BridgeCaptureError(
                    f"invalid target record at {path}:{line_number}"
                ) from error
    if len(records) != shard.record_count:
        raise BridgeCaptureError("target shard record count mismatch")
    target_experts = {(item.layer, item.expert) for item in shard.target_experts}
    identities: set[str] = set()
    if [record.row_index for record in records] != list(range(shard.record_count)):
        raise BridgeCaptureError("target shard row_index values are not contiguous")
    for record in records:
        if record.run_id != shard.run_id:
            raise BridgeCaptureError("target record run_id mismatch")
        if (
            record.source_manifest_hash != shard.source_manifest_hash
            or record.model_revision != shard.model_revision
            or record.tokenizer_fingerprint != shard.tokenizer_fingerprint
            or record.config_sha256 != shard.config_sha256
            or record.candidate_manifest_sha256 != shard.candidate_manifest_sha256
        ):
            raise BridgeCaptureError("target record provenance differs from shard")
        if (record.donor_layer, record.expert_id) not in target_experts:
            raise BridgeCaptureError("target record expert is not declared by shard")
        identity = target_event_id(
            record.sample_id,
            record.token_position,
            record.donor_layer,
            record.expert_id,
            record.route_rank,
        )
        if identity != record.event_id:
            raise BridgeCaptureError("target record event_id does not match its route identity")
        if identity in identities:
            raise BridgeCaptureError(f"duplicate target record identity: {identity}")
        identities.add(identity)
        if not math.isfinite(record.router_weight) or record.router_weight < 0:
            raise BridgeCaptureError("target record has invalid router weight")
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise BridgeCaptureError(
            "torch and safetensors are required to validate target tensors"
        ) from error
    expected_tensor_keys = {
        "expert_input",
        "replayed_expert_output",
        "weighted_replayed_expert_output",
    }
    with safe_open(str(tensors_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected_tensor_keys:
            raise BridgeCaptureError("target tensor keys do not match metadata records")
        for key in sorted(expected_tensor_keys):
            tensor = handle.get_tensor(key)
            if tuple(tensor.shape) != (shard.record_count, hidden_size) or (
                tensor.dtype != torch.bfloat16
            ):
                raise BridgeCaptureError(
                    f"target tensor {key} is not BF16 [{shard.record_count}, {hidden_size}]"
                )
            if not torch.isfinite(tensor.float()).all():
                raise BridgeCaptureError(f"target tensor {key} contains NaN or infinity")
    return {
        "valid": True,
        "path": str(path),
        "shard_id": shard.shard_id,
        "record_count": shard.record_count,
        "analyzed_tokens": shard.analyzed_tokens,
        "checksums": shard.checksums,
    }


def build_target_handoff(
    capture_root: Path,
    capture_manifest_path: Path,
    destination: Path,
    *,
    extraction_dir: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Build a hash-only handoff manifest after all capture shards validate.

    With ``allow_incomplete=True`` a hash-valid but coverage-incomplete
    checkpoint is bundled with an explicit ``coverage-incomplete``
    classification instead of raising; every integrity gate (checkpoint hash,
    manifest binding, shard inventory agreement, record membership,
    contiguity, uniqueness) still applies. The default preserves the
    success-only behavior.
    """
    capture_manifest = load_bridge_manifest(capture_manifest_path)
    shard_dirs = sorted(path for path in capture_root.glob("shard-*") if path.is_dir())
    if not shard_dirs:
        raise BridgeCaptureError(f"no completed target shards found under {capture_root}")
    shard_reports = [validate_target_shard(path) for path in shard_dirs]
    expected_targets = {(item.layer, item.expert) for item in capture_manifest.experts}
    sample_by_ordinal = {
        item.sample_ordinal: item for item in capture_manifest.samples
    }
    expected_ordinals = set(range(len(capture_manifest.samples)))
    if set(sample_by_ordinal) != expected_ordinals:
        raise BridgeCaptureError("capture manifest sample ordinals are not contiguous")
    record_count = 0
    token_identities: set[tuple[str, int]] = set()
    event_ids: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    state_path = capture_root / "capture-state.json"
    if not state_path.is_file():
        raise BridgeCaptureError("target capture is missing its progress checkpoint")
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    try:
        state = BridgeTargetCaptureState.model_validate(state_payload)
    except Exception as error:
        raise BridgeCaptureError("target capture checkpoint is invalid") from error
    expected_state_hash = hashlib.sha256(
        canonical_json(
            {key: value for key, value in state_payload.items() if key != "state_sha256"}
        )
    ).hexdigest()
    if expected_state_hash != state.state_sha256:
        raise BridgeCaptureError("target capture checkpoint is tampered")
    if not state.complete and not allow_incomplete:
        raise BridgeCaptureError("target capture checkpoint is incomplete")
    if state.run_id != capture_manifest.run_id:
        raise BridgeCaptureError("capture checkpoint run_id differs from capture manifest")
    if state.capture_manifest_sha256 != capture_manifest.manifest_sha256:
        raise BridgeCaptureError("capture checkpoint manifest hash differs from capture manifest")
    coverage = state.coverage
    if not isinstance(coverage, dict):
        raise BridgeCaptureError("target capture checkpoint has no coverage report")
    outcome = (
        "coverage-complete" if coverage.get("capture_success") else "coverage-incomplete"
    )
    if outcome == "coverage-incomplete" and not allow_incomplete:
        raise BridgeCaptureError("target capture checkpoint does not prove coverage success")
    if int(coverage.get("analyzed_tokens", -1)) < capture_manifest.target_analyzed_tokens:
        raise BridgeCaptureError("target capture checkpoint is below the target token count")
    if set(state.completed_shards) != {path.name for path in shard_dirs}:
        raise BridgeCaptureError("checkpoint and completed target shards disagree")
    if state.next_sample_index > len(capture_manifest.samples):
        raise BridgeCaptureError("checkpoint sample index exceeds capture manifest")
    artifacts.append(
        {
            "path": _relative_safe(capture_root.parent, state_path).as_posix(),
            "bytes": state_path.stat().st_size,
            "sha256": _file_sha256(state_path),
        }
    )
    artifacts.append(
        {
            "path": _relative_safe(capture_root.parent, capture_manifest_path).as_posix(),
            "bytes": capture_manifest_path.stat().st_size,
            "sha256": _file_sha256(capture_manifest_path),
        }
    )
    for shard_dir, report in zip(shard_dirs, shard_reports, strict=True):
        metadata = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
        shard_targets = {
            (item["layer"], item["expert"]) for item in metadata["target_experts"]
        }
        if shard_targets != expected_targets:
            raise BridgeCaptureError(
                "shard candidate set differs from capture manifest: "
                f"{shard_dir}"
            )
        for key in (
            "run_id",
            "source_manifest_hash",
            "config_sha256",
            "candidate_manifest_sha256",
        ):
            expected = {
                "run_id": capture_manifest.run_id,
                "source_manifest_hash": capture_manifest.source_manifest_sha256,
                "config_sha256": capture_manifest.config_sha256,
                "candidate_manifest_sha256": capture_manifest.candidate_manifest_sha256,
            }[key]
            if metadata[key] != expected:
                raise BridgeCaptureError(f"shard {key} differs from capture manifest: {shard_dir}")
        records_path = shard_dir / metadata["records_file"]
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = BridgeTargetRecord.model_validate_json(line)
                selected = sample_by_ordinal.get(record.sample_ordinal)
                if selected is None or record.sample_ordinal >= state.next_sample_index:
                    raise BridgeCaptureError(
                        "target record is outside the contiguous resume prefix"
                    )
                if (
                    record.sample_id != selected.sample.sample_id
                    or record.sample_content_sha256 != selected.sample.content_sha256
                    or record.domain != selected.sample.domain
                    or record.split != selected.sample.split
                    or record.token_position >= selected.token_count
                ):
                    raise BridgeCaptureError(
                        "target record does not match capture sample membership"
                    )
                if record.event_id in event_ids:
                    raise BridgeCaptureError(
                        f"duplicate target event across shards: {record.event_id}"
                    )
                event_ids.add(record.event_id)
                token_identities.add((record.sample_id, record.token_position))
        record_count += int(report["record_count"])
        for path in sorted(item for item in shard_dir.rglob("*") if item.is_file()):
            relative = _relative_safe(capture_root.parent, path)
            artifacts.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    extraction_artifact: str | None = None
    if extraction_dir is not None:
        manifest = extraction_dir / "extraction-manifest.json"
        verification = extraction_dir / "verification-report.json"
        if not manifest.is_file() or not verification.is_file():
            raise BridgeCaptureError("extraction directory is missing its required reports")
        extraction_artifact = _relative_safe(destination.parent, manifest).as_posix()
        for path in sorted(item for item in extraction_dir.rglob("*") if item.is_file()):
            relative = _relative_safe(destination.parent, path)
            artifacts.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    base: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bridge-target-handoff",
        "run_id": capture_manifest.run_id,
        "capture_manifest_sha256": capture_manifest.manifest_sha256,
        "config_sha256": capture_manifest.config_sha256,
        "candidate_manifest_sha256": capture_manifest.candidate_manifest_sha256,
        "target_experts": [item.model_dump(mode="json") for item in capture_manifest.experts],
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "record_count": record_count,
        # The checkpoint counts every teacher-forced non-padding token.  The
        # shard records necessarily cover only tokens that routed through one
        # of the selected experts, so their unique-token count is not the
        # experiment's analysed-token denominator.
        "analyzed_tokens": int(coverage["analyzed_tokens"]),
        "capture_outcome": outcome,
        "coverage": coverage,
        "extraction_artifact": extraction_artifact,
    }
    base["bundle_sha256"] = hashlib.sha256(canonical_json(base)).hexdigest()
    bundle = BridgeTargetBundle.model_validate(base)
    _write_json_atomic(destination, bundle.model_dump(mode="json"))
    return bundle.model_dump(mode="json")


def validate_target_handoff(path: Path) -> dict[str, Any]:
    """Revalidate every hash in a copied handoff manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    bundle = BridgeTargetBundle.model_validate(payload)
    expected = hashlib.sha256(
        canonical_json({key: value for key, value in payload.items() if key != "bundle_sha256"})
    ).hexdigest()
    if expected != bundle.bundle_sha256:
        raise BridgeCaptureError("target handoff bundle hash mismatch")
    root = path.parent
    for artifact in bundle.artifacts:
        artifact_path = root / artifact["path"]
        if _relative_safe(root, artifact_path).as_posix() != artifact["path"]:
            raise BridgeCaptureError("target handoff contains an unsafe artifact path")
        if not artifact_path.is_file() or artifact_path.stat().st_size != artifact["bytes"]:
            raise BridgeCaptureError(
                f"target handoff artifact missing or truncated: {artifact_path}"
            )
        if _file_sha256(artifact_path) != artifact["sha256"]:
            raise BridgeCaptureError(f"target handoff artifact hash mismatch: {artifact_path}")
    return {
        "valid": True,
        "artifact_count": len(bundle.artifacts),
        "bundle_sha256": bundle.bundle_sha256,
    }
