"""A small, hash-bound representation bridge for the Qwen3.5 handoff.

This module is deliberately separate from the Reverse-REAP v0 path.  The v0
donor tensors remain immutable; this code consumes a verified handoff and
trains only a bridge sidecar for a frozen ``Qwen/Qwen3.5-2B`` host.  Torch is an
optional dependency: configuration, split repair, and preflight work on a CPU
installation, while model construction and training fail with a useful error
when torch is not installed.

The bridge is intentionally conservative:

``host hidden -> trainable input adapter -> frozen donor SwiGLU expert
             -> trainable output adapter -> capped sigmoid residual gate``

The sidecar is attached at an explicitly configured host MLP layer.  It never
loads or saves host weights.  All rows retain their shard/row references and
all excluded rows receive an explicit reason in the repaired manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from reverse_reap.bridge_capture import (
    BridgeTargetRecord,
    load_bridge_manifest,
    validate_target_handoff,
    validate_target_shard,
)
from reverse_reap.config import StrictModel
from reverse_reap.datasets import canonical_json


class BridgeTrainingError(RuntimeError):
    """Raised when bridge inputs or training invariants are unsafe."""


__all__ = [
    "BridgeTrainingError",
    "BridgeHostConfig",
    "BridgeDonorConfig",
    "BridgeExpertMapping",
    "BridgeDataConfig",
    "BridgeRuntimeConfig",
    "BridgeLossConfig",
    "BridgeControlsConfig",
    "BridgeBudgetConfig",
    "BridgeTrainingConfig",
    "BridgeHostStateRecord",
    "BridgeHostStateManifest",
    "BridgeTrainingRow",
    "BridgeUnusedRow",
    "BridgeTrainingManifest",
    "BridgeCheckpoint",
    "BridgeExample",
    "FrozenSwiGLUExpert",
    "EqualParameterAdapter",
    "BridgeModel",
    "install_bridge_sidecars",
    "load_bridge_training_config",
    "freeze_host_state_manifest",
    "capture_host_hidden_states",
    "repair_bridge_manifest",
    "load_training_manifest",
    "load_bridge_examples",
    "assert_frozen_module",
    "bridge_trainable_state",
    "save_bridge_safetensors",
    "save_bridge_checkpoint",
    "load_bridge_checkpoint",
    "train_bridge",
    "bridge_control_keys",
    "bridge_control_status",
    "estimate_bridge_memory",
    "validate_bridge_training_config",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_without(payload: Mapping[str, Any], field: str) -> str:
    body = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _write_json_atomic(
    path: Path, payload: Mapping[str, Any], *, refuse_existing: bool = False
) -> None:
    """Write JSON durably, refusing accidental overwrite by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if refuse_existing and path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise BridgeTrainingError(f"refusing to overwrite existing artifact: {path}")
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


class BridgeHostConfig(StrictModel):
    """Pinned recipient model contract."""

    model_id: Literal["Qwen/Qwen3.5-2B"]
    revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    hidden_size: Literal[2048] = 2048
    precision: Literal["bf16"] = "bf16"
    num_layers: int = Field(gt=0)


class BridgeDonorConfig(StrictModel):
    """Immutable donor/extraction identity crossing the VM boundary."""

    model_id: Literal["Qwen/Qwen3.5-35B-A3B"]
    revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    extraction_dir: Path
    extraction_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BridgeExpertMapping(StrictModel):
    """Explicit mapping from one donor expert to one host MLP sidecar layer."""

    donor_layer: int = Field(ge=0)
    donor_expert: int = Field(ge=0)
    host_layer: int = Field(ge=0)

    @property
    def key(self) -> str:
        return f"{self.donor_layer}:{self.donor_expert}"


class BridgeDataConfig(StrictModel):
    """Hash-bound input inventory for a bridge run."""

    handoff_manifest: Path
    handoff_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    host_states_manifest: Path
    host_states_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_manifest: Path
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allow_observational_coverage_incomplete: bool = False


class BridgeRuntimeConfig(StrictModel):
    """Deterministic B1 training settings."""

    seed: int = Field(ge=0)
    deterministic: Literal[True] = True
    batch_size: Literal[1] = 1
    max_epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    learning_rate: float = Field(gt=0, lt=1)
    weight_decay: float = Field(ge=0, lt=1)
    max_grad_norm: float = Field(gt=0)
    bottleneck_size: int = Field(gt=0, le=2048)
    gate_cap: float = Field(gt=0, le=1)
    gate_init_bias: float = Field(ge=-20, le=0)
    device: Literal["cpu", "cuda"] = "cuda"


class BridgeLossConfig(StrictModel):
    input_weight: float = Field(ge=0)
    output_weight: float = Field(ge=0)
    retention_weight: float = Field(ge=0)

    @model_validator(mode="after")
    def positive_total(self) -> BridgeLossConfig:
        if self.input_weight + self.output_weight + self.retention_weight <= 0:
            raise ValueError("at least one bridge loss weight must be positive")
        return self


class BridgeControlsConfig(StrictModel):
    """Required controls; random experts are only available with a verified artifact."""

    conditions: list[
        Literal[
            "disabled",
            "untrained",
            "trained",
            "shuffled-pair",
            "random-expert",
            "equal-parameter-adapter",
        ]
    ] = Field(
        default=[
            "disabled",
            "untrained",
            "trained",
            "shuffled-pair",
            "equal-parameter-adapter",
        ],
        min_length=5,
    )

    @field_validator("conditions")
    @classmethod
    def unique_conditions(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("bridge controls must be unique")
        return value


class BridgeBudgetConfig(StrictModel):
    """Hard resource and deadline contract for a future GPU run."""

    max_gpu_hours: float = Field(gt=0)
    max_cost_usd: float = Field(gt=0)
    provider_rate_usd_per_hour: float = Field(gt=0)
    storage_limit_gb: float = Field(gt=0)
    deadline_utc: datetime

    @model_validator(mode="after")
    def timezone_required(self) -> BridgeBudgetConfig:
        if self.deadline_utc.tzinfo is None or self.deadline_utc.utcoffset() is None:
            raise ValueError("bridge budget deadline_utc must include a timezone")
        return self


class BridgeTrainingConfig(StrictModel):
    """Strict, intentionally unlaunchable-until-pinned bridge configuration."""

    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    host: BridgeHostConfig
    donor: BridgeDonorConfig
    data: BridgeDataConfig
    mappings: list[BridgeExpertMapping] = Field(min_length=1)
    runtime: BridgeRuntimeConfig
    losses: BridgeLossConfig
    controls: BridgeControlsConfig = BridgeControlsConfig()
    budget: BridgeBudgetConfig
    output_dir: Path

    @field_validator("run_id")
    @classmethod
    def resolved_run_id(cls, value: str) -> str:
        if value.lower() in {"unresolved", "required", "replace-me"}:
            raise ValueError("bridge training requires a resolved run_id")
        return value

    @field_validator("mappings")
    @classmethod
    def unique_mappings(cls, value: list[BridgeExpertMapping]) -> list[BridgeExpertMapping]:
        identities = [item.key for item in value]
        if len(identities) != len(set(identities)):
            raise ValueError("donor expert mappings must be unique")
        for item in value:
            if item.host_layer >= 10_000:
                raise ValueError("host layer is implausibly large")
        return value

    @model_validator(mode="after")
    def host_layers_in_range(self) -> BridgeTrainingConfig:
        if any(item.host_layer >= self.host.num_layers for item in self.mappings):
            raise ValueError("every mapped host layer must be in the host model")
        if "random-expert" in self.controls.conditions:
            raise ValueError(
                "random-expert requires a separately verified random extraction artifact; "
                "the v1 config intentionally does not admit one"
            )
        if self.budget.deadline_utc.astimezone(UTC) <= datetime.now(UTC):
            raise ValueError("bridge budget deadline has expired")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class BridgeHostStateRecord(StrictModel):
    """Index row for one frozen host hidden state."""

    schema_version: Literal[1] = 1
    row_index: int = Field(ge=0)
    sample_id: str = Field(min_length=1)
    sample_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    host_layer: int = Field(ge=0)
    tensor_key: str = Field(min_length=1)

    @field_validator("tensor_key")
    @classmethod
    def local_tensor_key(cls, value: str) -> str:
        if "/" in value or "\\" in value:
            raise ValueError("host tensor keys cannot contain path separators")
        return value


class BridgeHostStateManifest(StrictModel):
    """Hash inventory for host hidden states captured with the frozen host."""

    schema_version: Literal[1]
    kind: Literal["bridge-host-states"]
    run_id: str = Field(min_length=1)
    host_model_id: Literal["Qwen/Qwen3.5-2B"]
    host_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    hidden_size: Literal[2048] = 2048
    tensors_file: str = Field(min_length=1)
    records_file: str = Field(min_length=1)
    tensors_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    records: list[BridgeHostStateRecord] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def contiguous_rows(self) -> BridgeHostStateManifest:
        if [row.row_index for row in self.records] != list(range(self.row_count)):
            raise ValueError("host state rows must have contiguous row_index values")
        if len(self.records) != self.row_count:
            raise ValueError("host state row_count does not match records")
        return self


class BridgeTrainingRow(StrictModel):
    """Reference into an immutable target shard plus a host state row."""

    schema_version: Literal[1] = 1
    event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_path: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    host_state_row: int = Field(ge=0)
    sample_id: str = Field(min_length=1)
    sample_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain: Literal["coding", "control"]
    split: Literal["train", "validation", "test"]
    token_position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    donor_layer: int = Field(ge=0)
    donor_expert: int = Field(ge=0)
    host_layer: int = Field(ge=0)
    route_rank: int = Field(ge=0)
    router_weight: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("shard_path")
    @classmethod
    def relative_shard_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("training shard paths must be relative to the handoff root")
        return value


class BridgeUnusedRow(StrictModel):
    """A valid source row deliberately excluded, with a machine-readable reason."""

    event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_path: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    sample_id: str = Field(min_length=1)
    sample_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    donor_layer: int = Field(ge=0)
    donor_expert: int = Field(ge=0)
    domain: Literal["coding", "control"]
    reason: Literal["missing-host-state", "per-sample-cap", "invalid-token-alignment"]


class BridgeTrainingManifest(StrictModel):
    """Frozen, sample-grouped split and row-reference manifest."""

    schema_version: Literal[1]
    kind: Literal["bridge-training-manifest"]
    handoff_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    host_states_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mappings: list[BridgeExpertMapping] = Field(min_length=1)
    split_counts: dict[str, int]
    cell_counts: dict[str, dict[str, int]]
    rows: list[BridgeTrainingRow] = Field(min_length=1)
    unused_rows: list[BridgeUnusedRow] = Field(default_factory=list)
    seed: int = Field(ge=0)
    min_samples_per_cell: int = Field(gt=0)
    max_rows_per_sample: int = Field(gt=0)
    min_events_per_cell: int = Field(gt=0)
    capture_outcome: Literal["coverage-complete", "coverage-incomplete"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def row_counts(self) -> BridgeTrainingManifest:
        if sum(self.split_counts.values()) != len(self.rows):
            raise ValueError("split_counts do not match training rows")
        if set(self.split_counts) != {"train", "validation", "test"}:
            raise ValueError("training manifest must contain all three splits")
        if any(value <= 0 for value in self.split_counts.values()):
            raise ValueError("training manifest splits must all be non-empty")
        if len({row.event_id for row in self.rows}) != len(self.rows):
            raise ValueError("training rows contain duplicate events")
        mapping_layers = {
            (item.donor_layer, item.donor_expert): item.host_layer for item in self.mappings
        }
        computed_cells: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in self.rows:
            key = (row.donor_layer, row.donor_expert)
            if key not in mapping_layers or mapping_layers[key] != row.host_layer:
                raise ValueError("training row has no matching explicit expert mapping")
            computed_cells[f"{row.domain}:{row.donor_layer}:{row.donor_expert}"][row.split] += 1
        normalized = {
            key: dict(sorted(value.items())) for key, value in sorted(computed_cells.items())
        }
        if normalized != self.cell_counts:
            raise ValueError("cell_counts do not match training rows")
        for cell, counts in normalized.items():
            for split in ("train", "validation", "test"):
                if counts.get(split, 0) < self.min_events_per_cell:
                    raise ValueError(
                        f"cell {cell}/{split} has fewer than "
                        f"{self.min_events_per_cell} events"
                    )
        return self


class BridgeCheckpoint(StrictModel):
    """Hash-bound checkpoint metadata; tensor payloads contain bridge state only."""

    schema_version: Literal[1]
    kind: Literal["bridge-checkpoint"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    epoch: int = Field(ge=0)
    step: int = Field(ge=0)
    metrics: dict[str, Any]
    bridge_file: str = Field(min_length=1)
    optimizer_file: str = Field(min_length=1)
    bridge_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("bridge_file", "optimizer_file")
    @classmethod
    def local_checkpoint_file(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError("checkpoint files must be local filenames")
        return value


@dataclass(frozen=True)
class _SourceRow:
    record: BridgeTargetRecord
    shard_path: str
    host_state_row: int | None
    host_layer: int


def load_bridge_training_config(path: Path) -> BridgeTrainingConfig:
    """Load and strictly validate a bridge config without loading torch."""
    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return BridgeTrainingConfig.model_validate(payload)
    except Exception as error:
        raise BridgeTrainingError(f"invalid bridge training config: {path}: {error}") from error


def freeze_host_state_manifest(
    tensors_path: Path,
    records_path: Path,
    destination: Path,
    *,
    run_id: str,
    host_revision: str,
) -> dict[str, Any]:
    """Freeze a host hidden-state index without loading the host model.

    The capture process is responsible for writing the safetensors and JSONL
    rows.  This helper only binds their hashes and validates the exact row
    identity contract, making it safe to run on the CPU handoff machine.
    """
    if not tensors_path.is_file() or not records_path.is_file():
        raise BridgeTrainingError("host state tensors and records are required")
    try:
        records = [
            BridgeHostStateRecord.model_validate_json(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = BridgeHostStateManifest(
            schema_version=1,
            kind="bridge-host-states",
            run_id=run_id,
            host_model_id="Qwen/Qwen3.5-2B",
            host_revision=host_revision,
            tensors_file=tensors_path.name,
            records_file=records_path.name,
            tensors_sha256=_sha256_file(tensors_path),
            records_sha256=_sha256_file(records_path),
            row_count=len(records),
            records=records,
            manifest_sha256="0" * 64,
        )
    except Exception as error:
        raise BridgeTrainingError("invalid host state records") from error
    body = manifest.model_dump(mode="json")
    body["manifest_sha256"] = _hash_without(body, "manifest_sha256")
    frozen = BridgeHostStateManifest.model_validate(body)
    _write_json_atomic(destination, frozen.model_dump(mode="json"), refuse_existing=True)
    return frozen.model_dump(mode="json")


def _resolve_host_layers(model: Any) -> Sequence[Any]:
    """Find decoder layers across text-only and conditional Qwen wrappers."""
    for path in (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        value = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None and len(value) > 0 and all(hasattr(layer, "mlp") for layer in value):
            return value
    raise BridgeTrainingError("could not locate host decoder layers with MLP modules")


def capture_host_hidden_states(
    host_model_path: Path,
    tokenizer_path: Path,
    handoff_path: Path,
    capture_manifest_path: Path,
    destination: Path,
    *,
    mappings: Sequence[BridgeExpertMapping | Mapping[str, Any]],
    host_revision: str,
    run_id: str,
    allow_observational_coverage_incomplete: bool = False,
) -> dict[str, Any]:
    """Capture only host rows needed by a verified handoff.

    This is the one GPU-facing data-preparation helper.  It loads the selected
    host checkpoint read-only, uses the exact donor chat-template sequence, and
    saves hidden states at the configured host-layer MLP *input* using a
    temporary forward pre-hook.  It intentionally does not save the host
    model or any logits.
    """
    torch = _require_torch()
    try:
        from transformers import AutoTokenizer
        try:
            # Qwen3.5-2B is registered as conditional generation in current
            # Transformers.  Keep the causal-LM fallback for older pins.
            from transformers import AutoModelForImageTextToText as _HostModel
        except ImportError:  # pragma: no cover - depends on Transformers pin
            from transformers import AutoModelForCausalLM as _HostModel
    except ImportError as error:  # pragma: no cover - optional GPU dependency
        raise BridgeTrainingError("transformers is required for host-state capture") from error
    typed_mappings = [
        item
        if isinstance(item, BridgeExpertMapping)
        else BridgeExpertMapping.model_validate(item)
        for item in mappings
    ]
    mapping_lookup = _mapping_lookup(typed_mappings)
    capture_manifest = load_bridge_manifest(capture_manifest_path)
    # The host-states manifest records the bridge run ID, not the donor run
    # ID: manifest-hash bindings (checked by repair and training) already
    # prevent mixing donor generations. Requiring equality here would collapse
    # the two namespaces and contradict the documented --run-id <bridge-run-id>.
    source_rows = _load_source_rows(
        handoff_path,
        allow_observational_coverage_incomplete=allow_observational_coverage_incomplete,
    )
    needed: set[tuple[str, int, int]] = set()
    expected_ids: dict[tuple[str, int], int] = {}
    for record, _ in source_rows:
        mapping = mapping_lookup.get((record.donor_layer, record.expert_id))
        if mapping is None:
            raise BridgeTrainingError(
                f"no host mapping for handoff expert {record.donor_layer}:{record.expert_id}"
            )
        needed.add((record.sample_content_sha256, record.token_position, mapping.host_layer))
        previous = expected_ids.setdefault(
            (record.sample_content_sha256, record.token_position), record.token_id
        )
        if previous != record.token_id:
            raise BridgeTrainingError("handoff contains conflicting token IDs")
    sample_by_id = {
        item.sample.sample_id: item.sample for item in capture_manifest.samples
    }
    samples_by_content = {sample.content_sha256: sample for sample in sample_by_id.values()}
    if any(content not in samples_by_content for content, _, _ in needed):
        raise BridgeTrainingError("handoff contains a sample absent from its capture manifest")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    model = _HostModel.from_pretrained(
        str(host_model_path),
        revision=host_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
    )
    assert_frozen_module(model)
    host_layers = _resolve_host_layers(model)
    if any(item.host_layer >= len(host_layers) for item in typed_mappings):
        raise BridgeTrainingError("configured host layer is absent from the loaded host")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hidden_rows: list[Any] = []
    index_rows: list[BridgeHostStateRecord] = []
    captured_mlp_inputs: dict[int, Any] = {}

    def capture_mlp_input(layer_index: int):
        def hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]):
            value = kwargs.get("hidden_states")
            if value is None and args:
                value = args[0]
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise BridgeTrainingError(
                    f"host MLP layer {layer_index} did not receive "
                    "[batch,sequence,hidden]"
                )
            captured_mlp_inputs[layer_index] = value.detach()

        return hook

    hook_handles = [
        host_layers[layer_index].mlp.register_forward_pre_hook(
            capture_mlp_input(layer_index), with_kwargs=True
        )
        for layer_index in sorted({layer for _, _, layer in needed})
    ]
    try:
        for sample in sorted(samples_by_content.values(), key=lambda item: item.sample_id):
            sample_needed = sorted(
                (position, layer)
                for content, position, layer in needed
                if content == sample.content_sha256
            )
            if not sample_needed:
                continue
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": sample.prompt},
                    {"role": "assistant", "content": sample.reference or ""},
                ],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                enable_thinking=False,
            )
            if isinstance(rendered, Mapping):
                rendered = rendered["input_ids"]
            ids = rendered if isinstance(rendered, torch.Tensor) else torch.as_tensor(rendered)
            ids = ids.to(dtype=torch.long)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise BridgeTrainingError("host tokenizer did not return one sequence")
            for token_position, _ in sample_needed:
                expected = expected_ids[(sample.content_sha256, token_position)]
                actual = int(ids[0, token_position].item()) if token_position < ids.shape[1] else -1
                if actual != expected:
                    raise BridgeTrainingError(
                        f"host/donor token ID mismatch at {sample.sample_id}:{token_position}"
                    )
            captured_mlp_inputs.clear()
            with torch.inference_mode():
                model(
                    input_ids=ids.to(device),
                    attention_mask=torch.ones_like(ids, device=device),
                    use_cache=False,
                )
            for token_position, host_layer in sample_needed:
                hidden_states = captured_mlp_inputs.get(host_layer)
                if hidden_states is None or token_position >= hidden_states.shape[1]:
                    raise BridgeTrainingError(
                        "host MLP hidden-state sequence/layer is out of range"
                    )
                value = hidden_states[0, token_position].detach().cpu().contiguous()
                if value.ndim != 1 or value.shape[0] != 2048:
                    raise BridgeTrainingError("host hidden state is not a 2048-vector")
                hidden_rows.append(value)
                index_rows.append(
                    BridgeHostStateRecord(
                        row_index=len(index_rows),
                        sample_id=sample.sample_id,
                        sample_content_sha256=sample.content_sha256,
                        token_position=token_position,
                        token_id=int(ids[0, token_position].item()),
                        host_layer=host_layer,
                        tensor_key="hidden_states",
                    )
                )
    finally:
        for handle in hook_handles:
            handle.remove()
    if not hidden_rows:
        raise BridgeTrainingError("no handoff rows were matched for host-state capture")
    destination.mkdir(parents=True, exist_ok=True)
    tensors_path = destination / "host-states.safetensors"
    records_path = destination / "host-states.jsonl"
    manifest_path = destination / "host-states-manifest.json"
    if tensors_path.exists() or records_path.exists() or manifest_path.exists():
        raise BridgeTrainingError("refusing to overwrite host-state capture")
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise BridgeTrainingError("safetensors.torch is required") from error
    tensor_tmp = tensors_path.with_suffix(".tmp")
    save_file(
        {"hidden_states": torch.stack(hidden_rows)},
        str(tensor_tmp),
        metadata={"kind": "frozen-host-states", "host_revision": host_revision},
    )
    os.replace(tensor_tmp, tensors_path)
    records_tmp = records_path.with_suffix(".tmp")
    records_tmp.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in index_rows
        ),
        encoding="utf-8",
    )
    os.replace(records_tmp, records_path)
    return freeze_host_state_manifest(
        tensors_path,
        records_path,
        manifest_path,
        run_id=run_id,
        host_revision=host_revision,
    )


def _load_host_states(
    path: Path,
) -> tuple[BridgeHostStateManifest, Path, dict[tuple[str, int, int], int]]:
    """Validate the JSON index and return lookup keys; tensor bytes are lazy."""
    if not path.is_file():
        raise BridgeTrainingError(f"host state manifest is unavailable: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = BridgeHostStateManifest.model_validate(payload)
    except Exception as error:
        raise BridgeTrainingError(f"invalid host state manifest: {path}") from error
    if _hash_without(payload, "manifest_sha256") != manifest.manifest_sha256:
        raise BridgeTrainingError("host state manifest hash mismatch")
    root = path.parent
    tensors = root / manifest.tensors_file
    records = root / manifest.records_file
    if not tensors.is_file() or not records.is_file():
        raise BridgeTrainingError("host state manifest references missing artifacts")
    if _sha256_file(tensors) != manifest.tensors_sha256:
        raise BridgeTrainingError("host state tensor hash mismatch")
    if _sha256_file(records) != manifest.records_sha256:
        raise BridgeTrainingError("host state records hash mismatch")
    parsed_records = [
        BridgeHostStateRecord.model_validate_json(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if parsed_records != manifest.records:
        raise BridgeTrainingError("host state manifest/index records differ")
    lookup: dict[tuple[str, int, int], int] = {}
    for item in parsed_records:
        key = (item.sample_content_sha256, item.token_position, item.host_layer)
        if key in lookup:
            raise BridgeTrainingError(f"duplicate host state key: {key}")
        lookup[key] = item.row_index
    return manifest, tensors, lookup


def _load_source_rows(
    handoff_path: Path, *, allow_observational_coverage_incomplete: bool = False
) -> list[tuple[BridgeTargetRecord, str]]:
    """Read all record rows from the handoff's hash-verified shard inventory."""
    try:
        validate_target_handoff(handoff_path)
        bundle = json.loads(handoff_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BridgeTrainingError(f"invalid target handoff: {handoff_path}: {error}") from error
    capture_outcome = bundle.get("capture_outcome")
    if capture_outcome not in {"coverage-complete", "coverage-incomplete"}:
        raise BridgeTrainingError(f"unsupported handoff capture outcome: {capture_outcome}")
    if capture_outcome == "coverage-incomplete" and not allow_observational_coverage_incomplete:
        raise BridgeTrainingError(
            "coverage-incomplete handoff requires explicit observational opt-in"
        )
    root = handoff_path.parent
    rows: list[tuple[BridgeTargetRecord, str]] = []
    seen_events: set[str] = set()
    for artifact in bundle["artifacts"]:
        relative = str(artifact["path"])
        if not relative.endswith("/records.jsonl") and relative != "records.jsonl":
            continue
        records_path = root / relative
        shard_dir = records_path.parent
        validate_target_shard(shard_dir)
        for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                record = BridgeTargetRecord.model_validate_json(line)
            except Exception as error:
                raise BridgeTrainingError(
                    f"invalid handoff row: {records_path}:{line_number + 1}"
                ) from error
            if record.event_id in seen_events:
                raise BridgeTrainingError(f"duplicate handoff event: {record.event_id}")
            seen_events.add(record.event_id)
            rows.append((record, relative))
    if not rows:
        raise BridgeTrainingError("handoff contains no target rows")
    return rows


def _assert_token_alignment(
    record: BridgeTargetRecord,
    host_record: BridgeHostStateRecord,
    *,
    context: str,
) -> None:
    """Require exact sample, position, and token identity across artifacts."""
    if (
        host_record.sample_id != record.sample_id
        or host_record.sample_content_sha256 != record.sample_content_sha256
        or host_record.token_position != record.token_position
        or host_record.token_id != record.token_id
    ):
        raise BridgeTrainingError(f"host/donor token alignment mismatch for {context}")


def _mapping_lookup(
    mappings: Sequence[BridgeExpertMapping],
) -> dict[tuple[int, int], BridgeExpertMapping]:
    lookup = {(item.donor_layer, item.donor_expert): item for item in mappings}
    if len(lookup) != len(mappings):
        raise BridgeTrainingError("donor expert mapping contains duplicates")
    return lookup


def _assign_sample_groups(
    groups: Mapping[str, Sequence[_SourceRow]],
    labels_by_group: Mapping[str, set[str]],
    *,
    seed: int,
    min_samples_per_cell: int,
) -> dict[str, str]:
    """Deterministic iterative multilabel stratification over content-hash groups."""
    split_names = ("train", "validation", "test")
    labels = sorted(set().union(*(labels_by_group[key] for key in groups)))
    group_keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest(),
    )
    for label in labels:
        if sum(label in labels_by_group[key] for key in group_keys) < 3 * min_samples_per_cell:
            raise BridgeTrainingError(
                f"cannot represent cell {label} in all train/validation/test splits"
            )

    assignment: dict[str, str] = {}
    label_counts = {split: Counter() for split in split_names}
    sample_counts = Counter()

    # Seed each split with the rarest remaining label first.  A group is never
    # reused, preserving sample-content disjointness.
    for split in split_names:
        while any(label_counts[split][label] < min_samples_per_cell for label in labels):
            deficits = [
                label for label in labels if label_counts[split][label] < min_samples_per_cell
            ]
            label = min(
                deficits,
                key=lambda candidate: (
                    sum(
                        candidate in labels_by_group[key]
                        for key in group_keys
                        if key not in assignment
                    ),
                    candidate,
                ),
            )
            candidates = [
                key
                for key in group_keys
                if key not in assignment and label in labels_by_group[key]
            ]
            if not candidates:
                raise BridgeTrainingError(
                    f"iterative stratification exhausted groups for cell {label} in {split}"
                )
            chosen = max(
                candidates,
                key=lambda key: (
                    sum(
                        other in labels_by_group[key]
                        and label_counts[split][other] < min_samples_per_cell
                        for other in labels
                    ),
                    hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest()[::-1],
                ),
            )
            assignment[chosen] = split
            sample_counts[split] += 1
            for other in labels_by_group[chosen]:
                label_counts[split][other] += 1

    # Distribute remaining groups by the most underrepresented labels, then by
    # split size.  The score is deterministic and independent of tensor values.
    for key in group_keys:
        if key in assignment:
            continue
        def score(split: str, *, group_key: str = key) -> tuple[int, int, str]:
            label_need = sum(
                max(0, sample_counts.total() // 3 + 1 - label_counts[split][label])
                for label in labels_by_group[group_key]
            )
            return (
                label_need,
                -sample_counts[split],
                hashlib.sha256(f"{seed}\0{group_key}\0{split}".encode()).hexdigest(),
            )

        chosen_split = max(split_names, key=score)
        assignment[key] = chosen_split
        sample_counts[chosen_split] += 1
        for label in labels_by_group[key]:
            label_counts[chosen_split][label] += 1

    for split in split_names:
        for label in labels:
            if label_counts[split][label] < min_samples_per_cell:
                raise BridgeTrainingError(f"split coverage failed for {split}/{label}")
    return assignment


def repair_bridge_manifest(
    handoff_path: Path,
    host_states_path: Path,
    destination: Path,
    *,
    mappings: Sequence[BridgeExpertMapping | Mapping[str, Any]],
    seed: int = 20260909,
    min_samples_per_cell: int = 1,
    min_events_per_cell: int = 32,
    max_rows_per_sample: int = 128,
    allow_observational_coverage_incomplete: bool = False,
) -> dict[str, Any]:
    """Repair deterministic train/validation/test splits and freeze row refs.

    Rows are grouped by ``sample_content_sha256`` before stratification.  A
    missing host hidden state or a per-sample cap never disappears silently; it
    is recorded in ``unused_rows``.  The function fails closed if every mapped
    expert/domain cell cannot be represented in all three splits.
    """
    if min_samples_per_cell <= 0 or min_events_per_cell <= 0 or max_rows_per_sample <= 0:
        raise BridgeTrainingError("split minima and per-sample cap must be positive")
    typed_mappings = [
        item
        if isinstance(item, BridgeExpertMapping)
        else BridgeExpertMapping.model_validate(item)
        for item in mappings
    ]
    mapping_lookup = _mapping_lookup(typed_mappings)
    source_rows = _load_source_rows(
        handoff_path,
        allow_observational_coverage_incomplete=allow_observational_coverage_incomplete,
    )
    host_manifest, _, host_lookup = _load_host_states(host_states_path)
    bundle = json.loads(handoff_path.read_text(encoding="utf-8"))
    source_by_group: dict[str, list[_SourceRow]] = defaultdict(list)
    unused: list[BridgeUnusedRow] = []
    sample_id_to_content: dict[str, str] = {}
    for record, shard_path in source_rows:
        previous_content = sample_id_to_content.setdefault(
            record.sample_id, record.sample_content_sha256
        )
        if previous_content != record.sample_content_sha256:
            raise BridgeTrainingError(
                "sample_id maps to multiple content hashes; refusing ambiguous grouping"
            )
        mapping = mapping_lookup.get((record.donor_layer, record.expert_id))
        if mapping is None:
            raise BridgeTrainingError(
                f"handoff row has no explicit host mapping: {record.donor_layer}:{record.expert_id}"
            )
        host_row = host_lookup.get(
            (record.sample_content_sha256, record.token_position, mapping.host_layer)
        )
        if host_row is None:
            unused.append(
                BridgeUnusedRow(
                    event_id=record.event_id,
                    shard_path=shard_path,
                    row_index=record.row_index,
                    sample_id=record.sample_id,
                    sample_content_sha256=record.sample_content_sha256,
                    donor_layer=record.donor_layer,
                    donor_expert=record.expert_id,
                    domain=record.domain,
                    reason="missing-host-state",
                )
            )
            continue
        host_record = host_manifest.records[host_row]
        try:
            _assert_token_alignment(record, host_record, context=record.event_id)
        except BridgeTrainingError:
            unused.append(
                BridgeUnusedRow(
                    event_id=record.event_id,
                    shard_path=shard_path,
                    row_index=record.row_index,
                    sample_id=record.sample_id,
                    sample_content_sha256=record.sample_content_sha256,
                    donor_layer=record.donor_layer,
                    donor_expert=record.expert_id,
                    domain=record.domain,
                    reason="invalid-token-alignment",
                )
            )
            continue
        source_by_group[record.sample_content_sha256].append(
            _SourceRow(record, shard_path, host_row, mapping.host_layer)
        )
    if not source_by_group:
        raise BridgeTrainingError("no rows have matching frozen host states")

    # Deterministic per-sample caps keep one long sample from dominating while
    # preserving a complete audit trail for discarded rows.
    capped_groups: dict[str, list[_SourceRow]] = {}
    for group_key, rows in source_by_group.items():
        ordered = sorted(rows, key=lambda row: (row.record.token_position, row.record.event_id))
        capped_groups[group_key] = ordered[:max_rows_per_sample]
        for row in ordered[max_rows_per_sample:]:
            unused.append(
                BridgeUnusedRow(
                    event_id=row.record.event_id,
                    shard_path=row.shard_path,
                    row_index=row.record.row_index,
                    sample_id=row.record.sample_id,
                    sample_content_sha256=row.record.sample_content_sha256,
                    donor_layer=row.record.donor_layer,
                    donor_expert=row.record.expert_id,
                    domain=row.record.domain,
                    reason="per-sample-cap",
                )
            )
    labels_by_group = {
        key: {
            f"{row.record.domain}:{row.record.donor_layer}:{row.record.expert_id}"
            for row in rows
        }
        for key, rows in capped_groups.items()
    }
    assignment = _assign_sample_groups(
        capped_groups,
        labels_by_group,
        seed=seed,
        min_samples_per_cell=min_samples_per_cell,
    )
    rows: list[BridgeTrainingRow] = []
    for group_key in sorted(capped_groups):
        split = assignment[group_key]
        for source in sorted(
            capped_groups[group_key],
            key=lambda row: (row.record.token_position, row.record.event_id),
        ):
            record = source.record
            mapping = mapping_lookup[(record.donor_layer, record.expert_id)]
            rows.append(
                BridgeTrainingRow(
                    event_id=record.event_id,
                    shard_path=source.shard_path,
                    row_index=record.row_index,
                    host_state_row=source.host_state_row or 0,
                    sample_id=record.sample_id,
                    sample_content_sha256=record.sample_content_sha256,
                    domain=record.domain,
                    split=split,
                    token_position=record.token_position,
                    token_id=record.token_id,
                    donor_layer=record.donor_layer,
                    donor_expert=record.expert_id,
                    host_layer=mapping.host_layer,
                    route_rank=record.route_rank,
                    router_weight=record.router_weight,
                )
            )
    split_counts = dict(sorted(Counter(row.split for row in rows).items()))
    cell_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cell_counts[f"{row.domain}:{row.donor_layer}:{row.donor_expert}"][row.split] += 1
    body: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bridge-training-manifest",
        "handoff_manifest_sha256": _sha256_file(handoff_path),
        "capture_manifest_sha256": str(bundle["capture_manifest_sha256"]),
        "source_manifest_sha256": _source_manifest_hash_from_handoff(handoff_path, bundle),
        "candidate_manifest_sha256": str(bundle["candidate_manifest_sha256"]),
        "extraction_manifest_sha256": _extraction_manifest_hash_from_handoff(
            handoff_path, bundle
        ),
        "host_states_manifest_sha256": _sha256_file(host_states_path),
        "mappings": [item.model_dump(mode="json") for item in typed_mappings],
        "split_counts": split_counts,
        "cell_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(cell_counts.items())
        },
        "rows": [item.model_dump(mode="json") for item in rows],
        "unused_rows": [
            item.model_dump(mode="json") for item in sorted(unused, key=lambda item: item.event_id)
        ],
        "seed": seed,
        "min_samples_per_cell": min_samples_per_cell,
        "max_rows_per_sample": max_rows_per_sample,
        "min_events_per_cell": min_events_per_cell,
        "capture_outcome": str(bundle["capture_outcome"]),
    }
    body["manifest_sha256"] = _hash_without(body, "manifest_sha256")
    manifest = BridgeTrainingManifest.model_validate(body)
    _write_json_atomic(destination, manifest.model_dump(mode="json"), refuse_existing=True)
    return manifest.model_dump(mode="json")


def _source_manifest_hash_from_handoff(handoff_path: Path, bundle: Mapping[str, Any]) -> str:
    """Read the capture manifest hash without trusting an unbound path."""
    root = handoff_path.parent
    for item in bundle["artifacts"]:
        if not str(item["path"]).endswith("capture-manifest.json"):
            continue
        candidate = root / str(item["path"])
        # Artifact entries record file bytes; the manifest's internal hash is
        # validated separately on load. Comparing the two different digests
        # directly can never match.
        if candidate.is_file() and _sha256_file(candidate) == str(item["sha256"]):
            manifest = load_bridge_manifest(candidate)
            if manifest.manifest_sha256 != str(bundle["capture_manifest_sha256"]):
                raise BridgeTrainingError(
                    "handoff capture manifest does not match the handoff bundle"
                )
            return str(manifest.source_manifest_sha256)
    raise BridgeTrainingError("handoff does not contain a verified capture manifest")


def _extraction_manifest_hash_from_handoff(
    handoff_path: Path, bundle: Mapping[str, Any]
) -> str:
    """Require the lossless extraction manifest to be part of the handoff."""
    root = handoff_path.parent
    extraction = bundle.get("extraction_artifact")
    if not extraction:
        raise BridgeTrainingError("bridge training requires an extraction manifest in the handoff")
    path = root / str(extraction)
    expected = next(
        (
            str(item["sha256"])
            for item in bundle["artifacts"]
            if str(item["path"]) == str(extraction)
        ),
        None,
    )
    if not path.is_file() or expected is None or _sha256_file(path) != expected:
        raise BridgeTrainingError("handoff extraction manifest is missing or hash-invalid")
    return _sha256_file(path)


def load_training_manifest(path: Path) -> BridgeTrainingManifest:
    """Load and verify a repaired manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = BridgeTrainingManifest.model_validate(payload)
    except Exception as error:
        raise BridgeTrainingError(f"invalid bridge training manifest: {path}") from error
    if _hash_without(payload, "manifest_sha256") != manifest.manifest_sha256:
        raise BridgeTrainingError("bridge training manifest hash mismatch")
    split_contents = {
        split: {row.sample_content_sha256 for row in manifest.rows if row.split == split}
        for split in ("train", "validation", "test")
    }
    if split_contents["train"] & split_contents["validation"]:
        raise BridgeTrainingError("sample content leaks between train and validation")
    if split_contents["train"] & split_contents["test"]:
        raise BridgeTrainingError("sample content leaks between train and test")
    if split_contents["validation"] & split_contents["test"]:
        raise BridgeTrainingError("sample content leaks between validation and test")
    split_ids = {
        split: {row.sample_id for row in manifest.rows if row.split == split}
        for split in ("train", "validation", "test")
    }
    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise BridgeTrainingError("sample IDs leak between training splits")
    return manifest


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised on CPU-only installs
        raise BridgeTrainingError(
            "bridge model/training requires the optional 'gpu' dependencies (torch)"
        ) from error
    return torch


def assert_frozen_module(module: Any) -> Any:
    """Put a host or expert module in eval mode with gradients disabled."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _module_key(layer: int, expert: int) -> str:
    return f"l{layer}_e{expert}"


def _load_extracted_experts(
    extraction_dir: Path, mappings: Sequence[BridgeExpertMapping]
) -> dict[str, Any]:
    _require_torch()
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover
        raise BridgeTrainingError("safetensors.torch is required for bridge experts") from error
    manifest_path = extraction_dir / "extraction-manifest.json"
    tensor_path = extraction_dir / "experts.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise BridgeTrainingError("extraction directory lacks manifest or experts.safetensors")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        f"layers.{item.donor_layer}.experts.{item.donor_expert}.{suffix}"
        for item in mappings
        for suffix in ("gate_up_proj", "down_proj")
    }
    actual_keys = set(load_file(str(tensor_path), device="cpu"))
    if not expected_keys.issubset(actual_keys):
        raise BridgeTrainingError("extraction is missing one or more mapped expert tensors")
    if manifest.get("artifact_hash") != _sha256_file(tensor_path):
        raise BridgeTrainingError("extraction artifact hash mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    experts: dict[str, Any] = {}
    for mapping in mappings:
        gate_up = tensors[
            f"layers.{mapping.donor_layer}.experts.{mapping.donor_expert}.gate_up_proj"
        ]
        down = tensors[
            f"layers.{mapping.donor_layer}.experts.{mapping.donor_expert}.down_proj"
        ]
        if gate_up.ndim != 2 or down.ndim != 2 or gate_up.shape[0] != 2 * down.shape[1]:
            raise BridgeTrainingError(f"invalid SwiGLU shapes for {mapping.key}")
        experts[mapping.key] = FrozenSwiGLUExpert(gate_up, down)
    return experts


try:  # Keep importing this module possible without the optional torch package.
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _functional
except ImportError:  # pragma: no cover - branch depends on environment
    _torch = None
    _nn = None
    _functional = None


if _nn is None:

    class FrozenSwiGLUExpert:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

    class BridgeModel:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

    class EqualParameterAdapter:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

else:

    def _adapter(hidden_size: int, bottleneck_size: int) -> Any:
        return _nn.Sequential(
            _nn.Linear(hidden_size, bottleneck_size),
            _nn.SiLU(),
            _nn.Linear(bottleneck_size, hidden_size),
        )


    class FrozenSwiGLUExpert(_nn.Module):
        """Exact extracted BF16 SwiGLU forward, with no trainable parameters."""

        def __init__(self, gate_up_proj: Any, down_proj: Any) -> None:
            super().__init__()
            self.register_buffer("gate_up_proj", gate_up_proj.detach().contiguous())
            self.register_buffer("down_proj", down_proj.detach().contiguous())
            assert_frozen_module(self)

        def forward(self, hidden: Any) -> Any:
            # The extracted weights remain BF16.  Cast only the bridge output
            # into the donor expert's native dtype; the autograd path through
            # this cast remains live for the trainable input adapter.
            native_hidden = hidden.to(dtype=self.gate_up_proj.dtype)
            gate_up = _functional.linear(native_hidden, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            output = _functional.linear(_functional.silu(gate) * up, self.down_proj)
            return output.to(dtype=hidden.dtype)

        def train(self, mode: bool = True) -> FrozenSwiGLUExpert:
            super().train(False)
            return self


    class EqualParameterAdapter(_nn.Module):
        """A donor-free residual adapter with the bridge's trainable budget.

        This is a control, not a replacement for the frozen donor expert.  It
        has the same two 2048-bottleneck-2048 adapters and input-dependent gate head as a
        ``BridgeModel`` but no extracted tensors, so any comparison can be
        attributed to the donor expert path rather than parameter count.
        """

        def __init__(
            self,
            *,
            hidden_size: int = 2048,
            bottleneck_size: int = 256,
            gate_cap: float = 0.25,
            gate_init_bias: float = -6.0,
            mapped_expert_count: int = 4,
        ) -> None:
            super().__init__()
            if hidden_size != 2048:
                raise BridgeTrainingError("v1 bridge expects hidden size 2048")
            if mapped_expert_count <= 0:
                raise BridgeTrainingError("mapped_expert_count must be positive")
            self.gate_cap = float(gate_cap)
            self.input_adapters = _nn.ModuleList(
                [_adapter(hidden_size, bottleneck_size) for _ in range(mapped_expert_count)]
            )
            self.output_adapters = _nn.ModuleList(
                [_adapter(hidden_size, bottleneck_size) for _ in range(mapped_expert_count)]
            )
            self.gate_heads = _nn.ModuleList(
                [_nn.Linear(hidden_size, 1) for _ in range(mapped_expert_count)]
            )
            for output_adapter in self.output_adapters:
                _nn.init.zeros_(output_adapter[-1].weight)
                _nn.init.zeros_(output_adapter[-1].bias)
            for gate_head in self.gate_heads:
                _nn.init.constant_(gate_head.bias, float(gate_init_bias))

        def forward(self, host_hidden: Any, expert_slot: int = 0) -> Any:
            gate = self.gate_cap * _torch.sigmoid(self.gate_heads[expert_slot](host_hidden))
            residual = self.output_adapters[expert_slot](
                self.input_adapters[expert_slot](host_hidden)
            )
            return host_hidden + gate * residual


    class BridgeModel(_nn.Module):
        """Expert-specific trainable sidecars around frozen donor experts."""

        def __init__(
            self,
            mappings: Sequence[BridgeExpertMapping | Mapping[str, Any]],
            experts: Mapping[str, Any],
            *,
            hidden_size: int = 2048,
            bottleneck_size: int = 256,
            gate_cap: float = 0.25,
            gate_init_bias: float = -6.0,
        ) -> None:
            super().__init__()
            if hidden_size != 2048:
                raise BridgeTrainingError("v1 bridge expects hidden size 2048")
            typed = [
                item
                if isinstance(item, BridgeExpertMapping)
                else BridgeExpertMapping.model_validate(item)
                for item in mappings
            ]
            self.mappings = {item.key: item for item in typed}
            missing = set(self.mappings) - set(experts)
            if missing:
                raise BridgeTrainingError(f"missing mapped frozen experts: {sorted(missing)}")
            self.hidden_size = hidden_size
            self.bottleneck_size = bottleneck_size
            self.gate_cap = float(gate_cap)
            self.input_adapters = _nn.ModuleDict(
                {
                    _module_key(*map(int, key.split(":"))): _adapter(
                        hidden_size, bottleneck_size
                    )
                    for key in sorted(self.mappings)
                }
            )
            self.output_adapters = _nn.ModuleDict(
                {
                    _module_key(*map(int, key.split(":"))): _adapter(
                        hidden_size, bottleneck_size
                    )
                    for key in sorted(self.mappings)
                }
            )
            # Each expert has an input-dependent gate.  A finite -6 bias keeps
            # the initial residual small without making the gate derivative
            # identically zero; the head weights remain independently learned.
            self.gate_heads = _nn.ModuleDict(
                {
                    _module_key(*map(int, key.split(":"))): _nn.Linear(hidden_size, 1)
                    for key in sorted(self.mappings)
                }
            )
            for gate_head in self.gate_heads.values():
                _nn.init.constant_(gate_head.bias, float(gate_init_bias))
            for output_adapter in self.output_adapters.values():
                # A zero final output gives an identity sidecar at
                # initialization while preserving adapter gradients.
                _nn.init.zeros_(output_adapter[-1].weight)
                _nn.init.zeros_(output_adapter[-1].bias)
            self.experts = _nn.ModuleDict(
                {
                    _module_key(*map(int, key.split(":"))): experts[key]
                    for key in sorted(self.mappings)
                }
            )
            for expert in self.experts.values():
                assert_frozen_module(expert)

        def train(self, mode: bool = True) -> BridgeModel:
            super().train(mode)
            for expert in self.experts.values():
                expert.eval()
            return self

        def _expert(self, key: str) -> FrozenSwiGLUExpert:
            if key not in self.mappings:
                raise BridgeTrainingError(f"unmapped donor expert: {key}")
            layer, expert = (int(value) for value in key.split(":"))
            return self.experts[_module_key(layer, expert)]

        def _input_adapter(self, key: str) -> Any:
            if key not in self.mappings:
                raise BridgeTrainingError(f"unmapped donor expert: {key}")
            layer, expert = (int(value) for value in key.split(":"))
            return self.input_adapters[_module_key(layer, expert)]

        def _output_adapter(self, key: str) -> Any:
            if key not in self.mappings:
                raise BridgeTrainingError(f"unmapped donor expert: {key}")
            layer, expert = (int(value) for value in key.split(":"))
            return self.output_adapters[_module_key(layer, expert)]

        def _gate_head(self, key: str) -> Any:
            if key not in self.mappings:
                raise BridgeTrainingError(f"unmapped donor expert: {key}")
            layer, expert = (int(value) for value in key.split(":"))
            return self.gate_heads[_module_key(layer, expert)]

        def components(self, host_hidden: Any, key: str) -> tuple[Any, Any, Any, Any, Any]:
            if host_hidden.shape[-1] != self.hidden_size:
                raise BridgeTrainingError(
                    f"host hidden state must end in {self.hidden_size}, "
                    f"got {tuple(host_hidden.shape)}"
                )
            donor_input = self._input_adapter(key)(host_hidden)
            donor_output = self._expert(key)(donor_input)
            mapped_output = self._output_adapter(key)(donor_output)
            gate = self.gate_cap * _torch.sigmoid(self._gate_head(key)(host_hidden))
            gated_output = gate * mapped_output
            return donor_input, donor_output, mapped_output, gated_output, gate

        def forward(
            self,
            host_hidden: Any,
            key: str,
            *,
            enabled: bool = True,
        ) -> Any:
            if not enabled:
                return host_hidden
            _, _, _, gated_output, _ = self.components(host_hidden, key)
            return host_hidden + gated_output

        def forward_at_host_layer(
            self,
            host_hidden: Any,
            *,
            host_layer: int,
            donor_layer: int,
            donor_expert: int,
            enabled: bool = True,
        ) -> Any:
            """Apply the sidecar only at its pre-registered host MLP layer."""
            key = f"{donor_layer}:{donor_expert}"
            mapping = self.mappings.get(key)
            if mapping is None:
                raise BridgeTrainingError(f"unmapped donor expert: {key}")
            if mapping.host_layer != host_layer:
                raise BridgeTrainingError(
                    f"host layer {host_layer} does not match mapping {mapping.host_layer}"
                )
            return self.forward(host_hidden, key, enabled=enabled)

        def forward_routes(
            self,
            host_hidden: Any,
            keys: Sequence[str],
            *,
            enabled: bool = True,
        ) -> Any:
            if host_hidden.ndim != 2 or host_hidden.shape[0] != len(keys):
                raise BridgeTrainingError("host batch and route key count differ")
            if not enabled:
                return host_hidden
            return _torch.stack(
                [
                    self.forward(host_hidden[index], key, enabled=True)
                    for index, key in enumerate(keys)
                ]
            )


def install_bridge_sidecars(
    host_model: Any,
    bridge_model: Any,
    mappings: Sequence[BridgeExpertMapping | Mapping[str, Any]],
) -> list[Any]:
    """Attach frozen-host parallel sidecars at the configured MLP modules.

    The bridge v1 map has one fixed donor expert per host MLP layer for
    inference.  Training-time route batches can still use ``forward_routes``
    when several mappings are present.  The returned hook handles must be
    removed by the caller to restore the untouched host.
    """
    torch = _require_torch()
    typed = [
        item
        if isinstance(item, BridgeExpertMapping)
        else BridgeExpertMapping.model_validate(item)
        for item in mappings
    ]
    by_host: dict[int, BridgeExpertMapping] = {}
    for mapping in typed:
        if mapping.host_layer in by_host:
            raise BridgeTrainingError(
                "runtime sidecar attachment requires one mapped donor expert per host layer"
            )
        by_host[mapping.host_layer] = mapping
    layers = _resolve_host_layers(host_model)
    if any(layer >= len(layers) for layer in by_host):
        raise BridgeTrainingError("configured host layer is absent from host model")
    assert_frozen_module(host_model)
    bridge_model.eval()
    handles: list[Any] = []

    def make_hook(mapping: BridgeExpertMapping):
        def hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            output: Any,
        ) -> Any:
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if not isinstance(hidden, torch.Tensor) or hidden.shape[-1] != 2048:
                raise BridgeTrainingError("host MLP hook received an unexpected hidden shape")
            flat = hidden.reshape(-1, hidden.shape[-1])
            residual = bridge_model(flat, mapping.key) - flat
            residual = residual.reshape_as(hidden).to(dtype=hidden.dtype)
            if isinstance(output, torch.Tensor):
                return output + residual.to(dtype=output.dtype)
            if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                return (output[0] + residual.to(dtype=output[0].dtype), *output[1:])
            raise BridgeTrainingError("host MLP hook returned an unsupported output type")

        return hook

    for host_layer, mapping in sorted(by_host.items()):
        handles.append(
            layers[host_layer].mlp.register_forward_hook(make_hook(mapping), with_kwargs=True)
        )
    return handles


@dataclass(frozen=True)
class BridgeExample:
    """One loaded row with exact host/donor token identity."""

    row: BridgeTrainingRow
    host_hidden: Any
    expert_input: Any
    replayed_expert_output: Any
    weighted_replayed_expert_output: Any


def _load_examples(
    training_manifest_path: Path,
    host_states_path: Path,
    handoff_path: Path,
    *,
    split: str | None = None,
    allow_observational_coverage_incomplete: bool = False,
) -> list[BridgeExample]:
    torch = _require_torch()
    manifest = load_training_manifest(training_manifest_path)
    host_manifest, host_tensors_path, _ = _load_host_states(host_states_path)
    if _sha256_file(handoff_path) != manifest.handoff_manifest_sha256:
        raise BridgeTrainingError("handoff hash differs from training manifest")
    if _sha256_file(host_states_path) != manifest.host_states_manifest_sha256:
        raise BridgeTrainingError("host state hash differs from training manifest")
    target_rows = {
        (record.event_id, path): record
        for record, path in _load_source_rows(
            handoff_path,
            allow_observational_coverage_incomplete=allow_observational_coverage_incomplete,
        )
    }
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover
        raise BridgeTrainingError("safetensors is required to load bridge examples") from error
    examples: list[BridgeExample] = []
    with safe_open(
        str(host_tensors_path), framework="pt", device="cpu"
    ) as host_file, _target_file_handles(handoff_path, target_rows) as target_files:
        for row in manifest.rows:
            if split is not None and row.split != split:
                continue
            source = target_rows.get((row.event_id, row.shard_path))
            if source is None:
                raise BridgeTrainingError(f"training row has no source record: {row.event_id}")
            record, shard_path = source, row.shard_path
            if (
                record.row_index != row.row_index
                or record.sample_id != row.sample_id
                or record.sample_content_sha256 != row.sample_content_sha256
                or record.domain != row.domain
                or record.token_position != row.token_position
                or record.donor_layer != row.donor_layer
                or record.expert_id != row.donor_expert
            ):
                raise BridgeTrainingError(
                    f"training row metadata differs from handoff: {row.event_id}"
                )
            target_values = target_files[shard_path]
            host_record = host_manifest.records[row.host_state_row]
            if host_record.host_layer != row.host_layer:
                raise BridgeTrainingError(f"host layer mismatch for {row.event_id}")
            _assert_token_alignment(record, host_record, context=row.event_id)
            if row.token_id != record.token_id:
                raise BridgeTrainingError(f"manifest token ID mismatch for {row.event_id}")
            hidden = host_file.get_tensor(host_record.tensor_key)[host_record.row_index]
            expert_input = target_values.get_tensor("expert_input")[record.row_index]
            replayed = target_values.get_tensor("replayed_expert_output")[record.row_index]
            weighted = target_values.get_tensor(
                "weighted_replayed_expert_output"
            )[record.row_index]
            if hidden.ndim != 1 or hidden.shape[0] != 2048:
                raise BridgeTrainingError("host hidden state has unexpected shape")
            examples.append(
                BridgeExample(
                    row=row,
                    host_hidden=hidden.to(dtype=torch.float32),
                    expert_input=expert_input.to(dtype=torch.float32),
                    replayed_expert_output=replayed.to(dtype=torch.float32),
                    weighted_replayed_expert_output=weighted.to(dtype=torch.float32),
                )
            )
    return examples


class _target_file_handles:
    """Small context manager for lazily opening all referenced target shards."""

    def __init__(
        self,
        handoff_path: Path,
        rows: Mapping[tuple[str, str], BridgeTargetRecord],
    ) -> None:
        self.handoff_path = handoff_path
        self.rows = rows
        self.handles: dict[str, Any] = {}

    def __enter__(self) -> dict[str, Any]:
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover
            raise BridgeTrainingError("safetensors is required") from error
        seen = sorted({path for _, path in self.rows})
        for relative in seen:
            metadata_path = self.handoff_path.parent / relative
            shard = json.loads((metadata_path.parent / "shard.json").read_text(encoding="utf-8"))
            tensor_path = metadata_path.parent / str(shard["tensors_file"])
            handle = safe_open(str(tensor_path), framework="pt", device="cpu")
            handle.__enter__()
            self.handles[relative] = handle
        return self.handles

    def __exit__(self, *_args: Any) -> None:
        for handle in self.handles.values():
            handle.__exit__(None, None, None)


def load_bridge_examples(
    training_manifest_path: Path,
    host_states_path: Path,
    handoff_path: Path,
    *,
    split: str | None = None,
    allow_observational_coverage_incomplete: bool = False,
) -> list[BridgeExample]:
    """Load rows and fail closed on exact sample/token alignment."""
    if split is not None and split not in {"train", "validation", "test"}:
        raise BridgeTrainingError(f"unknown bridge split: {split}")
    return _load_examples(
        training_manifest_path,
        host_states_path,
        handoff_path,
        split=split,
        allow_observational_coverage_incomplete=allow_observational_coverage_incomplete,
    )


def _losses(
    model: Any,
    example: BridgeExample,
    *,
    input_weight: float,
    output_weight: float,
    retention_weight: float,
    enabled: bool = True,
    key_override: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    torch = _require_torch()
    key = key_override or f"{example.row.donor_layer}:{example.row.donor_expert}"
    if not enabled:
        zero = example.host_hidden.new_zeros(())
        return zero, {"input": zero, "output": zero, "retention": zero}
    donor_input, donor_output, mapped_output, gated_output, gate = model.components(
        example.host_hidden, key
    )
    input_loss = torch.mean((donor_input - example.expert_input) ** 2)
    # The captured target is already router-weighted.  Supervise the mapped
    # and gated sidecar output, not the raw frozen donor output; otherwise the
    # output adapter and gate receive no useful task gradient.
    weighted_prediction = gated_output
    output_loss = torch.mean(
        (weighted_prediction - example.weighted_replayed_expert_output) ** 2
    )
    retention_loss = torch.mean(gated_output**2)
    total = (
        input_weight * input_loss
        + output_weight * output_loss
        + retention_weight * retention_loss
    )
    return total, {
        "input": input_loss,
        "output": output_loss,
        "retention": retention_loss,
        "gate_mean": torch.mean(gate),
        "mapped_output_norm": torch.mean(mapped_output**2),
        "donor_output_norm": torch.mean(donor_output**2),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with suppress(RuntimeError):
        torch.use_deterministic_algorithms(True)


def bridge_trainable_state(model: Any) -> dict[str, Any]:
    """Return adapter/gate tensors only; frozen expert tensors never serialize."""
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if not key.startswith("experts.")
    }


def save_bridge_safetensors(destination: Path, model: Any) -> str:
    """Atomically publish bridge-only tensors and return their SHA-256."""
    _require_torch()
    if destination.exists():
        raise BridgeTrainingError(f"refusing to overwrite final bridge: {destination}")
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise BridgeTrainingError("safetensors.torch is required") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            bridge_trainable_state(model),
            str(temporary),
            metadata={"kind": "bridge-only", "schema_version": "1"},
        )
        os.replace(temporary, destination)
        return _sha256_file(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_bridge_checkpoint(
    destination: Path,
    model: Any,
    optimizer: Any,
    *,
    config_sha256: str,
    training_manifest_sha256: str,
    epoch: int,
    step: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save bridge safetensors plus optimizer state and hash metadata."""
    torch = _require_torch()
    if destination.exists():
        raise BridgeTrainingError(f"refusing to overwrite checkpoint: {destination}")
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise BridgeTrainingError("safetensors.torch is required") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        bridge_path = temporary / "bridge.safetensors"
        optimizer_path = temporary / "optimizer.pt"
        save_file(bridge_trainable_state(model), str(bridge_path), metadata={"kind": "bridge-only"})
        torch.save(optimizer.state_dict(), optimizer_path)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "bridge-checkpoint",
            "config_sha256": config_sha256,
            "training_manifest_sha256": training_manifest_sha256,
            "epoch": epoch,
            "step": step,
            "metrics": dict(metrics),
            "bridge_file": bridge_path.name,
            "optimizer_file": optimizer_path.name,
            "bridge_sha256": _sha256_file(bridge_path),
            "optimizer_sha256": _sha256_file(optimizer_path),
        }
        payload["checkpoint_sha256"] = _hash_without(payload, "checkpoint_sha256")
        # Validate the metadata before publishing the directory.  This keeps
        # the on-disk checkpoint contract identical to the checked-in schema.
        BridgeCheckpoint.model_validate(payload)
        _write_json_atomic(temporary / "checkpoint.json", payload)
        os.replace(temporary, destination)
        return payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_bridge_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any | None = None,
    *,
    config_sha256: str,
    training_manifest_sha256: str,
) -> dict[str, Any]:
    """Load a checkpoint only when both input hashes match exactly."""
    torch = _require_torch()
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover
        raise BridgeTrainingError("safetensors.torch is required") from error
    payload = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    try:
        BridgeCheckpoint.model_validate(payload)
    except Exception as error:
        raise BridgeTrainingError("invalid bridge checkpoint metadata") from error
    if _hash_without(payload, "checkpoint_sha256") != payload.get("checkpoint_sha256"):
        raise BridgeTrainingError("checkpoint metadata hash mismatch")
    if payload.get("config_sha256") != config_sha256:
        raise BridgeTrainingError("checkpoint config hash differs")
    if payload.get("training_manifest_sha256") != training_manifest_sha256:
        raise BridgeTrainingError("checkpoint training manifest hash differs")
    bridge_path = path / str(payload["bridge_file"])
    optimizer_path = path / str(payload["optimizer_file"])
    if _sha256_file(bridge_path) != payload["bridge_sha256"]:
        raise BridgeTrainingError("checkpoint bridge hash mismatch")
    if _sha256_file(optimizer_path) != payload["optimizer_sha256"]:
        raise BridgeTrainingError("checkpoint optimizer hash mismatch")
    model.load_state_dict(load_file(str(bridge_path), device="cpu"), strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=False)
        )
    return payload


def train_bridge(
    config: BridgeTrainingConfig | Path,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train the bridge with validation early stopping and a held-out test."""
    torch = _require_torch()
    if isinstance(config, Path):
        config = load_bridge_training_config(config)
    config_path_hash = config.fingerprint()
    if _sha256_file(config.data.handoff_manifest) != config.data.handoff_manifest_sha256:
        raise BridgeTrainingError("configured handoff hash mismatch")
    if _sha256_file(config.data.host_states_manifest) != config.data.host_states_manifest_sha256:
        raise BridgeTrainingError("configured host state hash mismatch")
    if _sha256_file(config.data.training_manifest) != config.data.training_manifest_sha256:
        raise BridgeTrainingError("configured training manifest hash mismatch")
    training_manifest = load_training_manifest(config.data.training_manifest)
    if training_manifest.capture_manifest_sha256 != config.data.capture_manifest_sha256:
        raise BridgeTrainingError("capture hash differs between config and training manifest")
    if training_manifest.source_manifest_sha256 != config.data.source_manifest_sha256:
        raise BridgeTrainingError("source hash differs between config and training manifest")
    if training_manifest.candidate_manifest_sha256 != config.data.candidate_manifest_sha256:
        raise BridgeTrainingError("candidate hash differs between config and training manifest")
    if training_manifest.extraction_manifest_sha256 != config.donor.extraction_manifest_sha256:
        raise BridgeTrainingError("extraction hash differs between config and training manifest")
    if (
        _sha256_file(config.donor.extraction_dir / "extraction-manifest.json")
        != config.donor.extraction_manifest_sha256
    ):
        raise BridgeTrainingError("extraction manifest hash mismatch")
    _seed_everything(config.runtime.seed)
    experts = _load_extracted_experts(config.donor.extraction_dir, config.mappings)
    model = BridgeModel(
        config.mappings,
        experts,
        hidden_size=config.host.hidden_size,
        bottleneck_size=config.runtime.bottleneck_size,
        gate_cap=config.runtime.gate_cap,
        gate_init_bias=config.runtime.gate_init_bias,
    )
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.runtime.learning_rate,
        weight_decay=config.runtime.weight_decay,
    )
    start_epoch = 0
    step = 0
    best_loss = math.inf
    best_epoch = -1
    if resume_checkpoint is not None:
        checkpoint = load_bridge_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            config_sha256=config_path_hash,
            training_manifest_sha256=training_manifest.manifest_sha256,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        step = int(checkpoint["step"])
        best_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint["metrics"]["validation"]["total"])
    examples = {
        split: load_bridge_examples(
            config.data.training_manifest,
            config.data.host_states_manifest,
            config.data.handoff_manifest,
            split=split,
            allow_observational_coverage_incomplete=(
                config.data.allow_observational_coverage_incomplete
            ),
        )
        for split in ("train", "validation", "test")
    }
    if not examples["train"] or not examples["validation"] or not examples["test"]:
        raise BridgeTrainingError("train/validation/test splits must all contain rows")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, config.runtime.max_epochs):
        model.train()
        train_total = 0.0
        order = sorted(
            range(len(examples["train"])),
            key=lambda index: hashlib.sha256(
                f"{config.runtime.seed}\0{epoch}\0{examples['train'][index].row.event_id}".encode()
            ).hexdigest(),
        )
        for index in order:
            example = examples["train"][index]
            host = example.host_hidden.to(device)
            # Training occurs one row at a time by contract (B1).  Test rows
            # are never visited here.
            current = BridgeExample(
                row=example.row,
                host_hidden=host,
                expert_input=example.expert_input.to(device),
                replayed_expert_output=example.replayed_expert_output.to(device),
                weighted_replayed_expert_output=example.weighted_replayed_expert_output.to(device),
            )
            loss, _ = _losses(
                model,
                current,
                input_weight=config.losses.input_weight,
                output_weight=config.losses.output_weight,
                retention_weight=config.losses.retention_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.runtime.max_grad_norm)
            optimizer.step()
            train_total += float(loss.detach().cpu())
            step += 1
        validation = _evaluate_bridge(model, examples["validation"], config.losses, device)
        row_metrics = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_total / len(examples["train"]),
            "validation": validation,
        }
        history.append(row_metrics)
        _append_jsonl(metrics_path, row_metrics)
        improved = validation["total"] < best_loss
        if improved:
            best_loss = validation["total"]
            best_epoch = epoch
            stale = 0
            save_bridge_checkpoint(
                output_dir / f"checkpoint-{epoch:04d}",
                model,
                optimizer,
                config_sha256=config_path_hash,
                training_manifest_sha256=training_manifest.manifest_sha256,
                epoch=epoch,
                step=step,
                metrics=row_metrics,
            )
        else:
            stale += 1
            if stale > config.runtime.patience:
                break
    # Load the best bridge state, then score test exactly once.
    if best_epoch < 0:
        raise BridgeTrainingError("validation produced no checkpoint")
    load_bridge_checkpoint(
        output_dir / f"checkpoint-{best_epoch:04d}",
        model,
        config_sha256=config_path_hash,
        training_manifest_sha256=training_manifest.manifest_sha256,
    )
    test = _evaluate_bridge(model, examples["test"], config.losses, device)
    final_path = output_dir / "bridge.safetensors"
    if final_path.exists():
        raise BridgeTrainingError(f"refusing to overwrite final bridge: {final_path}")
    bridge_hash = save_bridge_safetensors(final_path, model)
    result = {
        "status": "PASS",
        "run_id": config.run_id,
        "classification": "trained-observational-bridge-unvalidated",
        "config_sha256": config_path_hash,
        "training_manifest_sha256": training_manifest.manifest_sha256,
        "best_epoch": best_epoch,
        "validation_best_loss": best_loss,
        "test": test,
        "controls": list(config.controls.conditions),
        "control_status": bridge_control_status(),
        "causal_claim": False,
        "bridge_sha256": bridge_hash,
        "bridge_file": str(final_path),
        "host_weights_saved": False,
        "history": history,
    }
    _write_json_atomic(output_dir / "training-report.json", result)
    return result


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _evaluate_bridge(
    model: Any,
    examples: Sequence[BridgeExample],
    losses: BridgeLossConfig,
    device: Any,
) -> dict[str, float]:
    torch = _require_torch()
    model.eval()
    totals = Counter()
    with torch.no_grad():
        for example in examples:
            current = BridgeExample(
                row=example.row,
                host_hidden=example.host_hidden.to(device),
                expert_input=example.expert_input.to(device),
                replayed_expert_output=example.replayed_expert_output.to(device),
                weighted_replayed_expert_output=example.weighted_replayed_expert_output.to(device),
            )
            total, parts = _losses(
                model,
                current,
                input_weight=losses.input_weight,
                output_weight=losses.output_weight,
                retention_weight=losses.retention_weight,
            )
            totals["total"] += float(total.cpu())
            for key, value in parts.items():
                totals[key] += float(value.cpu())
    return {key: value / len(examples) for key, value in sorted(totals.items())}


def bridge_control_keys(
    keys: Sequence[str], *, condition: str, seed: int, epoch: int = 0
) -> list[str] | None:
    """Return route keys for controls; ``None`` denotes disabled sidecar."""
    if condition == "disabled":
        return None
    if condition in {"trained", "untrained", "equal-parameter-adapter"}:
        return list(keys)
    if condition == "shuffled-pair":
        if len(keys) < 2:
            raise BridgeTrainingError("shuffled-pair control requires at least two route keys")
        counts = Counter(keys)
        largest = max(counts.values())
        if largest * 2 > len(keys):
            raise BridgeTrainingError(
                "shuffled-pair control cannot derange an overrepresented route key"
            )
        ordered_indices = sorted(
            range(len(keys)),
            key=lambda index: (
                keys[index],
                hashlib.sha256(f"{seed}:{epoch}:{index}".encode()).hexdigest(),
            ),
        )
        ordered_values = [keys[index] for index in ordered_indices]
        rotated = ordered_values[largest:] + ordered_values[:largest]
        result = list(keys)
        for index, value in zip(ordered_indices, rotated, strict=True):
            result[index] = value
        if any(original == shuffled for original, shuffled in zip(keys, result, strict=True)):
            raise BridgeTrainingError("shuffled-pair derangement construction failed")
        return result
    if condition == "random-expert":
        raise BridgeTrainingError(
            "random-expert control unavailable without a separately verified extraction"
        )
    raise BridgeTrainingError(f"unknown bridge control: {condition}")


def bridge_control_status() -> dict[str, dict[str, Any]]:
    """Describe control availability without pretending absent controls ran."""
    return {
        "disabled": {"implemented": True, "evaluated": False, "description": "identity sidecar"},
        "untrained": {
            "implemented": True,
            "evaluated": False,
            "description": "initial bridge parameters",
        },
        "trained": {
            "implemented": True,
            "evaluated": False,
            "description": "best validation checkpoint",
        },
        "shuffled-pair": {
            "implemented": True,
            "evaluated": False,
            "description": "deterministically deranged donor row pairing",
        },
        "equal-parameter-adapter": {
            "implemented": True,
            "evaluated": False,
            "description": "separate equal-parameter adapter baseline",
        },
        "random-expert": {
            "implemented": False,
            "evaluated": False,
            "description": "requires separately verified random extraction",
        },
    }


def estimate_bridge_memory(
    *,
    vram_bytes: int,
    host_parameter_count: int = 2_000_000_000,
    hidden_size: int = 2048,
    bottleneck_size: int = 256,
    batch_size: int = 1,
    sequence_tokens: int = 1024,
    gate_cap: float = 0.25,
    mapped_expert_count: int = 4,
) -> dict[str, Any]:
    """Estimate B1 memory without loading a model; the 92% gate is explicit."""
    if (
        vram_bytes <= 0
        or host_parameter_count <= 0
        or batch_size != 1
        or mapped_expert_count <= 0
    ):
        raise BridgeTrainingError("GPU preflight requires positive memory and B1 batch_size=1")
    adapter_params_per_expert = (
        4 * hidden_size * bottleneck_size + 2 * bottleneck_size + 3 * hidden_size + 1
    )
    adapter_params = mapped_expert_count * adapter_params_per_expert
    host_bytes = host_parameter_count * 2  # BF16 frozen host
    bridge_bytes = adapter_params * 2
    optimizer_bytes = adapter_params * 12  # FP32 Adam states + gradient/master margin
    activation_bytes = batch_size * sequence_tokens * hidden_size * 2 * 8
    estimated = host_bytes + bridge_bytes + optimizer_bytes + activation_bytes
    ceiling = int(vram_bytes * 0.92)
    return {
        "passed": estimated <= ceiling,
        "batch_size": batch_size,
        "vram_bytes": vram_bytes,
        "safety_ceiling_bytes": ceiling,
        "host_bf16_bytes": host_bytes,
        "bridge_bf16_bytes": bridge_bytes,
        "mapped_expert_count": mapped_expert_count,
        "trainable_parameters": adapter_params,
        "optimizer_and_gradient_bytes": optimizer_bytes,
        "activation_margin_bytes": activation_bytes,
        "estimated_peak_bytes": estimated,
        "estimated_peak_fraction": estimated / vram_bytes,
        "gate_cap": gate_cap,
        "requires_empirical_probe": True,
    }


def validate_bridge_training_config(path: Path) -> dict[str, Any]:
    """CLI-friendly config validation; missing hashes remain a hard failure."""
    config = load_bridge_training_config(path)
    return {
        "valid": True,
        "config_sha256": config.fingerprint(),
        "host_model_id": config.host.model_id,
        "host_revision": config.host.revision,
        "donor_revision": config.donor.revision,
        "mapping_count": len(config.mappings),
        "controls": list(config.controls.conditions),
        "launchable": "REQUIRED" not in str(config.model_dump()),
    }
