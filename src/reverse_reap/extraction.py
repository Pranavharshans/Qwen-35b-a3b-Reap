"""Lossless fused-expert tensor extraction and independent byte verification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from reverse_reap.qwen35 import Qwen35Architecture


class ExtractionError(RuntimeError):
    """Raised when source or extracted tensor evidence is incomplete."""


@dataclass(frozen=True)
class ExtractedTensor:
    source_shard: str
    source_key: str
    output_key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    content_sha256: str


def architecture_from_weight_index(model_dir: Path) -> Qwen35Architecture:
    """Infer the exact fused key prefix while enforcing the approved donor dimensions."""
    weight_map = load_weight_map(model_dir)
    pattern = re.compile(r"^(.*\.layers)\.(\d+)\.mlp\.experts\.gate_up_proj$")
    matches = [match for key in weight_map if (match := pattern.match(key))]
    if not matches:
        raise ExtractionError("could not locate fused expert tensors in weight index")
    prefixes = {match.group(1) for match in matches}
    layers = {int(match.group(2)) for match in matches}
    if len(prefixes) != 1 or layers != set(range(40)):
        raise ExtractionError(
            f"expected one 40-layer fused expert prefix, got prefixes={prefixes}, layers={layers}"
        )
    return Qwen35Architecture(
        layers=tuple(None for _ in range(40)),
        num_experts=256,
        experts_per_token=8,
        hidden_size=2048,
        expert_intermediate_size=512,
        state_prefix=prefixes.pop(),
    )


def tensor_bytes(array: Any) -> bytes:
    if hasattr(array, "detach"):
        import torch

        return array.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return np.ascontiguousarray(array).view(np.uint8).tobytes()


def tensor_sha256(array: Any) -> str:
    return hashlib.sha256(tensor_bytes(array)).hexdigest()


def load_weight_map(model_dir: Path) -> dict[str, str]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ExtractionError(f"expected one safetensors index, found {len(indexes)}")
    payload = json.loads(indexes[0].read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ExtractionError("safetensors index has no weight_map")
    return {str(key): str(value) for key, value in weight_map.items()}


def weight_index_path(model_dir: Path) -> Path:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ExtractionError(f"expected one safetensors index, found {len(indexes)}")
    return indexes[0]


def _read_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> Any:
    shard = weight_map.get(key)
    if shard is None:
        raise ExtractionError(f"source tensor is absent from weight map: {key}")
    try:
        import torch  # noqa: F401
    except ImportError:
        framework = "numpy"
    else:
        framework = "pt"
    with safe_open(model_dir / shard, framework=framework, device="cpu") as handle:
        return handle.get_tensor(key)


def _contiguous(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().contiguous()
    return np.ascontiguousarray(value)


def _save_tensors(tensors: dict[str, Any], path: Path, metadata: dict[str, str]) -> None:
    first = next(iter(tensors.values()))
    if hasattr(first, "detach"):
        from safetensors.torch import save_file
    else:
        from safetensors.numpy import save_file

    save_file(tensors, path, metadata=metadata)


def extract_experts(
    model_dir: Path,
    architecture: Qwen35Architecture,
    selected: list[tuple[int, int]],
    destination: Path,
    *,
    model_id: str,
    model_revision: str,
    run_id: str = "unresolved",
    selection_status: str = "observational-candidates",
    selection_metrics: dict[str, Any] | None = None,
    causal_metrics: dict[str, Any] | None = None,
    tool_git_revision: str | None = None,
) -> dict[str, Any]:
    """Extract each selected expert slice and verify logical tensor bytes after reload."""
    if destination.exists():
        raise ExtractionError(f"refusing to overwrite extraction destination: {destination}")
    weight_map = load_weight_map(model_dir)
    tensors: dict[str, Any] = {}
    records: list[ExtractedTensor] = []
    for layer, expert in sorted(set(selected)):
        spec = architecture.tensor_spec(layer, expert)
        source_pairs = ((spec.gate_up_key, "gate_up_proj"), (spec.down_key, "down_proj"))
        for source_key, suffix in source_pairs:
            fused = _read_tensor(model_dir, weight_map, source_key)
            if fused.shape[0] != architecture.num_experts:
                raise ExtractionError(f"unexpected expert axis for {source_key}: {fused.shape}")
            output_key = f"layers.{layer}.experts.{expert}.{suffix}"
            value = _contiguous(fused[expert])
            tensors[output_key] = value
            records.append(
                ExtractedTensor(
                    source_shard=weight_map[source_key],
                    source_key=source_key,
                    output_key=output_key,
                    shape=tuple(value.shape),
                    dtype=str(value.dtype).removeprefix("torch."),
                    nbytes=len(tensor_bytes(value)),
                    content_sha256=tensor_sha256(value),
                )
            )
    destination.mkdir(parents=True)
    tensor_path = destination / "experts.safetensors"
    _save_tensors(
        tensors, tensor_path, metadata={"model_id": model_id, "revision": model_revision}
    )

    verified = []
    framework = "pt" if hasattr(next(iter(tensors.values())), "detach") else "numpy"
    with safe_open(tensor_path, framework=framework, device="cpu") as extracted:
        actual_keys = set(extracted.keys())
        if actual_keys != set(tensors):
            raise ExtractionError("extracted tensor key set differs after independent reload")
        for record in records:
            reloaded = extracted.get_tensor(record.output_key)
            dtype = str(reloaded.dtype).removeprefix("torch.")
            if tuple(reloaded.shape) != record.shape or dtype != record.dtype:
                raise ExtractionError(f"shape or dtype changed for {record.output_key}")
            if tensor_sha256(reloaded) != record.content_sha256:
                raise ExtractionError(f"byte verification failed for {record.output_key}")
            verified.append({**record.__dict__, "verified": True})
    manifest = {
        "schema_version": 1,
        "label": "extracted",
        "run_id": run_id,
        "source_model_id": model_id,
        "source_revision": model_revision,
        "source_weight_index_hash": hashlib.sha256(
            weight_index_path(model_dir).read_bytes()
        ).hexdigest(),
        "selection_status": selection_status,
        "selection_metrics": selection_metrics or {},
        "causal_metrics": causal_metrics or {},
        "tool_git_revisions": {"reverse_reap": tool_git_revision or "unknown"},
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experts": [{"layer": layer, "expert": expert} for layer, expert in sorted(set(selected))],
        "tensors": verified,
        "tensor_file": tensor_path.name,
        "artifact_hash": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "total_parameter_bytes": sum(record.nbytes for record in records),
    }
    manifest_path = destination / "extraction-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_map = {
        record.output_key: {
            "source_shard": record.source_shard,
            "source_key": record.source_key,
            "source_expert_axis_index": int(record.output_key.split(".")[3]),
        }
        for record in records
    }
    (destination / "source-to-extracted-map.json").write_text(
        json.dumps(source_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "checksums.sha256").write_text(
        f"{manifest['artifact_hash']}  {tensor_path.name}\n", encoding="utf-8"
    )
    verification = {
        "valid": True,
        "tensor_count": len(records),
        "all_tensor_bytes_verified": all(item["verified"] for item in verified),
    }
    (destination / "verification-report.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "# Extracted routed experts\n\n"
        "These tensors are not a standalone model. They depend on the donor representation, "
        "original layer context, router, shared expert path, attention stack, and residual "
        "stream.\n",
        encoding="utf-8",
    )
    return manifest


def verify_extraction(destination: Path, model_dir: Path) -> dict[str, Any]:
    """Reload both artifacts independently and re-prove every manifest assertion."""
    manifest = json.loads((destination / "extraction-manifest.json").read_text(encoding="utf-8"))
    tensor_path = destination / manifest["tensor_file"]
    if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != manifest["artifact_hash"]:
        raise ExtractionError("extracted safetensors file hash mismatch")
    expected_keys = {
        f"layers.{item['layer']}.experts.{item['expert']}.{suffix}"
        for item in manifest["experts"]
        for suffix in ("gate_up_proj", "down_proj")
    }
    manifest_keys = {item["output_key"] for item in manifest["tensors"]}
    if manifest_keys != expected_keys:
        raise ExtractionError("manifest does not account for every expected expert tensor")
    weight_map = load_weight_map(model_dir)
    try:
        import torch  # noqa: F401
    except ImportError:
        framework = "numpy"
    else:
        framework = "pt"
    with safe_open(tensor_path, framework=framework, device="cpu") as extracted:
        for record in manifest["tensors"]:
            source = _read_tensor(model_dir, weight_map, record["source_key"])
            expert = int(record["output_key"].split(".")[3])
            source_slice = _contiguous(source[expert])
            output = extracted.get_tensor(record["output_key"])
            if tensor_bytes(source_slice) != tensor_bytes(output):
                raise ExtractionError(f"source bytes differ for {record['output_key']}")
    source_index_valid = (
        hashlib.sha256(weight_index_path(model_dir).read_bytes()).hexdigest()
        == manifest["source_weight_index_hash"]
    )
    if not source_index_valid:
        raise ExtractionError("source weight index hash mismatch")
    return {
        "valid": True,
        "tensor_count": len(manifest["tensors"]),
        "source_weight_index_hash_valid": source_index_valid,
    }
