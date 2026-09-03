"""Metadata-first donor pinning and verified weight acquisition."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from reverse_reap.config import ExperimentConfig


class ModelPreflightError(RuntimeError):
    """Raised before weight download when the donor contract cannot be proven."""


EXPECTED_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "dtype": "bfloat16",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    mismatches = {}
    if payload.get("model_type") != "qwen3_5_moe":
        mismatches["model_type"] = {
            "expected": "qwen3_5_moe",
            "actual": payload.get("model_type"),
        }
    if payload.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]:
        mismatches["architectures"] = {
            "expected": ["Qwen3_5MoeForConditionalGeneration"],
            "actual": payload.get("architectures"),
        }
    text = payload.get("text_config", {})
    for key, expected in EXPECTED_TEXT_CONFIG.items():
        if text.get(key) != expected:
            mismatches[f"text_config.{key}"] = {"expected": expected, "actual": text.get(key)}
    if mismatches:
        raise ModelPreflightError(f"official donor metadata violates the v0 contract: {mismatches}")
    return {"compatible": True, "text_config": EXPECTED_TEXT_CONFIG}


def write_pinned_config(template: Path, destination: Path, revision: str) -> ExperimentConfig:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    payload["model"]["revision"] = revision
    config = ExperimentConfig.model_validate(payload)
    body = yaml.safe_dump(payload, sort_keys=False)
    if destination.exists() and destination.read_text(encoding="utf-8") != body:
        raise ModelPreflightError(f"refusing to overwrite pinned config: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(body, encoding="utf-8")
    return config


def preflight_model(
    template_config: Path,
    pinned_config: Path,
    metadata_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    template = ExperimentConfig.model_validate(
        yaml.safe_load(template_config.read_text(encoding="utf-8"))
    )
    info = HfApi().model_info(template.model.id, files_metadata=True)
    revision = str(info.sha)
    metadata_names = (
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "chat_template.jinja",
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in metadata_names:
        downloaded = Path(
            hf_hub_download(
                template.model.id,
                name,
                revision=revision,
                local_dir=metadata_dir,
            )
        )
        files[name] = {"bytes": downloaded.stat().st_size, "sha256": file_sha256(downloaded)}
    config_payload = json.loads((metadata_dir / "config.json").read_text(encoding="utf-8"))
    architecture = validate_model_config(config_payload)
    siblings = []
    total_weight_bytes = 0
    for sibling in info.siblings:
        if not sibling.rfilename.endswith(".safetensors"):
            continue
        size = int(sibling.size or getattr(sibling.lfs, "size", 0) or 0)
        sha = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
        siblings.append({"name": sibling.rfilename, "bytes": size, "sha256": sha})
        total_weight_bytes += size
    missing_hash = any(not item["sha256"] for item in siblings)
    if len(siblings) != 14 or total_weight_bytes <= 0 or missing_hash:
        raise ModelPreflightError("could not prove all 14 weight shard sizes and SHA-256 values")
    required_free_bytes = int(total_weight_bytes * 1.2)
    available_free_bytes = shutil.disk_usage(metadata_dir).free
    report = {
        "passed": available_free_bytes >= required_free_bytes,
        "model_id": template.model.id,
        "revision": revision,
        "architecture": architecture,
        "metadata_files": files,
        "weight_shards": siblings,
        "total_weight_bytes": total_weight_bytes,
        "required_free_bytes": required_free_bytes,
        "available_free_bytes": available_free_bytes,
        "pinned_config": str(pinned_config),
    }
    write_pinned_config(template_config, pinned_config, revision)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def download_verified_weights(report_path: Path, destination: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ModelPreflightError("model preflight did not pass")
    from huggingface_hub import snapshot_download

    expected = {item["name"]: item for item in report["weight_shards"]}
    snapshot_download(
        report["model_id"],
        revision=report["revision"],
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.txt",
            "*.safetensors",
        ],
    )
    verified = []
    for name, expectation in expected.items():
        path = destination / name
        if not path.exists() or path.stat().st_size != expectation["bytes"]:
            raise ModelPreflightError(f"weight shard size mismatch: {name}")
        actual = file_sha256(path)
        if actual != expectation["sha256"]:
            raise ModelPreflightError(f"weight shard SHA-256 mismatch: {name}")
        verified.append(name)
    return {"valid": True, "revision": report["revision"], "verified_weight_shards": verified}
