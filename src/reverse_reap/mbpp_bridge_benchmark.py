"""Four-condition MBPP+ benchmark using the official EvalPlus scorer."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from reverse_reap.bridge_benchmark import (
    BridgeBenchmarkBudget,
    BridgeBenchmarkError,
    BridgeGenerationTelemetry,
    _load_bridge,
    _load_host,
    _paired_report,
    _seed_runtime,
    _verify_host_files,
)
from reverse_reap.bridge_training import install_bridge_sidecars, load_bridge_training_config
from reverse_reap.config import StrictModel
from reverse_reap.datasets import canonical_json

OFFICIAL_MBPP_VERSION = "v0.2.0"
OFFICIAL_MBPP_TASKS = 378
OFFICIAL_MBPP_ARCHIVE_SHA256 = "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"
OFFICIAL_EVALPLUS_REVISION = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
CONDITIONS = (
    "base-thinking-off",
    "base-thinking-on",
    "bridge-thinking-off",
    "bridge-thinking-on",
)


class MbppRuntime(StrictModel):
    seed: int = Field(default=20260909, ge=0)
    deterministic: Literal[True] = True
    batch_size: Literal[1] = 1
    max_input_tokens: int = Field(default=2048, gt=0)
    direct_max_new_tokens: int = Field(default=512, gt=0)
    thinking_max_new_tokens: int = Field(default=2048, gt=0)
    heartbeat_seconds: Literal[300] = 300


class MbppDataset(StrictModel):
    archive: Path
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: Literal["v0.2.0"] = OFFICIAL_MBPP_VERSION
    expected_tasks: Literal[378] = OFFICIAL_MBPP_TASKS
    pilot_items: int = Field(default=50, ge=1, le=OFFICIAL_MBPP_TASKS)
    selection_seed: int = Field(default=20260909, ge=0)


class EvalPlusContract(StrictModel):
    revision: Literal["e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"] = OFFICIAL_EVALPLUS_REVISION
    parallel_workers: int = Field(default=8, ge=1, le=64)
    memory_gb: int = Field(default=16, ge=4, le=64)
    cpus: int = Field(default=8, ge=1, le=64)
    timeout_seconds_per_condition: int = Field(default=3600, ge=60, le=21600)


class MbppBridgeBenchmarkConfig(StrictModel):
    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    host_model: Path
    host_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    host_files_manifest: Path
    host_files_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bridge_config: Path
    bridge_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bridge_checkpoint: Path
    bridge_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: MbppDataset
    runtime: MbppRuntime
    evalplus: EvalPlusContract
    budget: BridgeBenchmarkBudget
    output_dir: Path
    proceed_full_regardless_of_pilot_score: Literal[True] = True

    @model_validator(mode="after")
    def immutable_upstream_artifacts(self) -> MbppBridgeBenchmarkConfig:
        if self.dataset.archive_sha256 != OFFICIAL_MBPP_ARCHIVE_SHA256:
            raise ValueError("MBPP+ archive SHA-256 differs from the approved v0.2.0 release")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


DETERMINISTIC_CUBLAS_WORKSPACE_VALUES = (":4096:8", ":16:8")


def _require_deterministic_cuda() -> None:
    """Fail closed when deterministic CUDA execution is requested but unsafe.

    The runtime enables deterministic algorithms; on CUDA >= 10.2 those raise
    at the first nondeterministic kernel unless a cuBLAS workspace is
    configured. This check runs before any model load so a misconfigured host
    fails fast instead of mid-generation. Hardware-free environments (no torch
    or no CUDA) and runs that did not request deterministic algorithms are
    unaffected. Prompts, seeds, decoding, mappings, experts, token limits and
    benchmark membership are unchanged.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    if not torch.are_deterministic_algorithms_enabled():
        return
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        not in DETERMINISTIC_CUBLAS_WORKSPACE_VALUES
    ):
        raise BridgeBenchmarkError(
            "deterministic CUDA requested but CUBLAS_WORKSPACE_CONFIG is missing "
            "or invalid; set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != body:
            raise BridgeBenchmarkError(f"refusing to overwrite MBPP artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != body:
            raise BridgeBenchmarkError(f"refusing to overwrite MBPP rows: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def load_mbpp_bridge_config(
    path: Path, *, allow_expired: bool = False
) -> MbppBridgeBenchmarkConfig:
    import yaml

    try:
        config = MbppBridgeBenchmarkConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        if not allow_expired and config.budget.deadline_utc.astimezone(UTC) <= datetime.now(UTC):
            raise ValueError("benchmark deadline has expired")
        return config
    except Exception as error:
        raise BridgeBenchmarkError(f"invalid MBPP bridge config: {path}: {error}") from error


def freeze_mbpp_tasks(config: MbppBridgeBenchmarkConfig) -> tuple[list[dict[str, Any]], dict]:
    if not config.dataset.archive.is_file():
        raise BridgeBenchmarkError("official MBPP+ archive is missing")
    actual_sha = _sha256_file(config.dataset.archive)
    if actual_sha != config.dataset.archive_sha256:
        raise BridgeBenchmarkError("official MBPP+ archive hash mismatch")
    try:
        with gzip.open(config.dataset.archive, "rt", encoding="utf-8") as handle:
            source = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeBenchmarkError("official MBPP+ archive is unreadable") from error
    required = {
        "task_id",
        "prompt",
        "entry_point",
        "canonical_solution",
        "base_input",
        "plus_input",
        "atol",
    }
    if len(source) != config.dataset.expected_tasks:
        raise BridgeBenchmarkError(
            f"MBPP+ task count differs: {len(source)} != {config.dataset.expected_tasks}"
        )
    if any(not required.issubset(row) for row in source):
        raise BridgeBenchmarkError("MBPP+ row lacks required official fields")
    ids = [str(row["task_id"]) for row in source]
    if len(ids) != len(set(ids)) or any(not value.startswith("Mbpp/") for value in ids):
        raise BridgeBenchmarkError("MBPP+ task IDs are duplicated or malformed")
    ordered = sorted(
        source,
        key=lambda row: hashlib.sha256(
            f"{config.dataset.selection_seed}:{row['task_id']}".encode()
        ).hexdigest(),
    )
    tasks = [
        {
            "task_id": str(row["task_id"]),
            "prompt": str(row["prompt"]),
            "entry_point": str(row["entry_point"]),
            "source_row_sha256": hashlib.sha256(canonical_json(row)).hexdigest(),
        }
        for row in ordered
    ]
    report = {
        "schema_version": 1,
        "kind": "official-mbppplus-task-freeze",
        "version": config.dataset.version,
        "archive_sha256": actual_sha,
        "task_count": len(tasks),
        "selection_seed": config.dataset.selection_seed,
        "pilot_items": config.dataset.pilot_items,
        "pilot_task_ids": [row["task_id"] for row in tasks[: config.dataset.pilot_items]],
        "full_task_ids": [row["task_id"] for row in tasks],
        "ordered_tasks_sha256": hashlib.sha256(canonical_json(tasks)).hexdigest(),
    }
    return tasks, report


def _render_prompt(tokenizer: Any, task_prompt: str, *, enable_thinking: bool) -> tuple[str, Any]:
    """Render one MBPP task through the native Qwen chat template.

    Only the user message is supplied; ``add_generation_prompt=True`` lets the
    template render the assistant generation header itself, including the
    thinking-mode switch. This function never provides an assistant message and
    never injects think tags, a response prefix, or a code-fence prefill. The
    generation boundary is the end of the rendered prompt, and the official
    EvalPlus sanitizer extracts the final executable code from the raw
    generation afterward. Task prompt wording is frozen by INSTRUCTION_PREFIX.
    """
    user = f"{INSTRUCTION_PREFIX}\n```\n{task_prompt.strip()}\n```\n"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered, str) or not rendered:
        raise BridgeBenchmarkError("host chat template produced an empty prompt")
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return rendered, ids


def _generate_one(
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    config: MbppBridgeBenchmarkConfig,
    *,
    condition: str,
) -> dict[str, Any]:
    import torch

    thinking = condition.endswith("thinking-on")
    prompt, input_ids = _render_prompt(tokenizer, task["prompt"], enable_thinking=thinking)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[1] > config.runtime.max_input_tokens:
        raise BridgeBenchmarkError(
            f"input exceeds frozen limit for {task['task_id']}: {input_ids.shape[1]}"
        )
    input_ids = input_ids.to(next(model.parameters()).device)
    max_new_tokens = (
        config.runtime.thinking_max_new_tokens if thinking else config.runtime.direct_max_new_tokens
    )
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.monotonic() - started
    generated_ids = output[0, input_ids.shape[1] :].detach().cpu().tolist()
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "solution": raw,
        "condition": condition,
        "bridge_enabled": condition.startswith("bridge-"),
        "thinking_enabled": thinking,
        "source_row_sha256": task["source_row_sha256"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_token_ids": input_ids[0].detach().cpu().tolist(),
        "generated_token_ids": generated_ids,
        "input_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(generated_ids),
        "hit_max_new_tokens": len(generated_ids) == max_new_tokens,
        "generation_seconds": elapsed,
        "raw_solution_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def _check_budget(config: MbppBridgeBenchmarkConfig, started: float) -> None:
    elapsed_hours = (time.monotonic() - started) / 3600
    if datetime.now(UTC) >= config.budget.deadline_utc.astimezone(UTC):
        raise BridgeBenchmarkError("MBPP benchmark deadline reached")
    if elapsed_hours >= config.budget.max_wall_hours * config.budget.usable_fraction:
        raise BridgeBenchmarkError("MBPP benchmark wall-hour ceiling reached")
    if (
        elapsed_hours * config.budget.provider_rate_usd_per_hour
        >= config.budget.max_cost_usd * config.budget.usable_fraction
    ):
        raise BridgeBenchmarkError("MBPP benchmark cost ceiling reached")


def _write_state(
    config: MbppBridgeBenchmarkConfig, status: str, stage: str, error: str | None = None
) -> None:
    _atomic_json(
        config.output_dir / "state.json",
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "task_id": "mbppplus-four-condition-bridge-benchmark",
            "status": status,
            "stage": stage,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "config_fingerprint": config.fingerprint(),
            "failure": error,
        },
        replace=True,
    )


def _write_scoring_state(
    config: MbppBridgeBenchmarkConfig,
    status: str,
    stage: str,
    *,
    artifacts: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> None:
    _atomic_json(
        config.output_dir / "official-evalplus-scoring-state.json",
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "task_id": "official-evalplus-mbppplus-score",
            "status": status,
            "stage": stage,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "config_fingerprint": config.fingerprint(),
            "artifacts": dict(artifacts or {}),
            "failure": error,
        },
        replace=True,
    )


def _heartbeat(
    config: MbppBridgeBenchmarkConfig,
    started: float,
    condition: str,
    completed: int,
    total: int,
) -> None:
    elapsed = time.monotonic() - started
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_id": config.run_id,
        "condition": condition,
        "completed_items": completed,
        "total_items": total,
        "throughput_items_per_minute": completed / max(elapsed / 60, 1e-9),
        "gpu_hours_consumed": elapsed / 3600,
        "estimated_cost_consumed": (elapsed / 3600 * config.budget.provider_rate_usd_per_hour),
    }
    try:
        import torch

        if torch.cuda.is_available():
            payload["gpu_memory_peak_bytes"] = torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    _atomic_json(config.output_dir / "heartbeat.json", payload, replace=True)


def _bridge_load_evidence(
    config: MbppBridgeBenchmarkConfig, bridge: Any, mappings: Any
) -> dict[str, Any]:
    bridge_config = load_bridge_training_config(config.bridge_config)
    extraction_dir = bridge_config.donor.extraction_dir
    manifest_path = extraction_dir / "extraction-manifest.json"
    tensors_path = extraction_dir / "experts.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != bridge_config.donor.extraction_manifest_sha256:
        raise BridgeBenchmarkError(
            "extraction manifest differs from the hash pinned by bridge training"
        )
    mapping_rows = [item.model_dump(mode="json") for item in mappings]
    expected = {
        f"layers.{item.donor_layer}.experts.{item.donor_expert}.{suffix}"
        for item in mappings
        for suffix in ("gate_up_proj", "down_proj")
    }
    actual = {
        f"layers.{item.donor_layer}.experts.{item.donor_expert}.{suffix}"
        for item in mappings
        for suffix in ("gate_up_proj", "down_proj")
        if hasattr(bridge.experts[f"l{item.donor_layer}_e{item.donor_expert}"], suffix)
    }
    if len(mapping_rows) != 4 or actual != expected:
        raise BridgeBenchmarkError("bridge did not expose exactly four complete expert mappings")
    if manifest.get("artifact_hash") != _sha256_file(tensors_path):
        raise BridgeBenchmarkError("loaded expert tensor artifact differs from extraction manifest")
    return {
        "host_revision": config.host_revision,
        "host_files_manifest_sha256": config.host_files_manifest_sha256,
        "bridge_config_sha256": _sha256_file(config.bridge_config),
        "bridge_checkpoint_sha256": _sha256_file(config.bridge_checkpoint),
        "extraction_manifest_sha256": manifest_sha,
        "experts_safetensors_sha256": _sha256_file(tensors_path),
        "selection_status": manifest.get("selection_status", "observational-candidates"),
        "mappings": mapping_rows,
        "expert_tensor_keys": sorted(expected),
        "checkpoint_loaded": True,
        "experts_loaded": True,
        "experts_frozen": not any(p.requires_grad for p in bridge.experts.parameters()),
    }


def _validate_bridge_engagement(
    snapshot: Mapping[str, Any], mappings: Any, *, condition: str
) -> dict[str, Any]:
    expected_keys = {item.key for item in mappings}
    if set(snapshot) != expected_keys:
        raise BridgeBenchmarkError(
            f"bridge engagement keys differ for {condition}: "
            f"{sorted(snapshot)} != {sorted(expected_keys)}"
        )
    normalized = dict(snapshot)
    for key, row in normalized.items():
        calls = row.get("calls")
        tokens = row.get("tokens")
        gate_mean = row.get("gate_mean")
        residual_l2_mean = row.get("residual_l2_mean")
        if not isinstance(calls, int) or calls <= 0:
            raise BridgeBenchmarkError(f"bridge sidecar {key} never executed in {condition}")
        if not isinstance(tokens, int) or tokens <= 0:
            raise BridgeBenchmarkError(f"bridge sidecar {key} saw no tokens in {condition}")
        if not isinstance(gate_mean, (int, float)) or not math.isfinite(gate_mean):
            raise BridgeBenchmarkError(f"bridge sidecar {key} gate is non-finite")
        if gate_mean <= 0:
            raise BridgeBenchmarkError(f"bridge sidecar {key} gate never opened")
        if not isinstance(residual_l2_mean, (int, float)) or not math.isfinite(residual_l2_mean):
            raise BridgeBenchmarkError(f"bridge sidecar {key} residual is non-finite")
        if residual_l2_mean <= 0:
            raise BridgeBenchmarkError(f"bridge sidecar {key} emitted no residual")
    return normalized


def validate_mbpp_generation(
    config: MbppBridgeBenchmarkConfig, *, expected_items: int | None = None
) -> dict[str, Any]:
    tasks, freeze = freeze_mbpp_tasks(config)
    expected_count = expected_items or len(tasks)
    if expected_count < 1 or expected_count > len(tasks):
        raise BridgeBenchmarkError("invalid MBPP validation prefix size")
    expected_tasks = tasks[:expected_count]
    expected_ids = [row["task_id"] for row in expected_tasks]
    files = {}
    prompt_hashes: dict[str, dict[str, str]] = {}
    for condition in CONDITIONS:
        path = config.output_dir / "conditions" / f"{condition}.jsonl"
        if not path.is_file():
            raise BridgeBenchmarkError(f"missing condition output: {condition}")
        all_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(all_rows) < expected_count:
            raise BridgeBenchmarkError(
                f"condition has too few rows: {condition}: {len(all_rows)} < {expected_count}"
            )
        if expected_items is None and len(all_rows) != expected_count:
            raise BridgeBenchmarkError(f"condition has extra rows: {condition}")
        rows = all_rows[:expected_count]
        if [row.get("task_id") for row in rows] != expected_ids:
            raise BridgeBenchmarkError(f"condition task order differs: {condition}")
        if any(row.get("condition") != condition for row in rows):
            raise BridgeBenchmarkError(f"condition identity differs: {condition}")
        if any(
            bool(row.get("thinking_enabled")) != condition.endswith("thinking-on") for row in rows
        ):
            raise BridgeBenchmarkError(f"thinking identity differs: {condition}")
        if any(bool(row.get("bridge_enabled")) != condition.startswith("bridge-") for row in rows):
            raise BridgeBenchmarkError(f"bridge identity differs: {condition}")
        if any(
            row.get("source_row_sha256") != task["source_row_sha256"]
            for row, task in zip(rows, expected_tasks, strict=True)
        ):
            raise BridgeBenchmarkError(f"source task hash differs: {condition}")
        files[condition] = {
            "prefix_sha256": hashlib.sha256(canonical_json(rows)).hexdigest(),
            "rows": len(rows),
        }
        if expected_items is None:
            files[condition]["file_sha256"] = _sha256_file(path)
        prompt_hashes[condition] = {row["task_id"]: row["prompt_sha256"] for row in rows}
    for bridge_state in ("base", "bridge"):
        off = prompt_hashes[f"{bridge_state}-thinking-off"]
        on = prompt_hashes[f"{bridge_state}-thinking-on"]
        if any(off[task_id] == on[task_id] for task_id in expected_ids):
            raise BridgeBenchmarkError("thinking toggle produced an identical rendered prompt")
    for thinking in ("off", "on"):
        if (
            prompt_hashes[f"base-thinking-{thinking}"]
            != prompt_hashes[f"bridge-thinking-{thinking}"]
        ):
            raise BridgeBenchmarkError(f"base and bridge prompts differ under thinking-{thinking}")
    return {
        "schema_version": 1,
        "kind": "mbppplus-four-condition-generation-validation",
        "passed": True,
        "expected_items": expected_count,
        "task_freeze_sha256": freeze["ordered_tasks_sha256"],
        "files": files,
    }


def run_mbpp_bridge_benchmark(config_path: Path) -> dict[str, Any]:
    config = load_mbpp_bridge_config(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        _write_state(config, "PREFLIGHTED", "freezing-tasks")
        tasks, freeze = freeze_mbpp_tasks(config)
        _atomic_json(config.output_dir / "config-snapshot.json", config.model_dump(mode="json"))
        _atomic_json(config.output_dir / "task-freeze.json", freeze)
        _write_jsonl(config.output_dir / "tasks.jsonl", tasks)
        host_evidence = _verify_host_files(config)
        _seed_runtime(config.runtime.seed)
        _require_deterministic_cuda()
        model, tokenizer = _load_host(config)
        bridge, mappings = _load_bridge(config, next(model.parameters()).device)
        load_evidence = {
            "host": host_evidence,
            "bridge": _bridge_load_evidence(config, bridge, mappings),
        }
        _atomic_json(config.output_dir / "load-verification.json", load_evidence)
        first_prompts = {}
        for enabled in (False, True):
            rendered, _ = _render_prompt(tokenizer, tasks[0]["prompt"], enable_thinking=enabled)
            first_prompts[enabled] = hashlib.sha256(rendered.encode()).hexdigest()
        if first_prompts[False] == first_prompts[True]:
            raise BridgeBenchmarkError("host tokenizer ignored the thinking-mode switch")
        _write_state(config, "RUNNING", "four-condition-generation")
        engagement: dict[str, dict[str, Any]] = {}
        tiers = (("pilot", config.dataset.pilot_items), ("full", len(tasks)))
        for tier, boundary in tiers:
            for condition in CONDITIONS:
                _check_budget(config, started)
                output = config.output_dir / "conditions" / f"{condition}.jsonl"
                rows = []
                if output.is_file():
                    rows = [
                        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                    ]
                    if [row.get("task_id") for row in rows] != [
                        task["task_id"] for task in tasks[: len(rows)]
                    ]:
                        raise BridgeBenchmarkError(f"resume prefix differs for {condition}")
                telemetry = BridgeGenerationTelemetry() if condition.startswith("bridge-") else None
                handles = (
                    install_bridge_sidecars(model, bridge, mappings, telemetry=telemetry)
                    if telemetry is not None and len(rows) < boundary
                    else []
                )
                try:
                    last_heartbeat = 0.0
                    pending = tasks[len(rows) : boundary]
                    for index, task in enumerate(pending, start=len(rows)):
                        _check_budget(config, started)
                        rows.append(
                            _generate_one(model, tokenizer, task, config, condition=condition)
                        )
                        _write_jsonl(output, rows, replace=True)
                        if time.monotonic() - last_heartbeat >= config.runtime.heartbeat_seconds:
                            _heartbeat(config, started, condition, index + 1, boundary)
                            last_heartbeat = time.monotonic()
                finally:
                    for handle in handles:
                        handle.remove()
                if telemetry is not None:
                    engagement_path = (
                        config.output_dir / "conditions" / f"{condition}-{tier}-engagement.json"
                    )
                    snapshot = telemetry.snapshot()
                    if not snapshot and len(rows) >= boundary and engagement_path.is_file():
                        snapshot = json.loads(engagement_path.read_text(encoding="utf-8"))
                    verified_engagement = _validate_bridge_engagement(
                        snapshot, mappings, condition=f"{condition}:{tier}"
                    )
                    engagement.setdefault(condition, {})[tier] = verified_engagement
                    _atomic_json(engagement_path, verified_engagement)
            tier_validation = validate_mbpp_generation(config, expected_items=boundary)
            _atomic_json(
                config.output_dir / f"{tier}-generation-report.json",
                {
                    "schema_version": 1,
                    "kind": "mbppplus-generation-tier",
                    "tier": tier,
                    "status": "PASS",
                    "run_id": config.run_id,
                    "validation": tier_validation,
                    "proceed_full_regardless_of_pilot_score": True,
                },
            )
        validation = validate_mbpp_generation(config)
        elapsed = time.monotonic() - started
        report = {
            "schema_version": 1,
            "kind": "mbppplus-four-condition-generation",
            "status": "PASS",
            "classification": "generation-complete-official-scoring-deferred",
            "run_id": config.run_id,
            "config_fingerprint": config.fingerprint(),
            "conditions": list(CONDITIONS),
            "validation": validation,
            "bridge_engagement": engagement,
            "elapsed_seconds": elapsed,
            "estimated_compute_cost_usd": (
                elapsed / 3600 * config.budget.provider_rate_usd_per_hour
            ),
            "scientific_claim": "paired capability benchmark; not causal proof",
        }
        _atomic_json(config.output_dir / "generation-report.json", report)
        _write_state(config, "COMPLETE", "generation-complete")
        return report
    except Exception as error:
        _write_state(config, "FAILED_TERMINAL", "generation", f"{type(error).__name__}: {error}")
        raise


def _docker_prefix(config: MbppBridgeBenchmarkConfig, image: str, *, work: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=512",
        f"--memory={config.evalplus.memory_gb}g",
        f"--cpus={config.evalplus.cpus}",
        "--tmpfs=/tmp:rw,exec,nosuid,size=2g",
        "--mount",
        f"type=bind,src={work},dst=/work",
        "--mount",
        f"type=bind,src={config.dataset.archive.resolve()},dst=/dataset/MbppPlus.jsonl.gz,readonly",
        "--env",
        "MBPP_OVERRIDE_PATH=/dataset/MbppPlus.jsonl.gz",
        "--env",
        "HOME=/tmp",
        image,
    ]


def _verify_evalplus_image(image: str, revision: str) -> None:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise BridgeBenchmarkError("EvalPlus image must be pinned by repository digest")
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            '{{index .Config.Labels "org.opencontainers.image.revision"}}',
            image,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != revision:
        raise BridgeBenchmarkError("EvalPlus image revision label differs from pinned config")


def _official_result_rows(path: Path, expected_ids: set[str]) -> dict[str, dict[str, bool]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluated = payload.get("eval")
    if not isinstance(evaluated, dict) or set(evaluated) != expected_ids:
        raise BridgeBenchmarkError("official EvalPlus result task universe differs")
    rows = {}
    for task_id, values in evaluated.items():
        if not isinstance(values, list) or len(values) != 1:
            raise BridgeBenchmarkError("official EvalPlus result is not pass@1")
        row = values[0]
        base = row.get("base_status") == "pass"
        plus_status = row.get("plus_status") == "pass"
        rows[task_id] = {"base": base, "plus": base and plus_status}
    return rows


def _metric_pair(
    base: dict[str, dict[str, bool]],
    bridge: dict[str, dict[str, bool]],
    task_ids: list[str],
    metric: Literal["base", "plus"],
) -> dict[str, Any]:
    return _paired_report(
        [{"sample_id": task_id, "passed": base[task_id][metric]} for task_id in task_ids],
        [{"sample_id": task_id, "passed": bridge[task_id][metric]} for task_id in task_ids],
    )


def _score_mbpp_bridge_benchmark(config_path: Path, *, evalplus_image: str) -> dict[str, Any]:
    config = load_mbpp_bridge_config(config_path, allow_expired=True)
    validation = validate_mbpp_generation(config)
    _verify_evalplus_image(evalplus_image, config.evalplus.revision)
    tasks, freeze = freeze_mbpp_tasks(config)
    expected = {task["task_id"] for task in tasks}
    scores_dir = config.output_dir / "official-evalplus"
    scores_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    logs = {}
    for condition in CONDITIONS:
        source = config.output_dir / "conditions" / f"{condition}.jsonl"
        relative_source = source.resolve().relative_to(config.output_dir.resolve())
        sanitized = scores_dir / f"{condition}-sanitized.jsonl"
        result_path = scores_dir / f"{condition}.eval_results.json"
        prefix = _docker_prefix(config, evalplus_image, work=config.output_dir.resolve())
        if not sanitized.is_file():
            command = prefix + [
                "python",
                "-m",
                "evalplus.sanitize",
                "--samples",
                f"/work/{relative_source}",
            ]
            run = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=config.evalplus.timeout_seconds_per_condition,
            )
            logs[f"{condition}-sanitize"] = {
                "return_code": run.returncode,
                "stdout": run.stdout[-8192:],
                "stderr": run.stderr[-8192:],
            }
            produced = source.with_name(source.stem + "-sanitized.jsonl")
            if run.returncode != 0 or not produced.is_file():
                raise BridgeBenchmarkError(f"official EvalPlus sanitize failed: {condition}")
            os.replace(produced, sanitized)
        if not result_path.is_file():
            command = prefix + [
                "python",
                "-m",
                "evalplus.evaluate",
                "--dataset",
                "mbpp",
                "--samples",
                f"/work/{sanitized.resolve().relative_to(config.output_dir.resolve())}",
                "--parallel",
                str(config.evalplus.parallel_workers),
                "--version",
                config.dataset.version,
                "--output-file",
                f"/work/{result_path.resolve().relative_to(config.output_dir.resolve())}",
            ]
            run = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=config.evalplus.timeout_seconds_per_condition,
            )
            logs[f"{condition}-evaluate"] = {
                "return_code": run.returncode,
                "stdout": run.stdout[-8192:],
                "stderr": run.stderr[-8192:],
            }
            if run.returncode != 0 or not result_path.is_file():
                raise BridgeBenchmarkError(f"official EvalPlus evaluate failed: {condition}")
        all_rows[condition] = _official_result_rows(result_path, expected)
    pilot_ids = freeze["pilot_task_ids"]
    full_ids = freeze["full_task_ids"]
    comparisons = {}
    for thinking in ("off", "on"):
        base = all_rows[f"base-thinking-{thinking}"]
        bridge = all_rows[f"bridge-thinking-{thinking}"]
        comparisons[f"thinking-{thinking}"] = {
            tier: {metric: _metric_pair(base, bridge, ids, metric) for metric in ("base", "plus")}
            for tier, ids in (("pilot", pilot_ids), ("full", full_ids))
        }
    official_artifacts = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(scores_dir.iterdir())
        if path.is_file()
        and (path.name.endswith("-sanitized.jsonl") or path.name.endswith(".eval_results.json"))
    }
    report = {
        "schema_version": 1,
        "kind": "official-evalplus-mbppplus-four-condition-score",
        "status": "PASS",
        "run_id": config.run_id,
        "evalplus_revision": config.evalplus.revision,
        "evalplus_image": evalplus_image,
        "dataset_version": config.dataset.version,
        "dataset_archive_sha256": config.dataset.archive_sha256,
        "generation_validation": validation,
        "official_artifacts": official_artifacts,
        "comparisons": comparisons,
        "logs": logs,
        "limitations": [
            "observational bridge comparison; not causal proof",
            "thinking-enabled and thinking-disabled results are separate experiments",
        ],
    }
    _atomic_json(config.output_dir / "official-evalplus-report.json", report)
    return report


def score_mbpp_bridge_benchmark(config_path: Path, *, evalplus_image: str) -> dict[str, Any]:
    config = load_mbpp_bridge_config(config_path, allow_expired=True)
    _write_scoring_state(config, "RUNNING", "official-evalplus")
    try:
        report = _score_mbpp_bridge_benchmark(config_path, evalplus_image=evalplus_image)
        _write_scoring_state(
            config,
            "COMPLETE",
            "official-evalplus-complete",
            artifacts=report["official_artifacts"],
        )
        return report
    except Exception as error:
        _write_scoring_state(
            config,
            "FAILED_TERMINAL",
            "official-evalplus",
            error=f"{type(error).__name__}: {error}",
        )
        raise
