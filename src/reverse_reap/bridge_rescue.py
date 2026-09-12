"""Governed runtime policies for the four-expert bridge rescue experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import Field, model_validator

from reverse_reap.config import StrictModel
from reverse_reap.datasets import canonical_json


class BridgeRescueError(RuntimeError):
    """Raised when a rescue policy or controller violates its frozen contract."""


class BridgeRuntimePolicy(StrictModel):
    """Hash-bound policy applied to an already-trained bridge residual."""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    mode: Literal["fixed", "late", "ramp", "learned"]
    residual_multiplier: float = Field(ge=0.0, le=1.0)
    active_experts: tuple[str, ...] = ()
    active_host_layers: tuple[int, ...] = ()
    thinking_only: bool = True
    start_token: int = Field(default=0, ge=0)
    full_strength_token: int = Field(default=0, ge=0)
    controller_checkpoint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def coherent(self) -> BridgeRuntimePolicy:
        if len(set(self.active_experts)) != len(self.active_experts):
            raise ValueError("active_experts contains duplicates")
        if len(set(self.active_host_layers)) != len(self.active_host_layers):
            raise ValueError("active_host_layers contains duplicates")
        if self.mode == "ramp" and self.full_strength_token <= self.start_token:
            raise ValueError("ramp full_strength_token must be greater than start_token")
        if self.mode != "ramp" and self.full_strength_token not in (0, self.start_token):
            raise ValueError("full_strength_token is only meaningful for ramp mode")
        if self.mode == "learned" and self.controller_checkpoint_sha256 is None:
            raise ValueError("learned policy requires controller_checkpoint_sha256")
        if self.mode != "learned" and self.controller_checkpoint_sha256 is not None:
            raise ValueError("controller checkpoint is only valid for learned mode")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


class RescueExperimentConfig(StrictModel):
    """One immutable experiment identity; experiments cannot be pooled silently."""

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    experiment: Literal["strength", "late", "expert-ablation", "layer-timing", "learned"]
    benchmark_config: Path
    benchmark_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest: Path
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["screening", "selection", "confirmation"]
    expected_tasks: int
    thinking_enabled: Literal[True] = True
    policies: tuple[BridgeRuntimePolicy, ...]
    output_dir: Path

    @model_validator(mode="after")
    def governed_size(self) -> RescueExperimentConfig:
        expected = {"screening": 12, "selection": 24, "confirmation": 100}[self.split]
        if self.expected_tasks != expected:
            raise ValueError(f"{self.split} split requires exactly {expected} tasks")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        if not names or names[0] != "base":
            raise ValueError("first policy must be the unmodified base control")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


@dataclass
class BridgePolicyState:
    """Per-generation mutable state; never shared across samples or workers."""

    generated_tokens: int = 0
    thinking_open: bool = True
    repetition_rate: float = 0.0
    gate_mean: float = 0.0
    residual_ratio: float = 0.0

    def advance(
        self,
        *,
        generated_tokens: int,
        thinking_open: bool,
        repetition_rate: float = 0.0,
        gate_mean: float = 0.0,
        residual_ratio: float = 0.0,
    ) -> None:
        if generated_tokens < self.generated_tokens:
            raise BridgeRescueError("generated token position moved backwards")
        values = (repetition_rate, gate_mean, residual_ratio)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise BridgeRescueError("policy state contains invalid feature values")
        self.generated_tokens = generated_tokens
        self.thinking_open = thinking_open
        self.repetition_rate = repetition_rate
        self.gate_mean = gate_mean
        self.residual_ratio = residual_ratio


class LinearGateCheckpoint(StrictModel):
    """Small deterministic controller; JSON only and never contains host weights."""

    schema_version: Literal[1] = 1
    feature_names: tuple[str, ...] = (
        "bias",
        "token_fraction",
        "repetition_rate",
        "gate_mean",
        "residual_ratio",
    )
    weights: tuple[float, ...]
    max_generated_tokens: int = Field(gt=0)
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def dimensions(self) -> LinearGateCheckpoint:
        if len(self.weights) != len(self.feature_names):
            raise ValueError("controller weight count differs from feature count")
        if any(not math.isfinite(value) for value in self.weights):
            raise ValueError("controller weights must be finite")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


def load_linear_gate(path: Path, expected_sha256: str) -> LinearGateCheckpoint:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise BridgeRescueError(f"controller checkpoint hash differs: {actual}")
    return LinearGateCheckpoint.model_validate_json(raw)


def save_linear_gate(checkpoint: LinearGateCheckpoint, destination: Path) -> str:
    body = json.dumps(checkpoint.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != body:
        raise BridgeRescueError(f"refusing to overwrite controller checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode()).hexdigest()


def fit_linear_gate(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    training_manifest_sha256: str,
    max_generated_tokens: int,
    steps: int = 500,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
) -> LinearGateCheckpoint:
    """Fit deterministic logistic regression for the Experiment 5 controller."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != 4:
        raise BridgeRescueError("controller training expects N x 4 features and N labels")
    if x.shape[0] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise BridgeRescueError("controller training data is empty or non-finite")
    if not set(y.tolist()).issubset({0.0, 1.0}):
        raise BridgeRescueError("controller labels must be binary")
    design = np.column_stack([np.ones(x.shape[0]), x])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - y) / y.size
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return LinearGateCheckpoint(
        weights=tuple(float(value) for value in weights),
        max_generated_tokens=max_generated_tokens,
        training_manifest_sha256=training_manifest_sha256,
    )


class BridgePolicyController:
    """Callable residual policy consumed by ``install_bridge_sidecars``."""

    def __init__(
        self,
        policy: BridgeRuntimePolicy,
        state: BridgePolicyState,
        *,
        learned: LinearGateCheckpoint | None = None,
    ) -> None:
        if policy.mode == "learned" and learned is None:
            raise BridgeRescueError("learned runtime policy has no loaded controller")
        self.policy = policy
        self.state = state
        self.learned = learned

    def _base_multiplier(self) -> float:
        position = self.state.generated_tokens
        if self.policy.mode == "fixed":
            return self.policy.residual_multiplier
        if self.policy.mode == "late":
            return self.policy.residual_multiplier if position >= self.policy.start_token else 0.0
        if self.policy.mode == "ramp":
            if position <= self.policy.start_token:
                return 0.0
            width = self.policy.full_strength_token - self.policy.start_token
            fraction = min(1.0, (position - self.policy.start_token) / width)
            return self.policy.residual_multiplier * fraction
        assert self.learned is not None
        vector = np.asarray(
            [
                1.0,
                position / self.learned.max_generated_tokens,
                self.state.repetition_rate,
                self.state.gate_mean,
                self.state.residual_ratio,
            ],
            dtype=np.float64,
        )
        score = float(np.dot(vector, np.asarray(self.learned.weights)))
        return self.policy.residual_multiplier / (1.0 + math.exp(-max(-30.0, min(30.0, score))))

    def __call__(self, mapping: Any, _hidden: Any, _residual: Any) -> float:
        if self.policy.name == "base":
            return 0.0
        if self.policy.active_experts and mapping.key not in self.policy.active_experts:
            return 0.0
        if (
            self.policy.active_host_layers
            and mapping.host_layer not in self.policy.active_host_layers
        ):
            return 0.0
        if self.policy.thinking_only and not self.state.thinking_open:
            return 0.0
        return self._base_multiplier()


class GenerationPolicyTracker:
    """Transformers-compatible logits processor that advances one sample's policy state."""

    def __init__(
        self, tokenizer: Any, state: BridgePolicyState, *, prompt_tokens: int | None = None
    ) -> None:
        if prompt_tokens is not None and prompt_tokens < 1:
            raise BridgeRescueError("prompt_tokens must be positive")
        self.tokenizer = tokenizer
        self.state = state
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
            raise BridgeRescueError("policy tracker requires deterministic batch size one")
        if self.prompt_tokens is None:
            self.prompt_tokens = int(input_ids.shape[1])
        generated = input_ids[0, self.prompt_tokens :].detach().cpu().tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        tail = generated[-64:]
        repeated = 0.0 if not tail else 1.0 - (len(set(tail)) / len(tail))
        self.state.advance(
            generated_tokens=len(generated),
            thinking_open="</think>" not in text,
            repetition_rate=repeated,
            gate_mean=self.state.gate_mean,
            residual_ratio=self.state.residual_ratio,
        )
        return scores


def policy_audit_record(
    policy: BridgeRuntimePolicy, state: BridgePolicyState, applied_multiplier: float
) -> Mapping[str, Any]:
    return {
        "policy": policy.name,
        "policy_fingerprint": policy.fingerprint(),
        "generated_tokens": state.generated_tokens,
        "thinking_open": state.thinking_open,
        "applied_multiplier": applied_multiplier,
    }


def load_rescue_experiment_config(path: Path) -> RescueExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise BridgeRescueError("rescue experiment config must be a mapping")
    return RescueExperimentConfig.model_validate(payload)


def fit_linear_gate_manifest(
    manifest: Path,
    destination: Path,
    *,
    expected_sha256: str,
    max_generated_tokens: int,
) -> Mapping[str, Any]:
    raw = manifest.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise BridgeRescueError(f"training manifest hash differs: {actual}")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    required = ("token_fraction", "repetition_rate", "gate_mean", "residual_ratio")
    features = [[float(row[key]) for key in required] for row in rows]
    labels = [int(row["bridge_helpful"]) for row in rows]
    checkpoint = fit_linear_gate(
        features,
        labels,
        training_manifest_sha256=actual,
        max_generated_tokens=max_generated_tokens,
    )
    checkpoint_sha256 = save_linear_gate(checkpoint, destination)
    return {
        "status": "PASS",
        "training_rows": len(rows),
        "training_manifest_sha256": actual,
        "checkpoint": str(destination),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze_mbpp_strength_screen(benchmark_config: Path, destination: Path) -> Mapping[str, Any]:
    """Freeze the first 12 hash-ordered MBPP+ tasks for post-hoc screening only."""
    from reverse_reap.mbpp_bridge_benchmark import (
        freeze_mbpp_tasks,
        load_mbpp_bridge_config,
    )

    benchmark = load_mbpp_bridge_config(benchmark_config, allow_expired=True)
    tasks, freeze = freeze_mbpp_tasks(benchmark)
    selected = tasks[:12]
    _write_jsonl_atomic(destination, selected)
    return {
        "status": "PASS",
        "classification": "post-hoc-exploratory-screen-not-confirmation",
        "tasks": len(selected),
        "task_ids": [row["task_id"] for row in selected],
        "manifest": str(destination),
        "manifest_sha256": _sha256_file(destination),
        "source_order_sha256": freeze["ordered_tasks_sha256"],
    }


def run_strength_screen(config_path: Path) -> Mapping[str, Any]:
    """Run Experiment 1: base plus five thinking-enabled fixed bridge strengths."""
    from reverse_reap.bridge_benchmark import (
        BridgeGenerationTelemetry,
        _load_bridge,
        _load_host,
        _seed_runtime,
        _verify_host_files,
    )
    from reverse_reap.bridge_training import install_bridge_sidecars
    from reverse_reap.mbpp_bridge_benchmark import (
        _check_budget,
        _generate_one,
        _require_deterministic_cuda,
        load_mbpp_bridge_config,
    )

    config = load_rescue_experiment_config(config_path)
    if config.experiment != "strength" or config.split != "screening":
        raise BridgeRescueError("strength screen requires experiment=strength and split=screening")
    expected = {
        "base": 0.0,
        "current": 1.0,
        "strength-0.025": 0.1,
        "strength-0.05": 0.2,
        "strength-0.10": 0.4,
        "strength-0.15": 0.6,
    }
    expected_names = tuple(expected)
    if tuple(policy.name for policy in config.policies) != expected_names:
        raise BridgeRescueError(f"strength policies differ from frozen order: {expected_names}")
    if any(
        policy.mode != "fixed"
        or policy.residual_multiplier != expected[policy.name]
        or not policy.thinking_only
        or policy.active_experts
        or policy.active_host_layers
        for policy in config.policies
    ):
        raise BridgeRescueError("strength screen policy definitions differ from frozen design")
    if _sha256_file(config.benchmark_config) != config.benchmark_config_sha256:
        raise BridgeRescueError("benchmark config hash mismatch")
    if _sha256_file(config.dataset_manifest) != config.dataset_manifest_sha256:
        raise BridgeRescueError("screening manifest hash mismatch")
    tasks = [
        json.loads(line)
        for line in config.dataset_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(tasks) != 12 or len({row.get("task_id") for row in tasks}) != 12:
        raise BridgeRescueError("screening manifest must contain 12 unique tasks")
    benchmark = load_mbpp_bridge_config(config.benchmark_config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _seed_runtime(benchmark.runtime.seed)
    _require_deterministic_cuda()
    host_evidence = _verify_host_files(benchmark)
    model, tokenizer = _load_host(benchmark)
    bridge, mappings = _load_bridge(benchmark, next(model.parameters()).device)
    results: dict[str, Any] = {}
    for policy in config.policies:
        _check_budget(benchmark, started)
        condition = (
            "base-thinking-on"
            if policy.name == "base"
            else f"bridge-{policy.name}-thinking-on"
        )
        destination = config.output_dir / "conditions" / f"{condition}.jsonl"
        rows = []
        if destination.is_file():
            rows = [json.loads(line) for line in destination.read_text().splitlines() if line]
            if [row.get("task_id") for row in rows] != [
                row["task_id"] for row in tasks[: len(rows)]
            ]:
                raise BridgeRescueError(f"resume prefix differs for {condition}")
        telemetry = BridgeGenerationTelemetry() if policy.name != "base" else None
        for task in tasks[len(rows) :]:
            _check_budget(benchmark, started)
            state = BridgePolicyState()
            controller = BridgePolicyController(policy, state)
            tracker = GenerationPolicyTracker(tokenizer, state)
            handles = (
                install_bridge_sidecars(
                    model,
                    bridge,
                    mappings,
                    telemetry=telemetry,
                    residual_policy=controller,
                )
                if policy.name != "base"
                else []
            )
            try:
                row = _generate_one(
                    model,
                    tokenizer,
                    task,
                    benchmark,
                    condition=condition,
                    logits_processor=tracker,
                    run_id=config.run_id,
                )
            finally:
                for handle in handles:
                    handle.remove()
            row["rescue_policy"] = policy.model_dump(mode="json")
            row["rescue_policy_fingerprint"] = policy.fingerprint()
            rows.append(row)
            _write_jsonl_atomic(destination, rows)
        results[condition] = {
            "rows": len(rows),
            "sha256": _sha256_file(destination),
            "telemetry": telemetry.snapshot() if telemetry is not None else {},
        }
    if any(result["rows"] != 12 for result in results.values()):
        raise BridgeRescueError("strength screen row reconciliation failed")
    report = {
        "status": "PASS",
        "classification": "post-hoc-exploratory-screen-not-confirmation",
        "run_id": config.run_id,
        "experiment_fingerprint": config.fingerprint(),
        "host": host_evidence,
        "conditions": results,
        "elapsed_seconds": time.monotonic() - started,
        "next_permitted_action": "official scoring on a Docker-capable scorer",
    }
    report_path = config.output_dir / "generation-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score_strength_screen(config_path: Path, *, evalplus_image: str) -> Mapping[str, Any]:
    """Officially score every Experiment 1 condition on the same 12 task IDs."""
    from reverse_reap.mbpp_bridge_benchmark import (
        _docker_prefix,
        _metric_pair,
        _official_result_path,
        _official_result_rows,
        _verify_evalplus_image,
        load_mbpp_bridge_config,
    )

    config = load_rescue_experiment_config(config_path)
    benchmark = load_mbpp_bridge_config(config.benchmark_config, allow_expired=True)
    _verify_evalplus_image(evalplus_image, benchmark.evalplus.revision)
    tasks = [
        json.loads(line)
        for line in config.dataset_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_ids = [row["task_id"] for row in tasks]
    expected = set(task_ids)
    scores_dir = config.output_dir / "official-evalplus"
    scores_dir.mkdir(parents=True, exist_ok=True)
    conditions = [
        "base-thinking-on" if policy.name == "base" else f"bridge-{policy.name}-thinking-on"
        for policy in config.policies
    ]
    scored: dict[str, dict[str, dict[str, bool]]] = {}
    logs: dict[str, Any] = {}
    for condition in conditions:
        source = config.output_dir / "conditions" / f"{condition}.jsonl"
        if not source.is_file():
            raise BridgeRescueError(f"missing generated condition: {condition}")
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        if [row.get("task_id") for row in rows] != task_ids:
            raise BridgeRescueError(f"condition task universe differs: {condition}")
        sanitized = scores_dir / f"{condition}-sanitized.jsonl"
        if not sanitized.is_file():
            prefix = _docker_prefix(benchmark, evalplus_image, work=config.output_dir.resolve())
            relative = source.resolve().relative_to(config.output_dir.resolve())
            command = prefix + [
                "python",
                "-m",
                "evalplus.sanitize",
                "--samples",
                f"/work/{relative}",
                "--mbpp_version",
                benchmark.dataset.version,
            ]
            run = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=benchmark.evalplus.timeout_seconds_per_condition,
            )
            logs[f"{condition}-sanitize"] = {
                "return_code": run.returncode,
                "stdout": run.stdout[-4096:],
                "stderr": run.stderr[-4096:],
            }
            produced = source.with_name(source.stem + "-sanitized.jsonl")
            if run.returncode != 0 or not produced.is_file():
                raise BridgeRescueError(f"official sanitizer failed: {condition}")
            os.replace(produced, sanitized)
        sanitized_rows = [
            json.loads(line) for line in sanitized.read_text().splitlines() if line
        ]
        if [row.get("task_id") for row in sanitized_rows] != task_ids:
            raise BridgeRescueError(f"sanitizer changed task universe: {condition}")
        result_path = _official_result_path(sanitized)
        if not result_path.is_file():
            prefix = _docker_prefix(benchmark, evalplus_image, work=config.output_dir.resolve())
            relative = sanitized.resolve().relative_to(config.output_dir.resolve())
            command = prefix + [
                "python",
                "-m",
                "evalplus.evaluate",
                "--dataset",
                "mbpp",
                "--samples",
                f"/work/{relative}",
                "--parallel",
                str(benchmark.evalplus.parallel_workers),
                "--version",
                benchmark.dataset.version,
            ]
            run = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=benchmark.evalplus.timeout_seconds_per_condition,
            )
            logs[f"{condition}-evaluate"] = {
                "return_code": run.returncode,
                "stdout": run.stdout[-4096:],
                "stderr": run.stderr[-4096:],
            }
            if run.returncode != 0 or not result_path.is_file():
                raise BridgeRescueError(f"official evaluation failed: {condition}")
        scored[condition] = _official_result_rows(result_path, expected)
    base = scored["base-thinking-on"]
    comparisons = {
        condition: {
            metric: _metric_pair(base, values, task_ids, metric)
            for metric in ("base", "plus")
        }
        for condition, values in scored.items()
        if condition != "base-thinking-on"
    }
    report = {
        "status": "PASS",
        "classification": "post-hoc-exploratory-screen-not-confirmation",
        "run_id": config.run_id,
        "task_ids": task_ids,
        "evalplus_image": evalplus_image,
        "comparisons": comparisons,
        "logs": logs,
        "next_permitted_action": "human selection of at most two strengths",
    }
    destination = config.output_dir / "official-strength-screen-report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
