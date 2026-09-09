"""Deterministic base-versus-bridge coding benchmark orchestration.

The benchmark deliberately separates score outcomes from integrity gates.  A
poor pilot score never suppresses the full tier, while a broken bridge load,
nondeterministic repeat, invalid task manifest, or exhausted budget stops the
run before it can produce misleading evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from reverse_reap.bridge_training import (
    BridgeModel,
    _load_extracted_experts,
    install_bridge_sidecars,
    load_bridge_training_config,
)
from reverse_reap.config import StrictModel
from reverse_reap.datasets import (
    NormalizedSample,
    canonical_json,
    freeze_manifest,
    load_manifest,
    normalize_sample,
)
from reverse_reap.evaluator import evaluate_python


class BridgeBenchmarkError(RuntimeError):
    """Raised when benchmark evidence would be invalid or unsafe."""


class BridgeGenerationTelemetry:
    """Streaming aggregate; full hidden activations are never retained."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.values: dict[str, dict[str, float]] = {}

    def record(self, key: str, gate: Any, residual: Any) -> None:
        row = self.values.setdefault(
            key, {"calls": 0.0, "tokens": 0.0, "gate_sum": 0.0, "residual_l2_sum": 0.0}
        )
        row["calls"] += 1
        row["tokens"] += float(gate.numel())
        row["gate_sum"] += float(gate.detach().float().sum().cpu())
        norms = residual.detach().float().reshape(-1, residual.shape[-1]).norm(dim=-1)
        row["residual_l2_sum"] += float(norms.sum().cpu())

    def snapshot(self) -> dict[str, Any]:
        result = {}
        for key, row in sorted(self.values.items()):
            tokens = max(row["tokens"], 1.0)
            result[key] = {
                "calls": int(row["calls"]),
                "tokens": int(row["tokens"]),
                "gate_mean": row["gate_sum"] / tokens,
                "residual_l2_mean": row["residual_l2_sum"] / tokens,
            }
        return result


class BridgeBenchmarkRuntime(StrictModel):
    seed: int = Field(ge=0)
    deterministic: Literal[True] = True
    enable_thinking: Literal[False] = False
    batch_size: Literal[1] = 1
    max_input_tokens: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    repeats_per_condition: Literal[2] = 2
    heartbeat_seconds: Literal[300] = 300


class BridgeBenchmarkDataset(StrictModel):
    source_manifest: Path
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: Literal[
        "evalplus/humanevalplus", "openai/openai_humaneval"
    ] = "evalplus/humanevalplus"
    pilot_items: int = Field(default=25, ge=1)
    full_items: int | None = Field(default=None, ge=1)
    selection_seed: int = Field(default=20260909, ge=0)


class BridgeBenchmarkBudget(StrictModel):
    max_wall_hours: float = Field(gt=0)
    max_cost_usd: float = Field(gt=0)
    provider_rate_usd_per_hour: float = Field(gt=0)
    usable_fraction: Literal[0.8] = 0.8
    deadline_utc: datetime

    @model_validator(mode="after")
    def valid_deadline(self) -> BridgeBenchmarkBudget:
        if self.deadline_utc.tzinfo is None or self.deadline_utc.utcoffset() is None:
            raise ValueError("benchmark deadline must be timezone-aware")
        return self


class BridgeBenchmarkConfig(StrictModel):
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
    dataset: BridgeBenchmarkDataset
    runtime: BridgeBenchmarkRuntime
    scoring_mode: Literal["deferred", "docker-local"] = "deferred"
    evaluator_image: str | None = Field(
        default=None, pattern=r"^[^\s]+@sha256:[0-9a-f]{64}$"
    )
    budget: BridgeBenchmarkBudget
    output_dir: Path
    proceed_full_regardless_of_pilot_score: Literal[True] = True

    @model_validator(mode="after")
    def scoring_contract_is_valid(self) -> BridgeBenchmarkConfig:
        if self.scoring_mode == "docker-local" and self.evaluator_image is None:
            raise ValueError("docker-local scoring requires a pinned evaluator image")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any], *, refuse: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and refuse:
        if path.read_text(encoding="utf-8") != body:
            raise BridgeBenchmarkError(
                f"refusing to overwrite benchmark artifact: {path}"
            )
        return
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


def _write_jsonl_atomic(
    path: Path, rows: list[dict[str, Any]], *, replace: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != rendered:
            raise BridgeBenchmarkError(f"refusing to overwrite benchmark rows: {path}")
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


def load_bridge_benchmark_config(
    path: Path, *, allow_expired: bool = False
) -> BridgeBenchmarkConfig:
    import yaml

    try:
        config = BridgeBenchmarkConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        if (
            not allow_expired
            and config.budget.deadline_utc.astimezone(UTC) <= datetime.now(UTC)
        ):
            raise ValueError("benchmark deadline has expired")
        return config
    except Exception as error:
        raise BridgeBenchmarkError(f"invalid bridge benchmark config: {path}: {error}") from error


def fetch_and_freeze_humanevalplus(
    destination: Path, *, revision: str, seed: int = 20260909
) -> dict[str, Any]:
    """Resolve one approved HumanEval+ revision and freeze normalized rows."""
    try:
        from huggingface_hub import HfApi

        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - optional data dependency
        raise BridgeBenchmarkError("datasets and huggingface_hub are required") from error
    resolved = HfApi().dataset_info(
        "evalplus/humanevalplus", revision=revision
    ).sha
    if resolved != revision:
        raise BridgeBenchmarkError(
            f"HumanEval+ revision moved: expected {revision}, resolved {resolved}"
        )
    dataset = load_dataset(
        "evalplus/humanevalplus",
        split="test",
        revision=revision,
        trust_remote_code=False,
    )
    samples = []
    for row in dataset:
        tests = str(row["test"]).rstrip() + f"\n\ncheck({row['entry_point']})\n"
        samples.append(
            normalize_sample(
                {
                    "source": "evalplus/humanevalplus",
                    "source_revision": revision,
                    "source_id": row["task_id"],
                    "domain": "coding",
                    "stratum": "function-synthesis",
                    "language": "python",
                    "prompt": row["prompt"],
                    "reference": row["canonical_solution"],
                    "tests": tests,
                    "entry_point": row["entry_point"],
                    "scorer": "unit_tests",
                    "prompt_template_version": "bridge-humanevalplus-v1",
                    "timeout_seconds": 30,
                },
                seed=seed,
            )
        )
    report = freeze_manifest(samples, destination)
    return {
        **report,
        "dataset_id": "evalplus/humanevalplus",
        "dataset_revision": revision,
        "license": "apache-2.0",
    }


def freeze_bridge_benchmark_tasks(
    config: BridgeBenchmarkConfig,
) -> tuple[list[NormalizedSample], list[NormalizedSample], dict[str, Any]]:
    """Select a hash-ordered pilot and fixed full HumanEval set."""
    source = config.dataset.source_manifest
    if _sha256_file(source) != config.dataset.source_manifest_sha256:
        raise BridgeBenchmarkError("source benchmark manifest hash mismatch")
    eligible = [
        sample
        for sample in load_manifest(source)
        if sample.source == config.dataset.source_id
        and sample.domain == "coding"
        and sample.language == "python"
        and sample.scorer == "unit_tests"
    ]
    if not eligible:
        raise BridgeBenchmarkError("source manifest has no eligible Python HumanEval tasks")
    ordered = sorted(
        eligible,
        key=lambda sample: hashlib.sha256(
            f"{config.dataset.selection_seed}\0{sample.content_sha256}".encode()
        ).hexdigest(),
    )
    full_count = config.dataset.full_items or len(ordered)
    if config.dataset.pilot_items > full_count or full_count > len(ordered):
        raise BridgeBenchmarkError(
            f"invalid pilot/full sizes for {len(ordered)} eligible tasks"
        )
    full = ordered[:full_count]
    pilot = full[: config.dataset.pilot_items]
    if len({sample.content_sha256 for sample in full}) != len(full):
        raise BridgeBenchmarkError("benchmark task set contains duplicate content")
    report = {
        "schema_version": 1,
        "kind": "bridge-benchmark-task-freeze",
        "source_manifest": str(source),
        "source_manifest_sha256": config.dataset.source_manifest_sha256,
        "selection_seed": config.dataset.selection_seed,
        "pilot_count": len(pilot),
        "full_count": len(full),
        "pilot_sample_ids": [sample.sample_id for sample in pilot],
        "full_sample_ids": [sample.sample_id for sample in full],
        "pilot_sha256": hashlib.sha256(
            canonical_json([sample.model_dump(mode="json") for sample in pilot])
        ).hexdigest(),
        "full_sha256": hashlib.sha256(
            canonical_json([sample.model_dump(mode="json") for sample in full])
        ).hexdigest(),
    }
    return pilot, full, report


def _verify_host_files(config: BridgeBenchmarkConfig) -> dict[str, Any]:
    path = config.host_files_manifest
    if _sha256_file(path) != config.host_files_manifest_sha256:
        raise BridgeBenchmarkError("host files manifest hash mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["revision"] != config.host_revision:
            raise BridgeBenchmarkError("host files manifest revision differs")
        files = payload["files"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise BridgeBenchmarkError("invalid host files manifest") from error
    if not files:
        raise BridgeBenchmarkError("host files manifest is empty")
    for item in files:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise BridgeBenchmarkError("host files manifest contains unsafe path")
        target = config.host_model / relative
        if not target.is_file() or _sha256_file(target) != item["sha256"]:
            raise BridgeBenchmarkError(f"host file hash mismatch: {relative}")
    return {"revision": config.host_revision, "verified_files": len(files)}


def _clean_code(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n?(.*?)```", stripped, flags=re.DOTALL)
    return (match.group(1) if match else stripped).strip()


def _prompt(sample: NormalizedSample) -> str:
    return (
        "Complete the Python function below. Return only the executable Python "
        "continuation that should be appended directly to the supplied prefix, "
        "with no Markdown fences or explanation.\n\n" + sample.prompt
    )


def _load_host(config: BridgeBenchmarkConfig) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
        try:
            from transformers import AutoModelForImageTextToText as HostModel
        except ImportError:  # pragma: no cover
            from transformers import AutoModelForCausalLM as HostModel
    except ImportError as error:  # pragma: no cover
        raise BridgeBenchmarkError("Torch and Transformers are required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.host_model), local_files_only=True, trust_remote_code=False
    )
    model = HostModel.from_pretrained(
        str(config.host_model),
        revision=config.host_revision,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    model_config = getattr(model.config, "text_config", model.config)
    actual_hidden = int(getattr(model_config, "hidden_size", -1))
    actual_layers = int(getattr(model_config, "num_hidden_layers", -1))
    if (actual_hidden, actual_layers) != (2048, 24):
        raise BridgeBenchmarkError(
            "host architecture differs from frozen Qwen3.5-2B contract: "
            f"hidden={actual_hidden}, layers={actual_layers}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def _seed_runtime(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with suppress(RuntimeError):
        torch.use_deterministic_algorithms(True)


def _load_bridge(config: BridgeBenchmarkConfig, device: Any) -> tuple[Any, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover
        raise BridgeBenchmarkError("safetensors.torch is required") from error
    if _sha256_file(config.bridge_config) != config.bridge_config_sha256:
        raise BridgeBenchmarkError("bridge config hash mismatch")
    if _sha256_file(config.bridge_checkpoint) != config.bridge_checkpoint_sha256:
        raise BridgeBenchmarkError("bridge checkpoint hash mismatch")
    bridge_config = load_bridge_training_config(config.bridge_config)
    if bridge_config.host.revision != config.host_revision:
        raise BridgeBenchmarkError("benchmark and bridge host revisions differ")
    experts = _load_extracted_experts(
        bridge_config.donor.extraction_dir, bridge_config.mappings
    )
    bridge = BridgeModel(
        bridge_config.mappings,
        experts,
        hidden_size=bridge_config.host.hidden_size,
        bottleneck_size=bridge_config.runtime.bottleneck_size,
        gate_cap=bridge_config.runtime.gate_cap,
        gate_init_bias=bridge_config.runtime.gate_init_bias,
    )
    state = load_file(str(config.bridge_checkpoint), device="cpu")
    expected = {
        key for key in bridge.state_dict() if not key.startswith("experts.")
    }
    if set(state) != expected:
        raise BridgeBenchmarkError(
            f"bridge checkpoint keys differ: missing={sorted(expected - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - expected)[:5]}"
        )
    result = bridge.load_state_dict(state, strict=False)
    expected_missing = {
        key for key in bridge.state_dict() if key.startswith("experts.")
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise BridgeBenchmarkError("bridge checkpoint did not load with exact expected keys")
    bridge.to(device)
    bridge.eval()
    if any(parameter.requires_grad for parameter in bridge.experts.parameters()):
        raise BridgeBenchmarkError("loaded donor experts are not frozen")
    return bridge, bridge_config.mappings


def _check_budget(config: BridgeBenchmarkConfig, started: float) -> None:
    elapsed_hours = (time.monotonic() - started) / 3600
    estimated_cost = elapsed_hours * config.budget.provider_rate_usd_per_hour
    if datetime.now(UTC) >= config.budget.deadline_utc.astimezone(UTC):
        raise BridgeBenchmarkError("benchmark deadline reached")
    if elapsed_hours >= config.budget.max_wall_hours * config.budget.usable_fraction:
        raise BridgeBenchmarkError("benchmark wall-hour ceiling reached")
    if estimated_cost >= config.budget.max_cost_usd * config.budget.usable_fraction:
        raise BridgeBenchmarkError("benchmark cost ceiling reached")


def _heartbeat(
    config: BridgeBenchmarkConfig,
    *,
    started: float,
    tier: str,
    condition: str,
    completed: int,
    total: int,
) -> None:
    elapsed = time.monotonic() - started
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_id": config.run_id,
        "tier": tier,
        "condition": condition,
        "completed": completed,
        "total": total,
        "elapsed_seconds": elapsed,
        "throughput_items_per_minute": completed / max(elapsed / 60, 1e-9),
        "estimated_compute_cost_usd": (
            elapsed / 3600 * config.budget.provider_rate_usd_per_hour
        ),
    }
    try:
        import torch

        if torch.cuda.is_available():
            payload["gpu_memory_peak_bytes"] = torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    _atomic_json(config.output_dir / "heartbeat.json", payload, refuse=False)


def _generate_one(model: Any, tokenizer: Any, sample: NormalizedSample, config: Any) -> dict:
    import torch

    prompt = _prompt(sample)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    input_ids = rendered if isinstance(rendered, torch.Tensor) else torch.as_tensor(rendered)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[1] > config.runtime.max_input_tokens:
        raise BridgeBenchmarkError(
            f"input exceeds frozen limit for {sample.sample_id}: {input_ids.shape[1]}"
        )
    input_ids = input_ids.to(next(model.parameters()).device)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=config.runtime.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.monotonic() - started
    generated_ids = output[0, input_ids.shape[1] :].detach().cpu().tolist()
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    code = _clean_code(raw)
    return {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "content_sha256": sample.content_sha256,
        "source_id": sample.source_id,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_token_ids": input_ids[0].detach().cpu().tolist(),
        "generated_token_ids": generated_ids,
        "raw_completion": raw,
        "completion": code,
        "completion_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "input_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(generated_ids),
        "generation_seconds": elapsed,
    }


def _score_rows(
    rows: list[dict[str, Any]],
    samples: dict[str, NormalizedSample],
    image: str,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        sample = samples[row["sample_id"]]
        result = evaluate_python(
            sample.prompt + row["completion"],
            sample.tests or "",
            image=image,
            timeout_seconds=sample.timeout_seconds,
        )
        scored.append(
            {
                **row,
                "passed": result.passed,
                "timed_out": result.timed_out,
                "return_code": result.return_code,
                "scorer_stdout": result.stdout,
                "scorer_stderr": result.stderr,
                "scored_program_sha256": result.program_sha256,
            }
        )
    return scored


def _validate_reference_scoring(
    samples: list[NormalizedSample], image: str
) -> dict[str, Any]:
    failures = []
    for sample in samples:
        result = evaluate_python(
            sample.prompt + (sample.reference or ""),
            sample.tests or "",
            image=image,
            timeout_seconds=sample.timeout_seconds,
        )
        if not result.passed:
            failures.append(
                {
                    "sample_id": sample.sample_id,
                    "return_code": result.return_code,
                    "timed_out": result.timed_out,
                    "stderr": result.stderr,
                }
            )
    if failures:
        raise BridgeBenchmarkError(
            f"reference scorer preflight failed for {len(failures)}/{len(samples)} tasks"
        )
    return {"passed": True, "references": len(samples), "failures": 0}


def _validate_repeats(
    first: list[dict[str, Any]], second: list[dict[str, Any]], condition: str
) -> dict[str, Any]:
    left = {row["sample_id"]: row for row in first}
    right = {row["sample_id"]: row for row in second}
    if set(left) != set(right):
        raise BridgeBenchmarkError(f"{condition} repeats have different sample IDs")
    mismatches = [
        sample_id
        for sample_id in sorted(left)
        if left[sample_id]["generated_token_ids"] != right[sample_id]["generated_token_ids"]
    ]
    if mismatches:
        raise BridgeBenchmarkError(
            f"{condition} repeat nondeterminism on {len(mismatches)} samples"
        )
    return {"condition": condition, "samples": len(left), "mismatches": 0, "passed": True}


def _paired_report(base: list[dict[str, Any]], bridge: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = {row["sample_id"]: row for row in base}
    bridge_by_id = {row["sample_id"]: row for row in bridge}
    if set(base_by_id) != set(bridge_by_id):
        raise BridgeBenchmarkError("base/bridge task identities differ")
    transitions = Counter()
    items = []
    for sample_id in sorted(base_by_id):
        base_pass = bool(base_by_id[sample_id]["passed"])
        bridge_pass = bool(bridge_by_id[sample_id]["passed"])
        label = {
            (False, False): "both-fail",
            (False, True): "bridge-fixes",
            (True, False): "bridge-breaks",
            (True, True): "both-pass",
        }[(base_pass, bridge_pass)]
        transitions[label] += 1
        items.append(
            {
                "sample_id": sample_id,
                "base_pass": base_pass,
                "bridge_pass": bridge_pass,
                "transition": label,
            }
        )
    total = len(items)
    fixes = transitions["bridge-fixes"]
    breaks = transitions["bridge-breaks"]
    discordant = fixes + breaks
    exact_p = 1.0
    if discordant:
        tail = sum(
            _binomial_probability(discordant, value)
            for value in range(0, min(fixes, breaks) + 1)
        )
        exact_p = min(1.0, 2 * tail)
    rng = np.random.default_rng(20260909)
    deltas = np.asarray(
        [int(row["bridge_pass"]) - int(row["base_pass"]) for row in items],
        dtype=np.float64,
    )
    bootstrap = np.asarray(
        [rng.choice(deltas, size=total, replace=True).mean() for _ in range(2000)]
    )
    return {
        "samples": total,
        "base_passes": sum(row["base_pass"] for row in items),
        "bridge_passes": sum(row["bridge_pass"] for row in items),
        "base_pass_rate": sum(row["base_pass"] for row in items) / total,
        "bridge_pass_rate": sum(row["bridge_pass"] for row in items) / total,
        "absolute_pass_rate_change": (
            sum(row["bridge_pass"] for row in items)
            - sum(row["base_pass"] for row in items)
        )
        / total,
        "transitions": dict(sorted(transitions.items())),
        "paired_exact_p_value": exact_p,
        "bootstrap_95pct_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "items": items,
    }


def _binomial_probability(trials: int, successes: int) -> float:
    return math.comb(trials, successes) * (0.5**trials)


def _freeze_artifact_hashes(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"artifact-manifest.json", "heartbeat.json"}:
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {"schema_version": 1, "kind": "bridge-benchmark-artifacts", "files": records}
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    _atomic_json(root / "artifact-manifest.json", payload, refuse=False)
    return payload


def _write_state(
    config: BridgeBenchmarkConfig,
    *,
    status: str,
    stage: str,
    error: str | None = None,
) -> None:
    _atomic_json(
        config.output_dir / "state.json",
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "task_id": "bridge-benchmark",
            "status": status,
            "stage": stage,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "config_fingerprint": config.fingerprint(),
            "failure": error,
        },
        refuse=False,
    )


def _run_bridge_benchmark(
    config_path: Path, config: BridgeBenchmarkConfig
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_state(config, status="PREFLIGHTED", stage="freezing-tasks")
    started = time.monotonic()
    _check_budget(config, started)
    pilot, full, freeze = freeze_bridge_benchmark_tasks(config)
    _atomic_json(config.output_dir / "task-freeze.json", freeze)
    _atomic_json(
        config.output_dir / "config-snapshot.json", config.model_dump(mode="json")
    )
    _write_jsonl_atomic(
        config.output_dir / "pilot-tasks.jsonl",
        [sample.model_dump(mode="json") for sample in pilot],
    )
    _write_jsonl_atomic(
        config.output_dir / "full-tasks.jsonl",
        [sample.model_dump(mode="json") for sample in full],
    )
    host_verification = _verify_host_files(config)
    if config.scoring_mode == "docker-local":
        assert config.evaluator_image is not None
        evaluator_probe = evaluate_python(
            "def bridge_benchmark_probe():\n    return 1",
            "assert bridge_benchmark_probe() == 1",
            image=config.evaluator_image,
            timeout_seconds=10,
        )
        if not evaluator_probe.passed:
            raise BridgeBenchmarkError("pinned evaluator image failed its preflight")
        _atomic_json(
            config.output_dir / "reference-scorer-preflight.json",
            _validate_reference_scoring(full, config.evaluator_image),
        )
    _seed_runtime(config.runtime.seed)
    model, tokenizer = _load_host(config)
    bridge, mappings = _load_bridge(config, next(model.parameters()).device)
    _write_state(config, status="RUNNING", stage="generation")
    sample_lookup = {sample.sample_id: sample for sample in full}
    all_reports: dict[str, Any] = {}
    for tier, samples in (("pilot", pilot), ("full", full)):
        tier_rows: dict[str, list[dict[str, Any]]] = {}
        for condition in ("base", "bridge"):
            handles = []
            telemetry = BridgeGenerationTelemetry() if condition == "bridge" else None
            if condition == "bridge":
                handles = install_bridge_sidecars(
                    model, bridge, mappings, telemetry=telemetry
                )
            try:
                for repeat in ("a", "b"):
                    run_name = f"{condition}-{repeat}"
                    run_path = config.output_dir / tier / f"{run_name}.jsonl"
                    rows = []
                    if run_path.is_file():
                        rows = [
                            json.loads(line)
                            for line in run_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ]
                        expected_prefix = [sample.sample_id for sample in samples[: len(rows)]]
                        if [row.get("sample_id") for row in rows] != expected_prefix:
                            raise BridgeBenchmarkError(
                                f"resume prefix differs for {tier}/{run_name}"
                            )
                        if any(
                            row.get("tier") != tier
                            or row.get("condition") != condition
                            or row.get("repeat") != repeat
                            for row in rows
                        ):
                            raise BridgeBenchmarkError(
                                f"resume metadata differs for {tier}/{run_name}"
                            )
                    last_heartbeat = 0.0
                    for index, sample in enumerate(samples[len(rows) :], start=len(rows)):
                        _check_budget(config, started)
                        if telemetry is not None:
                            telemetry.reset()
                        row = _generate_one(model, tokenizer, sample, config)
                        if telemetry is not None:
                            row["bridge_telemetry"] = telemetry.snapshot()
                        row.update(
                            {
                                "run_id": config.run_id,
                                "tier": tier,
                                "condition": condition,
                                "repeat": repeat,
                                "ordinal": index,
                            }
                        )
                        rows.append(row)
                        _write_jsonl_atomic(run_path, rows, replace=True)
                        if time.monotonic() - last_heartbeat >= config.runtime.heartbeat_seconds:
                            _heartbeat(
                                config,
                                started=started,
                                tier=tier,
                                condition=run_name,
                                completed=index + 1,
                                total=len(samples),
                            )
                            last_heartbeat = time.monotonic()
                    final_rows = rows
                    if config.scoring_mode == "docker-local":
                        assert config.evaluator_image is not None
                        final_rows = _score_rows(rows, sample_lookup, config.evaluator_image)
                    _write_jsonl_atomic(run_path, final_rows, replace=True)
                    tier_rows[run_name] = final_rows
            finally:
                for handle in handles:
                    handle.remove()
        base_determinism = _validate_repeats(
            tier_rows["base-a"], tier_rows["base-b"], "base"
        )
        bridge_determinism = _validate_repeats(
            tier_rows["bridge-a"], tier_rows["bridge-b"], "bridge"
        )
        paired = None
        if config.scoring_mode == "docker-local":
            paired = _paired_report(tier_rows["base-a"], tier_rows["bridge-a"])
        report = {
            "tier": tier,
            "base_determinism": base_determinism,
            "bridge_determinism": bridge_determinism,
            "paired": paired,
            "scoring_status": (
                "complete" if config.scoring_mode == "docker-local" else "deferred"
            ),
            "pilot_score_controls_full_execution": False,
        }
        _atomic_json(config.output_dir / tier / "report.json", report)
        all_reports[tier] = report
        _write_state(config, status="RUNNING", stage=f"{tier}-complete")
        # Full proceeds regardless of score. Integrity errors above still stop.
    elapsed = time.monotonic() - started
    final = {
        "schema_version": 1,
        "kind": "bridge-base-paired-benchmark",
        "status": "PASS",
        "classification": (
            "capability-comparison-complete"
            if config.scoring_mode == "docker-local"
            else "generation-complete-scoring-deferred"
        ),
        "run_id": config.run_id,
        "config_sha256": _sha256_file(config_path),
        "config_fingerprint": config.fingerprint(),
        "host_revision": config.host_revision,
        "host_verification": host_verification,
        "bridge_checkpoint_sha256": config.bridge_checkpoint_sha256,
        "task_freeze": freeze,
        "reports": all_reports,
        "elapsed_seconds": elapsed,
        "estimated_compute_cost_usd": (
            elapsed / 3600 * config.budget.provider_rate_usd_per_hour
        ),
        "scientific_claim": "paired benchmark result; not causal proof",
    }
    _atomic_json(config.output_dir / "benchmark-report.json", final)
    _write_state(config, status="COMPLETE", stage="generation-complete")
    final["artifact_manifest"] = _freeze_artifact_hashes(config.output_dir)[
        "manifest_sha256"
    ]
    return final


def run_bridge_benchmark(config_path: Path) -> dict[str, Any]:
    """Run pilot then full, with two base and two bridged repeats per tier."""
    config = load_bridge_benchmark_config(config_path)
    try:
        return _run_bridge_benchmark(config_path, config)
    except Exception as error:
        _write_state(
            config,
            status="FAILED_TERMINAL",
            stage="generation",
            error=f"{type(error).__name__}: {error}",
        )
        raise


def _score_bridge_benchmark(
    config_path: Path,
    *,
    evaluator_image: str,
    config: BridgeBenchmarkConfig,
) -> dict[str, Any]:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", evaluator_image):
        raise BridgeBenchmarkError("scoring requires an image pinned by sha256 digest")
    probe = evaluate_python(
        "def bridge_benchmark_probe():\n    return 1",
        "assert bridge_benchmark_probe() == 1",
        image=evaluator_image,
        timeout_seconds=10,
    )
    if not probe.passed:
        raise BridgeBenchmarkError("pinned evaluator image failed its preflight")
    full_samples = [
        NormalizedSample.model_validate_json(line)
        for line in (config.output_dir / "full-tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    _atomic_json(
        config.output_dir / "reference-scorer-preflight.json",
        _validate_reference_scoring(full_samples, evaluator_image),
    )
    reports: dict[str, Any] = {}
    for tier in ("pilot", "full"):
        tasks_path = config.output_dir / f"{tier}-tasks.jsonl"
        samples = {
            sample.sample_id: sample
            for sample in (
                NormalizedSample.model_validate_json(line)
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        scored_runs: dict[str, list[dict[str, Any]]] = {}
        for condition in ("base", "bridge"):
            for repeat in ("a", "b"):
                name = f"{condition}-{repeat}"
                raw_path = config.output_dir / tier / f"{name}.jsonl"
                rows = [
                    json.loads(line)
                    for line in raw_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if {row["sample_id"] for row in rows} != set(samples):
                    raise BridgeBenchmarkError(f"{tier}/{name} task IDs differ from freeze")
                scored = _score_rows(rows, samples, evaluator_image)
                _write_jsonl_atomic(
                    config.output_dir / tier / f"{name}-scored.jsonl", scored
                )
                scored_runs[name] = scored
        report = {
            "tier": tier,
            "base_determinism": _validate_repeats(
                scored_runs["base-a"], scored_runs["base-b"], "base"
            ),
            "bridge_determinism": _validate_repeats(
                scored_runs["bridge-a"], scored_runs["bridge-b"], "bridge"
            ),
            "paired": _paired_report(
                scored_runs["base-a"], scored_runs["bridge-a"]
            ),
            "scoring_status": "complete",
            "evaluator_image": evaluator_image,
            "pilot_score_controls_full_execution": False,
        }
        _atomic_json(config.output_dir / tier / "scored-report.json", report)
        reports[tier] = report
    result = {
        "schema_version": 1,
        "kind": "bridge-base-paired-scoring",
        "status": "PASS",
        "classification": "capability-comparison-complete",
        "run_id": config.run_id,
        "generation_config_sha256": _sha256_file(config_path),
        "evaluator_image": evaluator_image,
        "reports": reports,
        "scientific_claim": "paired benchmark result; not causal proof",
    }
    _atomic_json(config.output_dir / "scoring-report.json", result)
    _atomic_json(
        config.output_dir / "scoring-state.json",
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "task_id": "bridge-benchmark-scoring",
            "status": "COMPLETE",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "failure": None,
        },
        refuse=False,
    )
    result["artifact_manifest"] = _freeze_artifact_hashes(config.output_dir)[
        "manifest_sha256"
    ]
    return result


def score_bridge_benchmark(
    config_path: Path, *, evaluator_image: str
) -> dict[str, Any]:
    """Score an already-complete four-run bundle on a Docker-capable CPU host."""
    config = load_bridge_benchmark_config(config_path, allow_expired=True)
    _atomic_json(
        config.output_dir / "scoring-state.json",
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "task_id": "bridge-benchmark-scoring",
            "status": "RUNNING",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "failure": None,
        },
        refuse=False,
    )
    try:
        return _score_bridge_benchmark(
            config_path, evaluator_image=evaluator_image, config=config
        )
    except Exception as error:
        _atomic_json(
            config.output_dir / "scoring-state.json",
            {
                "schema_version": 1,
                "run_id": config.run_id,
                "task_id": "bridge-benchmark-scoring",
                "status": "FAILED_TERMINAL",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "failure": f"{type(error).__name__}: {error}",
            },
            refuse=False,
        )
        raise
