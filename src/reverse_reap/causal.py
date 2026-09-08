"""Deterministic masked generation, scoring, causal controls, and Gate D reporting."""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from reverse_reap.config import ExperimentConfig
from reverse_reap.datasets import NormalizedSample, balanced_subset, load_manifest
from reverse_reap.evaluator import EvaluationResult, evaluate_java, evaluate_python
from reverse_reap.instrumentation import intervene_qwen35
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import load_donor, validate_donor_contract


class CausalError(RuntimeError):
    """Raised when an intervention or causal comparison is invalid."""


def load_expert_set(path: Path) -> frozenset[tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experts = payload.get("experts")
    if not isinstance(experts, list) or not experts:
        raise CausalError("intervention manifest has no experts")
    selected = frozenset((int(item["layer"]), int(item["expert"])) for item in experts)
    if len(selected) != len(experts):
        raise CausalError("intervention manifest contains duplicate experts")
    return selected


def _extract_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else text.strip()


def _exact_answer(text: str) -> str:
    marker = re.findall(r"####\s*([^\n]+)", text)
    if marker:
        return marker[-1].strip().replace(",", "")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip().replace(",", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].replace(",", "") if lines else ""


def score_response(
    sample: NormalizedSample, response: str, *, evaluator_image: str
) -> dict[str, Any]:
    if sample.scorer == "exact_match":
        predicted = _exact_answer(response)
        reference = _exact_answer(sample.reference or "")
        return {"scoreable": True, "passed": predicted == reference, "predicted": predicted}
    if sample.scorer == "multiple_choice":
        choices = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", response.upper())
        predicted = choices[-1] if choices else ""
        return {
            "scoreable": True,
            "passed": predicted == (sample.reference or "").strip().upper(),
            "predicted": predicted,
        }
    if sample.scorer == "unit_tests":
        code = _extract_code(response)
        if (
            sample.entry_point
            and f"def {sample.entry_point}" not in code
            and "def " in sample.prompt
        ):
            code = sample.prompt.rstrip() + "\n" + code
        if sample.language == "java" and "class Solution" not in code:
            code = sample.prompt.rstrip() + "\n" + code
        evaluator = evaluate_java if sample.language == "java" else evaluate_python
        result: EvaluationResult = evaluator(
            code,
            sample.tests or "",
            image=evaluator_image,
            timeout_seconds=sample.timeout_seconds,
        )
        return {
            "scoreable": True,
            "passed": result.passed,
            "timed_out": result.timed_out,
            "return_code": result.return_code,
            "program_sha256": result.program_sha256,
            "stderr_tail": result.stderr[-1000:],
        }
    if sample.scorer == "swebench":
        return {
            "scoreable": False,
            "passed": False,
            "error": "requires the pinned SWE-bench repository harness stage",
        }
    raise CausalError(f"unsupported scorer: {sample.scorer}")


def _generate(
    model: Any, tokenizer: Any, sample: NormalizedSample, config: ExperimentConfig
) -> tuple[str, int, bool]:
    import torch

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=config.runtime.enable_thinking,
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to(
        dtype=torch.long, device=model.get_input_embeddings().weight.device
    )
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=config.runtime.max_new_tokens,
            use_cache=config.runtime.use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, ids.shape[1] :]
    return (
        tokenizer.decode(generated, skip_special_tokens=True),
        int(generated.numel()),
        generated.numel() >= config.runtime.max_new_tokens,
    )


def generate_condition(
    model: Any,
    tokenizer: Any,
    architecture: Any,
    dataset_manifest: Path,
    destination: Path,
    config: ExperimentConfig,
    *,
    split: str,
    condition_id: str,
    expert_manifest: Path | None = None,
    limit: int | None = None,
    instrument_noop: bool = False,
) -> dict[str, Any]:
    """Generate responses for one causal condition and write generation-only records.

    Scoring is deliberately excluded so GPU hosts without Docker can still
    produce generations; ``score_condition`` completes the records on a
    CPU host that has the pinned evaluator image.

    ``instrument_noop`` runs the optimized intervention-only path with an empty
    mask — a numerically transparent no-op used to prove the intervention
    wrapper itself cannot perturb generation (no-op equivalence gate).
    Causal generation uses :func:`intervene_qwen35` (native fused forward
    with zeroed router weights, no telemetry side path); the slow
    telemetry/replay path is preserved for capture and Gate A only.
    """
    masked = load_expert_set(expert_manifest) if expert_manifest else frozenset()
    samples = [sample for sample in load_manifest(dataset_manifest) if sample.split == split]
    samples = balanced_subset(samples, limit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CausalError(f"refusing to overwrite evaluation: {destination}")
    records = []
    for sample in samples:
        started = time.monotonic()
        if masked:
            with intervene_qwen35(architecture, masked=masked):
                response, generated_tokens, truncated = _generate(model, tokenizer, sample, config)
        elif instrument_noop:
            with intervene_qwen35(architecture, masked=frozenset()):
                response, generated_tokens, truncated = _generate(model, tokenizer, sample, config)
        else:
            response, generated_tokens, truncated = _generate(model, tokenizer, sample, config)
        latency_seconds = time.monotonic() - started
        records.append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "source_id": sample.source_id,
                "scorer": sample.scorer,
                "domain": sample.domain,
                "stratum": sample.stratum,
                "split": sample.split,
                "condition_id": condition_id,
                "masked_experts": len(masked),
                "response": response,
                "generated_tokens": generated_tokens,
                "truncated": truncated,
                "latency_seconds": latency_seconds,
            }
        )
    _atomic_write_jsonl(destination, records)
    return {
        "condition_id": condition_id,
        "samples": len(records),
        "masked_experts": len(masked),
        "truncation_rate": (
            float(np.mean([record["truncated"] for record in records])) if records else 0.0
        ),
        "mean_latency_seconds": (
            float(np.mean([record["latency_seconds"] for record in records])) if records else 0.0
        ),
    }


def _encode_left_padded_batch(
    tokenizer: Any, samples: list[NormalizedSample], config: ExperimentConfig, model: Any
) -> dict:
    """Left-padded batch encoding shared by the benchmark and production B8 path.

    Identical convention to the B8-qualified benchmark
    (scripts/benchmark_intervention_paths.py::_encode_batch): left padding,
    eos as the pad token when none is set, the pinned chat template with the
    configured thinking flag. Prompt order is the input order.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": sample.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.runtime.enable_thinking,
        )
        for sample in samples
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    return {k: v.to(model.get_input_embeddings().weight.device) for k, v in encoded.items()}


def _batched_generate(
    model: Any, tokenizer: Any, samples: list[NormalizedSample], config: ExperimentConfig
) -> tuple[list[str], list[list[int]], float]:
    """Greedy left-padded batch generation with the production decoding parameters.

    Same ``model.generate`` arguments as :func:`_generate`
    (``do_sample=False``, pinned ``max_new_tokens``/``use_cache``,
    ``pad_token_id=eos``); only the input shape differs. Returns
    (responses, generated-id lists with pad ids stripped, chunk wall seconds).
    """
    import torch

    encoded = _encode_left_padded_batch(tokenizer, samples, config, model)
    prompt_width = encoded["input_ids"].shape[1]
    pad_id_used = tokenizer.eos_token_id
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=config.runtime.max_new_tokens,
            use_cache=config.runtime.use_cache,
            pad_token_id=pad_id_used,
        )
    wall_seconds = time.monotonic() - started
    generated = output[:, prompt_width:]
    responses, gen_ids = [], []
    for row in range(generated.shape[0]):
        ids = generated[row]
        keep = ids != pad_id_used
        kept = ids[keep].tolist()
        gen_ids.append(kept)
        responses.append(tokenizer.decode(ids, skip_special_tokens=True))
    return responses, gen_ids, wall_seconds


def _write_heartbeat(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def _gpu_peak_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / 2**20)
    except Exception:
        pass
    return None


def generate_condition_batched(
    model: Any,
    tokenizer: Any,
    architecture: Any,
    dataset_manifest: Path,
    destination: Path,
    config: ExperimentConfig,
    *,
    split: str,
    condition_id: str,
    expert_manifest: Path | None = None,
    limit: int | None = None,
    instrument_noop: bool = False,
    checkpoint_dir: Path | None = None,
    heartbeat_path: Path | None = None,
    run_id: str = "unscoped",
) -> dict[str, Any]:
    """Batched production generation for one causal condition (B8-qualified path).

    Consumes the same frozen inputs and emits the exact record schema as
    :func:`generate_condition` (so ``generation_bundle.py`` and the scorer
    consume its output unchanged), but generates in deterministic manifest-order
    chunks of ``config.runtime.batch_size`` under a single arm context per
    chunk (null / empty-mask no-op / frozen-mask intervention — the optimized
    ``intervene_qwen35`` path qualified by the PRO 6000 B8 benchmark).

    Checkpointing: each completed chunk is written atomically to
    ``checkpoint_dir/<condition_id>.chunk-<index:04d>.json`` the moment it is
    validated; a re-entry loads and re-validates those files and only
    generates missing chunks (fail-closed: the final destination is never
    overwritten, and a chunk file whose sample ids disagree with the expected
    slice is a terminal error, not a silent reuse).

    Heartbeats: ``heartbeat_path`` is rewritten atomically after every chunk
    (chunks run ~1 min at B8 throughput, satisfying the 15-minute heartbeat
    rule trivially) with stage/progress/throughput/elapsed/GPU-peak fields.

    Every output row is validated (sample order, non-empty response,
    in-vocab ids); the first invalid row raises :class:`CausalError`.
    """
    import contextlib

    batch_size = config.runtime.batch_size
    if batch_size < 1:
        raise CausalError(f"invalid batch_size: {batch_size}")
    masked = load_expert_set(expert_manifest) if expert_manifest else frozenset()
    samples = [sample for sample in load_manifest(dataset_manifest) if sample.split == split]
    samples = balanced_subset(samples, limit)
    if not samples:
        raise CausalError(f"no samples for split {split!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CausalError(f"refusing to overwrite evaluation: {destination}")
    checkpoint_dir = checkpoint_dir or destination.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        vocab_size = len(tokenizer)
    except Exception:
        vocab_size = None

    total = len(samples)
    chunks = [samples[i : i + batch_size] for i in range(0, total, batch_size)]
    started_all = time.monotonic()
    latencies: list[float] = []
    completed_chunks = 0

    def heartbeat(chunk_index: int, status: str) -> None:
        elapsed = time.monotonic() - started_all
        done = sum(len(c) for c in chunks[:chunk_index])
        _write_heartbeat(
            heartbeat_path,
            {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": run_id,
                "condition_id": condition_id,
                "stage": f"generate-{condition_id}",
                "status": status,
                "completed_items": done,
                "total_items": total,
                "batch_size": batch_size,
                "completed_chunks": chunk_index,
                "total_chunks": len(chunks),
                "elapsed_seconds": round(elapsed, 1),
                "samples_per_minute": round(done / (elapsed / 60), 3) if elapsed > 0 and done else 0.0,
                "gpu_peak_mib": _gpu_peak_mib(),
                "masked_experts": len(masked),
            },
        )

    def arm_context() -> Any:
        if masked or instrument_noop:
            targets = masked if masked else frozenset()
            return intervene_qwen35(architecture, masked=targets)
        return contextlib.nullcontext()

    def chunk_path(index: int) -> Path:
        return checkpoint_dir / f"{condition_id}.chunk-{index:04d}.json"

    def load_validated_chunk(index: int, expected: list[NormalizedSample]) -> list[dict] | None:
        path = chunk_path(index)
        if not path.is_file():
            return None
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [row.get("sample_id") for row in rows] != [s.sample_id for s in expected]:
            raise CausalError(
                f"checkpoint {path} sample ids disagree with the manifest slice "
                "(refusing to reuse a stale or reordered chunk)"
            )
        return rows

    heartbeat(0, "started")
    all_rows: list[dict] = []
    for index, chunk in enumerate(chunks):
        cached = load_validated_chunk(index, chunk)
        if cached is not None:
            all_rows.extend(cached)
            completed_chunks += 1
            latencies.extend(row["latency_seconds"] for row in cached)
            heartbeat(index + 1, "resumed-chunk" if completed_chunks else "running")
            continue
        chunk_started = time.monotonic()
        with arm_context():
            responses, gen_ids, _wall = _batched_generate(model, tokenizer, chunk, config)
        chunk_wall = time.monotonic() - chunk_started
        per_sample_latency = chunk_wall / len(chunk)
        rows: list[dict] = []
        for sample, response, ids in zip(chunk, responses, gen_ids, strict=True):
            if not isinstance(response, str) or not response.strip():
                raise CausalError(f"{condition_id}/{sample.sample_id}: empty response")
            if vocab_size is not None and not all(0 <= i < vocab_size for i in ids):
                raise CausalError(f"{condition_id}/{sample.sample_id}: out-of-vocab id")
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "source_id": sample.source_id,
                    "scorer": sample.scorer,
                    "domain": sample.domain,
                    "stratum": sample.stratum,
                    "split": sample.split,
                    "condition_id": condition_id,
                    "masked_experts": len(masked),
                    "response": response,
                    "generated_tokens": len(ids),
                    "truncated": len(ids) >= config.runtime.max_new_tokens,
                    "latency_seconds": per_sample_latency,
                    "chunk": index,
                    "batch_size": batch_size,
                }
            )
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=checkpoint_dir,
            prefix=f".{chunk_path(index).name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        temporary.replace(chunk_path(index))
        all_rows.extend(rows)
        completed_chunks += 1
        latencies.extend([per_sample_latency] * len(rows))
        heartbeat(index + 1, "running")

    if [row["sample_id"] for row in all_rows] != [s.sample_id for s in samples]:
        raise CausalError(f"{condition_id}: assembled rows drifted from manifest order")
    _atomic_write_jsonl(destination, all_rows)
    heartbeat(len(chunks), "complete")
    return {
        "condition_id": condition_id,
        "samples": len(all_rows),
        "masked_experts": len(masked),
        "batch_size": batch_size,
        "chunks": len(chunks),
        "truncation_rate": (
            float(np.mean([row["truncated"] for row in all_rows])) if all_rows else 0.0
        ),
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else 0.0,
    }


def score_condition(
    generated_path: Path,
    dataset_manifest: Path,
    destination: Path,
    *,
    evaluator_image: str,
) -> dict[str, Any]:
    """Score generation-only records from ``generate_condition`` on a CPU host.

    Merges ``score_response`` fields into each record and writes the exact
    schema ``evaluate_condition`` produces, so ``causal_gate_report`` and
    ``compare_deterministic_evaluations`` consume scored files unchanged.
    """
    if destination.exists():
        raise CausalError(f"refusing to overwrite evaluation: {destination}")
    by_id = {sample.sample_id: sample for sample in load_manifest(dataset_manifest)}
    records = _read_jsonl(generated_path)
    scored = []
    for record in records:
        sample = by_id.get(record["sample_id"])
        if sample is None:
            raise CausalError(f"generation record has no manifest sample: {record['sample_id']}")
        score = score_response(sample, record["response"], evaluator_image=evaluator_image)
        scored.append({**record, **score})
    _atomic_write_jsonl(destination, scored)
    return _summarize_condition(scored)


def evaluate_condition(
    model_path: Path,
    dataset_manifest: Path,
    destination: Path,
    config: ExperimentConfig,
    *,
    split: str,
    condition_id: str,
    evaluator_image: str,
    expert_manifest: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    model, tokenizer = load_donor(model_path, config)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CausalError(f"refusing to overwrite evaluation: {destination}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    temporary.unlink()  # reserve a unique name only; generate_condition writes atomically
    try:
        generate_condition(
            model,
            tokenizer,
            architecture,
            dataset_manifest,
            temporary,
            config,
            split=split,
            condition_id=condition_id,
            expert_manifest=expert_manifest,
            limit=limit,
        )
        return score_condition(
            temporary, dataset_manifest, destination, evaluator_image=evaluator_image
        )
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(destination: Path, records: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(destination)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    scoreable = [record for record in records if record["scoreable"]]
    return {
        "condition_id": records[0]["condition_id"] if records else "",
        "samples": len(records),
        "scoreable_fraction": len(scoreable) / len(records) if records else 0.0,
        "pass_rate": (
            float(np.mean([record["passed"] for record in scoreable])) if scoreable else 0.0
        ),
        "parse_error_rate": (1 - len(scoreable) / len(records) if records else 1.0),
        "truncation_rate": (
            float(np.mean([record["truncated"] for record in records])) if records else 0.0
        ),
        "mean_latency_seconds": (
            float(np.mean([record["latency_seconds"] for record in records])) if records else 0.0
        ),
    }


def _read_results(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["sample_id"]: row for row in rows if row.get("scoreable")}


def _domain_drop(baseline: dict[str, Any], intervention: dict[str, Any], domain: str) -> float:
    common = [
        key
        for key in baseline.keys() & intervention.keys()
        if baseline[key]["domain"] == domain
    ]
    if not common:
        raise CausalError(f"no paired scoreable {domain} samples")
    differences = [
        float(baseline[key]["passed"]) - float(intervention[key]["passed"])
        for key in common
    ]
    return float(np.mean(differences))


def _paired_differences(
    baseline: dict[str, Any], intervention: dict[str, Any], domain: str
) -> list[float]:
    common = sorted(
        key
        for key in baseline.keys() & intervention.keys()
        if baseline[key]["domain"] == domain
    )
    return [
        float(baseline[key]["passed"]) - float(intervention[key]["passed"])
        for key in common
    ]


def paired_bootstrap_interval(
    differences: list[float], *, iterations: int = 2000, seed: int = 20260903
) -> list[float]:
    if not differences:
        raise CausalError("cannot bootstrap an empty paired sample")
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [
        float(rng.choice(values, size=len(values), replace=True).mean())
        for _ in range(iterations)
    ]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def causal_gate_report(
    baseline_path: Path,
    selected_path: Path,
    random_paths: list[Path],
    *,
    replication_baseline_path: Path | None = None,
    replication_selected_path: Path | None = None,
) -> dict[str, Any]:
    if len(random_paths) < 20:
        raise CausalError("Gate D requires at least 20 layer-matched random sets")
    baseline, selected = _read_results(baseline_path), _read_results(selected_path)
    coding_drop = _domain_drop(baseline, selected, "coding")
    control_drop = _domain_drop(baseline, selected, "control")
    random_drops = [_domain_drop(baseline, _read_results(path), "coding") for path in random_paths]
    median_random = float(np.median(random_drops))
    percentile_95 = float(np.percentile(random_drops, 95))
    broad_collapse = control_drop >= 0.95
    if (replication_baseline_path is None) != (replication_selected_path is None):
        raise CausalError("both replication paths are required together")
    replication = None
    replication_direction_passed = False
    if replication_baseline_path and replication_selected_path:
        replication_baseline = _read_results(replication_baseline_path)
        replication_selected = _read_results(replication_selected_path)
        replication_coding_drop = _domain_drop(
            replication_baseline, replication_selected, "coding"
        )
        replication_control_drop = _domain_drop(
            replication_baseline, replication_selected, "control"
        )
        replication_direction_passed = (
            replication_coding_drop > 0
            and replication_coding_drop > replication_control_drop
        )
        replication = {
            "coding_drop": replication_coding_drop,
            "control_drop": replication_control_drop,
        }
    criteria = {
        "twice_random_median": coding_drop >= 2 * median_random,
        "at_or_above_random_p95": coding_drop >= percentile_95,
        "coding_specificity_2pp": coding_drop - control_drop >= 0.02,
        "replication_direction": replication_direction_passed,
        "no_broad_output_collapse": not broad_collapse,
    }
    passed = all(criteria.values())
    validation_passed = all(
        value for key, value in criteria.items() if key != "replication_direction"
    )
    if passed:
        label = "coding-critical-v0"
    elif validation_passed:
        # Validation criteria hold but replication is absent (validation-stage
        # run) or contradicted the direction. Either way the candidates are
        # NOT coding-critical-v0 yet; the label records the stronger state.
        label = "unreplicated-candidates"
    else:
        label = "observational-candidates"
    coding_differences = _paired_differences(baseline, selected, "coding")
    control_differences = _paired_differences(baseline, selected, "control")
    baseline_coding_scores = [
        float(item["passed"]) for item in baseline.values() if item["domain"] == "coding"
    ]
    baseline_coding_rate = float(np.mean(baseline_coding_scores))
    return {
        "gate": "D",
        "passed": passed,
        "validation_passed": validation_passed,
        "label": label,
        "coding_drop": coding_drop,
        "control_drop": control_drop,
        "random_median_coding_drop": median_random,
        "random_p95_coding_drop": percentile_95,
        "random_coding_drops": random_drops,
        "coding_drop_95ci": paired_bootstrap_interval(coding_differences),
        "control_drop_95ci": paired_bootstrap_interval(control_differences),
        "relative_coding_drop": (
            coding_drop / baseline_coding_rate if baseline_coding_rate > 0 else None
        ),
        "replication": replication,
        "criteria": criteria,
    }


def compare_generation_determinism(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Response-only determinism check usable before any scoring has happened.

    Early-stop pre-gate: verifies the GPU generation path reproduced identical
    responses across two identical baseline conditions. Complements (does not
    replace) the official scored Gate B check.
    """
    first = {row["sample_id"]: row for row in _read_jsonl(first_path)}
    second = {row["sample_id"]: row for row in _read_jsonl(second_path)}
    if first.keys() != second.keys():
        raise CausalError("determinism runs contain different sample IDs")
    mismatches = [
        sample_id
        for sample_id in sorted(first)
        if first[sample_id].get("response") != second[sample_id].get("response")
    ]
    return {
        "passed": not mismatches,
        "samples": len(first),
        "mismatched_sample_ids": mismatches,
    }


def compare_deterministic_evaluations(first_path: Path, second_path: Path) -> dict[str, Any]:
    first_rows = [json.loads(line) for line in first_path.read_text().splitlines() if line]
    second_rows = [json.loads(line) for line in second_path.read_text().splitlines() if line]
    first = {row["sample_id"]: row for row in first_rows}
    second = {row["sample_id"]: row for row in second_rows}
    if first.keys() != second.keys():
        raise CausalError("determinism runs contain different sample IDs")
    mismatches = [
        sample_id
        for sample_id in sorted(first)
        if first[sample_id].get("response") != second[sample_id].get("response")
        or first[sample_id].get("passed") != second[sample_id].get("passed")
    ]
    scoreable_fraction = (
        sum(bool(row.get("scoreable")) for row in first.values()) / len(first) if first else 0.0
    )
    return {
        "passed": not mismatches and scoreable_fraction >= 0.95,
        "samples": len(first),
        "scoreable_fraction": scoreable_fraction,
        "mismatched_sample_ids": mismatches,
    }
