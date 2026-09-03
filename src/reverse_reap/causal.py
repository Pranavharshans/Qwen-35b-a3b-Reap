"""Deterministic masked generation, scoring, causal controls, and Gate D reporting."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from reverse_reap.config import ExperimentConfig
from reverse_reap.datasets import NormalizedSample, load_manifest
from reverse_reap.evaluator import EvaluationResult, evaluate_java, evaluate_python
from reverse_reap.instrumentation import instrument_qwen35
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
) -> str:
    import torch

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=config.runtime.enable_thinking,
    )
    ids = ids.to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=config.runtime.max_new_tokens,
            use_cache=config.runtime.use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True)


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
    masked = load_expert_set(expert_manifest) if expert_manifest else frozenset()
    samples = [sample for sample in load_manifest(dataset_manifest) if sample.split == split]
    if limit is not None:
        samples = samples[:limit]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CausalError(f"refusing to overwrite evaluation: {destination}")
    records = []
    for sample in samples:
        if masked:
            with instrument_qwen35(architecture, masked=masked):
                response = _generate(model, tokenizer, sample, config)
        else:
            response = _generate(model, tokenizer, sample, config)
        score = score_response(sample, response, evaluator_image=evaluator_image)
        records.append(
            {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "stratum": sample.stratum,
                "split": sample.split,
                "condition_id": condition_id,
                "masked_experts": len(masked),
                "response": response,
                **score,
            }
        )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(destination)
    scoreable = [record for record in records if record["scoreable"]]
    return {
        "condition_id": condition_id,
        "samples": len(records),
        "scoreable_fraction": len(scoreable) / len(records) if records else 0.0,
        "pass_rate": (
            float(np.mean([record["passed"] for record in scoreable])) if scoreable else 0.0
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


def causal_gate_report(
    baseline_path: Path,
    selected_path: Path,
    random_paths: list[Path],
    *,
    replication_direction_passed: bool,
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
    criteria = {
        "twice_random_median": coding_drop >= 2 * median_random,
        "at_or_above_random_p95": coding_drop >= percentile_95,
        "coding_specificity_2pp": coding_drop - control_drop >= 0.02,
        "replication_direction": replication_direction_passed,
        "no_broad_output_collapse": not broad_collapse,
    }
    passed = all(criteria.values())
    return {
        "gate": "D",
        "passed": passed,
        "label": "coding-critical-v0" if passed else "observational-candidates",
        "coding_drop": coding_drop,
        "control_drop": control_drop,
        "random_median_coding_drop": median_random,
        "random_p95_coding_drop": percentile_95,
        "random_coding_drops": random_drops,
        "criteria": criteria,
    }
