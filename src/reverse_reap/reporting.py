"""Auditable run-bundle inventory and conservative terminal classification."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reverse_reap.config import ExperimentConfig
from reverse_reap.controller import run_status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_run_bundle(
    run_dir: Path, state_dir: Path, config: ExperimentConfig, destination: Path
) -> dict[str, Any]:
    probe = _optional_json(run_dir / "probe.json")
    candidate = _optional_json(run_dir / "analysis" / "candidate-manifest.json")
    causal = _optional_json(run_dir / "causal-report.json")
    determinism = _optional_json(run_dir / "gates" / "determinism-report.json") or _optional_json(
        run_dir / "determinism-report.json"
    )
    extraction = _optional_json(run_dir / "extraction" / "extraction-manifest.json")
    if causal and causal.get("passed"):
        classification = "positive"
        evidence_label = "coding-critical-v0"
    elif determinism is not None and not determinism.get("passed"):
        classification = "feasibility-failure"
        evidence_label = "nondeterministic-generation"
    elif causal:
        classification = "null"
        evidence_label = causal.get("label") or "observational-candidates"
    elif candidate:
        classification = "null" if not candidate.get("gate_passed") else "incomplete"
        evidence_label = "observational-candidates"
    else:
        classification = (
            "feasibility-failure" if probe and not probe.get("passed") else "incomplete"
        )
        evidence_label = "none"
    files = []
    destination_resolved = destination.resolve()
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.resolve() == destination_resolved:
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "classification": classification,
        "evidence_label": evidence_label,
        "model": config.model.model_dump(mode="json"),
        "condition": {
            "thinking_enabled": config.runtime.enable_thinking,
            "execution_precision": config.model.execution_precision,
        },
        "config_sha256": config.fingerprint(),
        "controller": run_status(state_dir),
        "gates": {
            "A_instrumentation": bool(probe and probe.get("passed")),
            "B_determinism": bool(determinism and determinism.get("passed")),
            "C_candidates": bool(candidate and candidate.get("gate_passed")),
            "D_causal": bool(causal and causal.get("passed")),
            "E_extraction": bool(
                extraction
                and extraction.get("tensors")
                and all(item.get("verified") for item in extraction["tensors"])
            ),
        },
        "artifacts": files,
        "limitations": [
            "Extracted experts are not a standalone model.",
            "Routing and REAP saliency are observational unless Gate D passes.",
            "Thinking-enabled and thinking-disabled results are not pooled.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload
