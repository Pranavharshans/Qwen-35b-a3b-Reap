#!/usr/bin/env python3
"""Regenerate checked-in JSON schemas from authoritative Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from reverse_reap.artifacts import CandidateManifest, ExtractionManifest, RoutingRow
from reverse_reap.config import ExperimentConfig
from reverse_reap.state import RunState

SCHEMAS = {
    "experiment-config.schema.json": ExperimentConfig,
    "run-state.schema.json": RunState,
    "routing-row.schema.json": RoutingRow,
    "candidate-manifest.schema.json": CandidateManifest,
    "extraction-manifest.schema.json": ExtractionManifest,
}


def main() -> None:
    destination = Path("schemas")
    destination.mkdir(exist_ok=True)
    for filename, model in SCHEMAS.items():
        payload = model.model_json_schema()
        (destination / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

