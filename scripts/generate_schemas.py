#!/usr/bin/env python3
"""Regenerate checked-in JSON schemas from authoritative Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from reverse_reap.artifacts import CandidateManifest, ExtractionManifest, RoutingRow
from reverse_reap.bridge_benchmark import BridgeBenchmarkConfig
from reverse_reap.bridge_capture import (
    BridgeCaptureManifest,
    BridgeTargetBundle,
    BridgeTargetCaptureState,
    BridgeTargetRecord,
    BridgeTargetShard,
)
from reverse_reap.bridge_training import (
    BridgeCheckpoint,
    BridgeHostStateManifest,
    BridgeHostStateRecord,
    BridgeTrainingConfig,
    BridgeTrainingManifest,
    BridgeTrainingRow,
    BridgeUnusedRow,
)
from reverse_reap.config import ExperimentConfig
from reverse_reap.state import RunState

SCHEMAS = {
    "experiment-config.schema.json": ExperimentConfig,
    "run-state.schema.json": RunState,
    "routing-row.schema.json": RoutingRow,
    "candidate-manifest.schema.json": CandidateManifest,
    "extraction-manifest.schema.json": ExtractionManifest,
    "bridge-capture-manifest.schema.json": BridgeCaptureManifest,
    "bridge-target-record.schema.json": BridgeTargetRecord,
    "bridge-target-shard.schema.json": BridgeTargetShard,
    "bridge-target-bundle.schema.json": BridgeTargetBundle,
    "bridge-target-capture-state.schema.json": BridgeTargetCaptureState,
    "bridge-training-config.schema.json": BridgeTrainingConfig,
    "bridge-checkpoint.schema.json": BridgeCheckpoint,
    "bridge-host-state-manifest.schema.json": BridgeHostStateManifest,
    "bridge-host-state-record.schema.json": BridgeHostStateRecord,
    "bridge-training-manifest.schema.json": BridgeTrainingManifest,
    "bridge-training-row.schema.json": BridgeTrainingRow,
    "bridge-unused-row.schema.json": BridgeUnusedRow,
    "bridge-benchmark-config.schema.json": BridgeBenchmarkConfig,
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
