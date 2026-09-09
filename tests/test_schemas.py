import json


def test_checked_in_schema_files_exist_and_forbid_extra_fields():
    expected = {
        "experiment-config.schema.json",
        "run-state.schema.json",
        "routing-row.schema.json",
        "candidate-manifest.schema.json",
        "extraction-manifest.schema.json",
        "bridge-capture-manifest.schema.json",
        "bridge-target-record.schema.json",
        "bridge-target-shard.schema.json",
        "bridge-target-bundle.schema.json",
        "bridge-target-capture-state.schema.json",
        "bridge-training-config.schema.json",
        "bridge-checkpoint.schema.json",
        "bridge-host-state-manifest.schema.json",
        "bridge-host-state-record.schema.json",
        "bridge-training-manifest.schema.json",
        "bridge-training-row.schema.json",
        "bridge-unused-row.schema.json",
    }
    from pathlib import Path

    from reverse_reap.artifacts import CandidateManifest, ExtractionManifest, RoutingRow
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

    schemas = {
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
    }

    assert set(schemas) == expected
    for filename, model in schemas.items():
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert json.loads((Path("schemas") / filename).read_text()) == schema
