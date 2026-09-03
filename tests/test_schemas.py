import json


def test_checked_in_schema_files_exist_and_forbid_extra_fields():
    expected = {
        "experiment-config.schema.json",
        "run-state.schema.json",
        "routing-row.schema.json",
        "candidate-manifest.schema.json",
        "extraction-manifest.schema.json",
    }
    from pathlib import Path

    from reverse_reap.artifacts import CandidateManifest, ExtractionManifest, RoutingRow
    from reverse_reap.config import ExperimentConfig
    from reverse_reap.state import RunState

    schemas = {
        "experiment-config.schema.json": ExperimentConfig,
        "run-state.schema.json": RunState,
        "routing-row.schema.json": RoutingRow,
        "candidate-manifest.schema.json": CandidateManifest,
        "extraction-manifest.schema.json": ExtractionManifest,
    }

    assert set(schemas) == expected
    for filename, model in schemas.items():
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert json.loads((Path("schemas") / filename).read_text()) == schema
