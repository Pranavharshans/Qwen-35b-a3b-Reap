import json

import pytest

from reverse_reap.telemetry import TelemetryError, validate_telemetry


def rows(num_layers=2, top_k=2):
    result = []
    for token in range(3):
        for layer in range(num_layers):
            for rank in range(top_k):
                result.append(
                    {
                        "schema_version": 1,
                        "run_id": "run",
                        "sample_id": "sample",
                        "condition_id": "C0",
                        "segment": "prompt",
                        "token_index": token,
                        "token_id": token + 10,
                        "layer_index": layer,
                        "expert_index": (token + rank) % 4,
                        "route_rank": rank,
                        "router_weight": 0.5,
                        "expert_output_l2": 2.0,
                        "chunk_id": "chunk-0",
                    }
                )
    return result


def write(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


def test_validates_exact_tokens_layers_topk_identity(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    write(path, rows())
    report = validate_telemetry(path, num_layers=2, num_experts=4, top_k=2)
    assert report["routing_rows"] == 12
    assert report["analysed_tokens"] == 3


def test_rejects_duplicate_expert_within_token_layer(tmp_path):
    values = rows()
    values[1]["expert_index"] = values[0]["expert_index"]
    path = tmp_path / "telemetry.jsonl"
    write(path, values)
    with pytest.raises(TelemetryError, match="top-k unique"):
        validate_telemetry(path, num_layers=2, num_experts=4, top_k=2)


def test_rejects_nonfinite_values(tmp_path):
    values = rows()
    values[0]["expert_output_l2"] = float("nan")
    path = tmp_path / "telemetry.jsonl"
    write(path, values)
    with pytest.raises(TelemetryError, match="invalid numeric"):
        validate_telemetry(path, num_layers=2, num_experts=4, top_k=2)
