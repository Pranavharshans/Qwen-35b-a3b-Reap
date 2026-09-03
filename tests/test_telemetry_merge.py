import json

import pytest

from reverse_reap.telemetry import TelemetryError, merge_telemetry


def row(sample, split):
    return {
        "sample_id": sample,
        "split": split,
        "token_index": 0,
        "layer_index": 0,
        "route_rank": 0,
    }


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def test_merge_allows_only_candidate_development_splits(tmp_path):
    calibration, selection = tmp_path / "a", tmp_path / "b"
    write(calibration, row("a", "calibration"))
    write(selection, row("b", "selection"))
    output = tmp_path / "merged"
    report = merge_telemetry([calibration, selection], output)
    assert report["splits"] == ["calibration", "selection"]
    assert report["routing_rows"] == 2


def test_merge_rejects_held_out_split(tmp_path):
    calibration, replication = tmp_path / "a", tmp_path / "b"
    write(calibration, row("a", "calibration"))
    write(replication, row("b", "replication"))
    with pytest.raises(TelemetryError, match="leaks held-out"):
        merge_telemetry([calibration, replication], tmp_path / "merged")
