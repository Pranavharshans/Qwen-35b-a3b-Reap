import hashlib
import json

import pytest

from reverse_reap.bridge_capture import (
    AtomicTargetShardWriter,
    BridgeCaptureError,
    CoverageTracker,
    freeze_bridge_manifest,
    load_bridge_manifest,
    load_target_capture_state,
    target_event_id,
    validate_target_shard,
    write_target_capture_state,
    _tokenizer_ids,
)
from reverse_reap.datasets import freeze_manifest, normalize_sample


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        # Stable toy tokenization that works without torch/transformers.
        text = "|".join(message["content"] for message in messages)
        return list(range(1, len(text) + 1))


def _sample(source_id: str, domain: str):
    return normalize_sample(
        {
            "source": "fixture",
            "source_revision": "rev1",
            "source_id": source_id,
            "domain": domain,
            "stratum": "synthesis" if domain == "coding" else "general",
            "language": "python" if domain == "coding" else None,
            "prompt": f"prompt for {source_id}",
            "reference": "pass",
            "scorer": "exact_match",
        },
        seed=7,
    )


def test_bridge_manifest_freeze_is_deterministic_and_hash_checked(tmp_path):
    source = tmp_path / "source.jsonl"
    freeze_manifest([_sample("a", "coding"), _sample("b", "control")], source)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "gate_passed": True,
                "experts": [{"layer": 3, "expert": 26}],
            }
        )
    )
    destination = tmp_path / "bridge.json"
    payload = freeze_bridge_manifest(
        source,
        _Tokenizer(),
        destination,
        model_revision="a" * 40,
        tokenizer_fingerprint_value="tokenizer-fixture",
        config_sha256="c" * 64,
        candidate_manifest=candidate,
        run_id="run-1",
        target_tokens=1,
        hard_token_ceiling=20,
        max_input_tokens=100,
        min_coding_events_per_expert=2,
        min_control_events_per_expert=1,
        experts=[(3, 26)],
        allowed_splits=("calibration", "selection", "validation", "replication"),
    )
    assert payload["kind"] == "bridge-target-capture"
    assert payload["target_reached"] is True
    loaded = load_bridge_manifest(destination)
    assert loaded.manifest_sha256 == payload["manifest_sha256"]
    destination.write_text(destination.read_text().replace("tokenizer-fixture", "changed"))
    with pytest.raises(BridgeCaptureError, match="hash mismatch"):
        load_bridge_manifest(destination)


def test_coverage_stops_only_after_both_domain_minima_or_hard_ceiling():
    tracker = CoverageTracker(
        frozenset({(3, 26)}),
        min_coding_events=2,
        min_control_events=1,
        target_tokens=2,
        hard_token_ceiling=10,
    )
    tracker.add_tokens(2)
    tracker.add_event("coding", 3, 26)
    tracker.add_event("coding", 3, 26)
    assert tracker.should_stop is False
    tracker.add_event("control", 3, 26)
    assert tracker.should_stop is True
    assert tracker.stop_reason == "target-and-minimum-coverage"


def test_coverage_reports_hard_ceiling_when_domain_coverage_is_incomplete():
    tracker = CoverageTracker(
        frozenset({(3, 26)}),
        min_coding_events=2,
        min_control_events=1,
        target_tokens=2,
        hard_token_ceiling=3,
    )
    tracker.add_tokens(3)
    assert tracker.should_stop is True
    assert tracker.stop_reason == "hard-token-ceiling"
    assert tracker.report()["minimum_coverage_reached"] is False


def test_tokenizer_ids_accepts_mapping_batch_encoding_and_sequences():
    """transformers>=5 slow tokenizers return BatchEncoding (UserDict, not dict)."""
    from collections import UserDict

    class _FakeTokenizer:
        def __init__(self, payload):
            self._payload = payload

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs.get("enable_thinking") is False
            return self._payload

    expected = [1, 2, 3]
    assert (
        _tokenizer_ids(_FakeTokenizer({"input_ids": expected}), [], enable_thinking=False)
        == expected
    )
    assert (
        _tokenizer_ids(
            _FakeTokenizer(UserDict({"input_ids": expected})), [], enable_thinking=False
        )
        == expected
    )
    assert _tokenizer_ids(_FakeTokenizer([[7, 8]]), [], enable_thinking=False) == [7, 8]
    assert _tokenizer_ids(_FakeTokenizer([9]), [], enable_thinking=False) == [9]


def test_capture_checkpoint_is_hash_bound_and_resumable(tmp_path):
    tracker = CoverageTracker(
        frozenset({(3, 26)}),
        min_coding_events=2,
        min_control_events=1,
        target_tokens=4,
        hard_token_ceiling=10,
    )
    tracker.add_tokens(4)
    tracker.add_event("coding", 3, 26)
    checkpoint_path = tmp_path / "capture-state.json"
    write_target_capture_state(
        checkpoint_path,
        {
            "run_id": "run-1",
            "capture_manifest_sha256": "a" * 64,
            "next_sample_index": 2,
            "next_batch_number": 1,
            "completed_shards": ["shard-000000"],
            "coverage": tracker.report(),
            "complete": False,
        },
    )
    loaded = load_target_capture_state(
        checkpoint_path,
        run_id="run-1",
        capture_manifest_sha256="a" * 64,
    )
    assert loaded is not None
    assert loaded.next_sample_index == 2
    restored = CoverageTracker(
        frozenset({(3, 26)}),
        min_coding_events=2,
        min_control_events=1,
        target_tokens=4,
        hard_token_ceiling=10,
    )
    restored.restore(loaded.coverage)
    assert restored.report() == tracker.report()
    write_target_capture_state(
        checkpoint_path,
        {
            "run_id": "run-1",
            "capture_manifest_sha256": "a" * 64,
            "next_sample_index": 3,
            "next_batch_number": 2,
            "completed_shards": ["shard-000000", "shard-000001"],
            "coverage": tracker.report(),
            "complete": False,
        },
    )
    assert load_target_capture_state(
        checkpoint_path,
        run_id="run-1",
        capture_manifest_sha256="a" * 64,
    ).next_sample_index == 3
    checkpoint_path.write_text(
        checkpoint_path.read_text().replace('"next_sample_index": 3', '"next_sample_index": 4')
    )
    with pytest.raises(BridgeCaptureError, match="checkpoint hash mismatch"):
        load_target_capture_state(
            checkpoint_path,
            run_id="run-1",
            capture_manifest_sha256="a" * 64,
        )


def test_atomic_bf16_shard_roundtrip_and_corruption_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    root = tmp_path / "capture"
    writer = AtomicTargetShardWriter(
        root,
        run_id="run-1",
        shard_id="000000",
        source_manifest_hash="b" * 64,
        model_revision="a" * 40,
        tokenizer_fingerprint_value="tok",
        config_sha256="c" * 64,
        candidate_manifest_sha256="d" * 64,
        target_experts=frozenset({(3, 26)}),
        hidden_size=4,
        max_records=2,
    )
    vector = torch.arange(4, dtype=torch.bfloat16)
    record = {
        "schema_version": 1,
        "run_id": "run-1",
        "row_index": 0,
        "event_id": "0" * 64,
        "sample_ordinal": 0,
        "sample_id": "sample-1",
        "sample_content_sha256": "e" * 64,
        "domain": "coding",
        "split": "calibration",
        "token_position": 0,
        "token_id": 11,
        "donor_layer": 3,
        "expert_id": 26,
        "route_rank": 0,
        "router_weight": 0.5,
        "source_manifest_hash": "b" * 64,
        "model_revision": "a" * 40,
        "tokenizer_fingerprint": "tok",
        "config_sha256": "c" * 64,
        "candidate_manifest_sha256": "d" * 64,
        "condition_id": "C0",
        "chunk_id": "batch-0",
    }
    record["event_id"] = target_event_id("sample-1", 0, 3, 26, 0)
    writer.append(record, vector, vector + 1, vector + 2)
    shard = writer.finalize()
    assert validate_target_shard(shard, hidden_size=4)["valid"] is True
    checksums = json.loads((shard / "shard.json").read_text())
    checksums["checksums"]["records.jsonl"] = hashlib.sha256(b"wrong").hexdigest()
    (shard / "shard.json").write_text(json.dumps(checksums))
    with pytest.raises(BridgeCaptureError, match="metadata hash mismatch|checksum mismatch"):
        validate_target_shard(shard, hidden_size=4)


def test_atomic_shard_refuses_overwrite(tmp_path):
    root = tmp_path / "capture"
    (root / "shard-000000").mkdir(parents=True)
    with pytest.raises(BridgeCaptureError, match="refusing to overwrite"):
        AtomicTargetShardWriter(
            root,
            run_id="run-1",
            shard_id="000000",
            source_manifest_hash="b" * 64,
            model_revision="a" * 40,
            tokenizer_fingerprint_value="tok",
            config_sha256="c" * 64,
            candidate_manifest_sha256="d" * 64,
            target_experts=frozenset({(3, 26)}),
            hidden_size=4,
        )
