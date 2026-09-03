import json

import pytest

from reverse_reap.datasets import DatasetError, audit_samples, freeze_manifest, normalize_sample


def raw(source_id, prompt, domain="coding"):
    return {
        "source": "fixture",
        "source_revision": "abc123",
        "source_id": source_id,
        "domain": domain,
        "stratum": "synthesis" if domain == "coding" else "matched-technical",
        "language": "python" if domain == "coding" else None,
        "prompt": prompt,
        "reference": "answer",
        "scorer": "exact_match",
    }


def test_normalization_and_freeze_are_deterministic_and_immutable(tmp_path):
    samples = [
        normalize_sample(raw("a", "Write a function that adds two integers."), seed=7),
        normalize_sample(
            raw("b", "Explain the electrical resistance calculation.", "control"), seed=7
        ),
    ]
    destination = tmp_path / "manifest.jsonl"
    report = freeze_manifest(samples, destination)
    assert len(report["manifest_sha256"]) == 64
    assert freeze_manifest(list(reversed(samples)), destination) == report
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert {row["domain"] for row in rows} == {"coding", "control"}

    changed = normalize_sample(raw("c", "A genuinely different control prompt.", "control"), seed=7)
    with pytest.raises(DatasetError, match="refusing to overwrite"):
        freeze_manifest([*samples, changed], destination)


def test_cross_split_exact_duplicate_is_rejected():
    first = normalize_sample(raw("one", "Identical prompt body."), seed=1)
    second = normalize_sample(raw("two", "Identical prompt body."), seed=2)
    with pytest.raises(DatasetError, match="duplicate_content"):
        audit_samples([first, second])


def test_near_duplicate_is_rejected():
    first = normalize_sample(
        raw("one", "Implement a parser for comma separated integer values."), seed=1
    )
    second = normalize_sample(
        raw("two", "Implement a parser for comma separated integer values!"), seed=1
    )
    with pytest.raises(DatasetError, match="near_duplicates=1"):
        audit_samples([first, second])
