import json

import pytest

from reverse_reap.datasets import (
    DatasetError,
    _lexical_fingerprint,
    _near_duplicate_candidates,
    audit_samples,
    balanced_subset,
    freeze_manifest,
    freeze_tiers,
    load_manifest,
    normalize_sample,
)


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


def test_blocking_nominates_near_duplicate_without_all_pairs_scan():
    texts = [
        "Implement a parser for comma separated integer values.",
        "Implement a parser for comma separated integer values!",
        "Explain photosynthesis in a concise paragraph.",
    ]
    candidates = _near_duplicate_candidates([_lexical_fingerprint(text) for text in texts])
    assert (0, 1) in candidates


def test_frozen_tiers_are_nested_and_reproducible(tmp_path):
    samples = []
    for domain in ("coding", "control"):
        for index in range(40):
            samples.append(
                normalize_sample(
                    raw(f"{domain}-{index}", f"unique prompt {domain} {index}", domain),
                    seed=11,
                )
            )
    full = tmp_path / "source-full.jsonl"
    freeze_manifest(samples, full)
    report = freeze_tiers(full, tmp_path / "tiers")
    tier_ids = {}
    for tier in ("smoke", "pilot", "medium", "full"):
        path = tmp_path / "tiers" / f"{tier}.jsonl"
        tier_ids[tier] = {item.sample_id for item in load_manifest(path)}
        assert report["tiers"][tier]["manifest_sha256"]
    assert tier_ids["smoke"] <= tier_ids["pilot"] <= tier_ids["medium"] <= tier_ids["full"]


def test_limited_subset_balances_domains_and_rotates_strata():
    samples = []
    for domain in ("coding", "control"):
        for stratum in ("a", "b"):
            for index in range(5):
                value = raw(f"{domain}-{stratum}-{index}", f"{domain} {stratum} {index}", domain)
                value["stratum"] = stratum
                samples.append(normalize_sample(value, seed=11))
    selected = balanced_subset(samples, 8)
    assert [item.domain for item in selected] == ["coding", "control"] * 4
    assert {item.stratum for item in selected} == {"a", "b"}
