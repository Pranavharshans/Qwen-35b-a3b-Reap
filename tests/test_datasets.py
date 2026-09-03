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
    rebalance_controls_by_length,
    token_length_report,
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


def test_length_rebalance_preserves_coding_and_prefers_long_controls(tmp_path):
    """Regression: smoke-tier p90 gate failed 3.24 > 3.0 on SWE-bench tails.

    Rebalancing must keep every coding row byte-identical, reselect controls
    longest-first from the pool, record discarded/added IDs, and never touch
    the original manifests.
    """

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    samples = []
    samples.append(normalize_sample(raw("code-short", "a b", "coding"), seed=1))
    samples.append(
        normalize_sample(raw("code-long", " ".join(["w"] * 60), "coding"), seed=1)
    )
    for index in range(6):
        samples.append(
            normalize_sample(
                raw(f"ctrl-{index}", " ".join(["x"] * (index + 1)), "control"), seed=1
            )
        )
    full = tmp_path / "source-full.jsonl"
    freeze_manifest(samples, full)
    tiers_dir = tmp_path / "tiers"
    tiers_dir.mkdir()
    # Hand-built smoke tier: 2 coding + 2 shortest controls (fails a p90-style gate).
    smoke = [samples[0], samples[1], samples[2], samples[3]]
    freeze_manifest(smoke, tiers_dir / "smoke.jsonl")
    for tier in ("pilot", "medium", "full"):
        freeze_manifest(samples, tiers_dir / f"{tier}.jsonl")
    report = rebalance_controls_by_length(full, tiers_dir, Tokenizer(), seed=5)
    rebalanced = {
        item.sample_id: item for item in load_manifest(tiers_dir / "smoke-lengthmatched.jsonl")
    }
    for item in smoke:
        if item.domain == "coding":
            assert rebalanced[item.sample_id].prompt == item.prompt
    control_lengths = sorted(
        len(item.prompt.split()) for item in rebalanced.values() if item.domain == "control"
    )
    assert control_lengths == [5, 6]
    assert report["tiers"]["smoke"]["discarded_control_ids"]
    assert report["tiers"]["smoke"]["control_lengths_before"] == [1, 2]
    assert report["tiers"]["smoke"]["control_lengths_after"] == [5, 6]
    # Original manifest untouched: refreeze the same rows and expect identity.
    assert freeze_manifest(smoke, tiers_dir / "smoke.jsonl")["manifest_sha256"]


def test_token_length_report_uses_exact_tokenizer_and_rejects_gross_mismatch():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": text.split()}

    coding = normalize_sample(raw("c", "one two three four", "coding"), seed=1)
    control = normalize_sample(raw("g", "one two three", "control"), seed=1)
    report = token_length_report([coding, control], Tokenizer(), max_input_tokens=8)
    assert report["passed"]
    long = normalize_sample(raw("long", " ".join(["x"] * 40), "coding"), seed=1)
    report = token_length_report([long, control], Tokenizer(), max_input_tokens=100)
    assert not report["passed"]
