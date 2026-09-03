"""Deterministic dataset normalization, leakage audit, splitting, and freezing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from reverse_reap.config import StrictModel


class DatasetError(ValueError):
    """Raised when a dataset cannot safely be frozen."""


class NormalizedSample(StrictModel):
    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    domain: Literal["coding", "control"]
    stratum: str = Field(min_length=1)
    language: str | None = None
    prompt: str = Field(min_length=1)
    reference: str | None = None
    scorer: Literal["exact_match", "unit_tests", "multiple_choice", "swebench"]
    tests: str | None = None
    entry_point: str | None = None
    timeout_seconds: int = Field(default=10, ge=1, le=120)
    split: Literal["calibration", "selection", "validation", "replication"]
    prompt_template_version: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_code_language(self) -> NormalizedSample:
        if self.domain == "coding" and not self.language:
            raise ValueError("coding samples require a programming language")
        if self.scorer == "unit_tests" and not self.tests:
            raise ValueError("unit-test samples require tests")
        return self


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def assign_split(source: str, source_id: str, seed: int) -> str:
    bucket = int(sha256(f"{seed}\0{source}\0{source_id}".encode())[:8], 16) % 100
    if bucket < 40:
        return "calibration"
    if bucket < 60:
        return "selection"
    if bucket < 80:
        return "validation"
    return "replication"


def normalize_sample(raw: dict[str, Any], *, seed: int) -> NormalizedSample:
    required = ("source", "source_revision", "source_id", "domain", "stratum", "prompt", "scorer")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise DatasetError(f"raw sample missing required fields: {', '.join(missing)}")
    identity = f"{raw['source']}:{raw['source_id']}"
    content = {
        "prompt": str(raw["prompt"]).strip(),
        "reference": raw.get("reference"),
        "tests": raw.get("tests"),
    }
    return NormalizedSample(
        sample_id=sha256(identity.encode())[:24],
        source=str(raw["source"]),
        source_revision=str(raw["source_revision"]),
        source_id=str(raw["source_id"]),
        domain=raw["domain"],
        stratum=str(raw["stratum"]),
        language=raw.get("language"),
        prompt=content["prompt"],
        reference=content["reference"],
        tests=content["tests"],
        entry_point=raw.get("entry_point"),
        scorer=raw["scorer"],
        timeout_seconds=int(raw.get("timeout_seconds", 10)),
        split=assign_split(str(raw["source"]), str(raw["source_id"]), seed),
        prompt_template_version=str(raw.get("prompt_template_version", "v1")),
        content_sha256=sha256(canonical_json(content)),
    )


def _lexical_fingerprint(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if len(normalized) < 25:
        return {normalized}
    return {normalized[index : index + 25] for index in range(len(normalized) - 24)}


def _near_duplicate_candidates(
    fingerprints: list[set[str]], *, bottom_k: int = 16
) -> set[tuple[int, int]]:
    """High-recall deterministic MinHash blocking before exact Jaccard.

    For Jaccard 0.92, missing all 16 independent bottom-hash opportunities has
    negligible probability, while avoiding quadratic comparisons of unrelated prompts.
    """
    buckets: dict[int, list[int]] = {}
    for index, fingerprint in enumerate(fingerprints):
        hashes = sorted(
            int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest())
            for shingle in fingerprint
        )[:bottom_k]
        for value in hashes:
            buckets.setdefault(value, []).append(index)
    candidates = set()
    for members in buckets.values():
        for left_offset, left in enumerate(members):
            for right in members[left_offset + 1 :]:
                candidates.add((left, right))
    return candidates


def audit_samples(
    samples: list[NormalizedSample], *, near_duplicate_threshold: float = 0.92
) -> dict[str, Any]:
    if not samples:
        raise DatasetError("cannot freeze an empty dataset")
    ids = [sample.sample_id for sample in samples]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    content = [sample.content_sha256 for sample in samples]
    duplicate_content = [key for key, count in Counter(content).items() if count > 1]
    near_duplicates: list[dict[str, Any]] = []
    fingerprints = [_lexical_fingerprint(sample.prompt) for sample in samples]
    candidate_pairs = _near_duplicate_candidates(fingerprints)
    for left, right in sorted(candidate_pairs):
        union = fingerprints[left] | fingerprints[right]
        similarity = (
            len(fingerprints[left] & fingerprints[right]) / len(union) if union else 1.0
        )
        if similarity >= near_duplicate_threshold:
            near_duplicates.append(
                {
                    "left": ids[left],
                    "left_source": f"{samples[left].source}:{samples[left].source_id}",
                    "right": ids[right],
                    "right_source": f"{samples[right].source}:{samples[right].source_id}",
                    "jaccard": similarity,
                }
            )
    if duplicate_ids or duplicate_content or near_duplicates:
        raise DatasetError(
            f"leakage audit failed: duplicate_ids={len(duplicate_ids)}, "
            f"duplicate_content={len(duplicate_content)}, near_duplicates={len(near_duplicates)}; "
            f"examples={near_duplicates[:5]}"
        )
    split_counts = Counter(sample.split for sample in samples)
    domain_counts = Counter(sample.domain for sample in samples)
    return {
        "samples": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_method": "bottom-16-blake2b-blocking-plus-exact-25gram-jaccard",
        "near_duplicate_candidate_pairs": len(candidate_pairs),
        "leakage_pairs": 0,
    }


def freeze_manifest(samples: list[NormalizedSample], destination: Path) -> dict[str, Any]:
    audit = audit_samples(samples)
    ordered = sorted(samples, key=lambda item: (item.domain, item.source, item.source_id))
    body = b"".join(canonical_json(sample.model_dump(mode="json")) + b"\n" for sample in ordered)
    digest = sha256(body)
    if destination.exists():
        existing = destination.read_bytes()
        if existing != body:
            raise DatasetError(f"refusing to overwrite frozen manifest: {destination}")
        return {**audit, "manifest_sha256": digest, "path": str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {**audit, "manifest_sha256": digest, "path": str(destination)}


def load_manifest(path: Path) -> list[NormalizedSample]:
    return [
        NormalizedSample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def freeze_tiers(full_manifest: Path, destination_dir: Path) -> dict[str, Any]:
    """Create deterministic, nested cost-control tiers from one audited full manifest."""
    samples = load_manifest(full_manifest)
    audit_samples(samples)
    # Per source/split caps retain source and held-out split coverage at every tier.
    caps: dict[str, int | None] = {"smoke": 2, "pilot": 8, "medium": 32, "full": None}
    ordered = sorted(
        samples,
        key=lambda item: (
            item.source,
            item.split,
            sha256(f"tier-v1\0{item.sample_id}".encode()),
        ),
    )
    groups: dict[tuple[str, str], list[NormalizedSample]] = {}
    for sample in ordered:
        groups.setdefault((sample.source, sample.split), []).append(sample)
    destination_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    previous_ids: set[str] = set()
    for tier, cap in caps.items():
        selected = [
            sample
            for group in groups.values()
            for sample in (group if cap is None else group[:cap])
        ]
        selected_ids = {sample.sample_id for sample in selected}
        if not previous_ids <= selected_ids:
            raise DatasetError(f"tier {tier} is not a superset of the previous tier")
        path = destination_dir / f"{tier}.jsonl"
        reports[tier] = freeze_manifest(selected, path)
        previous_ids = selected_ids
    return {"schema_version": 1, "tiers": reports}
