"""Pinned Hugging Face dataset acquisition and source-specific normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from reverse_reap.config import StrictModel
from reverse_reap.datasets import NormalizedSample, freeze_manifest, normalize_sample


class SourceError(RuntimeError):
    """Raised when a source revision or schema cannot be reproduced."""


class SourceDefinition(StrictModel):
    name: str
    dataset_id: str
    config: str | None = None
    split: str
    revision: str | None = None
    adapter: Literal["humaneval", "mbpp", "humaneval_x", "cruxeval", "swebench", "gsm8k"]
    domain: Literal["coding", "control"]
    stratum: str
    license: str
    citation: str
    language: str | None = None
    limit: int | None = Field(default=None, ge=1)


class SourceCatalog(StrictModel):
    schema_version: Literal[1]
    seed: int
    sources: list[SourceDefinition]


def load_catalog(path: Path) -> SourceCatalog:
    return SourceCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _adapt(
    source: SourceDefinition, revision: str, row: dict[str, Any], index: int
) -> dict[str, Any]:
    common = {
        "source": source.dataset_id,
        "source_revision": revision,
        "source_id": str(row.get("task_id", row.get("id", row.get("instance_id", index)))),
        "domain": source.domain,
        "stratum": source.stratum,
        "language": source.language,
        "prompt_template_version": "source-v1",
    }
    if source.adapter == "humaneval":
        return {
            **common,
            "prompt": row["prompt"],
            "reference": row["canonical_solution"],
            "tests": row["test"] + f"\ncheck({row['entry_point']})\n",
            "entry_point": row["entry_point"],
            "scorer": "unit_tests",
        }
    if source.adapter == "mbpp":
        tests = "\n".join(row["test_list"])
        return {
            **common,
            "prompt": row["text"],
            "reference": row["code"],
            "tests": tests,
            "scorer": "unit_tests",
        }
    if source.adapter == "humaneval_x":
        return {
            **common,
            "prompt": row.get("prompt", row.get("declaration")),
            "reference": row.get("canonical_solution", row.get("canonical_solution")),
            "tests": row.get("test", row.get("test_code")),
            "entry_point": row.get("entry_point"),
            "language": source.language or row.get("language"),
            "scorer": "unit_tests",
        }
    if source.adapter == "cruxeval":
        code = row.get("code", row.get("function"))
        prompt = (
            f"Given this Python code:\n{code}\nInput: {row['input']}\nPredict the exact output."
        )
        return {
            **common,
            "prompt": prompt,
            "reference": str(row["output"]),
            "scorer": "exact_match",
        }
    if source.adapter == "swebench":
        prompt = f"Repository: {row['repo']}\nIssue:\n{row['problem_statement']}"
        return {
            **common,
            "prompt": prompt,
            "reference": row["patch"],
            "language": source.language or "mixed",
            "scorer": "swebench",
            "timeout_seconds": 120,
        }
    if source.adapter == "gsm8k":
        return {
            **common,
            "prompt": row["question"],
            "reference": row["answer"],
            "scorer": "exact_match",
        }
    raise SourceError(f"unsupported source adapter: {source.adapter}")


def fetch_and_freeze(catalog_path: Path, destination: Path) -> dict[str, Any]:
    """Resolve immutable Hub SHAs first, then fetch and freeze normalized records."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    catalog = load_catalog(catalog_path)
    api = HfApi()
    samples: list[NormalizedSample] = []
    resolved_sources = []
    for source in catalog.sources:
        info = api.dataset_info(source.dataset_id, revision=source.revision)
        revision = info.sha
        if source.revision is not None and revision != source.revision:
            raise SourceError(
                f"resolved revision differs for {source.dataset_id}: "
                f"{revision} != {source.revision}"
            )
        dataset = load_dataset(
            source.dataset_id,
            source.config,
            split=source.split,
            revision=revision,
            trust_remote_code=False,
        )
        limit = min(source.limit or len(dataset), len(dataset))
        for index in range(limit):
            adapted = _adapt(source, revision, dataset[index], index)
            samples.append(normalize_sample(adapted, seed=catalog.seed))
        resolved_sources.append(
            {
                "name": source.name,
                "dataset_id": source.dataset_id,
                "revision": revision,
                "split": source.split,
                "license": source.license,
                "citation": source.citation,
                "records": limit,
            }
        )
    report = freeze_manifest(samples, destination)
    source_path = destination.with_suffix(".sources.json")
    source_path.write_text(
        json.dumps({"sources": resolved_sources}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "source_manifest": str(source_path)}
