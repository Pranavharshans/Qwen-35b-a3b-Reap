"""Strict schemas for evidence artifacts crossing experiment stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpertIdentity(ArtifactModel):
    layer: int = Field(ge=0)
    expert: int = Field(ge=0)


class CandidateExpert(ExpertIdentity):
    differential: float


class CandidateManifest(ArtifactModel):
    schema_version: Literal[1]
    status: Literal["domain-differential candidate"]
    selection_method: str = Field(min_length=1)
    top_n: int = Field(gt=0)
    thresholds: dict[str, float]
    gate_passed: bool
    experts: list[CandidateExpert]
    source_hashes: dict[str, str]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RoutingRow(ArtifactModel):
    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    condition_id: str = Field(min_length=1)
    split: Literal["calibration", "selection", "validation", "replication"]
    domain: Literal["coding", "control"]
    stratum: str = Field(min_length=1)
    segment: Literal["prompt", "reference", "generated"]
    token_index: int = Field(ge=0)
    token_id: int = Field(ge=0)
    layer_index: int = Field(ge=0)
    expert_index: int = Field(ge=0)
    route_rank: int = Field(ge=0)
    router_weight: float = Field(ge=0, allow_inf_nan=False)
    expert_output_l2: float = Field(ge=0, allow_inf_nan=False)
    chunk_id: str = Field(min_length=1)


class ExtractedTensorRecord(ArtifactModel):
    source_shard: str = Field(min_length=1)
    source_key: str = Field(min_length=1)
    output_key: str = Field(min_length=1)
    shape: list[int]
    dtype: str = Field(min_length=1)
    nbytes: int = Field(gt=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified: Literal[True]


class ExtractionManifest(ArtifactModel):
    schema_version: Literal[1]
    label: Literal["extracted"]
    run_id: str = Field(min_length=1)
    source_model_id: Literal["Qwen/Qwen3.5-35B-A3B"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    source_weight_index_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_status: Literal[
        "observational-candidates", "unreplicated-candidates", "coding-critical-v0"
    ]
    selection_metrics: dict
    causal_metrics: dict
    tool_git_revisions: dict[str, str]
    created_at_utc: str
    experts: list[ExpertIdentity]
    tensors: list[ExtractedTensorRecord]
    tensor_file: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_parameter_bytes: int = Field(gt=0)

