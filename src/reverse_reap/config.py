"""Validated experiment configuration and immutable run identity."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    id: Literal["Qwen/Qwen3.5-35B-A3B"]
    revision: str = Field(min_length=40, max_length=64, pattern=r"^[0-9a-f]+$")
    source_precision: Literal["bf16"] = "bf16"
    execution_precision: Literal["bf16", "fp16"]
    text_only: Literal[True] = True


class RuntimeConfig(StrictModel):
    seed: int = Field(ge=0)
    deterministic: bool = True
    batch_size: int = Field(default=1, ge=1)
    max_input_tokens: int = Field(ge=64)
    max_new_tokens: int = Field(ge=0)
    enable_thinking: bool = False
    use_cache: bool = False
    speculative_decoding: Literal[False] = False

    @model_validator(mode="after")
    def validate_low_cost_primary(self) -> RuntimeConfig:
        if self.batch_size != 1:
            raise ValueError("v0 requires batch_size=1 until the GPU pilot passes")
        if not self.enable_thinking and self.max_new_tokens > 4096:
            raise ValueError("thinking-disabled v0 runs cap max_new_tokens at 4096")
        return self


class BudgetConfig(StrictModel):
    max_gpu_hours: float = Field(gt=0)
    max_cost_usd: float = Field(gt=0)
    provider_rate_usd_per_hour: float = Field(gt=0)
    storage_limit_gb: float = Field(gt=0)
    deadline_utc: datetime

    @model_validator(mode="after")
    def validate_deadline(self) -> BudgetConfig:
        if self.deadline_utc.tzinfo is None:
            raise ValueError("deadline_utc must include a timezone")
        return self


class DatasetConfig(StrictModel):
    manifest: Path
    split: Literal["smoke", "pilot", "medium", "full"]


class InterventionConfig(StrictModel):
    mode: Literal["none", "zero_contribution"] = "none"
    manifest: Path | None = None
    renormalize_router_weights: Literal[False] = False

    @model_validator(mode="after")
    def require_manifest_for_intervention(self) -> InterventionConfig:
        if self.mode != "none" and self.manifest is None:
            raise ValueError("an intervention manifest is required when mode is not none")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    run_id: str | None = None
    model: ModelConfig
    runtime: RuntimeConfig
    budget: BudgetConfig
    datasets: DatasetConfig
    intervention: InterventionConfig = InterventionConfig()

    def canonical_bytes(self) -> bytes:
        payload = self.model_dump(mode="json", exclude={"run_id"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def resolve_run_id(self, git_sha: str, now: datetime | None = None) -> str:
        moment = now or datetime.now(UTC)
        condition = "think" if self.runtime.enable_thinking else "direct"
        stamp = moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-qwen35a3b-{condition}-{git_sha[:8]}-{self.fingerprint()[:8]}"


def load_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)
