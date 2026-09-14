"""Safe, model-specific launch infrastructure for DeepSeek V4 Flash.

This module deliberately does not share configuration classes with the Qwen v0
pipeline.  It is an infrastructure-only surface: it validates a pinned model
and renders a vLLM command, but it does not download weights or make any claim
about hardware compatibility.

The default caller-facing operation is a read-only preview.  State-changing
operations are explicit and fail closed when an example still contains a
placeholder value.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEEPSEEK_PINNED_DATE = "2026-09-14"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^(?:>=)?\d+\.\d+(?:\.\d+)?$")
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,127}$")
_TIME_RE = re.compile(r"^(?:\d+-)?\d{1,3}:\d{2}:\d{2}$")
_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_SHELL_TEXT_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,\- ]+$")
_SAFE_SLURM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,\-]+$")
_PLACEHOLDER_RE = re.compile(
    r"(?:placeholder|change[_-]?me|replace[_-]?me|set[_-]?me|"
    r"required(?:[_-]?(?:value|path|account|partition|constraint))?|"
    r"/absolute/path|/path/to|your(?:[_-][a-z0-9]+)+|<[^>]+>|\bTODO\b)",
    re.IGNORECASE,
)
_EXAMPLE_RUN_ID_RE = re.compile(r"(?:^|[-_])example$")

# These flags are emitted by the renderer and may not be overridden through
# ``extra_args``.  This keeps the model/revision and launch semantics owned by
# the pinned config rather than by an arbitrary appended token.
PROTECTED_VLLM_FLAGS = frozenset(
    {
        "--revision",
        "--trust-remote-code",
        "--tokenizer-mode",
        "--reasoning-parser",
        "--tool-call-parser",
        "--enable-auto-tool-choice",
        "--kv-cache-dtype",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--max-model-len",
        "--host",
        "--port",
        "--gpu-memory-utilization",
        "--download-dir",
    }
)

_PLACEHOLDER_MARKERS = (
    "placeholder",
    "change_me",
    "change-me",
    "set_me",
    "set-me",
    "required_",
    "required-value",
    "/absolute/path",
    "/path/to",
    "your-account",
    "your_account",
    "your-fau-account",
    "your_fau_account",
    "your_partition",
    "your-partition",
    "your-constraint",
    "your_constraint",
    "todo",
)


class DeepSeekInfrastructureError(ValueError):
    """Base error for invalid or unsafe DeepSeek launch requests."""


class DeepSeekModeError(DeepSeekInfrastructureError):
    """Raised when an explicit action does not match the config transport."""


class ProtectedVLLMArgumentError(DeepSeekInfrastructureError):
    """Raised when ``extra_args`` attempts to replace canonical vLLM flags."""


class DeepSeekStrictModel(BaseModel):
    """Strict immutable base for every DeepSeek-only configuration object."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _reject_controls(value: str, field_name: str) -> str:
    if _CONTROL_RE.search(value):
        raise ValueError(f"{field_name} must not contain newline or control characters")
    return value


def _is_placeholder(value: object) -> bool:
    text = str(value).strip().lower()
    return any(marker in text for marker in _PLACEHOLDER_MARKERS) or bool(
        _PLACEHOLDER_RE.search(text)
    )


def _coerce_path(value: object) -> Path:
    """Accept YAML path strings, while keeping every other field strict."""

    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise TypeError("path must be a string or pathlib.Path")


PathValue = Annotated[Path, BeforeValidator(_coerce_path)]


def _validate_absolute_path(value: Path, field_name: str) -> Path:
    rendered = str(value)
    _reject_controls(rendered, field_name)
    if not value.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    return value


def _validate_shell_text(value: str, field_name: str) -> str:
    _reject_controls(value, field_name)
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class DeepSeekModelConfig(DeepSeekStrictModel):
    """The exact official 0731 checkpoint and its trust-remote-code contract."""

    id: Literal[DEEPSEEK_MODEL_ID]
    revision: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
    trust_remote_code: Literal[True] = True

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not _SHA_RE.fullmatch(value):
            raise ValueError("revision must be a 40-character lowercase hexadecimal SHA")
        if value == "0" * 40:
            raise ValueError("revision must not be an all-zero placeholder SHA")
        return value


class DeepSeekServingConfig(DeepSeekStrictModel):
    """Hardware-neutral serving settings used by both launch transports."""

    # This is a declaration for operator preflight, not a claim that this
    # repository has installed or tested that version.
    vllm_min_version: str = "0.20.0"
    tensor_parallel_size: int = Field(default=8, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    gpu_count: int | None = Field(default=None, ge=1)
    max_model_len: int = Field(default=32_768, ge=1)
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, lt=1.0)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("vllm_min_version")
    @classmethod
    def validate_vllm_min_version(cls, value: str) -> str:
        _validate_shell_text(value, "vllm_min_version")
        if not _VERSION_RE.fullmatch(value):
            raise ValueError("vllm_min_version must be a version such as 0.20.0 or >=0.20.0")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        _validate_shell_text(value, "host")
        if not _HOST_RE.fullmatch(value):
            raise ValueError("host contains unsupported characters")
        return value

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, values: list[str]) -> list[str]:
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value:
                raise ValueError(f"extra_args[{index}] must be a non-empty string")
            _reject_controls(value, f"extra_args[{index}]")
        return values

    @model_validator(mode="after")
    def validate_parallelism(self) -> DeepSeekServingConfig:
        if self.gpu_count is not None:
            product = self.tensor_parallel_size * self.pipeline_parallel_size
            if product != self.gpu_count:
                raise ValueError(
                    "tensor_parallel_size * pipeline_parallel_size must equal gpu_count "
                    f"({product} != {self.gpu_count})"
                )
        return self


class DeepSeekPathsConfig(DeepSeekStrictModel):
    """Absolute, operator-owned locations for one run and its caches/logs."""

    model_cache_dir: PathValue
    run_dir: PathValue
    output_dir: PathValue
    cache_dir: PathValue
    log_dir: PathValue

    @field_validator("model_cache_dir", "run_dir", "output_dir", "cache_dir", "log_dir")
    @classmethod
    def validate_paths(cls, value: Path, info: Any) -> Path:
        return _validate_absolute_path(value, info.field_name)


class DeepSeekDirectConfig(DeepSeekStrictModel):
    """Optional direct-process wrapper settings."""

    working_directory: PathValue | None = None

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return _validate_absolute_path(value, "working_directory")


class DeepSeekFauSlurmConfig(DeepSeekStrictModel):
    """Site-specific FAU Slurm resources and wrapper directives."""

    job_name: str
    account: str
    partition: str
    constraint: str
    # Multi-node orchestration is intentionally out of scope until a tested
    # distributed vLLM launcher exists.  Keep the generated wrapper honest.
    nodes: Literal[1] = 1
    tasks_per_node: Literal[1] = 1
    gpus_per_node: int = Field(default=8, ge=1)
    cpus_per_task: int = Field(default=16, ge=1)
    memory_gb: int = Field(default=192, ge=1)
    time_limit: str = "24:00:00"
    output_path: PathValue
    error_path: PathValue
    module_commands: list[str] = Field(default_factory=list)
    environment_activation: str | None = None

    @field_validator("job_name")
    @classmethod
    def validate_job_name(cls, value: str) -> str:
        _validate_shell_text(value, "job_name")
        if not _JOB_NAME_RE.fullmatch(value):
            raise ValueError("job_name must contain only Slurm-safe name characters")
        return value

    @field_validator("account", "partition", "constraint")
    @classmethod
    def validate_directive_text(cls, value: str, info: Any) -> str:
        # Site-specific values intentionally remain configurable.  Restricting
        # them to one inert Slurm token prevents a directive from smuggling in
        # shell syntax or relying on site-specific whitespace parsing.
        _validate_shell_text(value, info.field_name)
        if not _SAFE_SLURM_TOKEN_RE.fullmatch(value):
            raise ValueError(
                f"{info.field_name} must be one Slurm-safe token without spaces or "
                "shell metacharacters"
            )
        return value

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, value: str) -> str:
        _validate_shell_text(value, "time_limit")
        if not _TIME_RE.fullmatch(value):
            raise ValueError("time_limit must use D-HH:MM:SS or HH:MM:SS")
        return value

    @field_validator("output_path", "error_path")
    @classmethod
    def validate_slurm_paths(cls, value: Path, info: Any) -> Path:
        return _validate_absolute_path(value, info.field_name)

    @field_validator("module_commands")
    @classmethod
    def validate_module_commands(cls, values: list[str]) -> list[str]:
        for index, value in enumerate(values):
            _validate_shell_text(value, f"module_commands[{index}]")
            if not value.startswith("module "):
                raise ValueError(
                    f"module_commands[{index}] must begin with 'module '"
                )
            if not _SAFE_SHELL_TEXT_RE.fullmatch(value):
                raise ValueError(
                    f"module_commands[{index}] contains shell metacharacters; "
                    "use a simple module/activation command"
                )
        return values

    @field_validator("environment_activation")
    @classmethod
    def validate_environment_activation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _validate_shell_text(value, "environment_activation")
        if not (
            value.startswith("source ") or value.startswith("conda activate ")
        ):
            raise ValueError(
                "environment_activation must begin with 'source ' or 'conda activate '"
            )
        if not _SAFE_SHELL_TEXT_RE.fullmatch(value):
            raise ValueError("environment_activation contains unsupported shell metacharacters")
        return value


class DeepSeekLaunchConfig(DeepSeekStrictModel):
    """Complete DeepSeek launch contract, separate from Qwen ExperimentConfig."""

    schema_version: Literal[1]
    run_id: str
    mode: Literal["direct", "fau_slurm"]
    model: DeepSeekModelConfig
    serving: DeepSeekServingConfig
    paths: DeepSeekPathsConfig
    direct: DeepSeekDirectConfig | None = None
    fau_slurm: DeepSeekFauSlurmConfig | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        _validate_shell_text(value, "run_id")
        if not _RUN_ID_RE.fullmatch(value):
            raise ValueError("run_id must be a unique filesystem-safe identifier")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> DeepSeekLaunchConfig:
        if self.mode == "direct" and self.direct is None:
            raise ValueError("direct mode requires a direct configuration block")
        if self.mode == "direct" and self.fau_slurm is not None:
            raise ValueError("direct mode must not contain fau_slurm settings")
        if self.mode == "fau_slurm" and self.fau_slurm is None:
            raise ValueError("fau_slurm mode requires a fau_slurm configuration block")
        if self.mode == "fau_slurm" and self.direct is not None:
            raise ValueError("fau_slurm mode must not contain direct settings")
        if self.fau_slurm is not None and self.serving.gpu_count is not None:
            requested = self.fau_slurm.nodes * self.fau_slurm.gpus_per_node
            if requested != self.serving.gpu_count:
                raise ValueError(
                    "FAU nodes * gpus_per_node must equal serving.gpu_count "
                    f"({requested} != {self.serving.gpu_count})"
                )
        return self


# Short aliases make the public surface easy to discover while keeping all
# names DeepSeek-specific and avoiding collisions with Qwen ``ModelConfig``.
DeepSeekConfig = DeepSeekLaunchConfig


def load_deepseek_config(path: Path) -> DeepSeekLaunchConfig:
    """Load and strictly validate one YAML launch configuration."""

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DeepSeekInfrastructureError(f"unable to load DeepSeek config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DeepSeekInfrastructureError("DeepSeek config must contain a YAML mapping")
    try:
        return DeepSeekLaunchConfig.model_validate(data)
    except Exception as exc:  # Pydantic's useful field paths are preserved in the message.
        raise DeepSeekInfrastructureError(str(exc)) from exc


def _placeholder_fields(config: DeepSeekLaunchConfig) -> list[str]:
    fields: list[tuple[str, object]] = [
        ("run_id", config.run_id),
        ("paths.model_cache_dir", config.paths.model_cache_dir),
        ("paths.run_dir", config.paths.run_dir),
        ("paths.output_dir", config.paths.output_dir),
        ("paths.cache_dir", config.paths.cache_dir),
        ("paths.log_dir", config.paths.log_dir),
    ]
    if config.mode == "direct":
        if config.direct and config.direct.working_directory is not None:
            fields.append(("direct.working_directory", config.direct.working_directory))
    else:
        assert config.fau_slurm is not None
        fields.extend(
            [
                ("fau_slurm.account", config.fau_slurm.account),
                ("fau_slurm.partition", config.fau_slurm.partition),
                ("fau_slurm.constraint", config.fau_slurm.constraint),
                ("fau_slurm.output_path", config.fau_slurm.output_path),
                ("fau_slurm.error_path", config.fau_slurm.error_path),
            ]
        )
    return [
        name
        for name, value in fields
        if _is_placeholder(value)
        or (name == "run_id" and _EXAMPLE_RUN_ID_RE.search(str(value).strip().lower()))
    ]


def preflight_deepseek_config(
    config: DeepSeekLaunchConfig, *, state_changing: bool = False
) -> dict[str, Any]:
    """Return a hardware-free preflight report.

    ``state_changing=True`` is used only by direct execution and FAU
    submission.  It upgrades placeholder warnings to hard errors and also
    requires the exact official revision, preventing an operator from
    accidentally launching an unpinned checkpoint.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if config.model.id != DEEPSEEK_MODEL_ID:
        errors.append(f"model.id must be exactly {DEEPSEEK_MODEL_ID}")
    if not _SHA_RE.fullmatch(config.model.revision):
        errors.append("model.revision must be a 40-character lowercase hexadecimal SHA")
    if config.model.revision != DEEPSEEK_MODEL_REVISION:
        errors.append(
            "model.revision must use the approved official 0731 pin "
            f"{DEEPSEEK_MODEL_REVISION}"
        )
    if config.model.trust_remote_code is not True:
        errors.append("model.trust_remote_code must remain true for the official recipe")

    placeholders = _placeholder_fields(config)
    if placeholders:
        message = "placeholder values require operator replacement: " + ", ".join(placeholders)
        (errors if state_changing else warnings).append(message)

    # Existing paths are inspected read-only when present.  Missing paths are
    # valid at preview time and will be created by the explicit direct process
    # or by the generated FAU script.
    for name, path in (
        ("paths.model_cache_dir", config.paths.model_cache_dir),
        ("paths.run_dir", config.paths.run_dir),
        ("paths.output_dir", config.paths.output_dir),
        ("paths.cache_dir", config.paths.cache_dir),
        ("paths.log_dir", config.paths.log_dir),
    ):
        if path.exists() and not path.is_dir():
            errors.append(f"{name} exists but is not a directory")

    if config.mode == "fau_slurm" and config.fau_slurm is not None:
        for name, path in (
            ("fau_slurm.output_path", config.fau_slurm.output_path),
            ("fau_slurm.error_path", config.fau_slurm.error_path),
        ):
            if path.exists() and path.is_dir():
                message = f"{name} exists as a directory, expected a file path"
                (errors if state_changing else warnings).append(message)
            parent = path.parent
            if not parent.is_dir():
                message = f"{name}.parent must already exist as a directory: {parent}"
                (errors if state_changing else warnings).append(message)

    report: dict[str, Any] = {
        "valid": not errors,
        "state_changing": state_changing,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "approved_revision": DEEPSEEK_MODEL_REVISION,
        "pinned_date": DEEPSEEK_PINNED_DATE,
        "mode": config.mode,
        "vllm_min_version": config.serving.vllm_min_version,
        "tensor_parallel_size": config.serving.tensor_parallel_size,
        "pipeline_parallel_size": config.serving.pipeline_parallel_size,
        "gpu_count": config.serving.gpu_count,
        "max_model_len": config.serving.max_model_len,
        "gpu_memory_utilization": config.serving.gpu_memory_utilization,
        "warnings": warnings,
        "errors": errors,
    }
    return report


def _validate_extra_args(extra_args: Sequence[str]) -> None:
    for token in extra_args:
        option = token.split("=", 1)[0]
        if option in PROTECTED_VLLM_FLAGS:
            raise ProtectedVLLMArgumentError(
                f"extra_args cannot override protected vLLM flag {option}"
            )


def _validate_render_inputs(config: DeepSeekLaunchConfig) -> None:
    """Apply checks shared by preview and both state-changing transports."""

    report = preflight_deepseek_config(config)
    if not report["valid"]:
        raise DeepSeekInfrastructureError("; ".join(report["errors"]))
    _validate_extra_args(config.serving.extra_args)


def render_vllm_argv(config: DeepSeekLaunchConfig) -> list[str]:
    """Render the one canonical vLLM argv used by both transports."""

    _validate_render_inputs(config)
    serving = config.serving
    return [
        "vllm",
        "serve",
        config.model.id,
        "--revision",
        config.model.revision,
        "--trust-remote-code",
        "--tokenizer-mode",
        "deepseek_v4",
        "--reasoning-parser",
        "deepseek_v4",
        "--tool-call-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--kv-cache-dtype",
        "fp8",
        "--tensor-parallel-size",
        str(serving.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(serving.pipeline_parallel_size),
        "--max-model-len",
        str(serving.max_model_len),
        "--host",
        serving.host,
        "--port",
        str(serving.port),
        "--gpu-memory-utilization",
        str(serving.gpu_memory_utilization),
        "--download-dir",
        str(config.paths.model_cache_dir),
        *serving.extra_args,
    ]


def _directive_value(value: object) -> str:
    """Quote a Slurm directive value while retaining a single directive line."""

    rendered = str(value)
    _reject_controls(rendered, "Slurm directive")
    return shlex.quote(rendered)


def render_fau_slurm_script(config: DeepSeekLaunchConfig) -> str:
    """Render a deterministic, non-submitting FAU Slurm script."""

    if config.mode != "fau_slurm" or config.fau_slurm is None:
        raise DeepSeekModeError("FAU Slurm script rendering requires mode=fau_slurm")
    _validate_render_inputs(config)

    slurm = config.fau_slurm
    script_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={_directive_value(slurm.job_name)}",
        f"#SBATCH --account={_directive_value(slurm.account)}",
        f"#SBATCH --partition={_directive_value(slurm.partition)}",
        f"#SBATCH --constraint={_directive_value(slurm.constraint)}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --ntasks-per-node={slurm.tasks_per_node}",
        f"#SBATCH --gres=gpu:{slurm.gpus_per_node}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.memory_gb}G",
        f"#SBATCH --time={_directive_value(slurm.time_limit)}",
        f"#SBATCH --output={_directive_value(slurm.output_path)}",
        f"#SBATCH --error={_directive_value(slurm.error_path)}",
        "",
        "set -euo pipefail",
        "",
    ]

    directories = [
        config.paths.model_cache_dir,
        config.paths.run_dir,
        config.paths.output_dir,
        config.paths.cache_dir,
        config.paths.log_dir,
    ]
    unique_directories = list(dict.fromkeys(directories))
    script_lines.append("mkdir -p " + shlex.join(str(path) for path in unique_directories))
    script_lines.extend(slurm.module_commands)
    if slurm.environment_activation is not None:
        script_lines.append(slurm.environment_activation)
    script_lines.extend(["", f"exec {shlex.join(render_vllm_argv(config))}", ""])
    return "\n".join(script_lines)


def _state_changing_preflight(config: DeepSeekLaunchConfig) -> dict[str, Any]:
    report = preflight_deepseek_config(config, state_changing=True)
    if not report["valid"]:
        raise DeepSeekInfrastructureError("; ".join(report["errors"]))
    return report


def execute_direct(config: DeepSeekLaunchConfig) -> dict[str, Any]:
    """Run the direct argv after an explicit, state-changing preflight."""

    if config.mode != "direct":
        raise DeepSeekModeError("--execute is only valid for mode=direct")
    _state_changing_preflight(config)
    # Render before creating any run directories so a protected override or
    # another renderer error cannot leave state behind.
    argv = render_vllm_argv(config)
    assert config.direct is not None
    try:
        for directory in (
            config.paths.model_cache_dir,
            config.paths.run_dir,
            config.paths.output_dir,
            config.paths.cache_dir,
            config.paths.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DeepSeekInfrastructureError(
            f"unable to prepare direct launch directories: {exc}"
        ) from exc
    try:
        completed = subprocess.run(
            argv,
            cwd=str(config.direct.working_directory) if config.direct.working_directory else None,
            shell=False,
            check=False,
        )
    except OSError as exc:
        raise DeepSeekInfrastructureError(
            f"unable to start direct vLLM process: {exc}"
        ) from exc
    return {
        "action": "execute",
        "mode": config.mode,
        "argv": argv,
        "returncode": completed.returncode,
    }


def submit_fau_slurm(config: DeepSeekLaunchConfig) -> dict[str, Any]:
    """Submit exactly one generated script through ``sbatch`` stdin."""

    if config.mode != "fau_slurm":
        raise DeepSeekModeError("--submit is only valid for mode=fau_slurm")
    _state_changing_preflight(config)
    script = render_fau_slurm_script(config)
    try:
        completed = subprocess.run(
            ["sbatch"],
            input=script,
            text=True,
            capture_output=True,
            shell=False,
            check=False,
        )
    except OSError as exc:
        raise DeepSeekInfrastructureError(f"unable to invoke sbatch: {exc}") from exc
    return {
        "action": "submit",
        "mode": config.mode,
        "sbatch_argv": ["sbatch"],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def preview_deepseek_launch(config: DeepSeekLaunchConfig) -> dict[str, Any]:
    """Return a read-only preview; this never creates directories or calls a process."""

    report = preflight_deepseek_config(config)
    output: dict[str, Any] = {
        "action": "preview",
        "run_id": config.run_id,
        "mode": config.mode,
        "preflight": report,
    }
    if config.mode == "direct":
        argv = render_vllm_argv(config)
        output.update({"argv": argv, "command": shlex.join(argv)})
    else:
        script = render_fau_slurm_script(config)
        output["script"] = script
    return output


def deepseek_launch(
    config: DeepSeekLaunchConfig,
    *,
    check: bool = False,
    execute: bool = False,
    submit: bool = False,
) -> dict[str, Any]:
    """Dispatch the CLI action with a safe preview as the default."""

    selected = sum((check, execute, submit))
    if selected > 1:
        raise DeepSeekInfrastructureError("--check, --execute, and --submit are mutually exclusive")
    if check:
        _validate_render_inputs(config)
        return {
            "action": "check",
            "run_id": config.run_id,
            "mode": config.mode,
            "preflight": preflight_deepseek_config(config),
        }
    if execute:
        return execute_direct(config)
    if submit:
        return submit_fau_slurm(config)
    return preview_deepseek_launch(config)


def serialize_launch_result(result: dict[str, Any]) -> str:
    """Stable JSON output used by the CLI and easy for operators to capture."""

    return json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
