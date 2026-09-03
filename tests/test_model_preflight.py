import json
from pathlib import Path

import pytest
import yaml

from reverse_reap.model_preflight import (
    EXPECTED_TEXT_CONFIG,
    ModelPreflightError,
    validate_model_config,
    write_pinned_config,
)


def official_config():
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": dict(EXPECTED_TEXT_CONFIG),
    }


def test_validates_exact_official_metadata_contract():
    assert validate_model_config(official_config())["compatible"]


def test_rejects_metadata_architecture_drift():
    value = official_config()
    value["text_config"]["num_experts"] = 255
    with pytest.raises(ModelPreflightError, match="num_experts"):
        validate_model_config(value)


def test_writes_revision_pinned_config_without_mutating_template(tmp_path):
    template = Path(__file__).parents[1] / "configs" / "smoke-3090-bf16.yaml"
    destination = tmp_path / "pinned.yaml"
    config = write_pinned_config(template, destination, "f" * 40)
    assert config.model.revision == "f" * 40
    assert yaml.safe_load(destination.read_text())["model"]["revision"] == "f" * 40
    assert yaml.safe_load(template.read_text())["model"]["revision"] == "0" * 40


def test_verified_download_rejects_failed_preflight(tmp_path):
    from reverse_reap.model_preflight import download_verified_weights

    report = tmp_path / "report.json"
    report.write_text(json.dumps({"passed": False}))
    with pytest.raises(ModelPreflightError, match="did not pass"):
        download_verified_weights(report, tmp_path / "model")
