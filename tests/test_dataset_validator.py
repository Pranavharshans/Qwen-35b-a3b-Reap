import importlib.util
from pathlib import Path

from reverse_reap.datasets import freeze_manifest, normalize_sample


def validator():
    path = Path(__file__).parents[1] / "scripts" / "validate_dataset.py"
    spec = importlib.util.spec_from_file_location("validate_dataset", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_dataset_validator_detects_missing_required_coverage(tmp_path):
    sample = normalize_sample(
        {
            "source": "fixture",
            "source_revision": "abc",
            "source_id": "one",
            "domain": "coding",
            "stratum": "function-synthesis",
            "language": "python",
            "prompt": "Return one.",
            "reference": "1",
            "scorer": "exact_match",
        },
        seed=1,
    )
    path = tmp_path / "manifest.jsonl"
    freeze_manifest([sample], path)
    errors = validator().validate(path, require_full=True)
    assert any("500 per domain" in error for error in errors)
    assert any("two coding languages" in error for error in errors)
    assert any("matched and general" in error for error in errors)
