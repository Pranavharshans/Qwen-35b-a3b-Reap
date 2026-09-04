"""Tests for scripts/generation_bundle.py (phase-1/phase-2 hash hand-off)."""

import json
import subprocess
import sys
from pathlib import Path

from reverse_reap.datasets import normalize_sample


def _spec(tmp_path, conditions):
    """Conditions spec with the same shape as configs/causal-pilot-conditions.json."""
    payload = {
        "schema_version": 1,
        "splits": {"validation": {"samples": 3, "coding": 2, "control": 1}},
        "conditions": conditions,
    }
    path = tmp_path / "conditions.json"
    path.write_text(json.dumps(payload))
    return path


def _validation_manifest(tmp_path, count=3):
    rows, index = [], 0
    while len(rows) < count:
        sample = normalize_sample(
            {
                "source": "fixture",
                "source_revision": "abc",
                "source_id": f"v{index}",
                "domain": "coding" if index < 2 else "control",
                "stratum": "synthesis",
                "language": "python",
                "prompt": f"write code {index}",
                "reference": "#### 42",
                "scorer": "exact_match",
            },
            seed=1,
        )
        index += 1
        if sample.split == "validation":
            rows.append(sample)
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(s.model_dump_json() + "\n" for s in rows), encoding="utf-8")
    return path, [s.sample_id for s in rows]


def _write_generations(generations_dir, conditions, sample_ids, responses=None):
    generations_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        condition_id = condition["condition_id"]
        masked = 0 if condition.get("expert_manifest") is None else 1
        rows = [
            {
                "sample_id": sample_id,
                "source": "fixture",
                "source_id": sample_id,
                "scorer": "exact_match",
                "domain": "coding",
                "stratum": "synthesis",
                "split": "validation",
                "condition_id": condition_id,
                "masked_experts": masked,
                "response": (responses or {}).get(sample_id, f"resp-{sample_id}"),
                "generated_tokens": 3,
                "truncated": False,
                "latency_seconds": 0.1,
            }
            for sample_id in sample_ids
        ]
        (generations_dir / f"{condition_id}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def _run(script_args, cwd):
    return subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts" / "generation_bundle.py")]
        + script_args,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
    )


def _fixture(tmp_path):
    conditions = [
        {
            "condition_id": "c0-baseline-a",
            "phase": "validation-baselines",
            "split": "validation",
            "expert_manifest": None,
            "expert_manifest_sha256": None,
        },
        {
            "condition_id": "c2-selected",
            "phase": "validation-interventions",
            "split": "validation",
            "expert_manifest": str(tmp_path / "experts.json"),
            "expert_manifest_sha256": "a" * 64,
        },
    ]
    manifest_path, sample_ids = _validation_manifest(tmp_path)
    return conditions, manifest_path, sample_ids


def test_build_then_verify_roundtrip(tmp_path):
    conditions, manifest_path, sample_ids = _fixture(tmp_path)
    spec = _spec(tmp_path, conditions)
    generations = tmp_path / "generations"
    _write_generations(generations, conditions, sample_ids)
    bundle = tmp_path / "generation-bundle.json"
    report = tmp_path / "verify-generation-inputs.json"

    built = _run(
        [
            "build",
            "--conditions",
            str(spec),
            "--generations-dir",
            str(generations),
            "--dataset-manifest",
            str(manifest_path),
            "--output",
            str(bundle),
            "--run-id",
            "test-run",
        ],
        cwd=tmp_path,
    )
    assert built.returncode == 0, built.stderr
    payload = json.loads(bundle.read_text())
    assert payload["run_id"] == "test-run"
    assert set(payload["files"]) == {"c0-baseline-a", "c2-selected"}
    assert payload["total_rows"] == 6
    assert all(entry["sha256"] for entry in payload["files"].values())

    verified = _run(
        [
            "verify",
            "--bundle",
            str(bundle),
            "--generations-dir",
            str(generations),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
    )
    assert verified.returncode == 0, verified.stderr
    result = json.loads(report.read_text())
    assert result["passed"] and result["verified_files"] == 2 and result["mismatches"] == []


def test_build_rejects_row_drift_and_empty_responses(tmp_path):
    conditions, manifest_path, sample_ids = _fixture(tmp_path)
    spec = _spec(tmp_path, conditions)
    generations = tmp_path / "generations"

    # Row-count drift: drop one sample from the intervention condition.
    _write_generations(generations, conditions, sample_ids[:-1])
    built = _run(
        [
            "build",
            "--conditions",
            str(spec),
            "--generations-dir",
            str(generations),
            "--dataset-manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "b.json"),
            "--run-id",
            "test-run",
        ],
        cwd=tmp_path,
    )
    assert built.returncode == 1 and "expected 3 rows" in built.stderr

    # Empty response.
    _write_generations(generations, conditions, sample_ids, responses={sample_ids[0]: "  "})
    built = _run(
        [
            "build",
            "--conditions",
            str(spec),
            "--generations-dir",
            str(generations),
            "--dataset-manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "b.json"),
            "--run-id",
            "test-run",
        ],
        cwd=tmp_path,
    )
    assert built.returncode == 1 and "empty or missing response" in built.stderr

    # Masked-expert contradiction: intervention rows recorded one masked expert
    # but the spec declares no expert manifest for that condition.
    tampered_conditions = [{**condition, "expert_manifest": None} for condition in conditions]
    generations2 = tmp_path / "generations2"
    _write_generations(generations2, conditions, sample_ids)
    spec2 = _spec(tmp_path, tampered_conditions)
    built = _run(
        [
            "build",
            "--conditions",
            str(spec2),
            "--generations-dir",
            str(generations2),
            "--dataset-manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "b2.json"),
            "--run-id",
            "test-run",
        ],
        cwd=tmp_path,
    )
    assert built.returncode == 1 and "masked_experts" in built.stderr


def test_verify_detects_tampered_copy(tmp_path):
    conditions, manifest_path, sample_ids = _fixture(tmp_path)
    spec = _spec(tmp_path, conditions)
    generations = tmp_path / "generations"
    _write_generations(generations, conditions, sample_ids)
    bundle = tmp_path / "generation-bundle.json"
    assert (
        _run(
            [
                "build",
                "--conditions",
                str(spec),
                "--generations-dir",
                str(generations),
                "--dataset-manifest",
                str(manifest_path),
                "--output",
                str(bundle),
                "--run-id",
                "test-run",
            ],
            cwd=tmp_path,
        ).returncode
        == 0
    )

    # Simulate a corrupted transfer: flip one response byte.
    target = generations / "c2-selected.jsonl"
    target.write_text(target.read_text().replace("resp-", "tampered-"), encoding="utf-8")
    report = tmp_path / "verify-report.json"
    verified = _run(
        [
            "verify",
            "--bundle",
            str(bundle),
            "--generations-dir",
            str(generations),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
    )
    assert verified.returncode == 1 and "mismatch" in verified.stderr
    result = json.loads(report.read_text())
    assert not result["passed"]
    assert result["mismatches"][0]["condition_id"] == "c2-selected"
    assert result["mismatches"][0]["reason"] == "sha256 mismatch"


def test_build_refuses_overwrite(tmp_path):
    conditions, manifest_path, sample_ids = _fixture(tmp_path)
    spec = _spec(tmp_path, conditions)
    generations = tmp_path / "generations"
    _write_generations(generations, conditions, sample_ids)
    bundle = tmp_path / "generation-bundle.json"
    args = [
        "build",
        "--conditions",
        str(spec),
        "--generations-dir",
        str(generations),
        "--dataset-manifest",
        str(manifest_path),
        "--output",
        str(bundle),
        "--run-id",
        "test-run",
    ]
    assert _run(args, cwd=tmp_path).returncode == 0
    again = _run(args, cwd=tmp_path)
    assert again.returncode == 1 and "refusing to overwrite" in again.stderr
