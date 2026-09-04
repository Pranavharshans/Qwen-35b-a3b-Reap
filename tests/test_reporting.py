import json
from pathlib import Path

from reverse_reap.config import load_config
from reverse_reap.reporting import build_run_bundle


def config():
    return load_config(Path(__file__).parents[1] / "configs" / "smoke-3090-bf16.yaml")


def test_bundle_never_promotes_observational_candidate_without_causal_gate(tmp_path):
    run_dir = tmp_path / "run"
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "candidate-manifest.json").write_text(
        json.dumps({"gate_passed": True, "experts": [{"layer": 0, "expert": 1}]})
    )
    destination = run_dir / "bundle.json"
    bundle = build_run_bundle(run_dir, tmp_path / "state", config(), destination)
    assert bundle["classification"] == "incomplete"
    assert bundle["evidence_label"] == "observational-candidates"
    assert not bundle["gates"]["D_causal"]
    assert destination.exists()


def test_bundle_positive_requires_passed_causal_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "causal-report.json").write_text(json.dumps({"passed": True}))
    bundle = build_run_bundle(
        run_dir, tmp_path / "state", config(), run_dir / "bundle.json"
    )
    assert bundle["classification"] == "positive"
    assert bundle["evidence_label"] == "coding-critical-v0"


def test_bundle_failed_causal_report_is_null_not_incomplete(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "causal-report.json").write_text(
        json.dumps({"passed": False, "label": "unreplicated-candidates"})
    )
    bundle = build_run_bundle(run_dir, tmp_path / "state", config(), run_dir / "bundle.json")
    assert bundle["classification"] == "null"
    assert bundle["evidence_label"] == "unreplicated-candidates"
    assert not bundle["gates"]["D_causal"]


def test_bundle_failed_determinism_is_feasibility_failure_even_with_failed_gate_d(tmp_path):
    run_dir = tmp_path / "run"
    gates = run_dir / "gates"
    gates.mkdir(parents=True)
    (gates / "determinism-report.json").write_text(json.dumps({"passed": False}))
    (run_dir / "causal-report.json").write_text(
        json.dumps({"passed": False, "label": "observational-candidates"})
    )
    bundle = build_run_bundle(run_dir, tmp_path / "state", config(), run_dir / "bundle.json")
    assert bundle["classification"] == "feasibility-failure"
    assert bundle["evidence_label"] == "nondeterministic-generation"
    assert not bundle["gates"]["B_determinism"]


def test_bundle_passing_determinism_sets_gate_b(tmp_path):
    run_dir = tmp_path / "run"
    gates = run_dir / "gates"
    gates.mkdir(parents=True)
    (gates / "determinism-report.json").write_text(
        json.dumps({"passed": True, "scoreable_restricted": {"passed": True}})
    )
    bundle = build_run_bundle(run_dir, tmp_path / "state", config(), run_dir / "bundle.json")
    assert bundle["gates"]["B_determinism"]
