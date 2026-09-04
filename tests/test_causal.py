import json

import pytest

from reverse_reap.causal import (
    CausalError,
    causal_gate_report,
    compare_deterministic_evaluations,
    load_expert_set,
    score_response,
)
from reverse_reap.datasets import normalize_sample
from reverse_reap.evaluator import EvaluationResult


def sample(scorer="exact_match"):
    data = {
        "source": "fixture",
        "source_revision": "abc",
        "source_id": "one",
        "domain": "coding",
        "stratum": "understanding",
        "language": "python",
        "prompt": "What is the output?",
        "reference": "#### 42",
        "scorer": scorer,
    }
    if scorer == "unit_tests":
        data.update(
            {
                "prompt": "def answer():\n",
                "reference": "    return 42",
                "tests": "assert answer() == 42",
                "entry_point": "answer",
            }
        )
    return normalize_sample(data, seed=1)


def test_exact_scorer_extracts_final_answer():
    result = score_response(sample(), "Reasoning\n\\boxed{42}", evaluator_image="unused")
    assert result["passed"]


def test_multiple_choice_scorer_uses_final_standalone_letter():
    item = sample().model_copy(update={"scorer": "multiple_choice", "reference": "B"})
    result = score_response(item, "I considered A and C. Final: B", evaluator_image="unused")
    assert result["passed"]


def test_unit_test_scorer_uses_sandbox(monkeypatch):
    captured = {}

    def fake_evaluate(code, tests, **kwargs):
        captured.update(code=code, tests=tests, kwargs=kwargs)
        return EvaluationResult(True, 0, False, "", "", "a" * 64)

    monkeypatch.setattr("reverse_reap.causal.evaluate_python", fake_evaluate)
    result = score_response(
        sample("unit_tests"), "```python\n    return 42\n```", evaluator_image="image@sha256:x"
    )
    assert result["passed"]
    assert captured["code"].startswith("def answer")


def write_results(path, condition, coding_passes, control_passes):
    rows = []
    for domain, values in (("coding", coding_passes), ("control", control_passes)):
        for index, passed in enumerate(values):
            rows.append(
                {
                    "sample_id": f"{domain}-{index}",
                    "domain": domain,
                    "condition_id": condition,
                    "scoreable": True,
                    "passed": passed,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_gate_d_applies_all_preregistered_thresholds(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    selected = tmp_path / "selected.jsonl"
    replication_baseline = tmp_path / "replication-baseline.jsonl"
    replication_selected = tmp_path / "replication-selected.jsonl"
    write_results(baseline, "C0", [True] * 10, [True] * 10)
    write_results(selected, "C2", [False] * 8 + [True] * 2, [False] + [True] * 9)
    write_results(replication_baseline, "C0", [True] * 10, [True] * 10)
    write_results(replication_selected, "C2", [False] * 7 + [True] * 3, [False] + [True] * 9)
    random_paths = []
    for index in range(20):
        path = tmp_path / f"random-{index}.jsonl"
        write_results(path, "C3", [False] * 2 + [True] * 8, [True] * 10)
        random_paths.append(path)
    report = causal_gate_report(
        baseline,
        selected,
        random_paths,
        replication_baseline_path=replication_baseline,
        replication_selected_path=replication_selected,
    )
    assert report["passed"]
    assert report["label"] == "coding-critical-v0"
    assert report["coding_drop"] == pytest.approx(0.8)
    assert report["replication"]["coding_drop"] == pytest.approx(0.7)
    assert len(report["coding_drop_95ci"]) == 2


def test_gate_d_validation_stage_without_replication_is_unreplicated(tmp_path):
    """Validation-stage Gate D (directive: replication split stays untouched).

    The five criteria keys stay stable so the report schema is identical, but
    without replication paths ``passed`` is False while ``validation_passed``
    reflects the four validation criteria; a validation pass labels the
    candidates unreplicated-candidates, never coding-critical-v0 and never
    observational-candidates.
    """
    baseline = tmp_path / "baseline.jsonl"
    selected = tmp_path / "selected.jsonl"
    write_results(baseline, "C0", [True] * 10, [True] * 10)
    write_results(selected, "C2", [False] * 8 + [True] * 2, [False] + [True] * 9)
    random_paths = []
    for index in range(20):
        path = tmp_path / f"random-{index}.jsonl"
        write_results(path, "C3", [False] * 2 + [True] * 8, [True] * 10)
        random_paths.append(path)
    report = causal_gate_report(baseline, selected, random_paths)
    assert not report["passed"]  # replication_direction cannot hold without replication
    assert report["validation_passed"]
    assert report["label"] == "unreplicated-candidates"
    assert report["replication"] is None
    assert set(report["criteria"]) == {
        "twice_random_median",
        "at_or_above_random_p95",
        "coding_specificity_2pp",
        "replication_direction",
        "no_broad_output_collapse",
    }
    assert report["coding_drop"] == pytest.approx(0.8)

    # A validation-stage failure stays observational-candidates (coding drop
    # 0.1 fails twice_random_median 0.4 and specificity 0.02).
    write_results(selected, "C2", [False] + [True] * 9, [False] + [True] * 9)
    failed = causal_gate_report(baseline, selected, random_paths)
    assert not failed["validation_passed"]
    assert failed["label"] == "observational-candidates"


def test_expert_manifest_rejects_duplicate_identity(tmp_path):
    path = tmp_path / "experts.json"
    path.write_text(json.dumps({"experts": [{"layer": 1, "expert": 2}] * 2}))
    with pytest.raises(CausalError, match="duplicate"):
        load_expert_set(path)


def test_swebench_pins_task_repo_required_by_pinned_harness():
    """Measured 2026-09-04 (scoring-host probe): at harness revision 02e7a74
    the hosted SWE-bench_Lite rows lack the image/eval_script columns, so the
    task repo is part of the pinned evaluation contract and must stay pinned."""
    import re

    from reverse_reap import swebench

    assert swebench.SWE_BENCH_REVISION == "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e"
    assert swebench.SWE_BENCH_TASKS_REPOSITORY.endswith("swe-bench-tasks.git")
    assert re.fullmatch(r"[0-9a-f]{40}", swebench.SWE_BENCH_TASKS_REVISION)
    assert swebench.SWE_BENCH_DATASET == "princeton-nlp/SWE-bench_Lite"


def test_determinism_comparison_requires_exact_responses_and_95_percent_scoreable(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    rows = [
        {
            "sample_id": f"s-{index}",
            "domain": "coding",
            "response": "same",
            "passed": True,
            "scoreable": True,
        }
        for index in range(20)
    ]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    first.write_text(content)
    second.write_text(content)
    assert compare_deterministic_evaluations(first, second)["passed"]
    changed = list(rows)
    changed[0] = {**changed[0], "response": "different"}
    second.write_text("".join(json.dumps(row) + "\n" for row in changed))
    assert not compare_deterministic_evaluations(first, second)["passed"]


# --- causal-pilot generation/scoring decomposition -------------------------

from pathlib import Path  # noqa: E402


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        torch = pytest.importorskip("torch")
        assert kwargs.get("return_tensors") == "pt"
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids, skip_special_tokens=True):
        return "#### 42"


class _FakeModel:
    def get_input_embeddings(self):
        class _Weight:
            device = "cpu"

        class _Embedding:
            weight = _Weight()

        return _Embedding()

    def generate(self, input_ids, **kwargs):
        torch = pytest.importorskip("torch")
        return torch.cat([input_ids, torch.tensor([[7, 8, 9]])], dim=1)


def _patch_generation_runtime(monkeypatch):
    import contextlib

    monkeypatch.setattr(
        "reverse_reap.causal.load_donor", lambda *args, **kwargs: (_FakeModel(), _FakeTokenizer())
    )
    monkeypatch.setattr("reverse_reap.causal.inspect_qwen35_moe", lambda model: object())
    monkeypatch.setattr("reverse_reap.causal.validate_donor_contract", lambda model, arch: None)
    monkeypatch.setattr(
        "reverse_reap.causal.instrument_qwen35",
        lambda architecture, masked=None: contextlib.nullcontext(),
    )


def _validation_manifest(tmp_path, count=2):
    """Build a manifest whose samples all land in the validation split."""
    from reverse_reap.datasets import normalize_sample

    rows, index = [], 0
    while len(rows) < count:
        sample = normalize_sample(
            {
                "source": "fixture",
                "source_revision": "abc",
                "source_id": f"v{index}",
                "domain": "coding",
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
    return path


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_generate_and_score_decomposition_reproduces_evaluate_condition(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from reverse_reap.causal import evaluate_condition, generate_condition, score_condition
    from reverse_reap.config import load_config

    _patch_generation_runtime(monkeypatch)
    config = load_config(Path(__file__).parents[1] / "configs" / "pinned-3090-bf16-gen.yaml")
    manifest = _validation_manifest(tmp_path)
    experts = tmp_path / "experts.json"
    experts.write_text(json.dumps({"experts": [{"layer": 3, "expert": 26}]}))
    image = "eval@sha256:" + "a" * 64

    full = evaluate_condition(
        Path("unused-model"),
        manifest,
        tmp_path / "full.jsonl",
        config,
        split="validation",
        condition_id="cT",
        evaluator_image=image,
        expert_manifest=experts,
    )
    generate_condition(
        _FakeModel(),
        _FakeTokenizer(),
        object(),
        manifest,
        tmp_path / "gen.jsonl",
        config,
        split="validation",
        condition_id="cT",
        expert_manifest=experts,
    )
    scored = score_condition(tmp_path / "gen.jsonl", manifest, tmp_path / "scored.jsonl",
                             evaluator_image=image)

    def without_latency(rows):
        return [
            {key: value for key, value in row.items() if key != "latency_seconds"}
            for row in rows
        ]

    assert without_latency(_read_rows(tmp_path / "full.jsonl")) == without_latency(
        _read_rows(tmp_path / "scored.jsonl")
    )
    assert all(row["masked_experts"] == 1 for row in _read_rows(tmp_path / "scored.jsonl"))
    assert all(row["scoreable"] and row["passed"] for row in _read_rows(tmp_path / "scored.jsonl"))
    full_summary = {k: v for k, v in full.items() if k != "mean_latency_seconds"}
    scored_summary = {k: v for k, v in scored.items() if k != "mean_latency_seconds"}
    assert full_summary == scored_summary
    assert full["samples"] == 2
    assert full["pass_rate"] == 1.0


def test_condition_writers_refuse_overwrite(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from reverse_reap.causal import generate_condition, score_condition
    from reverse_reap.config import load_config

    _patch_generation_runtime(monkeypatch)
    config = load_config(Path(__file__).parents[1] / "configs" / "pinned-3090-bf16-gen.yaml")
    manifest = _validation_manifest(tmp_path)
    destination = tmp_path / "gen.jsonl"
    generate_condition(
        _FakeModel(), _FakeTokenizer(), object(), manifest, destination, config,
        split="validation", condition_id="cT",
    )
    with pytest.raises(CausalError, match="refusing to overwrite"):
        generate_condition(
            _FakeModel(), _FakeTokenizer(), object(), manifest, destination, config,
            split="validation", condition_id="cT",
        )
    scored = tmp_path / "scored.jsonl"
    scored.write_text("")
    with pytest.raises(CausalError, match="refusing to overwrite"):
        score_condition(destination, manifest, scored, evaluator_image="eval@sha256:" + "a" * 64)


def test_generation_determinism_pregate(tmp_path):
    from reverse_reap.causal import compare_generation_determinism

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows = [{"sample_id": f"s-{index}", "response": "same"} for index in range(3)]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    first.write_text(content)
    second.write_text(content)
    result = compare_generation_determinism(first, second)
    assert result["passed"] and result["samples"] == 3
    changed = list(rows)
    changed[1] = {**changed[1], "response": "different"}
    second.write_text("".join(json.dumps(row) + "\n" for row in changed))
    result = compare_generation_determinism(first, second)
    assert not result["passed"] and result["mismatched_sample_ids"] == ["s-1"]
    second.write_text("".join(json.dumps({"sample_id": "other", "response": "same"}) + "\n"))
    with pytest.raises(CausalError, match="different sample IDs"):
        compare_generation_determinism(first, second)


def test_instrument_noop_uses_empty_mask_and_records_zero_experts(tmp_path, monkeypatch):
    """c0-noop-masked: the intervention path with an empty mask is a numeric no-op.

    The instrumentation wrapper must be entered with masked=frozenset() (which
    instrument_qwen35 treats as a transparent passthrough) and the record must
    attribute zero masked experts, so the pre-gate can compare it against
    c0-baseline-a to prove the wrapper itself cannot perturb generation.
    """
    pytest.importorskip("torch")
    import contextlib

    from reverse_reap.causal import generate_condition
    from reverse_reap.config import load_config

    entered_with = []
    real_nullcontext = contextlib.nullcontext

    def recording_instrument(architecture, masked=None):
        entered_with.append(masked)
        return real_nullcontext()

    monkeypatch.setattr(
        "reverse_reap.causal.load_donor", lambda *a, **k: (_FakeModel(), _FakeTokenizer())
    )
    monkeypatch.setattr("reverse_reap.causal.instrument_qwen35", recording_instrument)
    config = load_config(Path(__file__).parents[1] / "configs" / "pinned-3090-bf16-gen.yaml")
    manifest = _validation_manifest(tmp_path)

    summary = generate_condition(
        _FakeModel(),
        _FakeTokenizer(),
        object(),
        manifest,
        tmp_path / "noop.jsonl",
        config,
        split="validation",
        condition_id="c0-noop-masked",
        expert_manifest=None,
        instrument_noop=True,
    )
    assert entered_with == [frozenset()] * 2  # once per sample, always empty mask
    rows = _read_rows(tmp_path / "noop.jsonl")
    assert len(rows) == 2
    assert all(row["masked_experts"] == 0 for row in rows)
    assert summary["masked_experts"] == 0

    # Without instrument_noop the baseline path must NOT touch the wrapper.
    entered_with.clear()
    generate_condition(
        _FakeModel(),
        _FakeTokenizer(),
        object(),
        manifest,
        tmp_path / "baseline.jsonl",
        config,
        split="validation",
        condition_id="c0-baseline-a",
    )
    assert entered_with == []
    assert all(row["masked_experts"] == 0 for row in _read_rows(tmp_path / "baseline.jsonl"))
