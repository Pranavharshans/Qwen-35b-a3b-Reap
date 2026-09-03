from types import SimpleNamespace

import numpy as np
import pytest

from reverse_reap.datasets import normalize_sample
from reverse_reap.instrumentation import CaptureState
from reverse_reap.qwen35 import Qwen35Architecture
from reverse_reap.routing import RouterBatch
from reverse_reap.runtime import RuntimeCompatibilityError, _segment_rows, validate_donor_contract


def architecture(num_layers=40):
    layers = tuple(
        SimpleNamespace(mlp=SimpleNamespace(shared_expert=object())) for _ in range(num_layers)
    )
    return Qwen35Architecture(layers, 256, 8, 2048, 512, "model.language_model.layers")


def test_exact_donor_contract_accepts_expected_shape():
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))
    report = validate_donor_contract(model, architecture())
    assert report["compatible"]


def test_donor_contract_fails_closed_on_architecture_drift():
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))
    with pytest.raises(RuntimeCompatibilityError, match="num_layers"):
        validate_donor_contract(model, architecture(num_layers=39))


def test_segment_subtraction_recovers_completion_only_statistics():
    prompt = CaptureState(1, 3)
    full = CaptureState(1, 3)
    prompt_batch = RouterBatch(
        indices=np.array([[0, 1]]),
        weights=np.array([[0.6, 0.4]]),
    )
    prompt.accumulators[0].update(prompt_batch, [[2.0, 3.0]])
    full.accumulators[0].update(prompt_batch, [[2.0, 3.0]])
    full.accumulators[0].update(
        RouterBatch(
            indices=np.array([[1, 2]]),
            weights=np.array([[0.7, 0.3]]),
        ),
        [[4.0, 5.0]],
    )
    sample = normalize_sample(
        {
            "source": "fixture",
            "source_revision": "abc",
            "source_id": "one",
            "domain": "coding",
            "stratum": "synthesis",
            "language": "python",
            "prompt": "write code",
            "reference": "pass",
            "scorer": "exact_match",
        },
        seed=1,
    )
    rows = _segment_rows(full, prompt, sample)
    completion = [row for row in rows if row["segment"] == "completion"]
    assert {(row["expert"], row["routed_count"]) for row in completion} == {(1, 1), (2, 1)}
    expert_one = next(row for row in completion if row["expert"] == 1)
    assert expert_one["reap_saliency"] == pytest.approx(2.8)
