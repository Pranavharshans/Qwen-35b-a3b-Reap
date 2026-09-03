import numpy as np
import pytest

from reverse_reap.routing import (
    RouterBatch,
    RoutingError,
    StreamingReapAccumulator,
    validate_router_batch,
)


def test_router_validation_and_exact_reap_formula():
    batch = validate_router_batch(
        np.array([[1, 2], [2, 0]]),
        np.array([[0.75, 0.25], [0.6, 0.4]]),
        num_experts=3,
        expected_top_k=2,
    )
    accumulator = StreamingReapAccumulator(3)
    accumulator.update(batch, np.array([[2.0, 4.0], [5.0, 10.0]]))
    records = accumulator.records()
    assert records[1]["reap_saliency"] == pytest.approx(1.5)
    assert records[2]["reap_saliency"] == pytest.approx((1.0 + 3.0) / 2)
    assert records[0]["reap_saliency"] == pytest.approx(4.0)
    assert records[2]["routed_count"] == 2


@pytest.mark.parametrize(
    ("indices", "weights", "message"),
    [
        ([[0, 0]], [[0.5, 0.5]], "same expert twice"),
        ([[0, 3]], [[0.5, 0.5]], "outside configured range"),
        ([[0, 1]], [[0.2, 0.2]], "normalized"),
    ],
)
def test_rejects_invalid_router_batches(indices, weights, message):
    with pytest.raises(RoutingError, match=message):
        validate_router_batch(np.array(indices), np.array(weights), num_experts=3)


def test_streaming_merge_matches_single_pass():
    batch = validate_router_batch(
        np.array([[0, 1], [1, 2]]), np.full((2, 2), 0.5), num_experts=3
    )
    left = StreamingReapAccumulator(3)
    right = StreamingReapAccumulator(3)
    whole = StreamingReapAccumulator(3)
    left.update(RouterBatch(batch.indices[:1], batch.weights[:1]), [[2.0, 4.0]])
    right.update(RouterBatch(batch.indices[1:], batch.weights[1:]), [[6.0, 8.0]])
    whole.update(batch, [[2.0, 4.0], [6.0, 8.0]])
    left.merge(right)
    assert left.records() == whole.records()
