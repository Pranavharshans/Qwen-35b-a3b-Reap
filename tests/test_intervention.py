"""Optimized intervention-only causal path (no telemetry side path)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from reverse_reap.instrumentation import (
    FROZEN_SELECTED_TOP4,
    instrument_qwen35,
    intervene_qwen35,
)
from reverse_reap.qwen35 import Qwen35Architecture

#: Declared bound for optimized-vs-slow-oracle agreement in BF16.
BF16_TOLERANCE = 0.0625


def test_frozen_selected_top4_matches_directive():
    assert frozenset({(35, 239), (3, 26), (7, 18), (37, 5)}) == FROZEN_SELECTED_TOP4


def test_intervention_path_accepts_no_observer_or_telemetry():
    assert "observer" not in inspect.signature(intervene_qwen35).parameters


def _make_spy(torch, num_experts=4, hidden=8, intermediate=8):
    class SpyExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(
                torch.randn(num_experts, 2 * intermediate, hidden)
            )
            self.down_proj = torch.nn.Parameter(
                torch.randn(num_experts, hidden, intermediate)
            )
            self.act_fn = torch.nn.functional.silu
            self.seen = []

        def forward(self, hidden_states, top_k_index, top_k_weights):
            self.seen.append(
                {
                    "index_obj": top_k_index,
                    "weight_obj": top_k_weights,
                    "indices": top_k_index.clone(),
                    "weights": top_k_weights.clone(),
                }
            )
            result = torch.zeros_like(hidden_states)
            mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=num_experts
            ).permute(2, 1, 0)
            for expert_tensor in torch.nonzero(mask.sum(dim=(1, 2))).flatten():
                expert = int(expert_tensor.item())
                ranks, tokens = torch.where(mask[expert_tensor])
                gate_up = torch.nn.functional.linear(
                    hidden_states[tokens], self.gate_up_proj[expert]
                )
                gate, up = gate_up.chunk(2, dim=-1)
                output = self.act_fn(gate) * up
                output = torch.nn.functional.linear(output, self.down_proj[expert])
                result.index_add_(0, tokens, output * top_k_weights[tokens, ranks, None])
            return result

    return SpyExperts()


def _architecture(spies, num_experts=4, top_k=2, hidden=8, intermediate=8):
    layers = tuple(SimpleNamespace(mlp=SimpleNamespace(experts=spy)) for spy in spies)
    return Qwen35Architecture(layers, num_experts, top_k, hidden, intermediate, "model.layers")


def _inputs(torch):
    torch.manual_seed(5)
    hidden = torch.randn(3, 8)
    indices = torch.tensor([[0, 1], [2, 3], [1, 2]])
    weights = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])
    return hidden, indices, weights


def test_empty_mask_matches_native_bit_for_bit():
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch)
    expected = spy(hidden, indices, weights)
    with intervene_qwen35(architecture, masked=frozenset()):
        actual = spy(hidden, indices, weights)
    assert torch.equal(actual, expected)
    # Zero additional tensor work: the original receives the identical objects.
    assert spy.seen[-1]["index_obj"] is indices
    assert spy.seen[-1]["weight_obj"] is weights


def test_router_indices_pass_through_unchanged():
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch)
    masked = frozenset({(0, 1)})
    with intervene_qwen35(architecture, masked=masked):
        spy(hidden, indices, weights)
    recorded = spy.seen[-1]
    assert recorded["index_obj"] is indices
    assert torch.equal(recorded["indices"], indices)


def test_only_targeted_weights_zeroed_and_survivors_unchanged():
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch)
    masked = frozenset({(0, 1)})
    weights_before = weights.clone()
    with intervene_qwen35(architecture, masked=masked):
        spy(hidden, indices, weights)
    recorded = spy.seen[-1]["weights"]
    # The caller's tensor is never mutated in place.
    assert torch.equal(weights, weights_before)
    # Only routes to expert 1 become zero; the clone proves the copy-on-write.
    assert spy.seen[-1]["weight_obj"] is not weights
    zero_expected = indices == 1
    assert zero_expected.any()
    assert (recorded[zero_expected] == 0).all()
    assert torch.equal(recorded[~zero_expected], weights[~zero_expected])


def test_no_renormalization_of_surviving_weights():
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch)
    masked = frozenset({(0, 1)})
    with intervene_qwen35(architecture, masked=masked):
        spy(hidden, indices, weights)
    recorded = spy.seen[-1]["weights"]
    # Row sums drop by exactly the masked weight instead of rescaling to 1.
    assert recorded[0].sum().item() == pytest.approx(0.6)
    assert recorded[2].sum().item() == pytest.approx(0.3)
    # Rows without a masked route keep their full normalized mass.
    assert recorded[1].sum().item() == pytest.approx(1.0)


def test_mask_applies_per_layer_only():
    torch = pytest.importorskip("torch")
    first, second = _make_spy(torch), _make_spy(torch)
    second.load_state_dict(first.state_dict())
    architecture = _architecture([first, second])
    hidden, indices, weights = _inputs(torch)
    masked = frozenset({(0, 1), (1, 2)})
    with intervene_qwen35(architecture, masked=masked):
        first(hidden, indices, weights)
        second(hidden, indices, weights)
    assert torch.equal(first.seen[-1]["weights"][indices == 2], weights[indices == 2])
    assert (first.seen[-1]["weights"][indices == 1] == 0).all()
    assert torch.equal(second.seen[-1]["weights"][indices == 1], weights[indices == 1])
    assert (second.seen[-1]["weights"][indices == 2] == 0).all()


def test_matches_slow_oracle_within_bf16_tolerance():
    torch = pytest.importorskip("torch")
    masked = frozenset({(0, 1), (0, 3)})
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(9)
        spy = _make_spy(torch)
        if dtype is torch.bfloat16:
            spy.to(torch.bfloat16)
        architecture = _architecture([spy])
        hidden = torch.randn(3, 8, dtype=dtype)
        indices = torch.tensor([[0, 1], [2, 3], [1, 2]])
        weights = torch.tensor(
            [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]], dtype=dtype
        )
        with intervene_qwen35(architecture, masked=masked):
            fast = spy(hidden, indices, weights)
        with instrument_qwen35(architecture, masked=masked):
            slow = spy(hidden, indices, weights)
        if dtype is torch.float32:
            assert torch.equal(fast, slow)
        else:
            maximum = float((fast.float() - slow.float()).abs().max().item())
            assert maximum <= BF16_TOLERANCE


def test_unrouted_selected_experts_change_nothing():
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    hidden = torch.randn(2, 8)
    indices = torch.tensor([[0, 1], [0, 1]])
    weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    expected = spy(hidden, indices, weights)
    # In-range experts that no token routes to, plus an out-of-range layer.
    masked = frozenset({(0, 2), (0, 3), (5, 0)})
    with intervene_qwen35(architecture, masked=masked):
        actual = spy(hidden, indices, weights)
    assert torch.equal(actual, expected)
    assert torch.equal(spy.seen[-1]["weights"], weights)


def test_context_manager_restores_after_success_and_exception():
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch)
    original_func = spy.forward.__func__
    with intervene_qwen35(architecture, masked=frozenset({(0, 0)})):
        spy(hidden, indices, weights)
    assert spy.forward.__func__ is original_func
    with pytest.raises(RuntimeError, match="boom"), intervene_qwen35(
        architecture, masked=frozenset({(0, 0)})
    ):
        spy(hidden, indices, weights)
        raise RuntimeError("boom")
    assert spy.forward.__func__ is original_func


def test_batched_and_partial_resumption_deterministic():
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    torch.manual_seed(13)
    hidden = torch.randn(4, 8)
    indices = torch.tensor([[0, 1], [2, 3], [1, 2], [0, 3]])
    weights = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8]])
    masked = frozenset({(0, 3)})
    with intervene_qwen35(architecture, masked=masked):
        full = spy(hidden, indices, weights)
        repeat = spy(hidden, indices, weights)
        first_half = spy(hidden[:2], indices[:2], weights[:2])
        second_half = spy(hidden[2:], indices[2:], weights[2:])
    assert torch.equal(repeat, full)
    # Chunked-vs-full is allclose, not bitwise: the native path itself diverges
    # ~3.8e-06 across batch chunkings (GEMM batch-shape sensitivity), and the
    # intervened diff (1.9e-06, measured on the 4x3090 GPU run) sits at the
    # same scale — so the bound below attributes nothing to the intervention.
    chunked = torch.cat([first_half, second_half])
    assert torch.allclose(chunked, full, atol=1e-4, rtol=1e-4)
    assert float((chunked - full).abs().max().item()) <= 1e-4


def test_no_cpu_copy_norms_or_telemetry_during_intervention(monkeypatch):
    torch = pytest.importorskip("torch")
    spy = _make_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch)
    masked = frozenset({(0, 1)})

    def _forbidden(*args, **kwargs):
        raise AssertionError("telemetry side path must stay untouched")

    # Oracle expectation computed on the untouched native path first.
    modified = weights.clone()
    modified[indices == 1] = 0
    expected = spy(hidden, indices, modified)

    monkeypatch.setattr(torch.Tensor, "cpu", _forbidden)
    monkeypatch.setattr(torch.linalg, "vector_norm", _forbidden)
    with intervene_qwen35(architecture, masked=masked) as yielded:
        actual = spy(hidden, indices, weights)
    assert yielded is None
    assert torch.equal(actual, expected)
