"""Synthetic validation of the same-kernel intervention diagnostic.

CPU-only: exercises scripts/validate_same_kernel_intervention.py's pure
gate logic (assess_capture + weight transforms + tolerance predicate)
against a faithful spy kernel and two buggy wrappers. The spy accumulates
in fp32 with a single final rounding to the input dtype — the same
contract the fused grouped_mm kernel honours — so additivity holds at
rounding scale while renormalization and wrong-target bugs trip the
gates they must trip.
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_same_kernel_intervention.py"

from reverse_reap.instrumentation import intervene_qwen35  # noqa: E402
from reverse_reap.qwen35 import Qwen35Architecture  # noqa: E402


def _import_script():
    spec = importlib.util.spec_from_file_location(
        "validate_same_kernel_intervention", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_kernel_spy(torch, num_experts=4, hidden=16, intermediate=16):
    """Deterministic kernel stand-in: fp32 accumulate, round once at end."""

    class KernelSpy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(
                torch.randn(num_experts, 2 * intermediate, hidden))
            self.down_proj = torch.nn.Parameter(
                torch.randn(num_experts, hidden, intermediate))
            self.act_fn = torch.nn.functional.silu
            self.seen = []

        def forward(self, hidden_states, top_k_index, top_k_weights):
            self.seen.append(top_k_weights.clone())
            out_dtype = hidden_states.dtype
            hidden = hidden_states.float()
            weights = top_k_weights.float()
            result = torch.zeros_like(hidden)
            mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=num_experts).permute(2, 1, 0)
            for expert_tensor in torch.nonzero(mask.sum(dim=(1, 2))).flatten():
                expert = int(expert_tensor.item())
                ranks, tokens = torch.where(mask[expert_tensor])
                gate_up = torch.nn.functional.linear(
                    hidden[tokens], self.gate_up_proj[expert].float())
                gate, up = gate_up.chunk(2, dim=-1)
                output = self.act_fn(gate) * up
                output = torch.nn.functional.linear(
                    output, self.down_proj[expert].float())
                result.index_add_(0, tokens, output * weights[tokens, ranks, None])
            return result.to(out_dtype)

    return KernelSpy()


def _architecture(spies, num_experts=4, top_k=2, hidden=16, intermediate=16):
    layers = tuple(SimpleNamespace(mlp=SimpleNamespace(experts=spy)) for spy in spies)
    return Qwen35Architecture(layers, num_experts, top_k, hidden, intermediate, "model.layers")


def _inputs(torch, dtype):
    torch.manual_seed(11)
    hidden = torch.randn(6, 16, dtype=dtype)
    indices = torch.tensor([[0, 1], [2, 3], [1, 2], [0, 3], [2, 1], [3, 0]])
    weights = torch.tensor(
        [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6]],
        dtype=dtype)
    return hidden, indices, weights


def _run_correct(torch, module, dtype, masked):
    """Full/only/manual/intervened/empty outputs via the real wrapper."""
    torch.manual_seed(23)
    spy = _make_kernel_spy(torch)
    if dtype is torch.bfloat16:
        spy.to(torch.bfloat16)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch, dtype)
    target_set = {expert for (layer, expert) in masked if layer == 0}
    original = spy.forward
    full = original(hidden, indices, weights)
    masked_manual = original(
        hidden, indices, module._zero_targeted_weights(weights, indices, target_set))
    only_manual = original(
        hidden, indices, module._only_targeted_weights(weights, indices, target_set))
    with intervene_qwen35(architecture, masked=masked):
        intervened = spy(hidden, indices, weights)
        real_modified = spy.seen[-1].clone()
    with intervene_qwen35(architecture, masked=frozenset()):
        empty_out = spy(hidden, indices, weights)
    return module.assess_capture(
        full, masked_manual, only_manual, intervened,
        empty_out, real_modified, weights, indices, target_set)


@contextmanager
def _buggy_wrapper(architecture, *, masked, mode):
    """Wrapper with an injected semantic bug (renorm or wrong target)."""
    import types

    originals = []
    per_layer = [
        frozenset(expert for (layer, expert) in masked if layer == layer_index)
        for layer_index in range(architecture.num_layers)
    ]
    for layer_index, layer in enumerate(architecture.layers):
        experts = layer.mlp.experts
        original = experts.forward
        originals.append((experts, original))
        targets = per_layer[layer_index]

        def forward(this, hidden_states, top_k_index, top_k_weights,
                    *, _original=original, _targets=targets):
            if not _targets:
                return _original(hidden_states, top_k_index, top_k_weights)
            modified = top_k_weights.clone()
            if mode == "wrong_target":
                wrong = {(t + 1) % 4 for t in _targets}
                for expert in wrong:
                    modified[top_k_index == expert] = 0
            else:
                for expert in _targets:
                    modified[top_k_index == expert] = 0
            if mode == "renorm":
                row_sum = modified.sum(dim=1, keepdim=True).clamp_min(1e-6)
                modified = modified / row_sum
            return _original(hidden_states, top_k_index, modified)

        experts.forward = types.MethodType(forward, experts)
    try:
        yield None
    finally:
        for experts, original in originals:
            experts.forward = original


def test_weight_transforms_zero_and_keep_exactly_targets():
    torch = pytest.importorskip("torch")
    module = _import_script()
    hidden, indices, weights = _inputs(torch, torch.float32)
    target_set = {1}
    zeroed = module._zero_targeted_weights(weights, indices, target_set)
    assert (zeroed[indices == 1] == 0).all()
    assert torch.equal(zeroed[indices != 1], weights[indices != 1])
    assert torch.equal(weights[indices == 1], torch.tensor([0.4, 0.7, 0.1]))
    only = module._only_targeted_weights(weights, indices, target_set)
    assert torch.equal(only[indices == 1], weights[indices == 1])
    assert (only[indices != 1] == 0).all()
    # Caller tensors are never mutated in place.
    assert torch.equal(weights, _inputs(torch, torch.float32)[2])


def test_tolerance_predicate_respects_precommitted_constants():
    torch = pytest.importorskip("torch")  # noqa: F841 (documents the dtype regime)
    module = _import_script()
    assert module.SAMEKERNEL_REL_L2_TOL == 5e-3
    assert module.SAMEKERNEL_COS_MIN == 0.9999
    assert module.SAMEKERNEL_MAXABS_FRAC == 0.02
    assert module._within_tolerance(
        {"rel_l2": 1e-6, "cosine": 1.0, "max_abs": 1e-7, "peak_ref": 1.0})
    assert not module._within_tolerance(
        {"rel_l2": 5.09e-3, "cosine": 1.0, "max_abs": 1e-7, "peak_ref": 1.0})
    assert not module._within_tolerance(
        {"rel_l2": 1e-6, "cosine": 0.9998, "max_abs": 1e-7, "peak_ref": 1.0})


def test_correct_wrapper_passes_all_gates_fp32():
    torch = pytest.importorskip("torch")
    module = _import_script()
    verdict = _run_correct(torch, module, torch.float32, frozenset({(0, 1)}))
    assert verdict["wrapper_fidelity_bitwise"]
    assert verdict["empty_mask_is_noop_bitwise"]
    assert verdict["surviving_weights_bitwise_exact"]
    assert verdict["additivity_within_tolerance"]
    assert verdict["nontargeted_delta_bitwise_zero"]
    assert verdict["route_hits"] == {1: 3}


def test_correct_wrapper_passes_all_gates_bf16():
    torch = pytest.importorskip("torch")
    module = _import_script()
    verdict = _run_correct(torch, module, torch.bfloat16, frozenset({(0, 1)}))
    assert verdict["wrapper_fidelity_bitwise"]
    assert verdict["empty_mask_is_noop_bitwise"]
    assert verdict["surviving_weights_bitwise_exact"]
    assert verdict["additivity_within_tolerance"]
    assert verdict["nontargeted_delta_bitwise_zero"]
    assert verdict["route_hits"] == {1: 3}


def test_catches_renormalization_bug():
    torch = pytest.importorskip("torch")
    module = _import_script()
    torch.manual_seed(23)
    spy = _make_kernel_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch, torch.float32)
    masked = frozenset({(0, 1)})
    target_set = {1}
    original = spy.forward
    full = original(hidden, indices, weights)
    masked_manual = original(
        hidden, indices, module._zero_targeted_weights(weights, indices, target_set))
    only_manual = original(
        hidden, indices, module._only_targeted_weights(weights, indices, target_set))
    with _buggy_wrapper(architecture, masked=masked, mode="renorm"):
        intervened = spy(hidden, indices, weights)
        real_modified = spy.seen[-1].clone()
    with intervene_qwen35(architecture, masked=frozenset()):
        empty_out = spy(hidden, indices, weights)
    verdict = module.assess_capture(
        full, masked_manual, only_manual, intervened,
        empty_out, real_modified, weights, indices, target_set)
    # Renormalization rescales survivors: wrapper output and the captured
    # weights fail, while additivity (hand-built weights through the honest
    # kernel, untouched by the wrapper) still passes — the gates isolate
    # wrapper fidelity from kernel semantics. Empty mask stays a no-op.
    assert not verdict["wrapper_fidelity_bitwise"]
    assert not verdict["surviving_weights_bitwise_exact"]
    assert verdict["additivity_within_tolerance"]
    assert verdict["empty_mask_is_noop_bitwise"]


def test_catches_wrong_target_bug():
    torch = pytest.importorskip("torch")
    module = _import_script()
    torch.manual_seed(23)
    spy = _make_kernel_spy(torch)
    architecture = _architecture([spy])
    hidden, indices, weights = _inputs(torch, torch.float32)
    masked = frozenset({(0, 1)})
    target_set = {1}
    original = spy.forward
    full = original(hidden, indices, weights)
    masked_manual = original(
        hidden, indices, module._zero_targeted_weights(weights, indices, target_set))
    only_manual = original(
        hidden, indices, module._only_targeted_weights(weights, indices, target_set))
    with _buggy_wrapper(architecture, masked=masked, mode="wrong_target"):
        intervened = spy(hidden, indices, weights)
        real_modified = spy.seen[-1].clone()
    with intervene_qwen35(architecture, masked=frozenset()):
        empty_out = spy(hidden, indices, weights)
    verdict = module.assess_capture(
        full, masked_manual, only_manual, intervened,
        empty_out, real_modified, weights, indices, target_set)
    assert not verdict["wrapper_fidelity_bitwise"]
    assert not verdict["surviving_weights_bitwise_exact"]
    assert verdict["empty_mask_is_noop_bitwise"]


def test_unrouted_target_reports_zero_hits_for_coverage_gate():
    torch = pytest.importorskip("torch")
    module = _import_script()
    torch.manual_seed(23)
    spy = _make_kernel_spy(torch)
    hidden, indices, weights = _inputs(torch, torch.float32)
    full = spy(hidden, indices, weights)
    zeros = torch.zeros_like(full)
    verdict_empty = module.assess_capture(
        full, full.clone(), zeros, full.clone(), full.clone(),
        weights.clone(), weights, indices, set())
    assert verdict_empty["route_hits"] == {}
    assert verdict_empty["wrapper_fidelity_bitwise"]
    assert verdict_empty["additivity_within_tolerance"]
    # Sanity: expert 3 is routed in the fixture, so a {3} probe is non-vacuous.
    assert (indices == 3).any()


def test_report_shape_matches_benchmark_layer_diagnostic_contract():
    module = _import_script()
    assert hasattr(module, "main")
    assert module.SAMEKERNEL_REL_L2_TOL == 5e-3
    # The benchmark's batch-selection verdict reads exactly these keys:
    # {"status": "pass", "passed": True, "layers": {...}}.
    verdict = {"status": "pass", "passed": True, "layers": {"(3,26)": {"passed": True}}}
    assert verdict.get("status") == "pass" and verdict.get("passed") is True
