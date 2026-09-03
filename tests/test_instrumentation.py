from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from reverse_reap.instrumentation import instrument_qwen35  # noqa: E402
from reverse_reap.qwen35 import Qwen35Architecture  # noqa: E402


class TinyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(3, 4, 4))
        self.down_proj = torch.nn.Parameter(torch.randn(3, 4, 2))
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=3).permute(2, 1, 0)
        for expert_tensor in torch.nonzero(mask.sum(dim=(1, 2))).flatten():
            expert = int(expert_tensor.item())
            ranks, tokens = torch.where(mask[expert_tensor])
            gate_up = torch.nn.functional.linear(hidden_states[tokens], self.gate_up_proj[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            output = torch.nn.functional.silu(gate) * up
            output = torch.nn.functional.linear(output, self.down_proj[expert])
            result.index_add_(0, tokens, output * top_k_weights[tokens, ranks, None])
        return result


def architecture(experts):
    layer = SimpleNamespace(mlp=SimpleNamespace(experts=experts))
    return Qwen35Architecture((layer,), 3, 2, 4, 2, "model.language_model.layers")


def test_instrumented_forward_is_exact_and_restores_original():
    torch.manual_seed(7)
    experts = TinyExperts()
    hidden = torch.randn(2, 4)
    indices = torch.tensor([[0, 1], [2, 1]])
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    expected = experts(hidden, indices, weights)
    original_func = experts.forward.__func__
    with instrument_qwen35(architecture(experts)) as capture:
        actual = experts(hidden, indices, weights)
        assert torch.equal(actual, expected)
        records = capture.accumulators[0].records()
        assert [record["routed_count"] for record in records] == [1, 2, 1]
        assert all(record["reap_saliency"] > 0 for record in records)
    assert experts.forward.__func__ is original_func


def test_mask_zeroes_only_selected_weighted_contribution_without_renormalizing():
    torch.manual_seed(11)
    experts = TinyExperts()
    hidden = torch.randn(1, 4)
    indices = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.25, 0.75]])
    with instrument_qwen35(architecture(experts), masked=frozenset({(0, 0)})):
        masked_output = experts(hidden, indices, weights)
    with instrument_qwen35(architecture(experts), masked=frozenset({(0, 1)})):
        other_output = experts(hidden, indices, weights)
    full = experts(hidden, indices, weights)
    assert torch.allclose(masked_output + other_output, full)


def test_observer_receives_exact_route_level_norm_matrix():
    torch.manual_seed(12)
    experts = TinyExperts()
    hidden = torch.randn(2, 4)
    indices = torch.tensor([[0, 1], [2, 1]])
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    observed = []
    with instrument_qwen35(
        architecture(experts),
        observer=lambda layer, batch, norms: observed.append((layer, batch, norms)),
    ):
        experts(hidden, indices, weights)
    assert len(observed) == 1
    layer, batch, norms = observed[0]
    assert layer == 0
    assert batch.indices.shape == (2, 2)
    assert norms.shape == (2, 2)
    assert (norms > 0).all()
