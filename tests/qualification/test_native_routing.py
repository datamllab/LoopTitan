"""Counterexamples to pooled counts as an exact logical routing oracle."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from compare_native import RouterCall, compare_routing, routing_trace
from moeloop.model import ModelConfig
from moeloop.moe import Router


def test_equal_counts_do_not_hide_routes_swapped_between_tokens():
    first = torch.tensor([[0], [1]])
    swapped = first.flip(0)
    assert torch.equal(torch.bincount(first.flatten()), torch.bincount(swapped.flatten()))
    valid = torch.ones(2, dtype=torch.bool)
    result = compare_routing([RouterCall("shared", valid, first)],
                             [RouterCall("shared", valid, swapped)])
    assert not result["equal"] and "routing call 0" in result["failures"][0]


def test_equal_counts_do_not_hide_routes_swapped_between_visits():
    valid = torch.ones(1, dtype=torch.bool)
    first = [RouterCall("shared", valid, torch.tensor([[expert]])) for expert in (0, 1)]
    swapped = list(reversed(first))
    result = compare_routing(first, swapped)
    assert not result["equal"] and len(result["failures"]) == 2


def test_trace_excludes_padding_and_ignores_top_k_order():
    router = Router(ModelConfig(vocab_size=4, dim=4, hidden_dim=8,
        num_heads=1, num_experts=3, top_k=2))
    with torch.no_grad():
        router.gate.weight.copy_(torch.eye(3, 4))
    inputs = torch.tensor([[3., 2., 0., 0.], [0., 0., 9., 0.]])
    valid = torch.tensor([True, False])
    with routing_trace(router) as first:
        output = router(inputs, valid)
    inputs[1] = torch.tensor([9., 0., 0., 0.])
    with routing_trace(router) as second:
        router(inputs, valid)
    assert output.weights.requires_grad
    assert first[0].ids.shape == (1, 2)
    assert torch.equal(first[0].valid_mask, valid)
    second[0] = second[0]._replace(ids=second[0].ids.flip(-1))
    report = compare_routing(first, second)
    assert report["equal"] and report["valid_token_expert_assignments"] == 2
    assert report["calls"][0]["native_ids_sha256"] == report["calls"][0]["reference_ids_sha256"]
    assert not router._forward_hooks
