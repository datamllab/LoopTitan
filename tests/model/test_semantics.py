"""Small equation/gradient checks for the public model, not training claims."""

from dataclasses import replace
import io
import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from moeloop.model import ModelConfig, RecursiveLM
from moeloop.moe import ExpertBank, Router


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(72)
    yield
    torch.set_num_threads(previous)


def tiny(**changes):
    return replace(ModelConfig(vocab_size=31, dim=16, hidden_dim=24, num_heads=4,
                               num_kv_heads=2, num_experts=4, top_k=2,
                               num_cells=1, num_visits=3), **changes)


def test_config_roundtrip_and_invalid_combinations():
    config = tiny(attention_per_visit=True, visit_capacity=5, input_injection="linear")
    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    for update in ({"unknown": 3}, {"num_heads": 3}, {"num_kv_heads": 3},
                   {"num_visits": True}, {"order": "mystery"},
                   {"order": "layer", "input_injection": "add"},
                   {"visit_capacity": 2}, {"rms_eps": float("nan")}):
        values = tiny().to_dict() | update
        with pytest.raises(ValueError):
            ModelConfig.from_dict(values)


def test_one_canonical_bank_and_independent_attention_owners():
    model = RecursiveLM(tiny(attention_per_visit=True, router_per_visit=True,
                             norm_per_visit=True, prefix_cells=1, suffix_cells=1))
    cell = model.core[0]
    assert len(cell.attentions) == len(cell.routers) == len(cell.ffn_norms) == 3
    assert len(model.prefix[0].attentions) == len(model.suffix[0].routers) == 1
    assert cell.attentions[0].q_proj.weight is not cell.attentions[1].q_proj.weight
    assert model.embedding.weight is not model.lm_head.weight
    parameters = list(model.canonical_parameters())
    assert len(parameters) == len({id(p) for _, p in parameters})
    assert [name for name, _ in parameters if "core.0.experts.w1" in name] == ["core.0.experts.w1_EFD"]
    assert "core.0.attentions.2.k_proj.weight" in model.state_dict()
    with pytest.raises(ValueError, match="capacity"):
        model(torch.tensor([[1, 2]]), num_visits=4)
    shared = RecursiveLM(tiny())
    assert shared(torch.tensor([[1, 2]]), num_visits=5).logits.shape == (1, 2, 31)


def test_factory_cannot_alias_parameters_across_physical_owners():
    config = tiny(num_cells=2)
    bank = ExpertBank(config)
    with pytest.raises(ValueError, match="canonical owner"):
        RecursiveLM(config, expert_factory=lambda _: bank)


def test_shared_forward_and_gradient_equal_explicit_unrolling():
    shared = RecursiveLM(tiny())
    unrolled = RecursiveLM(tiny(num_cells=3, num_visits=1))
    unrolled.embedding.load_state_dict(shared.embedding.state_dict())
    unrolled.final_norm.load_state_dict(shared.final_norm.state_dict())
    unrolled.lm_head.load_state_dict(shared.lm_head.state_dict())
    for cell in unrolled.core:
        cell.load_state_dict(shared.core[0].state_dict())
    ids = torch.tensor([[2, 4, 5], [1, 3, 9]])
    one, three = shared(ids), unrolled(ids)
    torch.testing.assert_close(one.logits, three.logits)
    for output in (one, three):
        (output.logits.square().sum() + .17 * output.balance_sum / output.aux_calls
         + .01 * output.z_sum / output.aux_calls).backward()
    for name, parameter in shared.core[0].named_parameters():
        total = sum(dict(cell.named_parameters())[name].grad for cell in unrolled.core)
        torch.testing.assert_close(parameter.grad, total, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(shared.embedding.weight.grad, unrolled.embedding.weight.grad)


def test_stack_and_layer_orders_are_different_functions():
    stack = RecursiveLM(tiny(num_cells=2, num_visits=2, init_std=.2))
    layer = RecursiveLM(replace(stack.config, order="layer"))
    layer.load_state_dict(stack.state_dict())
    ids = torch.tensor([[2, 5, 7, 9]])
    assert not torch.allclose(stack(ids).logits, layer(ids).logits)
    with pytest.raises(ValueError, match="stack"):
        layer(ids, grad_visits=1)


@pytest.mark.parametrize("injection,live_anchor", [("none", False), ("add", True), ("linear", True)])
def test_gradient_suffix_keeps_only_explicit_anchor_path(injection, live_anchor):
    model = RecursiveLM(tiny(num_experts=0, prefix_cells=1, input_injection=injection))
    output = model(torch.tensor([[2, 4, 7]]), grad_visits=1)
    output.logits.square().sum().backward()
    embedding_gradient = model.embedding.weight.grad
    prefix_gradient = model.prefix[0].attentions[0].q_proj.weight.grad
    assert (embedding_gradient is not None) == live_anchor
    assert (prefix_gradient is not None) == live_anchor
    if live_anchor:
        assert embedding_gradient.abs().sum() > 0
        assert prefix_gradient.abs().sum() > 0
    assert model.core[0].attentions[0].q_proj.weight.grad.abs().sum() > 0


def test_aux_calls_count_gradient_calls_and_usage_counts_all_calls():
    model = RecursiveLM(tiny(prefix_cells=1, suffix_cells=1))
    valid = torch.tensor([[True, True, False]])
    output = model(torch.tensor([[2, 4, 0]]), router_mask=valid, grad_visits=1)
    assert output.aux_calls == 3  # live prefix, final core visit, live suffix
    assert output.router_counts.sum(-1).tolist() == [4, 12, 4]
    assert output.balance_sum.requires_grad and output.z_sum.requires_grad
    assert not output.router_counts.requires_grad


def test_router_equations_separate_padding_and_dispatch_counts():
    config = tiny(dim=4, num_heads=1, num_kv_heads=1, hidden_dim=8)
    router = Router(config)
    with torch.no_grad():
        router.gate.weight.copy_(torch.eye(4))
    inputs = torch.tensor([[[3., 2., 0., -1.], [1., -2., 4., 0.], [5., 0., 2., 1.]]], requires_grad=True)
    valid = torch.tensor([[True, False, True]])
    routing = router(inputs, valid)
    probs = inputs.softmax(-1)
    selected = probs.gather(-1, routing.ids)
    torch.testing.assert_close(routing.weights, selected / selected.sum(-1, keepdim=True))
    expected_load = routing.counts.float() / 4
    expected_balance = 4 * (expected_load * probs[:, [0, 2]].mean((0, 1))).sum()
    torch.testing.assert_close(routing.balance, expected_balance)
    torch.testing.assert_close(routing.z_loss, inputs.logsumexp(-1).square()[:, [0, 2]].mean())
    assert routing.counts.sum() == 4
    assert routing.dispatch_counts.sum() == 6
    empty = router(inputs, torch.zeros_like(valid))
    assert empty.balance == empty.z_loss == 0
    assert empty.counts.sum() == 0 and empty.dispatch_counts.sum() == 6
    (empty.balance + empty.z_loss).backward()
    assert inputs.grad.count_nonzero() == 0


def test_reference_experts_match_explicit_weighted_swiglu():
    config = tiny()
    bank = ExpertBank(config)
    for parameter in bank.parameters():
        nn.init.normal_(parameter, std=.2)
    x = torch.randn(2, 3, config.dim)
    ids = torch.tensor([[[0, 2], [3, 1], [1, 2]], [[2, 0], [0, 1], [3, 2]]])
    weights = torch.rand(2, 3, 2)
    expected = torch.zeros_like(x)
    for b in range(2):
        for t in range(3):
            for k in range(2):
                expert = ids[b, t, k]
                hidden = torch.nn.functional.silu(bank.w1_EFD[expert] @ x[b, t]) * (bank.w3_EFD[expert] @ x[b, t])
                expected[b, t] += weights[b, t, k] * (bank.w2_EDF[expert] @ hidden)
    torch.testing.assert_close(bank(x, ids, weights), expected)


def test_gqa_matches_repeated_kv_heads():
    grouped = RecursiveLM(tiny(num_experts=0, num_visits=1))
    repeated = RecursiveLM(replace(grouped.config, num_kv_heads=4))
    state = grouped.state_dict()
    for name in state:
        if name.endswith(("k_proj.weight", "v_proj.weight")):
            state[name] = state[name].view(2, 4, 16).repeat_interleave(2, 0).reshape(16, 16)
    repeated.load_state_dict(state)
    ids = torch.tensor([[2, 3, 9, 7]])
    torch.testing.assert_close(grouped(ids).logits, repeated(ids).logits)


def test_causality_document_isolation_and_padding_invariance():
    model = RecursiveLM(tiny()).eval()
    ids = torch.tensor([[1, 2, 3, 7, 8]])
    documents = torch.tensor([[0, 0, 0, 1, 1]])
    positions = torch.tensor([[0, 1, 2, 0, 1]])
    same_document = documents[:, None, :, None] == documents[:, None, None, :]
    first = model(ids, positions=positions, attention_mask=same_document)
    changed = model(torch.tensor([[5, 6, 9, 7, 8]]), positions=positions, attention_mask=same_document)
    independent = model(torch.tensor([[7, 8]]))
    torch.testing.assert_close(first.logits[:, 3:], changed.logits[:, 3:])
    torch.testing.assert_close(first.logits[:, 3:], independent.logits, atol=2e-7, rtol=2e-5)
    future_changed = model(torch.tensor([[1, 2, 3, 7, 9]]))
    original = model(ids)
    torch.testing.assert_close(original.logits[:, :4], future_changed.logits[:, :4])
    padded_ids = torch.tensor([[0, 0, 7, 8], [7, 8, 0, 0]])
    valid = torch.tensor([[False, False, True, True], [True, True, False, False]])
    padded = model(padded_ids, attention_mask=valid)
    torch.testing.assert_close(padded.logits[0, 2:], independent.logits[0], atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(padded.logits[1, :2], independent.logits[0], atol=2e-7, rtol=2e-5)
    assert padded.router_counts.sum() == 4 * 2 * 3


def test_hidden_path_skips_head_and_uses_same_normalized_state():
    model = RecursiveLM(tiny())
    calls = []
    hook = model.lm_head.register_forward_hook(lambda *args: calls.append(1))
    ids = torch.tensor([[1, 2]])
    hidden = model(ids, return_hidden=True)
    assert hidden.logits is None and calls == []
    full = model(ids)
    assert calls == [1]
    torch.testing.assert_close(model.lm_head(hidden.hidden_states), full.logits)
    hook.remove()


def test_meta_materialization_initialization_and_strict_state_roundtrip():
    config = tiny(attention_per_visit=True, router_per_visit=True, norm_per_visit=True)
    with torch.device("meta"):
        model = RecursiveLM(config)
    model.to_empty(device="cpu")
    model.initialize_weights(generator=torch.Generator().manual_seed(9))
    for _, parameter in model.canonical_parameters():
        assert torch.isfinite(parameter).all()
        if parameter.ndim == 1:
            assert torch.equal(parameter, torch.ones_like(parameter))
    serialized = io.BytesIO()
    torch.save(model.state_dict(), serialized)
    serialized.seek(0)
    restored = RecursiveLM(ModelConfig.from_dict(config.to_dict()))
    restored.load_state_dict(torch.load(serialized, weights_only=True), strict=True)
    ids = torch.tensor([[2, 5, 1]])
    torch.testing.assert_close(restored(ids).logits, model(ids).logits)
