"""Region geometry, residual equations and logical-call statistics contracts."""

from dataclasses import replace
import io
import json

import pytest

torch = pytest.importorskip("torch")

from moeloop.cache import DecodeCache
from moeloop import model as model_module
from moeloop.model import ModelConfig, RecursiveLM


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(43)
    yield
    torch.set_num_threads(previous)


def tiny(**changes):
    return ModelConfig(**(dict(vocab_size=23, dim=16, hidden_dim=24,
        num_heads=4, num_kv_heads=2, num_experts=4, top_k=2,
        num_cells=2, num_visits=2, prefix_cells=1, suffix_cells=1,
        init_std=.2) | changes))


def dense_update(cell, state, positions, scale):
    """Independent residual equation; scales are supplied as literal oracle values."""
    state = state + scale * cell.attentions[0](cell.attention_norms[0](state), positions=positions)
    return state + scale * cell.dense_ffn(cell.ffn_norms[0](state))


def test_schema_one_weights_and_global_residual_path_remain_readable():
    old_values = dict(vocab_size=23, dim=16, hidden_dim=24, num_heads=4,
        num_kv_heads=2, num_experts=0, num_cells=1, num_visits=2,
        prefix_cells=1, suffix_cells=1, residual_scale=.3, schema_version=1)
    upgraded = ModelConfig.from_dict(old_values)
    assert upgraded.schema_version == 4
    assert ModelConfig.from_dict(json.loads(json.dumps(upgraded.to_dict()))) == upgraded
    model = RecursiveLM(upgraded)
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    restored = RecursiveLM(ModelConfig.from_dict(old_values))
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    ids = torch.tensor([[2, 3, 5]])
    positions = torch.arange(3).unsqueeze(0)
    state = restored.embedding(ids)
    for cell in (restored.prefix[0], restored.core[0], restored.core[0], restored.suffix[0]):
        state = dense_update(cell, state, positions, .3)
    expected = restored.lm_head(restored.final_norm(state))
    torch.testing.assert_close(restored(ids).logits, expected)
    torch.testing.assert_close(model(ids).logits, expected)


def test_region_geometry_is_resolved_without_creating_parameter_aliases():
    outer = model_module.RegionConfig(num_experts=2, hidden_dim=12, top_k=1)
    config = tiny(prefix=outer, core=model_module.RegionConfig(num_experts=6, hidden_dim=36), suffix=outer)
    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    model = RecursiveLM(config)
    for cell, experts, hidden in ((model.prefix[0], 2, 12), (model.core[0], 6, 36),
                                  (model.suffix[0], 2, 12)):
        assert cell.config.num_experts == experts and cell.config.hidden_dim == hidden
        assert cell.config.dim == 16 and cell.config.prefix is cell.config.core is cell.config.suffix is None
        assert cell.experts.w1_EFD.shape == (experts, hidden, 16)
        assert cell.routers[0].gate.out_features == experts
    assert model.prefix[0].experts.w1_EFD is not model.suffix[0].experts.w1_EFD
    parameters = list(model.named_parameters(remove_duplicate=False))
    assert len(parameters) == len({id(parameter) for _, parameter in parameters})
    ids = torch.tensor([[2, 3, 5]])
    model(ids).logits.square().sum().backward()
    assert all(parameter.grad is not None for _, parameter in parameters)
    with pytest.raises(ValueError, match="top_k"):
        tiny(core=model_module.RegionConfig(num_experts=1))
    with pytest.raises(ValueError, match="region"):
        ModelConfig.from_dict(config.to_dict() | {"core": {"dim": 8}})
    with pytest.raises(ValueError, match="schema"):
        ModelConfig.from_dict(config.to_dict() | {"schema_version": 1})


def test_only_core_residuals_are_scaled_and_shared_gradients_sum_across_unroll():
    config = tiny(num_experts=0, core=model_module.RegionConfig(residual_scale=.5))
    shared = RecursiveLM(config)
    unrolled = RecursiveLM(replace(config, num_cells=4, num_visits=1))
    for name in ("embedding", "prefix", "suffix", "final_norm", "lm_head"):
        getattr(unrolled, name).load_state_dict(getattr(shared, name).state_dict())
    for index, cell in enumerate(unrolled.core):
        cell.load_state_dict(shared.core[index % 2].state_dict())
    ids = torch.tensor([[2, 4, 5], [1, 3, 9]])
    positions = torch.arange(3).expand(2, -1)
    state = dense_update(unrolled.prefix[0], unrolled.embedding(ids), positions, 1.)
    for cell in unrolled.core:
        state = dense_update(cell, state, positions, .5)
    state = dense_update(unrolled.suffix[0], state, positions, 1.)
    expected = unrolled.lm_head(unrolled.final_norm(state))
    actual = shared(ids).logits
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for index, cell in enumerate(shared.core):
        for name, parameter in cell.named_parameters():
            total = sum(dict(unrolled.core[copy].named_parameters())[name].grad
                        for copy in (index, index + 2))
            torch.testing.assert_close(parameter.grad, total, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(shared.embedding.weight.grad, unrolled.embedding.weight.grad)


@pytest.mark.parametrize("order", ["stack", "layer"])
def test_heterogeneous_router_counts_keep_call_identity_and_exact_cache(order):
    config = tiny(order=order, attention_per_visit=True,
        prefix=model_module.RegionConfig(num_experts=2, top_k=1),
        core=model_module.RegionConfig(num_experts=6), suffix=model_module.RegionConfig(num_experts=0))
    model = RecursiveLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    valid = torch.tensor([[True, True, True, False]])
    output = model(ids, attention_mask=valid)
    core_slots = ([("core", 0, 0), ("core", 1, 0), ("core", 0, 1), ("core", 1, 1)]
                  if order == "stack" else
                  [("core", 0, 0), ("core", 0, 1), ("core", 1, 0), ("core", 1, 1)])
    assert output.router_call_slots == tuple([("prefix", 0, 0)] + core_slots + [("suffix", 0, 0)])
    assert output.router_call_widths == (2, 6, 6, 6, 6, 0)
    assert output.router_counts.shape == (4, 6)
    assert output.router_counts.sum(-1).tolist() == [3, 12, 12, 0]
    assert output.router_call_counts.sum(-1).tolist() == [3, 6, 6, 6, 6, 0]
    assert output.router_counts[0, 2:].count_nonzero() == 0
    assert output.router_counts[-1].count_nonzero() == 0
    assert output.aux_calls == 5
    assert not output.router_call_counts.requires_grad
    cache = DecodeCache(2)
    with torch.no_grad():
        pieces = [model(ids[:, :2], attention_mask=valid[:, :2], cache=cache).logits,
                  model(ids[:, 2:], attention_mask=valid[:, 2:], cache=cache).logits]
    # Full/chunked FP32 kernels reduce in different orders, amplified here by
    # init_std=.2. Use the predeclared FP32 criterion of the independent HF gate.
    torch.testing.assert_close(torch.cat(pieces, dim=1), output.logits, atol=1e-5, rtol=5e-5)
    assert set(cache.slots) == set(output.router_call_slots)
    assert len({id(key) for key, _ in cache.slots.values()}) == 6


def test_sparse_regions_contribute_auxiliary_losses_when_global_default_is_dense():
    model = RecursiveLM(tiny(num_experts=0, core=model_module.RegionConfig(num_experts=4)))
    output = model(torch.tensor([[1, 2, 3]]), grad_visits=1)
    assert output.aux_calls == 2
    assert output.balance_sum.requires_grad and output.z_sum.requires_grad
    assert output.router_counts.sum(-1).tolist() == [0, 12, 12, 0]
