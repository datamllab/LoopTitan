"""Per-visit core residual scales (LatentLoop's alpha_v): each visit of the shared core uses its own scale."""

from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes
from moeloop.model import ModelConfig, RecursiveLM, RegionConfig
from moeloop.untie import untied


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(47)
    yield
    torch.set_num_threads(previous)


def tiny(**changes):
    return ModelConfig(**(dict(vocab_size=23, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
        num_experts=0, num_cells=2, num_visits=2, prefix_cells=1, suffix_cells=1, init_std=.2) | changes))


def dense_update(cell, state, positions, scale):
    """Independent residual equation; the scale is a literal oracle value."""
    state = state + scale * cell.attentions[0](cell.attention_norms[0](state), positions=positions)
    return state + scale * cell.dense_ffn(cell.ffn_norms[0](state))


@pytest.mark.parametrize("order", ["stack", "layer"])
def test_each_core_visit_uses_its_own_scale_on_the_shared_weights(order):
    model = RecursiveLM(tiny(order=order, visit_residual_scales=(.25, 1.)))
    ids = torch.tensor([[2, 4, 5, 7], [1, 3, 9, 6]])
    positions = torch.arange(4).expand(2, -1)
    state = dense_update(model.prefix[0], model.embedding(ids), positions, 1.)
    calls = ([(cell, scale) for scale in (.25, 1.) for cell in model.core] if order == "stack"
             else [(cell, scale) for cell in model.core for scale in (.25, 1.)])
    for cell, scale in calls:
        state = dense_update(cell, state, positions, scale)
    state = dense_update(model.suffix[0], state, positions, 1.)
    torch.testing.assert_close(model(ids).logits, model.lm_head(model.final_norm(state)))
    # Non-vacuous: swapping or equalizing the two visits' scales changes the function.
    for other in ((1., .25), (.25, .25), (1., 1.)):
        changed = RecursiveLM(replace(model.config, visit_residual_scales=other))
        changed.load_state_dict(model.state_dict())
        assert not torch.allclose(changed(ids).logits, model(ids).logits)


def test_equal_visit_scales_are_bitwise_the_core_residual_scale():
    """(s, s) per visit and a core residual_scale of s run the same multiplications; MoE cells included."""
    moe = dict(num_experts=4, top_k=2)
    reference = RecursiveLM(tiny(**moe, core=RegionConfig(residual_scale=.5)))
    candidate = RecursiveLM(tiny(**moe, visit_residual_scales=(.5, .5)))
    assert list(candidate.state_dict()) == list(reference.state_dict())  # no new parameters or buffers
    candidate.load_state_dict(reference.state_dict())
    ids = torch.tensor([[2, 4, 5, 7], [1, 3, 9, 6]])
    assert torch.equal(candidate(ids).logits, reference(ids).logits)


def test_default_config_writes_and_builds_what_it_did_before():
    assert "visit_residual_scales" not in ModelConfig().to_dict()  # frozen specs keep their bytes
    for config in (tiny(), tiny(num_experts=4, core=RegionConfig(residual_scale=.5))):
        assert "visit_residual_scales" not in config.to_dict() and config.visit_residual_scales == ()
        model = RecursiveLM(config)
        assert all(cell.visit_residual_scales == () for cell in (*model.prefix, *model.core, *model.suffix))


def test_scales_round_trip_and_reach_only_the_core():
    config = tiny(visit_residual_scales=[.25, 1])  # a JSON list becomes a tuple
    assert config.visit_residual_scales == (.25, 1)
    assert config.to_dict()["visit_residual_scales"] == [.25, 1]
    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert config.region_config("core").visit_residual_scales == (.25, 1)
    assert config.region_config("prefix").visit_residual_scales == config.region_config("suffix").visit_residual_scales == ()


def test_invalid_scales_and_depths_are_refused():
    for changes in (dict(visit_residual_scales=(.25,)), dict(visit_residual_scales=(.25, 1., 1.)),
                    dict(visit_residual_scales=(.25, float("nan"))), dict(visit_residual_scales=(True, 1.)),
                    dict(visit_residual_scales=(.25, 1.), core=RegionConfig(residual_scale=.5)),
                    dict(visit_residual_scales=(.25, 1.), residual_scale=.5)):
        with pytest.raises(ValueError, match="visit_residual_scales"):
            tiny(**changes)
    model = RecursiveLM(tiny(visit_residual_scales=(.25, 1.)))
    with pytest.raises(ValueError, match="per-visit residual scales"):
        model(torch.tensor([[2, 4]]), num_visits=3)
    model(torch.tensor([[2, 4]]), num_visits=1)  # a shallower depth uses the first scales


def test_untie_refuses_per_visit_scales():
    spec = recipes.latentloop_residual_spec().to_dict()
    with pytest.raises(ValueError, match="untie needs"):
        untied(spec)


def test_latentloop_residual_constructor_is_smelt_with_asymmetric_core_scales():
    latent, smelt = recipes.latentloop_residual_spec(), recipes.smelt_spec()
    assert latent.model.visit_residual_scales == (.25, 1.)
    assert latent.model.region_config("core").residual_scale == 1.
    assert latent.model == replace(smelt.model, core=None, visit_residual_scales=(.25, 1.))
    assert (latent.training, latent.inference) == (smelt.training, smelt.inference)
