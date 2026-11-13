"""Live-anchor and gradient-tail oracle with explicit cuts in an unrolled graph."""
import copy

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from moeloop.model import ModelConfig, RecursiveLM, RegionConfig
from moeloop.training import TrainingRecipe


def reference_equations(model, tokens, depth, tail):
    cfg = model.config
    positions = torch.arange(tokens.size(1)).expand_as(tokens)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    balance, z_loss = [], []

    def cell_update(cell, state, *, differentiable=True):
        state = state + cell.residual_scale * cell.attentions[0](cell.attention_norms[0](state),
            positions=positions, attention_mask=None, cache=None, slot=("core", 0, 0))
        normalized = cell.ffn_norms[0](state)
        routing = cell.routers[0](normalized, valid)
        update = cell.experts(normalized, routing.ids, routing.weights, routing.dispatch_counts)
        if cell.always_on_ffn is not None:
            update = update + cell.always_on_ffn(normalized)
        if differentiable:
            balance.append(routing.balance)
            z_loss.append(routing.z_loss)
        return state + cell.residual_scale * update

    anchor = model.embedding(tokens)
    for cell in model.prefix:
        anchor = cell_update(cell, anchor)
    # Draw with the declared distribution, independently of initial_latent().
    state = torch.nn.init.trunc_normal_(torch.empty_like(anchor), std=cfg.latent_init_std,
        a=-cfg.latent_truncation * cfg.latent_init_std, b=cfg.latent_truncation * cfg.latent_init_std)
    for visit in range(depth):
        adapter = model.input_adapter
        state = adapter.state_proj(state) + cfg.injection_scale * adapter.input_proj(anchor)
        for cell in model.core:
            state = cell_update(cell, state, differentiable=visit >= depth - tail)
        # The oracle builds then cuts the graph, instead of entering no_grad().
        # The separately retained prelude anchor stays live at every visit.
        if visit < depth - tail:
            state = state.detach()
    for cell in model.suffix:
        state = cell_update(cell, state)
    hidden = F.rms_norm(state, (cfg.dim,), model.final_norm.weight, cfg.rms_eps)
    return model.lm_head(hidden), torch.stack(balance).mean(), torch.stack(z_loss).mean(), len(balance)


@pytest.mark.parametrize("depth,tail", [(5, 2), (1, 1), (3, 3)],
                         ids=["truncated-live-anchor", "single-visit", "full-bptt"])
def test_random_latent_logits_and_all_gradients_match_explicit_graph_cuts(depth, tail):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = RecursiveLM(ModelConfig(vocab_size=19, dim=16, hidden_dim=24,
            num_heads=4, num_kv_heads=1, num_experts=3, top_k=2,
            num_cells=3, num_visits=5, prefix_cells=2, suffix_cells=1, max_seq_len=16,
            input_injection="linear", injection_scale=.7, initial_state="random",
            latent_init_std=.35, latent_truncation=1.2, qk_norm=True, always_on_hidden_dim=12,
            prefix=RegionConfig(hidden_dim=12, num_experts=2, top_k=1, residual_scale=.3),
            core=RegionConfig(hidden_dim=28, num_experts=4, top_k=3, residual_scale=.7),
            suffix=RegionConfig(hidden_dim=20, num_experts=2, top_k=1, residual_scale=1.1)))
        for name, parameter in model.named_parameters():
            if "norm" in name and parameter.ndim == 1:
                with torch.no_grad():
                    parameter.copy_(torch.linspace(.7, 1.3, parameter.numel()))
        reference = copy.deepcopy(model)
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, -100]])
        rng = torch.get_rng_state()
        output = model(tokens, num_visits=depth, grad_visits=tail)
        torch.set_rng_state(rng)
        logits, balance, z_loss, calls = reference_equations(reference, tokens, depth, tail)
    torch.testing.assert_close(output.logits, logits, rtol=2e-5, atol=2e-7)
    assert output.aux_calls == calls == 3 + 3 * tail
    denominator = (targets != -100).sum()
    actual = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(),
        reduction="sum", ignore_index=-100) / denominator
    actual = actual + (.07 * output.balance_sum + .03 * output.z_sum) / output.aux_calls
    expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
        reduction="sum", ignore_index=-100) / denominator + .07 * balance + .03 * z_loss
    actual.backward()
    expected.backward()
    other = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and other[name].grad is not None, name
        torch.testing.assert_close(parameter.grad, other[name].grad,
            rtol=4e-5, atol=4e-7, msg=name)


# initial_latent and sampled_depth do not recheck these; the configs reject them.
@pytest.mark.parametrize("changes", [
    {"latent_init_std": 0}, {"latent_init_std": float("nan")}, {"latent_init_std": True},
    {"latent_truncation": -1}, {"latent_truncation": float("inf")},
])
def test_model_config_rejects_invalid_latent_distribution(changes):
    with pytest.raises(ValueError, match="latent"):
        ModelConfig(vocab_size=19, dim=16, hidden_dim=24, num_heads=4, initial_state="random", **changes)


@pytest.mark.parametrize("changes", [
    {"depth_mean": 1.0}, {"depth_mean": float("nan")}, {"depth_mean": float("inf")},
    {"depth_log_std": -.1}, {"depth_log_std": float("inf")},
])
def test_training_recipe_rejects_invalid_depth_distribution(changes):
    with pytest.raises(ValueError, match="depth_mean"):
        TrainingRecipe(num_visits=4, grad_visits=2, depth_rule="poisson_lognormal_micro_v1", **changes)
