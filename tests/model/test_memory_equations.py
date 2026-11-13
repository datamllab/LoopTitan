"""Model-level oracle using closed-form histories, not recursive EMA helpers."""
import copy
import math

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from moeloop.model import ModelConfig, RecursiveLM, RegionConfig


def history_average(writes, beta):
    # A normalized exponentially weighted history is independent of the
    # implementation's numerator/denominator carry and reset machinery.
    values = torch.stack(writes)
    powers = torch.arange(len(writes) - 1, -1, -1, dtype=values.dtype)
    weights = beta ** powers
    return (values * weights[:, None, None, None]).sum(0) / weights.sum()


def reference_equations(model, tokens, valid):
    cfg = model.config
    anchor = model.embedding(tokens)
    positions = valid.long().cumsum(-1).sub(1).clamp_min(0)
    causal = torch.arange(tokens.size(1))[None, :] <= torch.arange(tokens.size(1))[:, None]
    mask = causal[None, None] & valid[:, None, None, :] & valid[:, None, :, None]
    balance, z_loss = [], []

    def ffn(cell, state, visit):
        normalized = cell.ffn_norms[0](state)
        routed = cell.routers[visit if len(cell.routers) > 1 else 0](normalized, valid)
        update = cell.experts(normalized, routed.ids, routed.weights, routed.dispatch_counts)
        if cell.always_on_ffn is not None:
            update = update + cell.always_on_ffn(normalized)
        balance.append(routed.balance)
        z_loss.append(routed.z_loss)
        return update

    def peripheral(cell, state):
        state = state + cell.residual_scale * cell.attentions[0](cell.attention_norms[0](state),
            positions=positions, attention_mask=mask, cache=None, slot=("prefix", 0, 0))
        return state + cell.residual_scale * ffn(cell, state, 0)

    for cell in model.prefix:
        anchor = peripheral(cell, anchor)
    state = anchor
    gamma = cfg.memory_scale / (cfg.num_visits * math.sqrt(cfg.num_cells))
    writes = []
    for visit in range(cfg.num_visits):
        if visit:
            gain = cfg.memory_scale / ((visit + 1) * math.sqrt(cfg.num_cells))
            state = (1 - gain) * state + gain * anchor
        beginning = len(writes)
        for cell in model.core:
            attention = cell.attentions[visit if cfg.attention_per_visit else 0]
            writes.append(gamma * attention(cell.attention_norms[0](state),
                positions=positions, attention_mask=mask, cache=None,
                slot=("core", 0, visit)))
            mixed = state + history_average(writes, cfg.memory_beta)
            mixed = mixed + history_average(writes[beginning:], cfg.memory_beta)
            state = mixed + gamma * F.rms_norm(ffn(cell, mixed, visit), (cfg.dim,),
                cell.ffn_output_norm.weight, cfg.rms_eps)
    for cell in model.suffix:
        state = peripheral(cell, state)
    logits = model.lm_head(model.final_norm(state))
    return logits, torch.stack(balance).mean(), torch.stack(z_loss).mean()


@pytest.mark.parametrize("changes,dtype", [
    pytest.param({}, torch.float32, id="shared"),
    pytest.param({"attention_per_visit": True}, torch.float32, id="independent-attention"),
    # Output RMS can amplify FP32 differences between recursive and closed-form
    # history summation. Use higher precision here without loosening tolerances.
    pytest.param({"num_cells": 3, "num_visits": 4, "memory_beta": .8, "memory_scale": .7,
     "attention_per_visit": True, "qk_norm": True, "num_heads": 4,
     "num_experts": 1, "top_k": 1, "hidden_dim": 20, "always_on_hidden_dim": 12},
     torch.float64, id="depth-coefficients-geometry"),
    pytest.param({"num_cells": 1, "num_visits": 1, "memory_beta": 0., "memory_scale": .9,
     "top_k": 1}, torch.float32, id="single-visit-beta-zero"),
    pytest.param({"num_visits": 2, "memory_beta": .25, "memory_scale": .35, "prefix_cells": 1,
     "suffix_cells": 1, "qk_norm": True, "always_on_hidden_dim": 12,
     "prefix": RegionConfig(hidden_dim=12, num_experts=2, top_k=1, residual_scale=.3),
     "core": RegionConfig(hidden_dim=28, num_experts=4, top_k=3),
     "suffix": RegionConfig(hidden_dim=20, num_experts=2, top_k=1, residual_scale=1.1)},
     torch.float32, id="periphery"),
])
def test_complete_memory_model_logits_and_all_gradients_match_history_equations(changes, dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        config = dict(vocab_size=19, dim=16, hidden_dim=24, num_heads=2,
            num_kv_heads=1, num_experts=3, top_k=2, num_cells=2, num_visits=3,
            max_seq_len=16, state_update="ema", router_per_visit=True)
        model = RecursiveLM(ModelConfig(**(config | changes))).to(dtype)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and parameter.ndim == 1:
                parameter.copy_(torch.linspace(.7, 1.3, parameter.numel()))
    reference = copy.deepcopy(model)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 2, -100, -100]])
    denominator = (targets != -100).sum()
    output = model(tokens, attention_mask=valid)
    logits, balance, z_loss = reference_equations(reference, tokens, valid)
    torch.testing.assert_close(output.logits, logits, rtol=2e-5, atol=2e-7)
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


EMA = dict(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_experts=3, top_k=2,
           max_seq_len=16, state_update="ema", router_per_visit=True)


# ema_update and residual_gain do not recheck these; the config and forward reject them.
@pytest.mark.parametrize("changes", [
    {"memory_beta": -.1}, {"memory_beta": 1.0}, {"memory_beta": float("nan")},
    {"memory_beta": float("inf")}, {"memory_beta": True},
    {"memory_scale": -.1}, {"memory_scale": float("inf")},
    {"num_cells": 0}, {"num_cells": True}, {"num_visits": 0}, {"num_visits": 1.5},
])
def test_config_rejects_invalid_memory_coefficients_and_geometry(changes):
    with pytest.raises(ValueError):
        ModelConfig(**(EMA | changes))


@pytest.mark.parametrize("num_visits", [0, 1.5, True])
def test_forward_rejects_invalid_memory_depth(num_visits):
    model = RecursiveLM(ModelConfig(**EMA))
    with pytest.raises(ValueError, match="num_visits"):
        model(torch.tensor([[1, 2, 3]]), num_visits=num_visits)
