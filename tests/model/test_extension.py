"""Pin the alternate update's execution and residual/normalization semantics."""

import pytest

torch = pytest.importorskip("torch")

from moeloop.model import ModelConfig, RecursiveLM
from recipes.ffn_chain import forward_ffn_chain


def owner():
    torch.manual_seed(37)
    return RecursiveLM(ModelConfig(vocab_size=23, dim=16, hidden_dim=24,
        num_heads=2, num_kv_heads=1, num_experts=2, num_cells=1, num_visits=1))


def test_chain_runs_attention_and_normalization_once_and_accumulates_gradients():
    model = owner()
    cell = model.core[0]
    calls = dict(attention=0, norm=0, router=0, experts=0)
    handles = []
    for name, module in (("attention", cell.attentions[0]), ("norm", cell.ffn_norms[0]),
                         ("router", cell.routers[0]), ("experts", cell.experts)):
        def observe(module, inputs, output, key=name):
            calls[key] += 1
        handles.append(module.register_forward_hook(observe))
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    positions = torch.arange(4).expand(2, -1)
    try:
        hidden, balance, z_loss = forward_ffn_chain(cell, model.embedding(ids),
            ffn_steps=3, positions=positions)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == dict(attention=1, norm=1, router=3, experts=3)
    logits = model.lm_head(model.final_norm(hidden))
    (logits.square().mean() + 0.1 * balance / 3 + 0.03 * z_loss / 3).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert cell.experts.w1_EFD.grad.abs().sum() > 0


def test_zero_updates_expose_inner_outer_residual_difference_from_whole_cell():
    cell = owner().core[0]
    with torch.no_grad():
        cell.attentions[0].out_proj.weight.zero_()
        cell.experts.w2_EDF.zero_()
    inputs = torch.randn(1, 4, 16)
    positions = torch.arange(4).unsqueeze(0)
    actual, _, _ = forward_ffn_chain(cell, inputs, ffn_steps=3, positions=positions)
    torch.testing.assert_close(actual, inputs + cell.ffn_norms[0](inputs))
    repeated = inputs
    for _ in range(3):
        repeated, *_ = cell(repeated, visit=0, positions=positions, attention_mask=None,
            router_mask=torch.ones(1, 4, dtype=torch.bool), cache=None, slot=("core", 0, 0))
    torch.testing.assert_close(repeated, inputs)
    assert not torch.allclose(actual, repeated)
