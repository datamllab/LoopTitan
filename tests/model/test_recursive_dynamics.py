"""New state paths keep their own state, readout and gradient contracts."""
import pytest

torch = pytest.importorskip("torch")

from moeloop.memory import EMACarry
from moeloop.model import ModelConfig, RegionConfig, RecursiveLM


def geometry(**changes):
    return ModelConfig(vocab_size=19, dim=16, hidden_dim=24, num_heads=2,
                       num_kv_heads=1, num_experts=2, top_k=2, num_cells=2,
                       num_visits=3, max_seq_len=32, **changes)


def test_random_latent_tail_keeps_live_anchor_and_deterministic_eval():
    model = RecursiveLM(geometry(prefix_cells=1, suffix_cells=1,
                                initial_state="random", input_injection="linear"))
    tokens = torch.tensor([[1, 2, 3, 4]])
    model(tokens, grad_visits=1).logits.square().sum().backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    assert model.prefix[0].attentions[0].q_proj.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        first = model(tokens).logits
        torch.randn(37)
        torch.testing.assert_close(first, model(tokens).logits, atol=0, rtol=0)


def test_memory_is_forward_local_and_combines_with_independent_attention():
    model = RecursiveLM(geometry(state_update="ema", router_per_visit=True,
                                attention_per_visit=True))
    tokens = torch.tensor([[1, 2, 3]])
    first = model(tokens)
    other = model(torch.tensor([[4, 5]]))
    second = model(tokens)
    torch.testing.assert_close(first.logits, second.logits, atol=0, rtol=0)
    first.logits.square().sum().backward()
    assert len(model.core[0].attentions) == 3
    assert all(p.grad is not None for p in model.parameters())
    assert not list(model.named_buffers())


def test_cell_and_memory_cell_share_one_call_and_the_forward_is_their_plain_unroll():
    model = RecursiveLM(geometry(state_update="ema", router_per_visit=True, prefix_cells=1, suffix_cells=1))
    tokens = torch.tensor([[1, 2, 3]])
    inputs, *_ = model.parse_inputs(tokens)
    memory = (EMACarry(), EMACarry())
    anchor, routing, passed = model.prefix[0](model.embedding(tokens), visit=0, slot=("prefix", 0, 0), memory=memory, **inputs)
    assert passed is memory and routing[2].sum() == 3 * 2  # a Cell passes memory on; three tokens, top-2
    state, carries = anchor, []
    for visit in range(3):
        if visit:
            gain = .5 / ((visit + 1) * 2 ** .5)  # memory_scale / (one-based visit * sqrt(num_cells))
            state = (1 - gain) * state + gain * anchor
        memory = (memory[0], EMACarry())  # global carried, local restarted
        for index, cell in enumerate(model.core):
            state, routing, memory = cell(state, visit=visit, slot=("core", index, visit), memory=memory, **inputs)
            carries.append(memory)
    assert [carry[0].denominator for carry in carries] == [2 - .5 ** n for n in range(6)]
    assert [carry[1].denominator for carry in carries] == [1.0, 1.5] * 3
    torch.testing.assert_close(model(tokens).hidden_states, model.readout(state, 0, inputs), atol=0, rtol=0)


def test_exit_readouts_share_head_and_default_to_terminal_logits():
    model = RecursiveLM(geometry(exit_distribution=True, suffix_cells=1))
    output = model(torch.tensor([[1, 2, 3]]), return_hidden=True)
    assert len(output.visit_hidden_states) == 3
    assert output.hazard_logits.shape == (3, 1, 3)
    torch.testing.assert_close(output.hazard_logits, torch.zeros_like(output.hazard_logits))
    torch.testing.assert_close(output.visit_hidden_states[-1], output.hidden_states)
    assert len([n for n, _ in model.named_parameters() if n == "lm_head.weight"]) == 1


def test_old_config_keeps_old_parameter_layout_and_new_config_roundtrips():
    old = geometry().to_dict()
    old["schema_version"] = 2
    for key in ("state_update", "memory_scale", "memory_beta", "latent_init_std",
                "latent_truncation", "exit_distribution"):
        old.pop(key, None)
    loaded = ModelConfig.from_dict(old)
    assert set(RecursiveLM(loaded).state_dict()) == set(RecursiveLM(geometry()).state_dict())
    new = geometry(state_update="ema", router_per_visit=True)
    assert ModelConfig.from_dict(new.to_dict()) == new


@pytest.mark.parametrize("changes", [{"residual_scale": .5},
                                      {"core": RegionConfig(residual_scale=.5)}])
def test_ema_rejects_an_undefined_extra_core_residual_scale(changes):
    with pytest.raises(ValueError, match="EMA core residual_scale"):
        geometry(state_update="ema", router_per_visit=True, **changes)


def test_ema_core_keeps_explicit_peripheral_residual_scales():
    config = geometry(state_update="ema", router_per_visit=True, prefix_cells=1,
                      prefix=RegionConfig(residual_scale=.5))
    model = RecursiveLM(config)
    assert model.prefix[0].residual_scale == .5
    assert model.core[0].residual_scale == 1.
