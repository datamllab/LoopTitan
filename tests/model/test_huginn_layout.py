"""The three flags of the official Huginn layout: owned parameters, initialization and serialization.

``qk_bias`` adds a query and a key bias to every attention owner, zero at
initialization; ``residual_norm`` adds two norms with the ownership of the input
norms; ``core_final_norm`` adds no parameters. All three are written by
``to_dict`` only when on, so existing records and identities keep their bytes.
"""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes, testing
from moeloop.config import ModelConfig, TrainingRecipe
from moeloop.costs import parameter_counts
from moeloop.model import LoopedLM

FLAGS = dict(qk_bias=True, residual_norm=True, core_final_norm=True)
# The converted huginn-0125 layout (prelude 2, core 4 x 32, coda 2), scaled down.
HUGINN = dict(vocab_size=64, dim=32, hidden_dim=88, num_heads=2, num_experts=0, prefix_cells=1, num_cells=2,
              suffix_cells=1, num_visits=4, input_injection="linear", initial_state="random", **FLAGS)


@pytest.mark.parametrize("config", [
    ModelConfig(**HUGINN),
    ModelConfig(**HUGINN | {"norm_per_visit": True, "attention_per_visit": True}),
    ModelConfig(vocab_size=64, dim=32, hidden_dim=48, num_heads=4, num_kv_heads=2, num_experts=4,
                prefix_cells=1, num_cells=2, suffix_cells=1, num_visits=3, sandwich_norm=True, **FLAGS),
], ids=["huginn", "per_visit_owners", "sparse_gqa_sandwich"])
def test_parameter_count_is_the_actual_count(config):
    with torch.device("meta"):
        model = LoopedLM(config)
    assert parameter_counts(config)["unique_total"] == sum(p.numel() for p in model.parameters())
    plain = ModelConfig.from_dict(config.to_dict() | {name: False for name in FLAGS})
    d, kv = config.dim, config.num_kv_heads * config.dim // config.num_heads
    visits = config.num_visits
    attentions = config.num_cells * (visits if config.attention_per_visit else 1) + config.prefix_cells + config.suffix_cells
    norms = config.num_cells * (visits if config.norm_per_visit else 1) + config.prefix_cells + config.suffix_cells
    assert parameter_counts(config)["unique_total"] - parameter_counts(plain)["unique_total"] == (d + kv) * attentions + 2 * d * norms


def test_qk_biases_start_at_zero_and_the_rest_of_the_draw_is_unchanged():
    model, plain = LoopedLM(ModelConfig(**HUGINN)), LoopedLM(ModelConfig(**HUGINN | {name: False for name in FLAGS}))
    for each in (model, plain):
        each.initialize_weights(torch.Generator().manual_seed(0))
    biases = {name: p for name, p in model.named_parameters() if name.endswith("_proj.bias")}
    assert len(biases) == 2 * 4 and all(not p.any() for p in biases.values())
    for name, parameter in plain.named_parameters():
        torch.testing.assert_close(model.get_parameter(name), parameter, rtol=0, atol=0)


def test_the_flags_are_written_only_when_on():
    on = ModelConfig(**HUGINN)
    record = json.loads(json.dumps(on.to_dict()))
    assert {name: record[name] for name in FLAGS} == FLAGS
    assert "sandwich_norm" not in record and "visit_final_norm" not in record
    assert ModelConfig.from_dict(record) == on
    assert set(ModelConfig(**HUGINN | {name: False for name in FLAGS}).to_dict()).isdisjoint(FLAGS)


SMALL = dict(vocab_size=64, dim=32, hidden_dim=48, num_heads=4, num_kv_heads=2, num_visits=3)


@pytest.mark.parametrize("config", [
    ModelConfig(**HUGINN | {"num_visits": 3}),
    ModelConfig(**SMALL, num_experts=4, norm_per_visit=True, sandwich_norm=True, visit_final_norm=True, **FLAGS),
    ModelConfig(**SMALL, num_experts=4, prefix_cells=1, suffix_cells=1, input_injection="orthogonal", **FLAGS),
    ModelConfig(**SMALL, num_experts=0, lora_attention_rank=2, lora_ffn_rank=2, **FLAGS),
    ModelConfig(**SMALL, num_experts=0, order="layer", suffix_cells=1, **FLAGS),
    ModelConfig(**SMALL, num_experts=0, state_update="ema", qk_bias=True, core_final_norm=True),
], ids=["huginn", "moe_norm_per_visit_sandwich", "moe_orthogonal", "dense_lora", "layer_order", "ema"])
def test_the_flags_combine_with_other_structure(config):
    tiny = recipes.tiny_spec()
    testing.check_recipe(replace(tiny, name="huginn_flags", model=config, training=TrainingRecipe(num_visits=3),
                                 inference=tiny.inference | {"num_visits": 3}))
    model = LoopedLM(config)
    model(torch.randint(0, 64, (2, 8))).logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        if "residual_norms" in name or name.endswith("_proj.bias"):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_combinations_without_a_definition_are_refused():
    with pytest.raises(ValueError, match="residual norm is defined"):
        ModelConfig(**SMALL, num_experts=0, state_update="ema", residual_norm=True)
