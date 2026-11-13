"""The two flags of the official Ouro layout: owned parameters and serialization.

``sandwich_norm`` adds output norms with the ownership of the input norms;
``visit_final_norm`` adds no parameters. Both flags are written by ``to_dict``
only when on, so existing records and identities keep their bytes.
"""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes, testing
from moeloop.config import WRITTEN_WHEN_ON, ModelConfig, TrainingRecipe
from moeloop.costs import parameter_counts
from moeloop.model import LoopedLM

# The converted Ouro-1.4B layout (24 shared dense layers, 4 visits, exit gate), scaled down.
OURO = dict(vocab_size=64, dim=32, hidden_dim=88, num_heads=2, num_kv_heads=2, num_experts=0,
            num_cells=3, num_visits=4, exit_distribution=True, sandwich_norm=True, visit_final_norm=True)


@pytest.mark.parametrize("config", [
    ModelConfig(**OURO),
    ModelConfig(**OURO | {"norm_per_visit": True}),
    ModelConfig(vocab_size=64, dim=32, hidden_dim=48, num_heads=4, num_kv_heads=2, num_experts=4,
                prefix_cells=1, num_cells=2, suffix_cells=1, num_visits=3, sandwich_norm=True),
], ids=["ouro", "norm_per_visit", "sparse_regions"])
def test_parameter_count_is_the_actual_count(config):
    with torch.device("meta"):
        model = LoopedLM(config)
    assert parameter_counts(config)["unique_total"] == sum(p.numel() for p in model.parameters())
    plain = ModelConfig.from_dict(config.to_dict() | {"sandwich_norm": False})
    norm_owners = config.num_cells * (config.num_visits if config.norm_per_visit else 1) + config.prefix_cells + config.suffix_cells
    assert parameter_counts(config)["unique_total"] - parameter_counts(plain)["unique_total"] == 2 * config.dim * norm_owners


def test_the_flags_change_no_existing_record():
    built = {}
    for name, build in vars(recipes).items():
        if name.endswith(("_spec", "_specs")) and not name.startswith("_"):
            spec = build()
            built.update(spec if isinstance(spec, dict) else {name: spec})
    for name, spec in built.items():
        assert set(spec.to_dict()["model"]).isdisjoint(WRITTEN_WHEN_ON), name
    on = ModelConfig(**OURO)
    record = json.loads(json.dumps(on.to_dict()))
    assert {name: record[name] for name in WRITTEN_WHEN_ON if name in record} == {"sandwich_norm": True, "visit_final_norm": True}
    assert ModelConfig.from_dict(record) == on


SMALL = dict(vocab_size=64, dim=32, hidden_dim=48, num_heads=4, num_kv_heads=2, num_visits=3)


@pytest.mark.parametrize("config", [
    ModelConfig(**SMALL, num_experts=4, norm_per_visit=True, sandwich_norm=True, visit_final_norm=True),
    ModelConfig(**SMALL, num_experts=4, prefix_cells=1, suffix_cells=1, sandwich_norm=True, visit_final_norm=True),
    ModelConfig(**SMALL, num_experts=0, lora_attention_rank=2, lora_ffn_rank=2, sandwich_norm=True, visit_final_norm=True),
    ModelConfig(**SMALL, num_experts=0, state_update="ema", visit_final_norm=True),
], ids=["moe_norm_per_visit", "moe_prefix_suffix", "dense_lora", "ema"])
def test_the_flags_combine_with_other_structure(config):
    tiny = recipes.tiny_spec()
    testing.check_recipe(replace(tiny, name="ouro_flags", model=config, training=TrainingRecipe(num_visits=3),
                                 inference=tiny.inference | {"num_visits": 3}))
    model = LoopedLM(config)
    model(torch.randint(0, 64, (2, 8))).logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        if "output_norms" in name:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_combinations_without_a_definition_are_refused():
    with pytest.raises(ValueError, match="sandwich norm is defined"):
        ModelConfig(**SMALL, num_experts=0, state_update="ema", sandwich_norm=True)
    with pytest.raises(ValueError, match="visit_final_norm closes"):
        ModelConfig(**SMALL, num_experts=0, order="layer", visit_final_norm=True)
