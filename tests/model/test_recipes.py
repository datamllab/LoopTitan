"""A few method and arithmetic contracts, rather than field-by-field copies."""

import json
import math
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes
from moeloop.model import ModelConfig, RecursiveLM


def test_method_specs_are_portable_and_keep_irreducible_method_semantics():
    # Importing/resolving a paper recipe must not import the training adapter.
    program = """
import importlib.abc, json, sys
class NoTitan(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('torchtitan'):
            raise AssertionError('ordinary recipe imported TorchTitan')
sys.meta_path.insert(0, NoTitan())
from moeloop import recipes
from moeloop.recipe_spec import RecipeSpec
for make in (recipes.smelt_spec, recipes.meta_spec, recipes.foil_spec, recipes.expert_tying_spec):
    spec = make()
    assert RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
"""
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert completed.returncode == 0, completed.stderr
    smelt = recipes.smelt_spec(physical_layers=8)
    assert smelt.model.num_cells == smelt.model.prefix_cells + smelt.model.suffix_cells == 4
    assert smelt.model.num_visits == 2
    assert [smelt.model.region_config(r).residual_scale for r in ("prefix", "core", "suffix")] == [1, .5, 1]
    tying = recipes.expert_tying_spec(group_size=4)
    assert tying.model.order == "layer"
    assert tying.model.attention_per_visit and tying.model.router_per_visit and tying.model.norm_per_visit
    assert tying.optimizer["expert_lr_multiplier"] == 1 / math.sqrt(4)
    with pytest.raises(ValueError, match="sparse"):
        recipes.smelt_spec(num_experts=0)


def test_foil_flattens_only_core_and_preserves_attention_owners():
    base = recipes.foil_base_spec(num_experts=4, core_layers=2, passes=2)
    foil = recipes.foil_spec(num_experts=4, core_layers=2, passes=2)
    assert (foil.model.core.num_experts, foil.model.num_cells, foil.model.num_visits) == (8, 1, 4)
    assert base.model.num_cells * base.model.visit_capacity == foil.model.num_cells * foil.model.visit_capacity == 4
    for region in ("prefix", "suffix"):
        assert base.model.region_config(region).num_experts == foil.model.region_config(region).num_experts == 4
    assert foil.model.region_config("core").hidden_dim == base.model.region_config("core").hidden_dim
    with pytest.raises(ValueError, match="even"):
        recipes.foil_spec(core_layers=3)


def test_smelt_ten_physical_layers_keep_middle_half_and_existing_eight_layer_path():
    from moeloop.costs import estimate_cost

    ten = recipes.smelt_spec(physical_layers=10)
    assert (ten.model.prefix_cells, ten.model.num_cells, ten.model.suffix_cells) == (2, 5, 3)
    assert estimate_cost(ten, seq_len=16)["calls"]["forward_cells"] == 15
    assert [ten.model.region_config(r).residual_scale for r in ("prefix", "core", "suffix")] == [1, .5, 1]
    eight = recipes.smelt_spec(physical_layers=8)
    assert (eight.model.prefix_cells, eight.model.num_cells, eight.model.suffix_cells) == (2, 4, 2)
    for invalid in (3, 5, 9, True):
        with pytest.raises(ValueError, match="even"):
            recipes.smelt_spec(physical_layers=invalid)


def test_cost_uses_unique_storage_but_counts_all_visits_and_kv():
    from moeloop.costs import estimate_cost

    # One cell: attention=192, experts=1152, router=32, two norms=16.
    # Four physical cells and eight logical calls; untied embedding/head=80 each.
    config = ModelConfig(vocab_size=10, dim=8, hidden_dim=12, num_heads=2,
                         num_kv_heads=1, num_experts=4, top_k=2,
                         num_cells=2, num_visits=3, prefix_cells=1, suffix_cells=1)
    cost = estimate_cost(config, seq_len=4, batch_size=2, bytes_per_element=2)
    assert cost["parameters"]["unique_total"] == 5736 == sum(p.numel() for p in RecursiveLM(config).parameters())
    assert cost["parameters"]["unique_non_embedding"] == 5576
    assert cost["calls"]["forward_cells"] == cost["calls"]["backward_cells"] == 8
    assert cost["flops"]["forward_per_token"] == 13984
    assert cost["flops"]["backward_per_token"] == 27968
    assert cost["flops"]["training_per_token"] == 41952
    # Eight logical slots, K and V, four KV dimensions, two bytes, two sequences.
    assert cost["kv"]["bytes_per_cached_token"] == 128
    assert cost["kv"]["bytes_at_context"] == 1024


def test_unknown_cost_never_passes_matching_and_finite_pair_matches_three_budgets():
    from dataclasses import replace
    from moeloop.comparison import compare_recipes, smelt_budget_pair

    pair = smelt_budget_pair(seq_len=2048)
    assert pair["candidate_count"] <= 32
    assert set(pair["comparison"]["matched"]) == {"unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes"}
    base = recipes.moe_spec()
    truncated = replace(base, training=replace(base.training, grad_visits=1, sampled_visits=(1, 2)))
    comparison = compare_recipes(base, truncated, seq_len=16)
    assert "training_flops_per_token" not in comparison["matched"]
    assert "training_flops_per_token" in comparison["unknown"]
    unknown_ac = compare_recipes(base, base, seq_len=16, activation_checkpointing=None)
    assert "training_flops_per_token" in unknown_ac["unknown"]
    assert "activation_checkpointing" not in unknown_ac["controlled"]
    assert json.loads(json.dumps(pair))["comparison"]["scientific_conclusion"] is None


def test_inference_kv_uses_declared_depth_while_training_cost_stays_fixed():
    from dataclasses import replace
    from moeloop.cache import DecodeCache
    from moeloop.comparison import compare_recipes
    from moeloop.costs import estimate_cost

    base = recipes.meta_spec(vocab_size=23, dim=16, hidden_dim=24,
                             num_heads=4, num_kv_heads=2, num_experts=4)
    shallow = replace(base, inference=base.inference | {"num_visits": 1})
    model = RecursiveLM(base.model).eval()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for spec in (base, shallow):
            cache = DecodeCache(spec.inference["num_visits"])
            with torch.no_grad():
                model(torch.tensor([[2, 3, 5, 7]]), num_visits=cache.num_visits, cache=cache)
            actual_bytes = sum(tensor.numel() * tensor.element_size()
                               for pair in cache.slots.values() for tensor in pair)
            cost = estimate_cost(spec, seq_len=4, bytes_per_element=4)
            assert cost["kv"]["logical_slots"] == len(cache.slots)
            assert cost["kv"]["bytes_at_context"] == actual_bytes
            assert cost["calls"]["forward_cells"] == 10  # Full-BPTT training still uses R4.
    finally:
        torch.set_num_threads(previous_threads)
    comparison = compare_recipes(base, shallow, seq_len=4, bytes_per_element=4)
    assert "training_flops_per_token" in comparison["matched"]
    assert "logical_kv_bytes" not in comparison["matched"]
    assert comparison["measured-different"]["logical_kv_bytes"]["relative_difference"] == pytest.approx(-0.6)

    foil = recipes.foil_spec()
    too_deep = replace(foil, inference=foil.inference | {"num_visits": foil.model.visit_capacity + 1})
    with pytest.raises(ValueError, match="capacity"):
        estimate_cost(too_deep, seq_len=4)
