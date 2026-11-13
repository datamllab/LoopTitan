"""Parameter owners and executed readouts stay distinct from inference KV."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop.comparison import compare_recipes
from moeloop.costs import estimate_cost, kv_bytes, parameter_counts, training_flops
from moeloop.halting import expected_lm_loss
from moeloop.model import ModelConfig, RecursiveLM
from moeloop.recipe_spec import RecipeSpec
from moeloop.training import TrainingRecipe


def geometry(**overrides):
    values = dict(vocab_size=17, dim=8, hidden_dim=12, num_heads=2,
                  num_kv_heads=1, num_experts=4, top_k=2,
                  num_cells=2, num_visits=3, prefix_cells=1, suffix_cells=1,
                  max_seq_len=8)
    return ModelConfig(**(values | overrides))


@pytest.mark.parametrize("memory,gate", [(True, False), (False, True), (True, True)])
def test_dynamic_parameters_match_actual_unique_owners(memory, gate):
    config = geometry(state_update="ema" if memory else "residual",
                      router_per_visit=memory, exit_distribution=gate)
    plain = replace(config, state_update="residual", exit_distribution=False)
    actual = RecursiveLM(config)
    cost = estimate_cost(config, seq_len=4)
    base = estimate_cost(plain, seq_len=4)
    extra_norm = config.num_cells * config.dim if memory else 0
    extra_gate = config.dim + 1 if gate else 0
    assert cost["parameters"]["unique_total"] == sum(p.numel() for p in actual.parameters())
    assert parameter_counts(config)["unique_total"] == cost["parameters"]["unique_total"]
    assert cost["parameters"]["unique_total"] - base["parameters"]["unique_total"] == extra_norm + extra_gate
    if memory:
        assert cost["parameters"]["by_region"]["core"]["parameters"]["norms"] == 3 * config.num_cells * config.dim
        assert "ema_state" in cost["unknown"]
    if gate:
        assert cost["parameters"]["by_component"]["exit_gate"] == extra_gate


@pytest.mark.parametrize("checkpointing", ["none", "full_cells"])
def test_expected_objective_counts_readouts_and_real_head_replay(checkpointing):
    config = geometry(exit_distribution=True)
    spec = RecipeSpec("expected", config,
                      TrainingRecipe(num_visits=3, objective="expected_ce", entropy_coeff=.1),
                      run={"objective": "expected_ce_minus_entropy_plus_router_aux"})
    cost = estimate_cost(spec, seq_len=4, activation_checkpointing=checkpointing)
    model = RecursiveLM(config).train()
    counts = {"suffix": 0, "gate_positions": 0, "head_positions": 0}

    def suffix_hook(*_):
        counts["suffix"] += 1

    def gate_hook(module, args, output):
        counts["gate_positions"] += args[0].numel() // config.dim

    def head_hook(module, args, output):
        counts["head_positions"] += args[0].numel() // config.dim

    handles = [model.suffix[0].register_forward_hook(suffix_hook),
               model.exit_gate.register_forward_hook(gate_hook),
               model.lm_head.register_forward_hook(head_hook)]
    try:
        output = model(torch.tensor([[2, 3, 4, 5]]), return_hidden=True)
        loss, _ = expected_lm_loss(output.visit_hidden_states, output.hazard_logits,
                                   model.lm_head, torch.tensor([[3, 4, -100, 6]]),
                                   torch.tensor(3), num_chunks=2)
        before_backward = dict(counts)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    assert before_backward["suffix"] == cost["parameters"]["by_region"]["suffix"]["logical_calls"] == 3
    assert before_backward["head_positions"] == cost["calls"]["head"] * 4 == 12
    assert before_backward["gate_positions"] == cost["calls"]["exit_gate"] * 4 == 12
    assert counts["head_positions"] - before_backward["head_positions"] == cost["calls"]["recompute_head"] * 4 == 12
    assert cost["calls"]["forward_cells"] == 10
    assert cost["calls"]["inference_cells"] == 8
    assert cost["parameters"]["by_region"]["suffix"]["inference_logical_calls"] == 1
    assert cost["formula_version"] == "fixed_expected_matrix_v1"
    forward = cost["flops"]["forward_components"]
    assert forward["head"] == 3 * 2 * config.dim * config.vocab_size
    assert forward["exit_gate"] == 3 * 2 * config.dim
    assert cost["flops"]["recompute_components"]["head"] == forward["head"]
    body_replay = sum(value for name, value in forward.items()
                      if name not in ("head", "input_adapter", "exit_gate")) if checkpointing == "full_cells" else 0
    assert cost["flops"]["recompute_per_token"] == forward["head"] + body_replay


def test_sampled_depth_has_expected_training_cost_and_fixed_inference_kv():
    config = geometry(num_visits=4, initial_state="random", input_injection="linear")
    recipe = TrainingRecipe(num_visits=4, grad_visits=2,
                            depth_rule="poisson_lognormal_micro_v1")
    spec = RecipeSpec("sampled", config, recipe, inference={"num_visits": 2})
    result = estimate_cost(spec, seq_len=4)
    assert result["formula_version"] == "sampled_depth_matrix_v1"
    assert result["flops"]["training_per_token"] is not None
    assert result["calls"]["forward_cells"] is None
    assert result["calls"]["head"] == 1  # The terminal projection count is known even when recurrence is not.
    assert result["calls"]["nominal"]["forward_cells"] == 10
    assert result["parameters"]["by_region"]["core"]["logical_calls"] is None
    assert result["parameters"]["by_region"]["core"]["nominal_logical_calls"] == 8
    assert result["flops"]["nominal_forward_per_token"] > 0
    assert result["kv"]["logical_slots"] == 6
    assert result["kv"]["num_visits"] == 2
    comparison = compare_recipes(spec, spec, seq_len=4)
    assert "training_flops_per_token" in comparison["matched"]
    assert "training_execution" in comparison["unknown"]


def test_training_objective_difference_is_explicit_and_not_a_quality_claim():
    terminal_config = geometry()
    terminal = RecipeSpec("terminal", terminal_config, TrainingRecipe(num_visits=3))
    expected = RecipeSpec("expected", replace(terminal_config, exit_distribution=True),
                          TrainingRecipe(num_visits=3, objective="expected_ce", entropy_coeff=.1))
    result = compare_recipes(terminal, expected, seq_len=4)
    assert result["measured-different"]["training.objective"] == {
        "reference": "terminal_ce", "candidate": "expected_ce", "basis": "declared_objective"}
    assert "objective_comparability" in result["unknown"]
    assert "cost_formula_version" not in result["controlled"]
    assert result["scientific_conclusion"] is None


def test_public_expected_recipe_uses_the_supported_objective_cost_path():
    from moeloop.recipes import ouro_spec

    spec = ouro_spec(vocab_size=17, dim=8, hidden_dim=12, num_heads=2,
                     num_kv_heads=1, num_experts=4, top_k=2, max_seq_len=8)
    result = estimate_cost(spec, seq_len=4)
    assert result["flops"]["training_per_token"] is not None
    assert result["calls"]["head"] == spec.training.num_visits
    assert result["calls"]["recompute_head"] == spec.training.num_visits


def test_published_p1_budget_values_stay_unchanged():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "p1_moe": (712_656_672, 1_284_757_348.671257),
        "p1_smelt": (737_845_056, 1_248_742_044.8630443),
    }
    for name, (parameters, flops) in expected.items():
        spec = RecipeSpec.from_dict(json.loads((root / "recipes" / (name + ".json")).read_text()))
        result = estimate_cost(spec, seq_len=4096,
                               attention_pairs_per_sequence=4_180_216.9229844734)
        assert result["parameters"]["unique_non_embedding"] == parameters
        assert result["flops"]["training_per_token"] == flops
        assert result["kv"]["bytes_at_context"] == 15_728_640
        assert result["formula_version"] == "fixed_terminal_document_matrix_v1"


def test_report_assembles_the_three_budgets():
    from moeloop import recipes

    small = dict(vocab_size=17, dim=8, hidden_dim=12, num_heads=2, num_kv_heads=1,
                 num_experts=4, top_k=2, max_seq_len=8)
    for spec in (recipes.moe_spec(layers=2, **small), recipes.ouro_spec(**small),
                 recipes.huginn_spec(**small), recipes.relaxed_spec(rank=2, **small)):
        options = dict(seq_len=4, batch_size=3, activation_checkpointing="full_cells")
        cost = estimate_cost(spec, bytes_per_element=4, attention_pairs_per_sequence=7.0, **options)
        training = training_flops(spec.model, spec.training, attention_pairs_per_sequence=7.0, **options)
        kv = kv_bytes(spec.model, spec.inference["num_visits"], seq_len=4, batch_size=3, bytes_per_element=4)
        assert cost["flops"] == training["flops"] and cost["tokens"] == training["tokens"]
        assert cost["calls"] == training["calls"] | {"inference_cells": kv["logical_slots"]}
        assert cost["kv"] == kv and cost["formula_version"] == training["formula_version"]
        parameters = parameter_counts(spec.model)
        for region, entry in cost["parameters"]["by_region"].items():
            calls = {key: entry.pop(key) for key in ("logical_calls", "inference_logical_calls", "nominal_logical_calls")
                     if key in entry}
            assert calls["logical_calls"] == training["region_calls"][region]["logical_calls"]
        assert cost["parameters"] == parameters


def test_kv_counts_every_logical_call_at_the_fixed_inference_depth():
    kv = kv_bytes(geometry(), 2, seq_len=4, batch_size=3, bytes_per_element=2)
    # Prefix, two core cells at two visits, suffix: six slots of K and V, one 4-wide KV head each.
    assert kv["logical_slots"] == 6 and kv["head_dim"] == 4
    assert kv["bytes_per_cached_token"] == 6 * 2 * 4 * 2
    assert kv["bytes_at_context"] == 6 * 2 * 4 * 2 * 4 * 3


@pytest.mark.parametrize("gate", [False, True])
def test_bare_model_costs_as_its_fixed_full_bptt_recipe(gate):
    config = geometry(exit_distribution=gate)
    spec = RecipeSpec("fixed", config, TrainingRecipe(num_visits=3, objective="expected_ce" if gate else "terminal_ce"))
    for checkpointing in ("none", "full_cells", None):
        assert estimate_cost(config, seq_len=4, activation_checkpointing=checkpointing) == estimate_cost(
            spec, seq_len=4, activation_checkpointing=checkpointing)


def test_training_cost_reads_the_objective_from_the_training_recipe():
    from moeloop.recipes import ouro_spec

    spec = ouro_spec(vocab_size=17, dim=8, hidden_dim=12, num_heads=2,
                     num_kv_heads=1, num_experts=4, top_k=2, max_seq_len=8)
    described = replace(spec, run=spec.run | {"objective": "terminal_ce_plus_router_aux"})
    # run.objective is a description; the training recipe selects the costed objective.
    assert estimate_cost(described, seq_len=4) == estimate_cost(spec, seq_len=4)
    assert estimate_cost(described, seq_len=4)["formula_version"] == "fixed_expected_matrix_v1"
