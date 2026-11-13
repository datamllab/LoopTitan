"""Sampled depth and tail gradients: expected training FLOPs and quantiles over the depth rule."""

from collections import Counter
from dataclasses import replace
import math

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes
from moeloop.comparison import compare_recipes
from moeloop.costs import depth_probability, estimate_cost, training_depth
from moeloop.latent import sampled_depth
from moeloop.model import Cell, RecursiveLM
from moeloop.recipe_spec import RecipeSpec
from moeloop.training import TrainingRecipe


GEOMETRY = dict(vocab_size=17, dim=8, hidden_dim=12, num_heads=2, num_kv_heads=1,
                num_experts=4, top_k=2, max_seq_len=8)
POISSON = TrainingRecipe(num_visits=4, grad_visits=3, depth_rule="poisson_lognormal_micro_v1",
                         depth_mean=4.0, depth_log_std=0.5)
DEPTHS = range(1, 200)  # Poisson-lognormal mass beyond 199 visits is far below 1e-12 here.


def specs():
    huginn = recipes.huginn_spec(grad_visits=2, **GEOMETRY)
    fixed_tail = replace(huginn, training=TrainingRecipe(num_visits=4, grad_visits=2))
    return {"input_anchor": recipes.input_anchor_spec(**GEOMETRY), "huginn": huginn,
            "fixed_tail": fixed_tail}


def fixed_flops(spec, visits, **options):
    """The unchanged fixed-depth/full-BPTT formula at one depth."""
    model = replace(spec.model, num_visits=visits)
    return estimate_cost(RecipeSpec("fixed", model, TrainingRecipe(num_visits=visits)),
                         seq_len=4, **options)["flops"]


def pass_flops(spec, visits, **options):
    """One pass: forward at ``visits``; backward and replay at the differentiated tail."""
    tail = fixed_flops(spec, min(spec.training.grad_visits or visits, visits), **options)
    return (fixed_flops(spec, visits, **options)["forward_per_token"]
            + tail["backward_per_token"] + tail["recompute_per_token"])


def test_poisson_lognormal_probabilities_reduce_to_poisson_without_spread():
    recipe = replace(POISSON, depth_log_std=0.0)
    for visits in range(1, 25):
        n = visits - 1
        assert depth_probability(recipe, visits) == pytest.approx(math.exp(-3) * 3**n / math.factorial(n), rel=1e-10)
    assert depth_probability(recipe, 0) == 0


def test_poisson_lognormal_probabilities_match_moments_and_the_sampler():
    probabilities = [depth_probability(POISSON, visits) for visits in DEPTHS]
    assert sum(probabilities) == pytest.approx(1, abs=1e-12)
    # k = 1 + N, N | rate ~ Poisson(rate), E[rate] = 3, E[rate^2] = 9 exp(sigma^2).
    assert sum(v * p for v, p in zip(DEPTHS, probabilities)) == pytest.approx(4.0, rel=1e-10)
    assert sum(v * v * p for v, p in zip(DEPTHS, probabilities)) == pytest.approx(
        1 + 2 * 3 + 3 + 9 * math.exp(0.25), rel=1e-10)
    draws = [sampled_depth(seed=7, completed_updates=update, microbatch=micro, mean=4.0, log_std=0.5)
             for update in range(1000) for micro in range(4)]
    for visits in (1, 2, 3, 4, 6, 9):
        exact = sum(probabilities[:visits])
        empirical = sum(depth <= visits for depth in draws) / len(draws)
        assert abs(empirical - exact) < 4 * math.sqrt(exact * (1 - exact) / len(draws))


def test_uniform_and_fixed_depth_probabilities_are_exact():
    uniform = TrainingRecipe(num_visits=4, sampled_visits=(2, 4, 6), grad_visits=2)
    assert [depth_probability(uniform, v) for v in range(1, 8)] == [0, 1 / 3, 0, 1 / 3, 0, 1 / 3, 0]
    fixed = TrainingRecipe(num_visits=3, grad_visits=1)
    assert [depth_probability(fixed, v) for v in range(1, 5)] == [0, 0, 1, 0]


def test_training_depth_reports_expected_tail_and_quantile_visits():
    assert training_depth(TrainingRecipe(num_visits=4, sampled_visits=(2, 4, 6), grad_visits=2)) == {
        "rule": "uniform_update_v1", "sampled_per": "update", "grad_visits": 2,
        "expected_visits": 4.0, "expected_grad_visits": 2.0,
        "quantile_visits": {"p50": 4, "p95": 6}}
    assert training_depth(TrainingRecipe(num_visits=3, grad_visits=1))["quantile_visits"] == {"p50": 3, "p95": 3}
    # Twenty equally likely depths: exactly the 10th and 19th, not one past a float boundary.
    assert training_depth(TrainingRecipe(num_visits=4, sampled_visits=tuple(range(1, 21))))["quantile_visits"] == {
        "p50": 10, "p95": 19}
    summary = training_depth(POISSON)
    probabilities = [depth_probability(POISSON, visits) for visits in DEPTHS]
    cdf = [sum(probabilities[:visits]) for visits in DEPTHS]
    assert summary["sampled_per"] == "microbatch"
    assert summary["expected_visits"] == 4.0
    assert summary["expected_grad_visits"] == pytest.approx(
        sum(min(3, v) * p for v, p in zip(DEPTHS, probabilities)), rel=1e-12)
    assert summary["quantile_visits"] == {
        "p50": next(v for v, c in zip(DEPTHS, cdf) if c >= 0.5),
        "p95": next(v for v, c in zip(DEPTHS, cdf) if c >= 0.95)}
    full = training_depth(replace(POISSON, grad_visits=None))
    assert full["expected_grad_visits"] == full["expected_visits"] == 4.0


@pytest.mark.parametrize("pairs", [None, 7.0])
@pytest.mark.parametrize("checkpointing", ["none", "full_cells"])
@pytest.mark.parametrize("name", ["input_anchor", "huginn", "fixed_tail"])
def test_expected_training_flops_average_fixed_depth_passes(name, checkpointing, pairs):
    spec = specs()[name]
    options = {"activation_checkpointing": checkpointing}
    if pairs is not None:
        options["attention_pairs_per_sequence"] = pairs
    cost = estimate_cost(spec, seq_len=4, **options)
    weights = [(visits, depth_probability(spec.training, visits)) for visits in DEPTHS]
    weights = [(visits, p) for visits, p in weights if p]
    flops = cost["flops"]
    assert flops["training_per_token"] == pytest.approx(
        sum(p * pass_flops(spec, visits, **options) for visits, p in weights), rel=1e-9)
    assert flops["forward_per_token"] == pytest.approx(
        sum(p * fixed_flops(spec, visits, **options)["forward_per_token"] for visits, p in weights), rel=1e-9)
    assert flops["training_per_token"] == flops["forward_per_token"] + flops["backward_per_token"] + flops["recompute_per_token"]
    assert flops["training_per_forward_batch"] == flops["training_per_token"] * 4
    for quantile in flops["training_depth"]["quantiles"].values():
        assert quantile["training_per_token"] == pytest.approx(pass_flops(spec, quantile["visits"], **options), rel=1e-12)
    assert cost["formula_version"] == ("sampled_depth_matrix_v1" if pairs is None else "sampled_depth_document_matrix_v1")
    # Only the fixed depth has a deterministic tail: prefix, last two visits of two cells, suffix.
    backward = 1 + 2 * 2 + 1 if name == "fixed_tail" else None
    assert cost["calls"]["backward_cells"] == backward
    assert cost["calls"]["recompute_cells"] == (None if backward is None else
                                                backward if checkpointing == "full_cells" else 0)


def test_tail_gradients_differentiate_the_prefix_last_visits_and_suffix():
    spec = recipes.huginn_spec(grad_visits=2, **GEOMETRY)
    model = RecursiveLM(spec.model).train()
    counts = Counter()

    def hook(module, args, output):
        counts["forward"] += 1
        if output[0].requires_grad:
            output[0].register_hook(lambda grad: counts.update(["backward"]))

    handles = [module.register_forward_hook(hook) for module in model.modules() if isinstance(module, Cell)]
    try:
        model(torch.tensor([[2, 3, 4, 5]]), num_visits=5, grad_visits=2).logits.sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    fixed = {visits: estimate_cost(RecipeSpec("fixed", replace(spec.model, num_visits=visits),
                                              TrainingRecipe(num_visits=visits)), seq_len=4)["calls"]
             for visits in (2, 5)}
    assert counts["forward"] == fixed[5]["forward_cells"] == 1 + 2 * 5 + 1
    assert counts["backward"] == fixed[2]["forward_cells"] == 1 + 2 * 2 + 1
    assert all(parameter.grad is not None for parameter in model.prefix.parameters())


def test_huginn_and_ouro_enter_the_budget_table_with_expectation_and_p95():
    baseline = recipes.moe_spec(layers=4, **GEOMETRY)
    huginn = recipes.huginn_spec(**GEOMETRY)
    ouro = recipes.ouro_spec(**GEOMETRY)
    for spec in (huginn, ouro):
        result = compare_recipes(baseline, spec, seq_len=4)
        assert "training_flops_per_token" not in result["unknown"]
        entry = (result["matched"] | result["measured-different"])["training_flops_per_token"]
        assert entry["candidate"] == result["costs"]["candidate"]["flops"]["training_per_token"]
    result = compare_recipes(baseline, huginn, seq_len=4)
    entry = (result["matched"] | result["measured-different"])["training_flops_per_token"]
    costs = result["costs"]
    assert entry["basis"] == "analytical_expectation_over_training_depth"
    assert entry["p95"] == {
        "reference": costs["reference"]["flops"]["training_per_token"],
        "candidate": costs["candidate"]["flops"]["training_depth"]["quantiles"]["p95"]["training_per_token"]}
    assert result["measured-different"]["cost_formula_version"]["candidate"] == "sampled_depth_matrix_v1"
    assert list(result["unknown"]["training_execution"]) == ["candidate"]
    assert "training_flops_per_token" in compare_recipes(huginn, huginn, seq_len=4)["matched"]
    fixed = compare_recipes(baseline, ouro, seq_len=4)
    entry = (fixed["matched"] | fixed["measured-different"])["training_flops_per_token"]
    assert entry["basis"] == "analytical_estimate" and "p95" not in entry


def test_fixed_full_bptt_recipes_keep_the_old_formula():
    for spec in (recipes.moe_spec(layers=2, **GEOMETRY), recipes.ouro_spec(**GEOMETRY),
                 recipes.recursive_moe_spec(**GEOMETRY), recipes.loom_spec(**GEOMETRY)):
        cost = estimate_cost(spec, seq_len=4)
        assert cost["formula_version"] in ("fixed_terminal_matrix_v1", "fixed_expected_matrix_v1")
        assert "training_depth" not in cost["flops"]
        assert cost["calls"]["backward_cells"] == cost["calls"]["forward_cells"]


def test_tail_gradients_without_input_injection_have_no_training_estimate():
    base = recipes.recursive_moe_spec(**GEOMETRY)
    for training in (TrainingRecipe(num_visits=4, grad_visits=1),
                     TrainingRecipe(num_visits=4, sampled_visits=(2, 4), grad_visits=1)):
        cost = estimate_cost(replace(base, training=training), seq_len=4)
        assert cost["flops"]["training_per_token"] is None
        assert "training_path" in cost["unknown"]
        assert "training_depth" not in cost["flops"]


def test_unknown_checkpointing_keeps_expected_training_flops_unknown():
    cost = estimate_cost(recipes.huginn_spec(**GEOMETRY), seq_len=4, activation_checkpointing=None)
    assert cost["flops"]["training_per_token"] is None
    assert cost["flops"]["forward_per_token"] is not None
    assert all(quantile["training_per_token"] is None
               for quantile in cost["flops"]["training_depth"]["quantiles"].values())
