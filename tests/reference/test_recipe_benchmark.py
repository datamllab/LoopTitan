"""Bounded contracts for an independently implemented public reference."""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

import recipe_benchmark as benchmark
from moeloop.model import ModelConfig, RecursiveLM, RegionConfig


def tiny(**changes):
    return replace(ModelConfig(vocab_size=43, dim=24, hidden_dim=32,
        num_heads=4, num_kv_heads=2, num_experts=4, top_k=2,
        num_cells=2, num_visits=1, max_seq_len=16, qk_norm=True), **changes)


@pytest.mark.parametrize("visits,ac", [(1, "none"), (2, "full")])
def test_cpu_sanity_checks_full_update_without_claiming_gpu_performance(visits, ac):
    pytest.importorskip("transformers")
    pytest.importorskip("torchtitan")
    settings = benchmark.Settings(mode="sanity", reference="transformers",
        candidate="plain", attention="sdpa", device="cpu", dtype="float32",
        batch_size=2, seq_len=8, head_chunks=2, activation_checkpoint=ac)
    report = benchmark.run(tiny(num_visits=visits), settings)
    assert report["status"] == "passed"
    assert report["correctness"]["physical_parameter_elements"] > 0
    assert report["correctness"]["all_parameter_gradients_checked"]
    assert report["correctness"]["optimizer_updates"] == 1
    assert report["batch"]["valid_labels"] == 16
    assert report["performance"]["status"] == "not_measured"
    assert "tokens_per_second" not in report["performance"]
    assert report["conditions"]["balance_coeff"] == 0
    assert report["conditions"]["z_loss_coeff"] == 0
    assert report["conditions"]["seed"] == 42
    json.dumps(report, allow_nan=False)


def test_bad_reference_weights_fail_the_gate():
    pytest.importorskip("transformers")
    pytest.importorskip("torchtitan")
    settings = benchmark.Settings(mode="sanity", reference="transformers",
        candidate="plain", attention="sdpa", device="cpu", dtype="float32",
        batch_size=1, seq_len=8, head_chunks=2)
    actual, reference, pairs = benchmark.make_models(tiny(), settings)
    with torch.no_grad():
        reference.lm_head.weight[0].add_(0.2)
    batch = benchmark.make_batch(tiny(), settings)
    result = benchmark.check_correctness(actual, reference, pairs, batch, settings)
    assert result["status"] == "failed"
    assert any("initial parameters" in error for error in result["failures"])


def test_torchtitan_reference_maps_unique_owners_and_fixed_loop_order():
    pytest.importorskip("torchtitan")
    settings = benchmark.Settings(reference="torchtitan", candidate="plain",
        attention="flex", device="cpu", seq_len=8, head_chunks=2)
    config = tiny(qk_norm=False, num_visits=2, prefix_cells=1, suffix_cells=1)
    torch.manual_seed(42)
    actual = RecursiveLM(config)
    reference, pairs = benchmark.make_torchtitan_reference(actual, attention="flex")
    layers = list(reference.layers.values())
    assert type(reference).__module__ == "torchtitan.models.qwen3.model"
    assert len(layers) == 6
    assert layers[1].moe.routed_experts.inner_experts.w1_EFD is layers[3].moe.routed_experts.inner_experts.w1_EFD
    assert layers[1].attention.qkv_linear.wq.weight is layers[3].attention.qkv_linear.wq.weight
    assert layers[1].moe.routed_experts.inner_experts.w1_EFD is not layers[2].moe.routed_experts.inner_experts.w1_EFD
    assert sum(p.numel() for p in actual.parameters()) == sum(p.numel() for p in reference.parameters())
    assert {pair.name for pair in pairs} == {name for name, _ in actual.canonical_parameters()}
    assert all(torch.equal(pair.actual, pair.reference_value()) for pair in pairs)
    assert all(layer.moe.expert_bias_E is None for layer in layers)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_native_expert_replacement_preserves_all_physical_parameter_keys(qk_norm):
    pytest.importorskip("torchtitan")
    config = tiny(qk_norm=qk_norm, prefix_cells=1, suffix_cells=1, num_visits=2)
    settings = benchmark.Settings(reference="torchtitan", candidate="native",
        attention="flex", device="cpu", seq_len=8, head_chunks=2)
    torch.manual_seed(42)
    expected = RecursiveLM(config)
    expected_parameters = dict(expected.canonical_parameters())
    actual, _, pairs = benchmark.make_models(config, settings)
    actual_nonexperts = {name: param for name, param in actual.named_parameters() if ".experts." not in name}
    expected_nonexperts = {name for name in expected_parameters if ".experts." not in name}
    assert set(actual_nonexperts) == expected_nonexperts
    assert {pair.name for pair in pairs} == set(expected_parameters)
    assert {id(pair.actual) for pair in pairs} == {id(param) for param in actual.parameters()}
    assert ".routed.inner_experts." in next(name for name, _ in actual.named_parameters() if ".experts." in name)
    for pair in pairs:
        assert torch.equal(pair.actual, expected_parameters[pair.name])
        assert torch.equal(pair.actual, pair.reference_value())
        if pair.name in actual_nonexperts:
            assert pair.actual is actual_nonexperts[pair.name]


def test_unmatched_method_and_cpu_throughput_are_rejected():
    settings = benchmark.Settings(device="cpu", seq_len=8, head_chunks=2)
    with pytest.raises(ValueError, match="residual"):
        benchmark.describe(tiny(core=RegionConfig(residual_scale=0.5)), settings)
    with pytest.raises(ValueError, match="CUDA"):
        benchmark.run(tiny(), replace(settings, mode="benchmark"))


def test_describe_and_lag_policy_distinguish_measurement_from_qualification():
    report = benchmark.describe(tiny(), benchmark.Settings(device="cpu", seq_len=8, head_chunks=2))
    assert report["status"] == "not_run"
    assert report["correctness"] is None
    assert report["performance"]["status"] == "not_measured"
    assert report["lag_policy"]["max_step_ratio"] == 1.2
    assert benchmark.performance_verdict(0.119, 0.1) == "passed"
    assert benchmark.performance_verdict(0.121, 0.1) == "needs_investigation"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_independent_same_input_components_check_all_derivatives(dtype):
    pytest.importorskip("transformers")
    settings = benchmark.Settings(reference="transformers", candidate="plain", attention="sdpa",
        device="cpu", dtype=dtype, batch_size=1, seq_len=8, head_chunks=2)
    torch.manual_seed(42)
    actual = RecursiveLM(tiny())
    result = benchmark.check_component_math(actual, benchmark.make_batch(tiny(), settings), settings)
    assert result["status"] == "passed"
    assert result["timed_routing"] == "natural_unmodified_router_outputs"
    checks = result["checks"]
    assert set(checks["attention"]["errors"]) == {"forward", "gradient.query", "gradient.key", "gradient.value"}
    assert set(checks["router"]["errors"]) == {"logits", "weights", "gradient.input", "gradient.gate.weight"}
    assert checks["router"]["expert_sets_equal"]
    assert set(checks["experts"]["errors"]) == {"forward", "gradient.input", "gradient.route_weights",
        "gradient.w1_EFD", "gradient.w3_EFD", "gradient.w2_EDF"}
    assert checks["experts"]["coverage"]["experts"] == 4
    json.dumps(result, allow_nan=False)


def test_component_oracle_rejects_an_incorrect_expert_operator():
    pytest.importorskip("transformers")
    settings = benchmark.Settings(reference="transformers", candidate="plain", attention="sdpa",
        device="cpu", dtype="float32", batch_size=1, seq_len=8, head_chunks=2)
    torch.manual_seed(42)
    actual = RecursiveLM(tiny())
    original = actual.core[0].experts.forward
    actual.core[0].experts.forward = lambda *args: original(*args) * 1.1
    result = benchmark.check_component_math(actual, benchmark.make_batch(tiny(), settings), settings)
    assert result["status"] == "failed"
    assert result["checks"]["experts"]["status"] == "failed"
    assert result["checks"]["experts"]["errors"]["forward"]["relative_l2"] > .09


def test_component_math_is_explicit_and_cannot_qualify_missing_gradients():
    assert not benchmark.Settings().component_math
    result = {"status": "failed", "initial_parameters_equal": True, "losses_finite": True,
        "ce_close": True, "all_gradients_finite": True, "all_updated_parameters_finite": True,
        "optimizer_updates": 1, "all_parameter_gradients_checked": False}
    assert not benchmark.full_step_safe(result)
    result["all_parameter_gradients_checked"] = True
    assert benchmark.full_step_safe(result)
