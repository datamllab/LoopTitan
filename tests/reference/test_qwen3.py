"""An independent upstream implementation guards common architectural errors."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from compare_baselines import cases, make_reference, run_suite, tiny_config
from moeloop.model import RecursiveLM


def test_official_qwen3_forward_gradient_and_short_training_trajectories():
    report = run_suite(steps=3)
    assert set(report["cases"]) == set(cases())
    for result in report["cases"].values():
        assert result["physical_parameter_elements"] == result["reference_physical_parameter_elements"]
        assert result["losses"][-1] < result["losses"][0]


def test_reference_sharing_and_order_are_observable():
    torch.manual_seed(123)
    actual = RecursiveLM(tiny_config(num_visits=3))
    reference, pairs = make_reference(actual)
    layers = reference.model.layers
    assert layers[0].mlp.experts is layers[2].mlp.experts is layers[4].mlp.experts
    assert layers[0].self_attn.q_proj.weight is layers[2].self_attn.q_proj.weight
    assert layers[0].self_attn.layer_idx != layers[2].self_attn.layer_idx
    assert layers[0].mlp.experts is not layers[1].mlp.experts
    assert len({pair.name for pair in pairs}) == len(list(actual.parameters()))
    ids = torch.tensor([[2, 5, 11, 8, 3]])
    with torch.no_grad():
        correct = reference(input_ids=ids).logits
        # A,A,A,B,B,B is a different computation from A,B,A,B,A,B.
        reference.model.layers = torch.nn.ModuleList([layers[index] for index in (0, 2, 4, 1, 3, 5)])
        wrong_order = reference(input_ids=ids).logits
    assert (correct - wrong_order).abs().max() > 1e-3


def test_reference_rejects_methods_outside_its_matching_scope():
    with pytest.raises(ValueError, match="outside this reference"):
        make_reference(RecursiveLM(tiny_config(input_injection="add")))
