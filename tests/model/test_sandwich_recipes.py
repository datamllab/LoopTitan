"""Part I sandwich recipes: geometry, parameter counts, injection and conformance."""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes, testing
from moeloop.costs import estimate_cost
from moeloop.model import RecursiveLM
from moeloop.recipe_spec import RecipeSpec


SMALL = dict(vocab_size=31, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
             num_experts=4, top_k=2, max_seq_len=32)


@pytest.mark.parametrize("dense", [False, True])
def test_sandwich_geometry_and_parameter_counts(dense):
    spec = recipes.sandwich_spec(dense=dense)
    model = spec.model
    assert spec.name == ("sandwich_dense" if dense else "sandwich")
    assert (model.prefix_cells, model.num_cells, model.num_visits, model.suffix_cells) == (2, 3, 8, 2)
    assert (model.input_injection, model.injection_dt, model.injection_decay, model.initial_state) == (
        "diag_decay", 1.0, 1.0, "input")
    assert model.num_experts == (0 if dense else 8) and model.order == "stack"
    assert (spec.training.num_visits, spec.training.grad_visits, spec.training.sampled_visits) == (8, None, ())
    assert spec.training.balance_coeff == (0.0 if dense else 0.001)
    assert spec.source["P_R_T_C"] == [2, 3, 8, 2]
    assert spec.source["selected_path"] == ("dense_middle_prelude_injection" if dense else "moe_middle_input_injection")
    # Per cell (d=128, 4 heads, 2 KV heads of width 32): attention 2*128*128 + 2*128*64 = 49152,
    # two norms 256, and either a dense SwiGLU 3*128*256 = 98304 or a router 128*8 = 1024 plus
    # eight experts 8*3*128*256 = 786432. Seven physical cells; untied embedding and head
    # 2*50257*128 = 12865792, final norm 128, injection W 128*128 plus dt and a 2*128 = 16640.
    cell = 49152 + 256 + (98304 if dense else 1024 + 786432)
    expected = 7 * cell + 12865792 + 128 + 16640
    assert expected == (13916544 if dense else 18740608)
    assert sum(p.numel() for p in RecursiveLM(model).parameters()) == expected
    cost = estimate_cost(spec, seq_len=16)
    assert cost["parameters"]["unique_total"] == expected
    assert cost["calls"]["forward_cells"] == 2 + 3 * 8 + 2


def test_sandwich_state_starts_at_the_prelude_output_and_takes_the_diagonal_write():
    model = RecursiveLM(recipes.sandwich_spec(**SMALL).model).eval()
    prelude, writes = [], []
    model.prefix[-1].register_forward_hook(lambda module, args, output: prelude.append(output[0]))
    model.input_adapter.register_forward_hook(lambda module, args, output: writes.append((*args, output)))
    model(torch.randint(0, 31, (2, 8)))
    assert len(writes) == 8
    state, anchor, written = writes[0]
    assert torch.equal(anchor, prelude[0]) and torch.equal(state, prelude[0])
    # At initialization W = I, a = 1 and dt = 1: exp(-1) * e + 1 * e.
    torch.testing.assert_close(written, (1 + math.exp(-1)) * prelude[0])
    assert all(torch.equal(later_anchor, prelude[0]) for _, later_anchor, _ in writes[1:])


@pytest.mark.parametrize("dense", [False, True])
def test_sandwich_conforms_and_round_trips(dense):
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    spec = recipes.sandwich_spec(dense=dense, **SMALL)
    report = testing.check_recipe(spec)
    assert {result["status"] for result in report["checks"].values()} == {"pass"}
    assert RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_sandwich_records_its_source_and_requires_prelude_and_coda():
    spec = recipes.sandwich_spec()
    assert spec.source["url"] == "https://arxiv.org/html/2610.23036v1"
    assert spec.preserved and spec.adaptations
    # The MoE path departs from the paper's random-state MoE tables; the record says so.
    assert any("moe_middle_input_injection" in line for line in spec.adaptations)
    assert not any("moe_middle_input_injection" in line for line in recipes.sandwich_spec(dense=True).adaptations)
    for missing in ({"prelude_layers": 0}, {"coda_layers": 0}):
        with pytest.raises(ValueError, match="prelude and coda"):
            recipes.sandwich_spec(**missing)


def test_sandwich_without_experts_requires_the_dense_flag():
    with pytest.raises(ValueError, match="sparse"):
        recipes.sandwich_spec(num_experts=0)
    assert recipes.sandwich_spec(dense=True, num_experts=0).model.num_experts == 0
