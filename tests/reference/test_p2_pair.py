"""Independent owner counts and the predeclared 32-shape P2 budget screen."""

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes
from moeloop.costs import estimate_cost
from moeloop.model import RecursiveLM


METRICS = {"unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes"}


def test_p2_screen_is_cost_only_and_selects_the_predeclared_feasible_minimum(monkeypatch):
    def reject_allocation(*args, **kwargs):
        raise AssertionError("arithmetic screen constructed a model")
    monkeypatch.setattr(RecursiveLM, "__init__", reject_allocation)
    from moeloop.comparison import smelt_p2_pair
    pair = smelt_p2_pair(attention_pairs_per_sequence=4_180_216.9229844734)
    assert pair["anchor"] == "P2" and pair["candidate_count"] == 32
    rows = pair["candidates"]
    assert {(r["num_experts"], r["hidden_dim"]) for r in rows} == {
        (e, f) for e in (88, 92, 96, 100) for f in (336, 344, 352, 360, 368, 376, 384, 392)}
    feasible = [i for i, r in enumerate(rows) if set(r["matched_budgets"]) == METRICS]
    assert pair["selected_candidate_index"] in feasible
    chosen = rows[pair["selected_candidate_index"]]
    assert (chosen["num_experts"], chosen["hidden_dim"]) == (88, 384)
    scores = {i: (max(abs(v) for v in rows[i]["relative_differences"].values()),
                  sum(abs(v) for v in rows[i]["relative_differences"].values()),
                  abs(rows[i]["num_experts"] - 96), rows[i]["hidden_dim"]) for i in feasible}
    assert scores[pair["selected_candidate_index"]] == min(scores.values())
    comparison = pair["comparison"]
    assert set(comparison["matched"]) == METRICS
    assert comparison["controlled"]["attention_pairs_per_sequence"] == 4_180_216.9229844734
    assert pair["attention_statistics"]["formal_p1_prefix_mean"] == 4_180_216.9229844734
    assert comparison["endpoint"]["nominal_successful_updates"] == 19537
    assert comparison["scientific_conclusion"] is None
    costs = comparison["costs"]
    assert costs["reference"]["parameters"]["unique_non_embedding"] == 1_318_288_640
    assert costs["candidate"]["parameters"]["unique_non_embedding"] == 1_314_771_744
    assert costs["reference"]["flops"]["training_per_token"] == pytest.approx(2_573_863_361.5343013)
    assert costs["candidate"]["flops"]["training_per_token"] == pytest.approx(2_515_805_029.898698)
    assert pair["active_parameters"]["reference"]["physical_non_embedding"] == 203_521_280
    assert pair["active_parameters"]["candidate"] == {
        "physical_non_embedding": 146_920_224, "logical_visited_non_embedding": 220_379_808}
    dense = pair["dense_comparison"]
    assert set(dense["matched"]) == METRICS
    assert dense["costs"]["reference"]["flops"]["training_per_token"] == 3_140_728_320
    assert dense["costs"]["candidate"]["flops"]["training_per_token"] == 3_217_300_416
    assert dense["arms"] == comparison["arms"]


@pytest.mark.parametrize("make,total,nonembedding,slots", [
    ("p2_moe_spec", 1_706_561_280, 1_318_288_640, 12),
    ("p2_smelt_spec", 1_635_096_672, 1_314_771_744, 18),
])
def test_p2_cost_owners_match_real_meta_models(make, total, nonembedding, slots):
    spec = getattr(recipes, make)()
    with torch.device("meta"):
        model = RecursiveLM(spec.model)
    assert sum(p.numel() for p in model.parameters()) == total
    assert sum(p.numel() for name, p in model.named_parameters()
               if name not in ("embedding.weight", "lm_head.weight")) == nonembedding
    assert model.embedding.weight is not model.lm_head.weight
    assert len(model.cache_slots()) == slots
    assert all(len(c.attentions) == len(c.routers) == 1 for c in model.core)
    assert estimate_cost(spec, seq_len=4096)["parameters"]["unique_total"] == total


def test_c_component_parameters_follow_actual_owners_and_are_not_iso_parameter():
    totals = {}
    for name, spec in recipes.sharing_panel_specs().items():
        with torch.device("meta"):
            model = RecursiveLM(spec.model)
        report = estimate_cost(spec, seq_len=2048)
        totals[name] = sum(p.numel() for p in model.parameters())
        assert totals[name] == report["parameters"]["unique_total"]
        assert report["calls"]["forward_cells"] == 8
        assert report["kv"]["logical_slots"] == 8
        owners = report["parameters"]["by_region"]["core"]["owners"]
        assert owners["attention"] == (8 if name[1] == "1" else 2)
        assert owners["router"] == (8 if name[-1] == "1" else 2)
        assert owners["norm_pairs"] == owners["expert_banks"] == 2
    assert totals["a1_r0"] - totals["a0_r0"] == 3_932_160
    assert totals["a0_r1"] - totals["a0_r0"] == 24_576
    assert totals["a1_r1"] - totals["a0_r0"] == 3_956_736


def test_p2_unspecified_recompute_is_not_silently_matched():
    from moeloop.comparison import smelt_p2_pair
    with pytest.raises(ValueError, match="activation_checkpointing"):
        smelt_p2_pair(activation_checkpointing=None)


def test_p2_dense_default_and_explicit_document_geometry_remain_distinct():
    from moeloop.comparison import smelt_p2_pair
    dense = smelt_p2_pair()
    chosen = dense["comparison"]["arms"]["candidate"]["model"]
    assert (chosen["num_experts"], chosen["hidden_dim"]) == (96, 352)
    assert dense["comparison"]["costs"]["candidate"]["formula_version"] == "fixed_terminal_matrix_v1"
    assert recipes.p2_smelt_spec().model.num_experts == 88
    replay = smelt_p2_pair(activation_checkpointing="full_cells",
                          attention_pairs_per_sequence=4_180_216.9229844734)
    assert replay["comparison"]["arms"]["candidate"]["run"]["activation_checkpointing"] == "full_cells"
    assert replay["comparison"]["costs"]["candidate"]["calls"]["recompute_cells"] == 18
    assert set(replay["comparison"]["matched"]) == METRICS
