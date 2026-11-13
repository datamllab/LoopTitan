"""Phase 1 arm specs: written by scripts/phase1_specs.py, on the P2 track and within its budget."""
import copy
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop import bench
from moeloop.comparison import compare_recipes
from moeloop.config import ModelConfig
from moeloop.recipe_spec import RecipeSpec

import phase1_specs

ROOT = Path(__file__).resolve().parents[2]

SPECS = phase1_specs.specs()
TRACK = bench.load_tracks()["P2"]
ELR = {"m7-elr-adamw": "m7-adamw", "m4-elr-adamw": "m4-adamw"}  # step-size arms (#264) and their sources
SCHEDULE = ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")


def base(name):
    return json.loads((ROOT / f"recipes/{name}.json").read_text())


def test_files_are_what_the_script_writes():
    files = {path.stem: json.loads(path.read_text()) for path in (ROOT / "recipes/phase1").glob("*.json")}
    assert files == SPECS
    assert len(SPECS) == 20  # 12 arms; m1s, m2, m7-elr and m4-elr run AdamW only.


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_arm_takes_schedule_and_optimizer_from_the_p2_track(name):
    spec = SPECS[name]
    run, optimizer, visits = spec["run"], spec["optimizer"], spec["model"]["num_visits"]
    assert run["successful_updates"] == run["schedule_total_steps"] == TRACK.horizon_updates
    assert {key: run["schedule"][key] for key in SCHEDULE} == {key: TRACK.schedule[key] for key in SCHEDULE}
    assert (run["seq_len"], run["global_batch_positions"], run["seed"]) == (TRACK.seq_len, TRACK.global_batch_positions, 42)
    assert (run["local_batch_size"], run["keep_latest_k"]) == (16, 0)
    fixed = TRACK.optimizers["fixed"][optimizer["name"]]
    assert {key: optimizer[key] for key in fixed if key != "muon"} == {key: fixed[key] for key in fixed if key != "muon"}
    assert (optimizer["base_lr"],) == TRACK.lr_grid[optimizer["name"]]  # the one calibrated LR
    assert optimizer.get("muon") == (fixed["muon"] | {"lr": optimizer["base_lr"]} if "muon" in fixed else None)
    assert optimizer["expert_lr_multiplier"] == (math.sqrt(2) if name in ELR else 1.0)
    assert spec["training"]["num_visits"] == spec["inference"]["num_visits"] == visits


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_arm_but_m0_2e_is_within_the_track_budget(name):
    comparison = compare_recipes(TRACK.reference_spec(), RecipeSpec.from_dict(SPECS[name]), seq_len=TRACK.seq_len,
                                 tolerance=TRACK.budget["tolerance"], **TRACK.budget["accounting"])
    outside = sorted(set(TRACK.budget["metrics"]) - set(comparison["matched"]))
    assert outside == (["unique_non_embedding_parameters"] if name.startswith("m0-2e") else [])


def test_the_ladder_shares_one_more_module_at_each_step():
    moe, smelt = base("p2_moe")["model"], base("p2_smelt")["model"]
    model = {name: spec["model"] for name, spec in SPECS.items()}
    assert ModelConfig.from_dict(model["m0-adamw"]) == ModelConfig.from_dict(moe)
    assert ModelConfig.from_dict(model["m3-adamw"]) == ModelConfig.from_dict(smelt)
    owners = ("attention_per_visit", "router_per_visit", "norm_per_visit")
    # m4 unties the 6 shared core layers of 88 experts into 12 layers of 44; m7 shares only the
    # experts, m5 also the router, m3 (SMELT) also attention and norms.
    assert (model["m4-adamw"]["num_cells"], model["m4-adamw"]["num_visits"]) == (12, 1)
    assert 12 * model["m4-adamw"]["core"]["num_experts"] == smelt["num_cells"] * smelt["num_experts"]
    assert [tuple(model[f"{arm}-adamw"][key] for key in owners) for arm in ("m7", "m5", "m3")] == [
        (True, True, True), (True, False, True), (False, False, False)]
    for arm in ("m4", "m7", "m5"):  # Only ownership changes along the ladder.
        assert {key: model[f"{arm}-adamw"][key] for key in ("dim", "num_heads", "hidden_dim", "num_experts", "top_k",
                                                              "prefix_cells", "suffix_cells")} == {
            key: smelt[key] for key in ("dim", "num_heads", "hidden_dim", "num_experts", "top_k",
                                        "prefix_cells", "suffix_cells")}


def test_the_rails_change_only_how_moe_experts_are_organised():
    moe = base("p2_moe")["model"]
    pool = {name: (spec["model"]["num_cells"], spec["model"]["num_visits"], spec["model"]["num_experts"],
                   spec["model"]["hidden_dim"], spec["model"]["top_k"], spec["model"]["order"],
                   spec["optimizer"]["expert_group_size"]) for name, spec in SPECS.items()}
    assert pool["m1-adamw"] == (6, 2, 128, 432, 8, "layer", 2)  # 6 banks, each shared by 2 adjacent layers
    assert pool["m1s-adamw"] == (6, 2, 128, 432, 8, "stack", 2)  # the same banks in loop order
    assert pool["m2-adamw"] == (3, 4, 256, 432, 8, "layer", 4)
    assert pool["m6-adamw"] == (12, 1, 128, 216, 16, "stack", 1)  # fine-grained: same pool as m1, no sharing
    assert pool["m0-2e-adamw"] == (12, 1, 128, 432, 8, "stack", 1)  # twice the expert parameters
    for name in ("m1-adamw", "m1s-adamw", "m2-adamw", "m6-adamw", "m0-2e-adamw"):
        assert {key: SPECS[name]["model"][key] for key in ("dim", "num_heads", "prefix_cells", "suffix_cells")} == {
            key: moe[key] for key in ("dim", "num_heads", "prefix_cells", "suffix_cells")}


@pytest.mark.parametrize("name,source", ELR.items())
def test_a_step_size_arm_differs_from_its_source_only_in_name_notes_and_the_multiplier(name, source):
    spec, expected = SPECS[name], copy.deepcopy(SPECS[source])
    assert spec["optimizer"]["expert_lr_multiplier"] == 1.4142135623730951 == math.sqrt(2)
    assert "sqrt(2)" in spec["preserved"][0]
    expected["name"], expected["preserved"] = f"phase1-{name}", spec["preserved"]
    expected["source"]["arm"] = name.removesuffix("-adamw")
    expected["optimizer"]["expert_lr_multiplier"] = 1.4142135623730951
    assert spec == expected


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_round_trips_through_recipe_spec(name):
    spec = RecipeSpec.from_dict(SPECS[name])
    assert spec.to_dict() == SPECS[name] and RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
