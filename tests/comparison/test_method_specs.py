"""Method specs: written by scripts/method_specs.py, on Track P2, within m0's budget, with each method's mechanism.

M0-Loop, the budget-convention control, matches m0's parameters only. LatentLoop-Residual is M3 with
per-visit core residual scales (LatentLoop Section 3.1), not the paper's final method."""
import hashlib
import json
from pathlib import Path

import pytest

from moeloop import bench, recipes
from moeloop.comparison import compare_recipes, diff_recipes
from moeloop.config import ModelConfig
from moeloop.recipe_spec import RecipeSpec
from moeloop.recipes import resolve

import method_specs

ROOT = Path(__file__).resolve().parents[2]

SPECS = method_specs.specs()
TRACK = bench.load_tracks()["P2"]
SCHEDULE = ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")
# The constructor defaults each method spec starts from.
DEFAULTS = {"meta": recipes.meta_spec(), "foil": recipes.foil_spec(), "expert-tying": recipes.expert_tying_spec(),
            "huginn": recipes.huginn_spec(), "loom": recipes.loom_spec(), "ouro": recipes.ouro_spec(),
            "sandwich": recipes.sandwich_spec(), "relaxed": recipes.relaxed_spec(),
            "latentloop-residual": recipes.latentloop_residual_spec()}
# The only model fields a method spec may change: its size. Region overrides may change only their sizes too.
GEOMETRY = {"vocab_size", "dim", "hidden_dim", "num_heads", "num_kv_heads", "num_experts", "top_k", "max_seq_len",
            "prefix_cells", "num_cells", "suffix_cells", "lora_attention_rank"}
REGION_GEOMETRY = {"hidden_dim", "num_experts", "top_k"}
# Source fields that restate the geometry, and the two the script adds.
SOURCE_GEOMETRY = {"base_E_D_L", "selected_E_D_L", "group_topology", "P_R_T_C", "layers_loops_rank", "track",
                   "constructor"}
# M0-Loop takes these model fields from m3 and every other field from m0.
TOPOLOGY = ("prefix_cells", "num_cells", "suffix_cells", "num_visits", "core")
# sha256 of the 20 method spec files, in name order, as #226 and #233 froze them.
EARLIER_SPECS_SHA256 = "69431780040c57d823ef6d0c3d78a48fd979f77de36f93deea953e710f27fbfc"
# sha256 of the 22 method spec files, in name order, as #244 froze them (the 20 and M0-Loop).
SPECS_BEFORE_LATENTLOOP_SHA256 = "7dca12db812ce814b5162638a69c1d675a0c6a07c9c70e944d3aaa932ee652b4"


def phase1(arm: str, optimizer: str) -> RecipeSpec:
    return RecipeSpec.from_dict(json.loads((ROOT / f"recipes/phase1/{arm}-{optimizer}.json").read_text()))


def mechanism(model: dict) -> dict:
    """``model`` without its geometry: every field that defines how the method computes."""
    regions = {region: None if model[region] is None else
               {key: value for key, value in model[region].items() if key not in REGION_GEOMETRY}
               for region in ("prefix", "core", "suffix")}
    return {key: value for key, value in model.items() if key not in GEOMETRY} | regions


def track_free(optimizer: dict) -> dict:
    """``optimizer`` without the settings Track P2 declares (name, LR and its fixed settings)."""
    declared = {"name", "base_lr"} | set(TRACK.optimizers["fixed"][optimizer["name"]])
    return {key: value for key, value in optimizer.items() if key not in declared}


def test_files_are_what_the_script_writes():
    files = {path.stem: json.loads(path.read_text()) for path in (ROOT / "recipes/methods").glob("*.json")}
    assert files == SPECS
    assert sorted(SPECS) == sorted(f"{name}-{optimizer}" for name in [*DEFAULTS, "smelt-v3", "smelt-v4", "m0-loop"]
                                   for optimizer in ("adamw", "muon"))


def files_sha256(names: list[str]) -> str:
    return hashlib.sha256(b"".join((ROOT / f"recipes/methods/{name}.json").read_bytes() for name in names)).hexdigest()


def test_adding_m0_loop_leaves_the_20_earlier_specs_byte_identical():
    names = sorted(name for name in SPECS if not name.startswith(("m0-loop-", "latentloop-residual-")))
    assert len(names) == 20 and files_sha256(names) == EARLIER_SPECS_SHA256


def test_adding_latentloop_residual_leaves_the_22_earlier_specs_byte_identical():
    names = sorted(name for name in SPECS if not name.startswith("latentloop-residual-"))
    assert len(names) == 22 and files_sha256(names) == SPECS_BEFORE_LATENTLOOP_SHA256


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_round_trips(name):
    assert RecipeSpec.from_dict(json.loads(json.dumps(SPECS[name]))).to_dict() == SPECS[name]
    assert SPECS[name]["name"] == f"method-{name}"


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_takes_schedule_and_optimizer_from_the_p2_track(name):
    spec = SPECS[name]
    run, optimizer, model = spec["run"], spec["optimizer"], spec["model"]
    assert run["successful_updates"] == run["schedule_total_steps"] == TRACK.horizon_updates
    assert {key: run["schedule"][key] for key in SCHEDULE} == {key: TRACK.schedule[key] for key in SCHEDULE}
    assert (run["seq_len"], run["global_batch_positions"], run["seed"]) == (TRACK.seq_len, TRACK.global_batch_positions, 42)
    assert run["mixed_precision_param"] == TRACK.mixed_precision_param
    fixed = TRACK.optimizers["fixed"][optimizer["name"]]
    assert {key: optimizer[key] for key in fixed if key != "muon"} == {key: fixed[key] for key in fixed if key != "muon"}
    assert (optimizer["base_lr"],) == TRACK.lr_grid[optimizer["name"]]  # the one calibrated LR
    assert optimizer.get("muon") == (fixed["muon"] | {"lr": optimizer["base_lr"]} if "muon" in fixed else None)
    assert spec["training"]["num_visits"] == spec["inference"]["num_visits"] == model["num_visits"]
    assert (model["vocab_size"], model["num_kv_heads"], model["max_seq_len"]) == (151669, 1, TRACK.seq_len)


@pytest.mark.parametrize("name", sorted(name for name in SPECS if not name.startswith("m0-loop-")))
def test_every_spec_is_within_5_percent_of_m0_on_all_three_budgets(name):
    m0 = phase1("m0", name.rsplit("-", 1)[1])
    comparison = compare_recipes(m0, RecipeSpec.from_dict(SPECS[name]), seq_len=TRACK.seq_len,
                                 tolerance=0.05, **TRACK.budget["accounting"])
    assert set(TRACK.budget["metrics"]) <= set(comparison["matched"])


@pytest.mark.parametrize("method", sorted(DEFAULTS))
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_each_method_changes_only_geometry_and_track_settings(method, optimizer):
    spec, default = SPECS[f"{method}-{optimizer}"], DEFAULTS[method].to_dict()
    assert mechanism(spec["model"]) == mechanism(default["model"])
    # Relaxed's LoRA rank keeps the constructor's rank/width ratio (16/128); every other method has no LoRA.
    assert (spec["model"]["lora_attention_rank"] / spec["model"]["dim"]
            == default["model"]["lora_attention_rank"] / default["model"]["dim"])
    for key in ("training", "inference", "preserved"):  # objective, depth rule, gradient tail, aux coefficient, readout
        assert spec[key] == default[key]
    assert track_free(spec["optimizer"]) == track_free(default["optimizer"])  # expert group size and LR multiplier
    assert {key: value for key, value in spec["source"].items() if key not in SOURCE_GEOMETRY} == {
        key: value for key, value in default["source"].items() if key not in SOURCE_GEOMETRY}


def test_method_specific_parts_scale_as_their_constructors_declare():
    model = {name: ModelConfig.from_dict(SPECS[f"{name}-adamw"]["model"]) for name in DEFAULTS}
    foil = model["foil"]
    # Foil flattens only the core: twice the Base experts per bank; prelude and coda keep the Base count.
    assert foil.region_config("core").num_experts == 2 * foil.region_config("prefix").num_experts == 128
    assert (foil.prefix_cells, 2 * foil.num_cells, foil.suffix_cells) == (2, 8, 2)  # the default 1:4:1 ratio
    assert SPECS["expert-tying-adamw"]["optimizer"]["expert_lr_multiplier"] == 0.5  # 1/sqrt(g), g = 4
    assert model["sandwich"].top_k == 2  # the preserved top-2 banks


@pytest.mark.parametrize("visits", [3, 4])
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_smelt_variants_run_the_p2_core_more_often_at_18_effective_layers(visits, optimizer):
    smelt, spec = recipes.p2_smelt_spec().to_dict(), SPECS[f"smelt-v{visits}-{optimizer}"]
    model = ModelConfig.from_dict(spec["model"])
    assert (model.prefix_cells, model.num_cells * model.num_visits, model.suffix_cells, model.num_visits) == (3, 12, 3, visits)
    assert spec["model"]["core"] == {"hidden_dim": None, "num_experts": model.region_config("core").num_experts,
                                     "top_k": None, "always_on_hidden_dim": None, "residual_scale": 1 / visits}
    changed = {"num_cells", "num_visits", "core"}
    assert {key: value for key, value in spec["model"].items() if key not in changed} == {
        key: value for key, value in smelt["model"].items() if key not in changed}
    for key in ("training", "inference"):
        assert spec[key] == smelt[key] | {"num_visits": visits}
    assert track_free(spec["optimizer"]) == track_free(smelt["optimizer"])


def test_every_rewrite_still_matches_a_constructor_note():
    """A reworded constructor note would make its rewrite a silent no-op and bring a false note back."""
    notes = " ".join(" ".join(method_specs.METHODS[name][0](**method_specs.METHODS[name][1]).adaptations)
                     for name in method_specs.METHODS)
    missing = [text for text, _ in method_specs.REWRITES if text not in notes]
    assert missing == []


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_m0_loop_matches_m0_parameters_but_not_flops_or_kv(optimizer):
    comparison = compare_recipes(phase1("m0", optimizer), RecipeSpec.from_dict(SPECS[f"m0-loop-{optimizer}"]),
                                 seq_len=TRACK.seq_len, tolerance=0.005, **TRACK.budget["accounting"])
    assert set(comparison["matched"]) == {"unique_non_embedding_parameters"}  # within 0.5% (exactly equal)
    different = {metric: round(1 + entry["relative_difference"], 2)
                 for metric, entry in comparison["measured-different"].items()}
    assert different == {"training_flops_per_token": 1.31, "logical_kv_bytes": 1.5}  # the ratios its notes state


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_m0_loop_is_m0_with_m3s_loop_and_checkpointing(optimizer):
    loop_spec = RecipeSpec.from_dict(SPECS[f"m0-loop-{optimizer}"])
    spec, m0, m3 = (resolve(recipe).to_dict() for recipe in (loop_spec, phase1("m0", optimizer), phase1("m3", optimizer)))
    assert spec["model"] == m0["model"] | {key: m3["model"][key] for key in TOPOLOGY}
    # M3's other loop settings (stack order, shared attention, router and norms) are m0's too.
    loop = ("order", "visit_capacity", "attention_per_visit", "router_per_visit", "norm_per_visit")
    assert {key: m3["model"][key] for key in loop} == {key: m0["model"][key] for key in loop}
    assert (spec["training"], spec["inference"]) == (m3["training"], m3["inference"])
    assert spec["optimizer"] == m0["optimizer"]  # Track P2's calibrated LR
    assert spec["run"] == m0["run"] | {"panel": "methods", "activation_checkpointing": "full_cells"}


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_diff_recipes_reports_only_what_the_trainer_reads(optimizer):
    # phase1 specs still carry keys no code reads (REMOVED_RUN_KEYS, REMOVED_OPTIMIZER_KEYS); the method specs drop them.
    differences = diff_recipes(phase1("m0", optimizer), RecipeSpec.from_dict(SPECS[f"m0-loop-{optimizer}"]))
    assert sorted(differences) == sorted([*(f"model.{key}" for key in TOPOLOGY[:4]), "model.core.residual_scale",
                                          "training.num_visits", "inference.num_visits",
                                          "run.panel", "run.activation_checkpointing"])

@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_latentloop_residual_is_m3_with_asymmetric_core_residual_scales(optimizer):
    latent_spec, m3_spec = RecipeSpec.from_dict(SPECS[f"latentloop-residual-{optimizer}"]), phase1("m3", optimizer)
    spec, m3 = resolve(latent_spec).to_dict(), resolve(m3_spec).to_dict()
    # M3's geometry and owners; only the core residual scales differ: (0.25, 1) per visit against 0.5 on both.
    assert m3["model"]["core"]["residual_scale"] == 0.5
    assert spec["model"] == m3["model"] | {"core": None, "visit_residual_scales": [0.25, 1.0]}
    assert (spec["training"], spec["inference"]) == (m3["training"], m3["inference"])
    assert spec["optimizer"] == m3["optimizer"]  # Track P2's calibrated LR
    assert spec["run"] == m3["run"] | {"panel": "methods"}  # no activation checkpointing, as M3
    comparison = compare_recipes(m3_spec, latent_spec, seq_len=TRACK.seq_len, tolerance=0.0,
                                 **TRACK.budget["accounting"])
    assert set(TRACK.budget["metrics"]) <= set(comparison["matched"])  # all three budgets equal M3's
