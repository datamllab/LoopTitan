"""0.5B size-study specs: written by scripts/p05_specs.py, budget-matched and on Track P2's rules."""
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop import bench
from moeloop.comparison import compare_recipes, p1_attention_pairs
from moeloop.costs import estimate_cost
from moeloop.recipe_spec import RecipeSpec

import p05_specs

ROOT = Path(__file__).resolve().parents[2]

SPECS = p05_specs.specs()
TRACK = bench.load_tracks()["P2"]
MANIFEST = json.loads((ROOT / TRACK.data["manifest"]).read_text())
TRAIN_LABELS = MANIFEST["provenance"]["selection"]["actual_label_tokens"]["train"]


def test_files_are_what_the_script_writes():
    files = {path.stem: json.loads(path.read_text()) for path in (ROOT / "recipes/p05").glob("*")}
    assert files == SPECS
    assert sorted(SPECS) == ["m0-adamw", "m0-muon", "m3-adamw", "m3-muon", "u-muon"]


def test_the_m0_and_m3_files_are_unchanged_by_u():
    # sha256 of the files merged in #234 and #237, before U was added (#255).
    frozen = {"m0-adamw": "09c4dfeedc4ae681f9c6a75863a4418c7dd032eb8135096d3f5a6ca5482692e3",
              "m0-muon": "a09e354816ea78c878a2f115dbcd7e0ef0b4dbc7c7c6c063bde4d73a686784ce",
              "m3-adamw": "8afb77ae88aaa5ce1f4b297d544f796de732c2e80d7cae95215439670b72fa66",
              "m3-muon": "4976195154e061cc5c64d388d20604e66a919cfa4fd3ce69b83ecdf2ea794016"}
    for name, digest in frozen.items():
        assert hashlib.sha256((ROOT / f"recipes/p05/{name}.json").read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_round_trips(name):
    assert RecipeSpec.from_dict(json.loads(json.dumps(SPECS[name]))).to_dict() == SPECS[name]


@pytest.mark.parametrize("name", ["m3-adamw", "m3-muon", "u-muon"])
@pytest.mark.parametrize("recompute", ["none", "full_cells"])  # Track P2 counts none; the runs use full_cells
@pytest.mark.parametrize("pairs", ["dense", "documents"])  # dense S^2, or P2's measured document pairs
def test_every_arm_matches_m0_on_the_three_budgets(name, recompute, pairs):
    m0 = RecipeSpec.from_dict(SPECS["m0-" + name.split("-")[1]])
    candidate = RecipeSpec.from_dict(SPECS[name])
    accounting = TRACK.budget["accounting"] | {"activation_checkpointing": recompute}
    if pairs == "documents":
        accounting["attention_pairs_per_sequence"] = p1_attention_pairs()
    comparison = compare_recipes(m0, candidate, seq_len=TRACK.seq_len, tolerance=TRACK.budget["tolerance"],
                                 **accounting)
    assert sorted(comparison["matched"]) == sorted(TRACK.budget["metrics"])


def test_m0_has_about_half_a_billion_active_parameters():
    # Attention, norms, router and 8 of 64 experts per layer, plus the final norm; no embedding or head.
    parameters = estimate_cost(RecipeSpec.from_dict(SPECS["m0-adamw"]), seq_len=4096)["parameters"]
    experts = parameters["by_component"]["routed_experts"]
    active = parameters["unique_non_embedding"] - experts + experts * 8 // 64
    assert active == 497_092_608


def test_m3_loops_the_middle_half_of_a_narrower_wider_pool():
    m0, m3 = SPECS["m0-adamw"]["model"], SPECS["m3-adamw"]["model"]
    assert (m0["dim"] // m0["num_heads"], m0["num_kv_heads"], m0["num_experts"], m0["top_k"]) == (128, 1, 64, 8)
    assert (m3["prefix_cells"], m3["num_cells"], m3["num_visits"], m3["suffix_cells"]) == (4, 8, 2, 4)
    assert m3["prefix_cells"] + 2 * m3["num_cells"] + m3["suffix_cells"] == 1.5 * m0["num_cells"]
    assert m3["core"]["residual_scale"] == 0.5
    assert m3["dim"] < m0["dim"] and m3["num_experts"] > m0["num_experts"] and m3["top_k"] == 8


def test_u_is_m3_unrolled_with_m0s_parameters():
    def without_sharing(spec):  # drop what U may change: sharing, expert count, name and notes
        spec = json.loads(json.dumps(spec))
        for key in ("name", "preserved", "adaptations"):
            del spec[key]
        del spec["source"]["arm"], spec["training"]["num_visits"], spec["inference"]["num_visits"]
        for key in ("prefix_cells", "num_cells", "suffix_cells", "num_visits", "core", "num_experts"):
            del spec["model"][key]
        return spec
    m3, u = SPECS["m3-muon"], SPECS["u-muon"]
    assert without_sharing(u) == without_sharing(m3)
    shape, model = m3["model"], u["model"]
    depth = shape["prefix_cells"] + shape["num_visits"] * shape["num_cells"] + shape["suffix_cells"]
    assert (model["prefix_cells"], model["num_cells"], model["suffix_cells"], model["num_visits"]) == (0, depth, 0, 1)
    assert depth == 24 and model["core"] is None and model["residual_scale"] == 1.0
    assert u["training"]["num_visits"] == u["inference"]["num_visits"] == 1
    assert u["adaptations"][:-1] == m3["adaptations"]  # U adds one note
    m0 = estimate_cost(RecipeSpec.from_dict(SPECS["m0-muon"]), seq_len=4096)["parameters"]["unique_non_embedding"]
    unique = estimate_cost(RecipeSpec.from_dict(u), seq_len=4096)["parameters"]["unique_non_embedding"]
    assert abs(unique / m0 - 1) < 0.005  # 60 experts per layer: the closest count to M0 (59: -1.2%, 61: +2.0%)


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_follows_track_p2_rules_over_the_whole_train_split(name):
    spec = SPECS[name]
    run, optimizer, schedule = spec["run"], spec["optimizer"], spec["run"]["schedule"]
    horizon = TRAIN_LABELS // TRACK.global_batch_positions
    assert horizon == 22888 and horizon * TRACK.global_batch_positions <= TRAIN_LABELS  # no label twice
    assert run["successful_updates"] == run["schedule_total_steps"] == schedule["total_steps"] == horizon
    assert schedule["warmup_steps"] == round(0.02 * horizon) == 458
    assert (schedule["name"], schedule["decay_ratio"], schedule["decay_type"], schedule["min_lr_factor"]) == (
        "warmup_stable_decay", 0.2, "linear", 0.0)
    assert (run["seq_len"], run["global_batch_positions"], run["seed"]) == (4096, 1048576, 42)
    assert (run["local_batch_size"], run["keep_latest_k"], run["activation_checkpointing"]) == (16, 0, "full_cells")
    fixed = TRACK.optimizers["fixed"][optimizer["name"]]
    assert {key: optimizer[key] for key in fixed if key != "muon"} == {key: fixed[key] for key in fixed if key != "muon"}
    assert (optimizer["base_lr"],) == TRACK.lr_grid[optimizer["name"]]  # the one calibrated LR
    assert optimizer.get("muon") == (fixed["muon"] | {"lr": optimizer["base_lr"]} if "muon" in fixed else None)
    assert optimizer["expert_lr_multiplier"] == 1.0
    assert spec["training"]["num_visits"] == spec["inference"]["num_visits"] == spec["model"]["num_visits"]


def test_the_forks_are_checkpoints():
    horizon, interval = 22888, SPECS["m0-adamw"]["run"]["checkpoint_interval"]
    forks = [1144, 2288, 4576, 9152]  # D/N 3, 6, 12, 24: 80% of about H/16, H/8, H/4, H/2
    for share, fork in zip((16, 8, 4, 2), forks):
        assert fork % interval == 0 and fork * 5 % 4 == 0  # each decays a quarter of its fork step
        assert abs(fork * 5 / 4 / (horizon / share) - 1) < 1e-3
