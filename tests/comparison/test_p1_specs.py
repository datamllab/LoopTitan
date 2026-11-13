"""0.1B size-study specs: written by scripts/p1_specs.py, on Track M and within its budget."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop import bench
from moeloop.comparison import compare_recipes
from moeloop.recipe_spec import RecipeSpec

import p1_specs

ROOT = Path(__file__).resolve().parents[2]

SPECS = p1_specs.specs()
TRACK = bench.load_tracks()["M"]
SCHEDULE = ("total_steps", "warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")


def test_files_are_what_the_script_writes():
    files = {path.stem: json.loads(path.read_text()) for path in (ROOT / "recipes/p1").glob("*.json")}
    assert files == SPECS
    assert sorted(SPECS) == ["m0-adamw", "m0-muon", "m3-adamw", "m3-muon"]


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_takes_schedule_and_optimizer_from_track_m(name):
    spec = SPECS[name]
    run, optimizer = spec["run"], spec["optimizer"]
    assert spec["name"] == f"p1-{name}"
    assert run["successful_updates"] == run["schedule_total_steps"] == TRACK.horizon_updates
    assert {key: run["schedule"][key] for key in SCHEDULE} == {key: TRACK.schedule[key] for key in SCHEDULE}
    assert (run["seq_len"], run["global_batch_positions"], run["seed"]) == (TRACK.seq_len, TRACK.global_batch_positions, 42)
    assert (run["checkpoint_interval"], run["keep_latest_k"]) == (238, 0)
    assert run["local_batch_size"] == 16
    # study #223 forks at these steps (D/N 3, 6, 12, 25 and the extension to D/N 100)
    assert all(step % run["checkpoint_interval"] == 0 for step in (238, 476, 952, 1904, 3332))
    fixed = TRACK.optimizers["fixed"][optimizer["name"]]
    assert {key: optimizer[key] for key in fixed if key != "muon"} == {key: fixed[key] for key in fixed if key != "muon"}
    assert optimizer["base_lr"] == TRACK.lr_grid[optimizer["name"]][1]
    assert optimizer.get("muon") == (fixed["muon"] | {"lr": optimizer["base_lr"]} if "muon" in fixed else None)
    assert optimizer["expert_lr_multiplier"] == 1.0


def test_m3_is_within_the_track_budget_of_m0():
    comparison = compare_recipes(RecipeSpec.from_dict(SPECS["m0-adamw"]), RecipeSpec.from_dict(SPECS["m3-adamw"]),
                                 seq_len=TRACK.seq_len, tolerance=TRACK.budget["tolerance"], **TRACK.budget["accounting"])
    assert sorted(comparison["matched"]) == sorted(TRACK.budget["metrics"])


@pytest.mark.parametrize("name", sorted(SPECS))
def test_spec_round_trips(name):
    assert RecipeSpec.from_dict(SPECS[name]).to_dict() == SPECS[name]
