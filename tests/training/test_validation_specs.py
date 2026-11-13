import json
import subprocess
import sys
from pathlib import Path

import pytest

import validation_specs
from moeloop.bench import load_tracks
from moeloop.recipe_spec import RecipeSpec

ROOT = Path(__file__).resolve().parents[2]
CHANGED_RUN_FIELDS = {"schedule", "local_batch_size"}


def test_specs_keep_the_track_s_reference_apart_from_schedule_local_batch_and_optimizer():
    reference = json.loads((ROOT / "recipes/c_a0_r0.json").read_text())
    specs = validation_specs.specs()
    assert set(specs) == {"b5-adamw", "b5-muon"}
    for name, spec in specs.items():
        assert set(spec) == set(reference)
        assert {key for key in spec if spec[key] != reference[key]} == {"name", "run", "optimizer"}
        assert {key for key in spec["run"] | reference["run"]
                if spec["run"].get(key) != reference["run"].get(key)} == CHANGED_RUN_FIELDS
        assert RecipeSpec.from_dict(spec).name == name
    fixed = load_tracks()["S"].optimizers["fixed"]
    assert specs["b5-adamw"]["optimizer"] == reference["optimizer"] | fixed["AdamW"]


def test_schedule_follows_track_s_and_update_150_is_at_peak():
    track, run = load_tracks()["S"], validation_specs.specs()["b5-adamw"]["run"]
    schedule = run["schedule"]
    shape = ("name", "coordinate", "decay_type", "decay_ratio", "min_lr_factor")
    assert {key: schedule[key] for key in shape} == {key: track.schedule[key] for key in shape}
    assert schedule["total_steps"] == run["successful_updates"] == run["schedule_total_steps"]
    assert schedule["warmup_steps"] == round(track.schedule["warmup_fraction"] * schedule["total_steps"]) == 41
    decay_start = schedule["total_steps"] - round(schedule["decay_ratio"] * schedule["total_steps"])
    assert schedule["warmup_steps"] < 150 and 150 + 50 < decay_start  # #49 forks at 150 with a 50-update tail


def test_local_batch_divides_the_global_batch_on_4_and_8_gpus():
    run = validation_specs.specs()["b5-adamw"]["run"]
    sequences = run["global_batch_positions"] // run["seq_len"]
    assert all(sequences % (gpus * run["local_batch_size"]) == 0 for gpus in (4, 8))


def test_muon_uses_track_s_fixed_settings_at_the_middle_of_its_grid():
    track = load_tracks()["S"]
    fixed, optimizer = track.optimizers["fixed"]["Muon"], validation_specs.specs()["b5-muon"]["optimizer"]
    assert optimizer["name"] == "Muon" and optimizer["base_lr"] == track.lr_grid["Muon"][1] == 6e-4
    assert optimizer["muon"] == fixed["muon"] | {"lr": 6e-4}
    assert all(optimizer[key] == value for key, value in fixed.items() if key != "muon")


def test_specs_build_trainer_configs(tmp_path):
    pytest.importorskip("torchtitan")
    from moeloop import recipes
    subprocess.run([sys.executable, str(ROOT / "scripts/validation_specs.py"), "--output", str(tmp_path)],
                   check=True, capture_output=True)
    for name in ("b5-adamw", "b5-muon"):
        spec = RecipeSpec.from_dict(json.loads((tmp_path / f"{name}.json").read_text()))
        config = recipes.trainer_config(spec)
        assert config.lr_scheduler.warmup_steps == spec.run["schedule"]["warmup_steps"]
