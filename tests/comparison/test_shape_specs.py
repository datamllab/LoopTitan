"""Shape-decomposition specs: written by scripts/shape_specs.py, differing from Phase 1 only as the issue lists, within the P2 budget."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop import bench
from moeloop.comparison import compare_recipes
from moeloop.config import ModelConfig
from moeloop.recipe_spec import RecipeSpec

import shape_specs

ROOT = Path(__file__).resolve().parents[2]

SPECS = shape_specs.specs()
TRACK = bench.load_tracks()["P2"]
NO_CORE = {"hidden_dim": None, "num_experts": None, "top_k": None, "always_on_hidden_dim": None}
# arm: (source arm, the model fields it changes)
CHANGES = {
    "g": ("m0", dict(hidden_dim=384, num_experts=72, top_k=9)),
    "u": ("m4", dict(prefix_cells=0, num_cells=18, suffix_cells=0, num_experts=58, core=None)),
    "r": ("m4", dict(core=NO_CORE | {"num_experts": 44, "residual_scale": 1.0})),
    "m0r": ("m0", dict(prefix_cells=3, num_cells=6, suffix_cells=3, core=NO_CORE | {"residual_scale": 0.5})),
    "m0lr": ("m0", dict(prefix_cells=3, num_cells=6, suffix_cells=3,
                        core=NO_CORE | {"residual_scale": 1.0, "output_lr_scale": 0.5})),
    "m0i": ("m0", dict(prefix_cells=3, num_cells=6, suffix_cells=3,
                       core=NO_CORE | {"residual_scale": 1.0, "output_init_scale": 0.5})),
    "d24": ("m0", dict(dim=896, num_heads=14, hidden_dim=320, num_experts=62, num_cells=24)),
    "d36": ("m0", dict(dim=704, num_heads=16, hidden_dim=256, num_experts=66, num_cells=36)),
}
FULL_CELLS = {"d24", "d36"}  # the depth arms also checkpoint activations
SPLIT = {"m0lr": "output_lr_scale", "m0i": "output_init_scale"}  # M0R's residual x 0.5, split; AdamW only
NAMES = [f"{arm}-{optimizer}" for arm in CHANGES for optimizer in (("adamw",) if arm in SPLIT else ("adamw", "muon"))]
METRICS = ("unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes")


def phase1(arm, optimizer):
    return json.loads((ROOT / f"recipes/phase1/{arm}-{optimizer}.json").read_text())


def compare(candidate, reference):
    return compare_recipes(RecipeSpec.from_dict(reference), RecipeSpec.from_dict(candidate), seq_len=TRACK.seq_len,
                           tolerance=TRACK.budget["tolerance"], **TRACK.budget["accounting"])


def test_files_are_what_the_script_writes():
    files = {path.stem: json.loads(path.read_text()) for path in (ROOT / "recipes/shape").iterdir()}
    assert files == SPECS
    assert sorted(SPECS) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_each_spec_differs_from_its_source_only_in_the_listed_fields(name):
    arm, optimizer = name.split("-")
    source_arm, changes = CHANGES[arm]
    source, spec = phase1(source_arm, optimizer), SPECS[name]
    assert spec["name"] == f"shape-{name}"
    assert len(spec["preserved"]) == 1
    expected = source | {"name": spec["name"], "preserved": spec["preserved"], "model": source["model"] | changes,
                         "source": {"kind": "shape_arm", "version": 1, "arm": arm,
                                    "base_recipe": f"recipes/phase1/{source_arm}-{optimizer}.json"}}
    if arm in FULL_CELLS:
        expected["run"] = source["run"] | {"activation_checkpointing": "full_cells"}
    assert spec == RecipeSpec.from_dict(expected).to_dict()
    assert spec["optimizer"]["base_lr"] == source["optimizer"]["base_lr"]


@pytest.mark.parametrize("name", NAMES)
def test_every_arm_is_within_the_p2_budget_of_m0(name):
    optimizer = name.split("-")[1]
    comparison = compare(SPECS[name], phase1("m0", optimizer))
    assert sorted(comparison["matched"]) == sorted(METRICS)
    if name.split("-")[0] in ("m0r", *SPLIT):
        assert all(comparison["matched"][metric]["relative_difference"] == 0 for metric in METRICS)


@pytest.mark.parametrize("arm", SPLIT)
def test_the_split_arms_differ_from_m0r_only_in_the_residual_scale_and_their_scale(arm):
    m0r, spec = SPECS["m0r-adamw"], SPECS[f"{arm}-adamw"]
    core = m0r["model"]["core"] | {"residual_scale": 1.0, SPLIT[arm]: 0.5}
    assert spec == m0r | {"name": spec["name"], "preserved": spec["preserved"], "model": m0r["model"] | {"core": core},
                          "source": m0r["source"] | {"arm": arm}}
    assert spec["run"]["activation_checkpointing"] == "none"


@pytest.mark.parametrize("name", NAMES)
def test_the_model_round_trips(name):
    model = ModelConfig.from_dict(SPECS[name]["model"])
    assert model.to_dict() == SPECS[name]["model"]
    assert RecipeSpec.from_dict(SPECS[name]).to_dict() == SPECS[name]
