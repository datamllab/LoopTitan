"""Freeze P2 controls and the independent attention/router experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from moeloop import recipes
from moeloop.comparison import diff_recipes
from moeloop.recipe_spec import RecipeSpec


ROOT = Path(__file__).resolve().parents[2]


def test_p2_preserves_p1_schedule_but_has_fresh_comparison_identity():
    # Catches accidentally changing the long schedule to a pilot horizon or
    # inheriting a P1 name/source while constructing the new scale.
    p1 = recipes.p1_moe_spec()
    for make, name, dim, hidden, experts, heads in (
        (recipes.p2_moe_spec, "p2_moe", 1280, 432, 64, 10),
        (recipes.p2_smelt_spec, "p2_smelt", 1056, 384, 88, 12),
    ):
        spec = make()
        assert spec.name == name
        assert (spec.model.dim, spec.model.hidden_dim, spec.model.num_experts,
                spec.model.num_heads, spec.model.num_kv_heads, spec.model.top_k) == (
                    dim, hidden, experts, heads, 1, 8)
        assert spec.model.vocab_size == 151669
        assert spec.model.init_std == .02
        assert spec.training == (recipes.p1_smelt_spec().training if "smelt" in name else p1.training)
        assert spec.optimizer == p1.optimizer
        assert spec.run["schedule"] == p1.run["schedule"]
        assert spec.run["successful_updates"] == spec.run["schedule_total_steps"] == 19537
        assert spec.run["global_batch_positions"] == 256 * 4096
        assert spec.run["local_batch_size"] == 16
        for key in ("seed", "training_seeds", "data_infinite", "validation_interval",
                    "validation_label_tokens", "mixed_precision_param"):
            assert spec.run[key] == p1.run[key]
        assert spec.source["anchor"] == spec.run["anchor"] == spec.run["panel"] == "P2"
        assert spec.source["paper_reference"]["baseline_active_tier"] == "200M"
        assert spec.source["paper_reference"]["non_embedding_unique_tier"] == "1.32B"
        assert spec.run["manifest_identity"] is None and not spec.evidence
        assert RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    smelt = recipes.p2_smelt_spec().model
    assert (smelt.prefix_cells, smelt.num_cells, smelt.num_visits, smelt.suffix_cells) == (3, 6, 2, 3)
    assert [smelt.region_config(r).residual_scale for r in ("prefix", "core", "suffix")] == [1., .5, 1.]


def test_c_grid_changes_only_attention_and_router_owners():
    # C must retain the same objective/shape/schedule instead of silently
    # introducing LOOM state or iso-parameter adjustments.
    panel = recipes.sharing_panel_specs()
    assert list(panel) == ["a0_r0", "a0_r1", "a1_r0", "a1_r1"]
    base = panel["a0_r0"]
    for name, spec in panel.items():
        attention, router = name[1] == "1", name[-1] == "1"
        assert spec.name == name and spec.run["panel"] == "C"
        assert spec.source["kind"] == "attention_router_ownership_grid"
        assert (spec.model.dim, spec.model.hidden_dim, spec.model.num_experts,
                spec.model.top_k, spec.model.num_heads, spec.model.num_kv_heads) == (512, 1024, 8, 2, 8, 2)
        assert (spec.model.num_cells, spec.model.num_visits, spec.model.prefix_cells,
                spec.model.suffix_cells, spec.model.order) == (2, 4, 0, 0, "stack")
        assert spec.model.attention_per_visit is attention
        assert spec.model.router_per_visit is router
        assert not spec.model.norm_per_visit
        assert spec.model.state_update == "residual" and spec.model.input_injection == "none"
        assert spec.training.objective == "terminal_ce" and spec.training.grad_visits is None
        assert spec.run["seed"] == 42
        assert spec.run["seq_len"] == 2048 and spec.run["global_batch_positions"] == 32768
        assert spec.run["successful_updates"] == spec.run["schedule_total_steps"] == 2048
        assert spec.run["schedule"]["total_steps"] == 2048
        assert spec.run["schedule"]["warmup_steps"] == 64
        assert spec.run["schedule"]["decay_type"] == "cosine"
        assert spec.run["schedule"]["min_lr_factor"] == .1
        assert spec.optimizer["base_lr"] == 3e-4
        assert set(diff_recipes(base, spec)) <= {"model.attention_per_visit", "model.router_per_visit", "model.visit_capacity"}
    shorter = recipes.sharing_panel_specs(horizon=1024)
    assert all(s.run["successful_updates"] == s.run["schedule"]["total_steps"] == 1024 for s in shorter.values())
    with pytest.raises(ValueError, match="horizon"):
        recipes.sharing_panel_specs(horizon=64)


def test_v3_public_json_is_portable_and_matches_its_factory():
    # test_frozen_arms.py compares the P2 JSONs, which also carry study policy.
    expected = {f"c_{name}": spec for name, spec in recipes.sharing_panel_specs().items()}
    for name, spec in expected.items():
        actual = RecipeSpec.from_dict(json.loads((ROOT / f"recipes/{name}.json").read_text()))
        assert recipes.resolve(actual) == spec
    program = """
import importlib.abc, json, sys
class NoTitan(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('torchtitan'):
            raise AssertionError('portable recipe imported TorchTitan')
sys.meta_path.insert(0, NoTitan())
from moeloop import recipes
from moeloop.recipe_spec import RecipeSpec
for spec in [recipes.p2_moe_spec(), recipes.p2_smelt_spec(), *recipes.sharing_panel_specs().values()]:
    assert RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert result.returncode == 0, result.stderr
