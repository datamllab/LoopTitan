"""New named recipes preserve real state/objective paths and portable configs."""

from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from moeloop import recipes
from moeloop.comparison import diff_recipes
from moeloop.model import RecursiveLM
from moeloop.recipe_spec import RecipeSpec


GEOMETRY = dict(vocab_size=31, dim=16, hidden_dim=24, num_heads=4,
                num_kv_heads=2, num_experts=4, top_k=2, max_seq_len=32)


def test_new_method_specs_roundtrip_without_importing_torchtitan():
    program = """
import importlib.abc, json, sys
class NoTitan(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('torchtitan'):
            raise AssertionError('ordinary method recipe imported TorchTitan')
sys.meta_path.insert(0, NoTitan())
from moeloop import recipes
from moeloop.recipe_spec import RecipeSpec
for make in (recipes.huginn_spec, recipes.loom_spec, recipes.ouro_spec):
    spec = make()
    restored = RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.preserved and restored.adaptations
    assert restored.source['url'].startswith('https://arxiv.org/')
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert result.returncode == 0, result.stderr


def test_latent_recipe_samples_microbatch_depth_with_a_separate_eval_policy():
    spec = recipes.huginn_spec(grad_visits=2, **GEOMETRY)
    restored = RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    depths = [restored.training.depth_at(0, microbatch=m) for m in range(64)]
    assert min(depths) == 1 and max(depths) > restored.training.grad_visits
    assert len(set(depths)) > 2
    assert restored.model.initial_state == "random"
    assert restored.model.input_injection == "linear"
    assert restored.model.prefix_cells and restored.model.suffix_cells
    assert restored.inference["initial_state"] == "zero"
    assert restored.inference["num_visits"] == restored.model.num_visits == 4
    assert restored.inference["cache"] == "uncached"


def test_latent_recipe_eval_depth_does_not_limit_training_gradient_tail():
    spec = recipes.huginn_spec(eval_passes=1, grad_visits=2, **GEOMETRY)
    assert spec.model.num_visits == spec.inference["num_visits"] == 1
    assert spec.training.grad_visits == 2
    assert max(spec.training.depth_at(0, microbatch=m) for m in range(16)) > 2


def test_memory_recipe_hybrid_changes_attention_ownership_only():
    base = recipes.loom_spec(core_layers=2, passes=3, **GEOMETRY)
    hybrid = recipes.loom_spec(core_layers=2, passes=3, attention_per_visit=True, **GEOMETRY)
    assert set(diff_recipes(base, hybrid)) == {"model.attention_per_visit"}
    assert base.model.state_update == "ema" and base.model.router_per_visit
    assert not base.model.norm_per_visit
    assert base.training.grad_visits is None
    assert base.inference["cache"] == "uncached"
    shared, independent = RecursiveLM(base.model), RecursiveLM(hybrid.model)
    assert len(shared.core[0].attentions) == 1
    assert len(independent.core[0].attentions) == 3
    assert len(shared.core[0].routers) == len(independent.core[0].routers) == 3
    with pytest.raises(ValueError, match="router"):
        replace(base, model=replace(base.model, router_per_visit=False, visit_capacity=None))


def test_expected_loss_recipe_retains_shared_gate_and_terminal_evaluation():
    spec = recipes.ouro_spec(passes=3, entropy_coeff=.2, **GEOMETRY)
    assert spec.model.exit_distribution
    assert spec.training.objective == "expected_ce"
    assert spec.training.entropy_coeff == .2
    assert spec.training.grad_visits is None
    assert spec.inference["readout"] == "terminal"
    assert spec.inference["labels_loss"] == "terminal_ce_only"
    with pytest.raises(ValueError, match="expected CE"):
        replace(spec, training=replace(spec.training, objective="terminal_ce"))


@pytest.mark.parametrize("name", ["huginn_spec", "loom_spec", "ouro_spec"])
def test_new_recipe_reaches_the_common_native_adapter(name):
    pytest.importorskip("torchtitan")
    spec = getattr(recipes, name)(**GEOMETRY)
    native = recipes.trainer_config(spec)
    assert native.recipe == spec.training
    assert native.model_spec.model.core == spec.model
    assert native.recipe_metadata == spec.to_dict()


@pytest.mark.parametrize("name, kwargs", [
    ("huginn_spec", {"grad_visits": 0}),
    ("huginn_spec", {"depth_mean": 1}),
    ("loom_spec", {"memory_beta": 1}),
    ("ouro_spec", {"entropy_coeff": -.1}),
])
def test_new_recipe_rejects_invalid_training_dynamics(name, kwargs):
    with pytest.raises(ValueError):
        getattr(recipes, name)(**GEOMETRY, **kwargs)
