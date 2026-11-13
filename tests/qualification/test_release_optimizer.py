"""Release checks bind a run's native optimizer, AdamW or Muon, to its frozen recipe."""

import copy
import importlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from moeloop import recipes
from moeloop.recipe_spec import RecipeSpec

GEOMETRY = dict(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, top_k=2, max_seq_len=64)
MUON = {"lr": .02, "momentum": .95, "nesterov": True, "ns_steps": 5,
        "ns_coefficients": [3.4445, -4.7750, 2.0315], "eps": 1e-7, "weight_decay": .1,
        "adjust_lr_fn": "match_rms_adamw"}
EXECUTION = {"native_experts": True, "flex_attention": True, "activation_checkpoint": None,
             "compile": {"enable": False, "enable_async_tensor_parallel": False,
                         "components": ["model", "loss"], "backend": "inductor"}}
SIDE_ADAMW = ("core.embedding.weight", "core.lm_head.weight", "core.final_norm.weight")


def frozen_spec(name):
    # Core experts carry expert_lr_multiplier, so a Muon run has two DistMuon instances.
    spec = recipes.smelt_spec(physical_layers=4, num_experts=4, **GEOMETRY)
    if name == "Muon":
        spec = replace(spec, optimizer=spec.optimizer | {"name": "Muon", "muon": MUON, "expert_lr_multiplier": .5})
    return RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))


def muon_matrices(identity):
    return [fqn for kwargs in identity["optimizer"]["optimizer_factory_kwargs_by_name"].values()
            for fqn in kwargs["compute_sharding_by_fqn"]]


def save_state(path, state):
    """A final DCP in TorchTitan's flat FQN-keyed optimizer layout.

    ``state`` maps FQN to state names. The names are those the real optimizers
    store: tests/training/test_muon.py pins ``momentum_buffer`` on a real
    DistMuon, and torch AdamW keeps ``exp_avg``, ``exp_avg_sq`` and ``step``.
    """
    import torch.distributed.checkpoint as dcp
    flat = {f"state.{fqn}.{key}": torch.ones(2) for fqn, keys in state.items() for key in keys}
    flat |= {f"param_groups.{fqn}.lr": .02 for fqn in state}
    dcp.save({"core": {"embedding.weight": torch.ones(2)}, "optimizer": flat}, checkpoint_id=str(path), no_dist=True)
    return str(path)


def native_run(name, tmp_path):
    """A run as verify_training returns it: frozen spec, native identity, final checkpoint."""
    spec = frozen_spec(name)
    # LoopTrainer writes config.optimizer.to_dict() into continuation_identity.json.
    native = json.loads(json.dumps(recipes.trainer_config(spec).optimizer.to_dict()))
    identity = {"optimizer": native, "max_norm": 1., "execution": copy.deepcopy(EXECUTION)}
    state = dict.fromkeys(SIDE_ADAMW, ("exp_avg", "exp_avg_sq", "step"))
    if name == "Muon":
        state |= dict.fromkeys(muon_matrices(identity), ("momentum_buffer",))
    checkpoint = save_state(tmp_path / "step-8", state)
    return {"spec": spec, "identity": identity, "receipt": {"checkpoint": checkpoint}}, state


def verify(training):
    checker = importlib.import_module("check_release")
    controls = {"optimizer": training["spec"].optimizer, "activation_checkpointing": "none"}
    checker.verify_native_optimizer_execution(training, controls, "arm")


def test_adamw_native_groups_are_what_the_trainer_writes(tmp_path):
    # The AdamW branch writes the native groups out as data; they must equal the
    # real trainer output, so AdamW evidence keeps its exact previous result.
    training, _ = native_run("AdamW", tmp_path)
    verify(training)
    del training["receipt"]  # AdamW is checked from the identity alone, as before
    verify(training)
    training["identity"]["optimizer"]["param_groups"][1]["optimizer_kwargs"]["lr"] = 1e-3
    with pytest.raises(ValueError, match="native optimizer differs from frozen spec"):
        verify(training)


@pytest.mark.parametrize("name", ["SGD", None])
def test_other_optimizer_names_are_still_rejected(name, tmp_path):
    training, _ = native_run("AdamW", tmp_path)
    training["spec"] = replace(training["spec"], optimizer=training["spec"].optimizer | {"name": name})
    with pytest.raises(ValueError, match="unsupported frozen optimizer roles"):
        verify(training)


def test_muon_run_is_bound_to_its_frozen_settings_and_state(tmp_path):
    training, _ = native_run("Muon", tmp_path)
    groups = training["identity"]["optimizer"]["param_groups"]
    assert [group["optimizer_name"] for group in groups] == ["DistMuonCoreExperts", "DistMuon", "AdamW"]
    verify(training)


@pytest.mark.parametrize("override", ["muon_lr", "ns_steps", "core_expert_group", "layout", "adamw_side"])
def test_muon_identity_rejects_settings_the_frozen_spec_does_not_declare(override, tmp_path):
    training, _ = native_run("Muon", tmp_path)
    native = training["identity"]["optimizer"]
    groups = {group["optimizer_name"]: group for group in native["param_groups"]}
    if override == "muon_lr":  # a CLI override with a stale recipe
        groups["DistMuon"]["optimizer_kwargs"]["lr"] = .01
    elif override == "ns_steps":
        groups["DistMuonCoreExperts"]["optimizer_kwargs"]["ns_steps"] = 6
    elif override == "core_expert_group":  # core experts lose their own LR
        native["param_groups"] = [group for group in native["param_groups"] if group["optimizer_name"] != "DistMuonCoreExperts"]
    elif override == "layout":
        native["optimizer_factory_kwargs_by_name"]["DistMuon"]["compute_sharding_by_fqn"].popitem()
    else:
        groups["AdamW"]["optimizer_kwargs"]["weight_decay"] = .1
    with pytest.raises(ValueError, match="native optimizer differs from frozen spec"):
        verify(training)


def test_an_adamw_identity_cannot_pass_as_a_muon_run_or_back(tmp_path):
    muon, _ = native_run("Muon", tmp_path / "muon")
    adamw, _ = native_run("AdamW", tmp_path / "adamw")
    muon["identity"]["optimizer"], adamw["identity"]["optimizer"] = adamw["identity"]["optimizer"], muon["identity"]["optimizer"]
    for training in (muon, adamw):
        with pytest.raises(ValueError, match="native optimizer differs from frozen spec"):
            verify(training)


@pytest.mark.parametrize("damage", ["adam_moments_on_matrix", "missing_momentum", "momentum_on_side", "no_side_adamw"])
def test_muon_checkpoint_must_hold_momentum_for_matrices_and_adamw_moments_elsewhere(damage, tmp_path):
    training, state = native_run("Muon", tmp_path)
    matrix = muon_matrices(training["identity"])[0]
    if damage == "adam_moments_on_matrix":
        state[matrix] = ("exp_avg", "exp_avg_sq", "step")
    elif damage == "missing_momentum":
        del state[matrix]
    elif damage == "momentum_on_side":
        state[SIDE_ADAMW[0]] = ("momentum_buffer",)
    else:
        state = {fqn: keys for fqn, keys in state.items() if fqn not in SIDE_ADAMW}
    training["receipt"]["checkpoint"] = save_state(tmp_path / "damaged", state)
    with pytest.raises(ValueError, match="optimizer state"):
        verify(training)
