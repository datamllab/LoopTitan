"""Release checks compare archived recipe records by meaning across schema versions."""

import copy
import importlib
import json

import pytest

pytest.importorskip("torch")

# Archived records are built from today's to_dict() by dropping exactly the
# keys an older checkout did not write and setting its model schema version.
# The key sets were checked against the records in-repo commits write. P2 and
# the C grid (17d51e4) use model schema 3, before #29 added the decay-injection
# fields. P1 uses model schema 2 (docs/validation.md); its own training source
# 1bea4d6 is outside this repo's history, so the schema-2 set was checked
# against 98bed07, the in-repo commit that writes schema 2, before #7 also
# added the recurrent-state and objective fields. All other values are equal.
SCHEMA_3 = (3, {"injection_dt", "injection_decay", "injection_eps"}, set())
SCHEMA_2 = (2, SCHEMA_3[1] | {"state_update", "latent_init_std", "latent_truncation",
                              "memory_scale", "memory_beta", "exit_distribution"},
            {"objective", "entropy_coeff", "depth_mean", "depth_log_std"})


def archived(recipe, era):
    """The record an older checkout wrote for the same recipe."""
    version, model_fields, training_fields = era
    record = copy.deepcopy(recipe)
    record["model"] = {key: value for key, value in record["model"].items()
                       if key not in model_fields} | {"schema_version": version}
    record["training"] = {key: value for key, value in record["training"].items() if key not in training_fields}
    return record


def current(spec):
    return json.loads(json.dumps(spec.to_dict()))


@pytest.mark.parametrize("era", [SCHEMA_3, SCHEMA_2], ids=["schema3", "schema2"])
@pytest.mark.parametrize("factory", ["p1_moe_spec", "p1_smelt_spec", "p2_moe_spec", "p2_smelt_spec"])
def test_archived_record_is_the_same_recipe(era, factory):
    from moeloop import recipes
    from moeloop.model import ModelConfig
    checker = importlib.import_module("check_release")
    spec = getattr(recipes, factory)()
    record = archived(current(spec), era)
    assert record != current(spec)
    assert checker.same_recipe(record, current(spec))
    # Model records (native identity, HF provenance) and the native training
    # record parse the same way.
    assert ModelConfig.from_dict(record["model"]) == spec.model
    assert checker.same_recipe(current(spec) | {"training": record["training"]}, current(spec))


@pytest.mark.parametrize("change", ["model", "new_model_field", "training", "optimizer", "run"])
def test_a_changed_recorded_value_is_a_different_recipe(change):
    from moeloop.recipes import p2_smelt_spec
    checker = importlib.import_module("check_release")
    spec = current(p2_smelt_spec())
    record = archived(spec, SCHEMA_3)
    if change == "model":
        record["model"]["num_experts"] += 8
    elif change == "new_model_field":
        record = copy.deepcopy(spec)  # Current schema, a non-default value of a field schema 3 lacks.
        record["model"]["injection_dt"] = .7
    elif change == "training":
        record["training"]["balance_coeff"] = .02
    elif change == "optimizer":
        record["optimizer"]["base_lr"] = 1e-4
    else:
        record["run"]["seed"] = 7
    assert not checker.same_recipe(record, spec)


@pytest.mark.parametrize("anchor", ["P1", "P2"])
def test_archived_anchor_arms_pass_the_anchor_comparison(anchor):
    from moeloop.recipe_spec import RecipeSpec
    checker = importlib.import_module("check_release")
    if anchor == "P1":
        trainings, comparison = importlib.import_module("test_recipe_release")._p1_native_controls()
        era = SCHEMA_2
    else:
        trainings, comparison = importlib.import_module("test_v3_release")._p2_native_controls()
        era = SCHEMA_3
    for role, training in trainings.items():
        arm = comparison["arms"][role] = archived(comparison["arms"][role], era)
        # verify_training parses each run's own archived spec.json the same way.
        training["spec"] = RecipeSpec.from_dict(copy.deepcopy(arm))
    checker.verify_anchor_comparison(trainings, comparison, anchor_name=anchor)
    comparison["arms"]["candidate"]["optimizer"]["base_lr"] = 1e-3
    with pytest.raises(ValueError, match="candidate: frozen comparison recipe differs"):
        checker.verify_anchor_comparison(trainings, comparison, anchor_name=anchor)


def test_archived_p2_pair_passes_the_recomputed_screen():
    from moeloop.comparison import smelt_p2_pair
    from moeloop.recipe_spec import RecipeSpec
    checker = importlib.import_module("check_release")
    pair = smelt_p2_pair(attention_pairs_per_sequence=4_180_216.9229844734)
    arms = pair["comparison"]["arms"]
    for role in arms:
        arms[role] = archived(arms[role], SCHEMA_3)
    # verify_training parses each run's archived spec.json the same way.
    trainings = {role: {"spec": RecipeSpec.from_dict(arm)} for role, arm in arms.items()}
    checker.verify_p2_screen(pair, trainings)
    arms["candidate"]["model"]["hidden_dim"] += 8
    with pytest.raises(ValueError, match="candidate is not the frozen actual recipe"):
        checker.verify_p2_screen(pair, trainings)


def test_archived_p1_comparison_passes_and_changed_arm_fails():
    from moeloop.comparison import compare_recipes
    from moeloop.recipes import p1_moe_spec, p1_smelt_spec
    checker = importlib.import_module("check_release")
    actual = json.loads(json.dumps(compare_recipes(p1_moe_spec(), p1_smelt_spec(), seq_len=4096)))
    recorded = copy.deepcopy(actual)
    for role in recorded["arms"]:
        recorded["arms"][role] = archived(recorded["arms"][role], SCHEMA_2)
    original = copy.deepcopy(recorded)
    checker.verify_legacy_comparison(actual, recorded)
    assert recorded == original
    recorded["arms"]["reference"]["optimizer"]["base_lr"] = 1e-3
    with pytest.raises(ValueError, match="budget-pair contract differs"):
        checker.verify_legacy_comparison(actual, recorded)
