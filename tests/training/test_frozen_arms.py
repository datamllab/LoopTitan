"""The P1/P2 constructors build the arms that recipes/*.json records as trained or frozen."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from moeloop import recipes
from moeloop.comparison import p1_attention_pairs, smelt_p1_pair, smelt_p2_pair
from moeloop.recipe_spec import RecipeSpec


ROOT = Path(__file__).resolve().parents[2]
ARMS = ["p1_moe", "p1_smelt", "p2_moe", "p2_smelt"]

# Run keys only the archived JSONs carry, excluded from the comparison. No
# constructor produces them; code reads them only from the archived JSON. They
# record the study around an arm, not the arm itself. recipes.resolve drops the
# removed single-valued keys (recipes.REMOVED_RUN_KEYS, REMOVED_OPTIMIZER_KEYS).
ARCHIVE_ONLY_RUN = {
    "frozen_data": "P1 data identities (manifest, document-pair statistic) recorded at the freeze",
    "frozen_runtime": "P1 runtime and HSDP topology recorded at the freeze",
    "formal_identity_status": "a status note of the freeze",
    "lr_trial_peak_lrs": "study policy: the predeclared peak-LR trials",
    "lr_trial_max_label_tokens": "study policy: the LR-trial label budget",
    "lr_trial_max_updates": "study policy: the LR-trial update budget",
    "diagnostic_updates": "study policy: updates where diagnostics were read",
    "final_validation_label_tokens": "study policy: the final held-out evaluation size",
    "final_test_label_tokens": "study policy: the final held-out evaluation size",
}


def archived(name):
    return RecipeSpec.from_dict(json.loads((ROOT / f"recipes/{name}.json").read_text()))


def resolved(frozen):
    """The archived arm as trainer_config reads it, without the study keys."""
    spec = recipes.resolve(frozen)
    return replace(spec, run={key: value for key, value in spec.run.items() if key not in ARCHIVE_ONLY_RUN})


@pytest.mark.parametrize("name", ARMS)
def test_constructor_builds_the_archived_arm(name):
    spec, frozen = getattr(recipes, f"{name}_spec")(), resolved(archived(name))
    assert spec.model == frozen.model
    assert spec.training == frozen.training
    assert spec.optimizer == frozen.optimizer
    assert spec.run == frozen.run
    # Name, source, preserved, adaptations, inference and evidence are equal too.
    assert spec == frozen


def test_excluded_keys_are_exactly_the_archive_only_keys():
    run, optimizer = set(), set()
    for name in ARMS:
        spec, frozen = getattr(recipes, f"{name}_spec")(), archived(name)
        run |= set(frozen.run) - set(spec.run)
        optimizer |= set(frozen.optimizer) - set(spec.optimizer)
    assert run == set(ARCHIVE_ONLY_RUN) | set(recipes.REMOVED_RUN_KEYS)
    assert optimizer == set(recipes.REMOVED_OPTIMIZER_KEYS)


@pytest.mark.parametrize("name", ARMS)
def test_each_predeclared_lr_trial_changes_only_the_peak_lr(name):
    frozen = resolved(archived(name))
    for lr in archived(name).run["lr_trial_peak_lrs"]:
        spec = getattr(recipes, f"{name}_spec")(base_lr=lr)
        assert spec.optimizer == frozen.optimizer | {"base_lr": lr}
        assert replace(spec, optimizer=frozen.optimizer) == getattr(recipes, f"{name}_spec")()


def test_smelt_arms_are_the_measured_document_pair_screen_choices():
    # Both screens use the frozen P1-prefix statistic; their dense-S^2
    # defaults choose other shapes (F424 for P1, E96/F352 for P2).
    pairs = p1_attention_pairs()
    assert pairs == 4_180_216.9229844734
    for screen, make in ((smelt_p1_pair, recipes.p1_smelt_spec), (smelt_p2_pair, recipes.p2_smelt_spec)):
        chosen = screen(attention_pairs_per_sequence=pairs)["comparison"]["arms"]["candidate"]
        assert RecipeSpec.from_dict(chosen) == make()
    assert recipes.p1_smelt_spec().model.hidden_dim == 440
