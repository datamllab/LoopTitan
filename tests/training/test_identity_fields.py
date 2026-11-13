"""The spec-derived continuation identity is defined once (``training.spec_identity``).

LoopTrainer.identity is built for real on one CPU gloo rank: the specs go through
``trainer_config`` (and ``experiment`` for launched runs and forks) and the real
LoopTrainer.__init__. Only the pinned Trainer.__init__ is replaced, by the parts
LoopTrainer.__init__ reads before it builds the identity.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from torchtitan.trainer import Trainer

from moeloop import recipes, testing
from moeloop.experiment import experiment
from moeloop.prepare import prepare_dataset
from moeloop.recipe_spec import RecipeSpec
from moeloop.titan import LoopTrainer
from moeloop.training import continuation_fingerprint

REPO = Path(__file__).resolve().parents[2]
# continuation_fingerprint of the spec-derived fields of LoopTrainer.identity for
# recipes/p1_smelt.json launched through experiment(), computed on origin/main e51ba7e
# before spec_identity existed. A change here changes the identity of existing runs.
# model is ModelConfig.to_dict(), recipe asdict(TrainingRecipe) and recipe_metadata the
# whole RecipeSpec.to_dict(), so a new field in any of the three, even one with a
# default, changes every run's identity and moves this pin, as does an edit of
# recipes/p1_smelt.json itself (#46's LoRA ranks moved it from c5b78a96... on f63af66).
P1_SMELT_FINGERPRINT = "9193bed7f2f9e7694b2f7d72bd2fd5ad9f983467a0d69ebd96ebbaae268ca4d7"
FORK = {"checkpoint": "/parent/checkpoint/step-4", "metadata_sha256": "0" * 64, "step": 4}

CONSTRUCTED = {}
for _name, _build in vars(recipes).items():
    if _name.endswith(("_spec", "_specs")) and not _name.startswith("_"):
        _built = _build()
        CONSTRUCTED.update(_built if isinstance(_built, dict) else {_name: _built})
# relaxed_spec() is constructed above; its dense path is another model (LoRA on FFNs too).
CONSTRUCTED["relaxed_spec(dense=True)"] = recipes.relaxed_spec(dense=True)
# Every constructor states expert_lr_multiplier; a third-party spec may leave it to the default.
_tiny = recipes.tiny_spec()
CONSTRUCTED["no_expert_lr_multiplier"] = replace(_tiny, optimizer={
    key: value for key, value in _tiny.optimizer.items() if key != "expert_lr_multiplier"})


def native_init(self, config):
    """What LoopTrainer.__init__ reads from the pinned Trainer.__init__; the model and the
    validator only take the EMA when ema_decay is set, so they stay empty."""
    self.config, self.model_config = config, config.model_spec.model
    self.parallel_dims = SimpleNamespace(dp_enabled=False)
    self.gradient_accumulation_steps = 1
    self.loss_fn = config.loss.build(compile_config=config.compile)
    self.model_parts, self.validator = [torch.nn.Module()], SimpleNamespace()


@pytest.fixture(scope="module")
def titan(tmp_path_factory):
    """``identity(spec, launch=False)``: LoopTrainer.identity of ``trainer_config(spec)``,
    or with ``launch`` of ``experiment()`` on the spec as recipe_experiments launches it."""
    import torch.distributed as dist
    root = tmp_path_factory.mktemp("identity")
    lengths = [str(n) for n in (9, 4, 13, 6, 17, 5, 8, 11, 3, 12)]
    manifest = prepare_dataset(root / "data", {"a": {"train": lengths, "validation": lengths}},
        encode=lambda text: [3 + (5 * i + int(text)) % 16 for i in range(int(text))],
        tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 19, "bos_token_id": 1, "eos_token_id": 2})
    count = iter(range(10**6))

    def identity(spec, *, launch=False):
        output = root / f"run-{next(count)}"
        with pytest.MonkeyPatch.context() as patch:
            if launch:
                output.mkdir()
                (output / "spec.json").write_text(json.dumps(spec.to_dict()))
                patch.setenv("MOELOOP_EXPERIMENT_SPEC", str(output / "spec.json"))
                patch.setenv("MOELOOP_EXPERIMENT_MANIFEST", str(manifest.path))
                patch.setenv("MOELOOP_EXPERIMENT_OUTPUT", str(output))
                if spec.run.get("fork"):
                    patch.setenv("MOELOOP_EXPERIMENT_RESUME", spec.run["fork"]["checkpoint"])
                config = experiment()
            else:
                config = recipes.trainer_config(spec)
                config.dataloader = replace(config.dataloader, manifest_path=str(manifest.path))
                config.dump_folder = str(output)
            return LoopTrainer(config).identity

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Trainer, "__init__", native_init)
        dist.init_process_group("gloo", init_method=f"file://{root}/init", rank=0, world_size=1)
        try:
            yield identity
        finally:
            dist.destroy_process_group()


def small_run():
    """A small SMELT spec with an explicit launch, as recipe_experiments runs it."""
    spec = recipes.smelt_spec(physical_layers=4, vocab_size=19, dim=16, hidden_dim=24, num_heads=2,
                              num_kv_heads=1, num_experts=2, top_k=1, max_seq_len=8)
    return replace(spec, run=spec.run | {"seq_len": 8, "global_batch_positions": 16, "local_batch_size": 2,
                                         "successful_updates": 8, "schedule_total_steps": 8})


def spec_fields(identity, spec):
    """The fields of ``identity`` that testing derives from ``spec``."""
    return {key: identity[key] for key in testing._spec_identity(spec)}


@pytest.mark.parametrize("name", CONSTRUCTED)
def test_titan_and_testing_agree_on_spec_fields_for_every_recipe_constructor(titan, name):
    spec = CONSTRUCTED[name]
    expected = testing._spec_identity(spec)
    assert spec_fields(titan(spec), spec) == expected
    assert expected["fork"] == {}


def test_titan_and_testing_agree_on_a_fork(titan):
    spec = replace(small_run(), run=small_run().run | {"fork": FORK})
    identity = titan(spec, launch=True)
    assert spec_fields(identity, spec) == testing._spec_identity(spec)
    assert identity["fork"] == FORK


def test_fingerprint_of_an_existing_run_spec_is_unchanged(titan):
    spec = RecipeSpec.from_dict(json.loads((REPO / "recipes/p1_smelt.json").read_text()))
    assert continuation_fingerprint(spec_fields(titan(spec, launch=True), spec)) == P1_SMELT_FINGERPRINT
    assert continuation_fingerprint(testing._spec_identity(spec)) == P1_SMELT_FINGERPRINT


def test_ema_decay_enters_the_identity_only_when_set(titan):
    """Absent, the identity is the one runs had before EMA existed; set, it is its own field."""
    base = small_run()
    specs = [base] + [replace(base, run=base.run | {"ema_decay": decay}) for decay in (.9, .5)]
    identities = [spec_fields(titan(spec, launch=True), spec) for spec in specs]
    assert "ema_decay" not in identities[0] and [identity["ema_decay"] for identity in identities[1:]] == [.9, .5]
    for spec, identity in zip(specs, identities):
        assert identity == testing._spec_identity(spec)
    assert len({continuation_fingerprint(identity) for identity in identities}) == 3


@pytest.mark.parametrize("change,fields", [
    (lambda spec: replace(spec, model=replace(spec.model, dim=32)), {"model"}),
    (lambda spec: replace(spec, training=replace(spec.training, balance_coeff=.02)), {"recipe"}),
    (lambda spec: replace(spec, optimizer=spec.optimizer | {"expert_lr_multiplier": .5}), {"expert_lr_multiplier"}),
    (lambda spec: replace(spec, run=spec.run | {"fork": FORK}), {"fork"}),
    (lambda spec: replace(spec, run=spec.run | {"seed": 43}), set()),
], ids=["model", "recipe", "expert_lr_multiplier", "fork", "run_seed"])
def test_a_changed_spec_field_changes_the_fingerprint(titan, change, fields):
    """Each change moves its own field and recipe_metadata (the whole spec), nothing else."""
    base = small_run()
    changed = change(base)
    before, after = (spec_fields(titan(spec, launch=True), spec) for spec in (base, changed))
    assert {key for key in before if before[key] != after[key]} == fields | {"recipe_metadata"}
    assert continuation_fingerprint(after) != continuation_fingerprint(before)
