"""Untie and continue: a fully shared stack-order run's step-S checkpoint becomes the start of
a run whose core visits own their own cells (core cell c of visit v -> cell v*C + c).

The conversion keeps the function (bitwise logits) and the training state (optimizer
moments, data cursor, RNG, schedule), so any later difference comes from training without
sharing. The launcher path runs like tests/training/test_fork.py: the real launcher, config
factory, LoopTrainer and native CheckpointManager on one CPU gloo rank; the converter runs
as its own process. The Muon test runs first: it starts its own gloo rank, and the module's
``lab`` fixture keeps one until the module ends.
"""

from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from torch.distributed.tensor import DTensor

import recipe_experiments as runner
from recipe_qualification import digest
from test_fork import RUN, assert_same_state, cpu_native_init, native_train
from test_muon import fsdp_model, muon_spec, one_cpu_rank  # noqa: F401  (one_cpu_rank is the fixture)
from torchtitan.trainer import Trainer
import untie_specs

from moeloop import recipes
from moeloop.comparison import compare_recipes
from moeloop.experiment import experiment
from moeloop.model import LoopedLM
from moeloop.prepare import prepare_dataset
from moeloop.recipe_spec import RecipeSpec
from moeloop.titan import UNTIED_FIELDS, LoopTrainer, configure_optimizer
from moeloop.training import continuation_fingerprint, spec_identity
from moeloop.untie import check_untied, untie_state, untied

ROOT = Path(__file__).resolve().parents[2]
S = 4
GEOMETRY = dict(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, max_seq_len=8)


def source_spec(**geometry):
    """SMELT on 4 layers: 1 prefix cell, 2 shared core cells visited twice, 1 suffix cell, MoE."""
    return recipes.smelt_spec(physical_layers=4, **(GEOMETRY | geometry))


def untied_spec(spec):
    return RecipeSpec.from_dict(untied(spec.to_dict()) | {"name": "untie-unit"})


def test_the_untied_model_computes_the_shared_models_logits_bitwise():
    source = source_spec(num_experts=4, top_k=2)
    target = untied_spec(source)
    assert (target.model.num_cells, target.model.num_visits, target.training.num_visits) == (4, 1, 1)
    torch.manual_seed(3)
    shared = LoopedLM(source.model)
    with torch.no_grad():  # random weights, not the initialization: norms and every matrix differ
        for parameter in shared.parameters():
            parameter.normal_()
    model = LoopedLM(target.model)
    model.load_state_dict(untie_state(shared.state_dict(), num_cells=2, num_visits=2), strict=True)
    assert torch.equal(model.core[3].experts.w1_EFD, shared.core[1].experts.w1_EFD)
    tokens = torch.randint(0, 19, (3, 8))
    for mode in (False, True):
        expected, actual = shared.train(mode)(tokens), model.train(mode)(tokens)
        assert actual.logits.dtype == torch.float32 and torch.equal(actual.logits, expected.logits)
        assert torch.equal(actual.balance_sum, expected.balance_sum) and actual.aux_calls == expected.aux_calls == 6


def test_every_muon_momentum_copy_loads_as_the_source_entry(one_cpu_rank, tmp_path):
    def build(spec):
        config = recipes.trainer_config(spec)
        model = fsdp_model(config, one_cpu_rank, tmp_path)
        optimizers = replace(config.optimizer, implementation="for-loop").build(model_parts=[model])
        configure_optimizer(optimizers, [model], None)
        return model, optimizers

    source = muon_spec(source_spec(num_experts=4, top_k=2))
    shared, optimizers = build(source)
    tokens = torch.randint(0, 19, (2, 8))
    torch.nn.functional.cross_entropy(shared(tokens).logits.flatten(0, 1), tokens.flatten()).backward()
    optimizers.step()
    expected = untie_state(optimizers.state_dict(), num_cells=2, num_visits=2)
    model, untied_optimizers = build(untied_spec(source))
    model.load_state_dict(untie_state(shared.state_dict(), num_cells=2, num_visits=2))
    untied_optimizers.load_state_dict(expected)
    state = untied_optimizers.state_dict()
    assert state.keys() == expected.keys()
    momentum = [key for key in state if key.endswith("momentum_buffer")]
    assert "state.core.core.3.experts.routed.inner_experts.w1_EFD.momentum_buffer" in momentum
    assert "state.core.core.2.attentions.0.q_proj.weight.momentum_buffer" in momentum
    full = lambda value: value.full_tensor() if isinstance(value, DTensor) else value
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(full(state[key]), full(value)), key
        else:
            assert state[key] == value, key


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """One gloo rank, prepared data, a shared source run stopped at 6 and its untie from step S."""
    import torch.distributed as dist
    root = tmp_path_factory.mktemp("untie")
    lengths = [str(n) for n in (9, 4, 13, 6, 17, 5, 8, 11, 3, 12)]
    manifest = prepare_dataset(root / "data", {"a": {"train": lengths, "validation": lengths}},
        encode=lambda text: [3 + (5 * i + int(text)) % 16 for i in range(int(text))],
        tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 19, "bos_token_id": 1, "eos_token_id": 2})
    spec = source_spec(num_experts=2, top_k=1)
    spec = replace(spec, run=spec.run | RUN)
    (root / "source.json").write_text(json.dumps(spec.to_dict()))
    (root / "u0.json").write_text(json.dumps(untied_spec(spec).to_dict()))
    trainers = {}

    def launch(argv, directory, *, env, **kwargs):
        """The torchrun child of recipe_experiments.launch, run here on the gloo rank."""
        with pytest.MonkeyPatch.context() as patch:
            for name in ("SPEC", "MANIFEST", "OUTPUT", "STOP", "RESUME"):
                key = "MOELOOP_EXPERIMENT_" + name
                if key in env:
                    patch.setenv(key, env[key])
                else:
                    patch.delenv(key, raising=False)
            config = experiment()
            config.model_spec.model.flex_attention = False  # flex attention binds a GPU kernel
            config.validator.enable = False  # validation is pinned by test_validation_bpb
            trainer = LoopTrainer(config)
            trainers.setdefault(Path(directory).name, []).append(trainer)
            trainer.train()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Trainer, "__init__", cpu_native_init)
        patch.setattr(Trainer, "train", native_train)
        patch.setattr(runner, "run_command", launch)
        patch.setattr(runner, "runtime_identity", lambda python: {"torch": torch.__version__})
        dist.init_process_group("gloo", init_method=f"file://{root}/init", rank=0, world_size=1)
        try:
            lab = SimpleNamespace(root=root, manifest=manifest.path, trainers=trainers)
            runner.run(root / "source.json", manifest.path, root / "source", gpus=1, stop=6)
            lab.checkpoint = root / "source/checkpoint" / f"step-{S}"
            lab.metadata = digest(lab.checkpoint / ".metadata")
            patch.setenv("RUN_DIR", str(root / "record-untie"))
            runner.untie(lab.checkpoint, root / "u0.json", root / "untie", gpus=1)
            yield lab
        finally:
            dist.destroy_process_group()


def test_the_untied_run_starts_from_the_sources_state_copied_per_visit(lab):
    source, target = lab.trainers["source"][0], lab.trainers["untie"][0]
    expected = {part: untie_state(lab.trainers["source"][0].states[S][part], num_cells=2, num_visits=2)
                for part in ("model", "optimizer")}
    assert_same_state(target.loaded, expected)  # every copy is the source entry, bitwise
    for name in ("exp_avg", "exp_avg_sq", "step"):
        for cell in (0, 2):  # visit 0 and visit 1 of source cell 0
            key = f"state.core.core.{cell}.attentions.0.q_proj.weight.{name}"
            assert torch.equal(target.loaded["optimizer"][key],
                               source.states[S]["optimizer"][f"state.core.core.0.attentions.0.q_proj.weight.{name}"])
    # Update S+1 reads the source's update S+1 data with the same function: the same loss.
    records = {name: [json.loads(line) for line in (lab.root / name / "metrics.jsonl").read_text().splitlines()]
               for name in ("source", "untie")}
    first, reference = records["untie"][0], records["source"][S]
    assert first["step"] == reference["step"] == S + 1
    for key in ("ce", "aux", "global_progress", "global_source_counts"):
        assert first[key] == reference[key], key
    # The updates then differ: each copy receives only its own visit's gradient.
    assert not torch.equal(target.states[S + 1]["model"]["core.core.2.attentions.0.q_proj.weight"],
                           target.states[S + 1]["model"]["core.core.0.attentions.0.q_proj.weight"])
    assert [item["step"] for item in records["untie"]] == list(range(S + 1, 13))


def test_the_untie_records_its_source_and_keeps_the_source_schedule(lab):
    source_dir, untie_dir = lab.root / "source", lab.root / "untie"
    record = {"source": "smelt", "step": S, "checkpoint": str(lab.checkpoint), "metadata_sha256": lab.metadata}
    spec, source = (json.loads((path / "spec.json").read_text()) for path in (untie_dir, source_dir))
    assert spec["run"]["untie"] == record
    assert {k: v for k, v in spec["run"].items() if k != "untie"} == source["run"]
    assert json.loads((lab.root / "record-untie/config.json").read_text())["run"]["untie"] == record
    assert "untie" not in source["run"]
    identity, source_identity = (json.loads((path / "continuation_identity.json").read_text())
                                 for path in (untie_dir, source_dir))
    assert identity["lr_scheduler"] == source_identity["lr_scheduler"]
    assert {key for key in identity if identity[key] != source_identity[key]} == {
        "model", "recipe", "loss", "recipe_metadata"}  # AdamW groups name no cells
    assert json.loads((untie_dir / "outcome.json").read_text())["completed_updates"] == 12
    # The converted checkpoint is retained in the untie's own directory with its receipt.
    protected = untie_dir / "protected" / f"step-{S}"
    assert json.loads((protected / "retention.json").read_text()) == {"original_path": str(lab.checkpoint),
        "protected_path": str(protected), "metadata_sha256": digest(protected / ".metadata")}
    assert digest(lab.checkpoint / ".metadata") == lab.metadata  # the source is only read
    # The same command continues the untie from its own last checkpoint.
    runner.untie(lab.checkpoint, lab.root / "u0.json", untie_dir, gpus=1)
    rerun = lab.trainers["untie"][-1]
    assert rerun.step == 12 and rerun.factors == {}


def fake_source(root, name, change):
    """A source run directory with a changed spec and a placeholder step-S checkpoint."""
    spec = json.loads((root / "source/spec.json").read_text())
    change(spec)
    checkpoint = root / name / "checkpoint" / f"step-{S}"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"placeholder")
    (root / name / "spec.json").write_text(json.dumps(spec))
    return checkpoint


def test_sources_that_are_not_fully_shared_stack_loops_are_refused(lab):
    changes = {"layer": lambda spec: spec["model"].update(order="layer"),
               "sampled-depth": lambda spec: spec["training"].update(depth_rule="poisson_lognormal_micro_v1"),
               "own-attention": lambda spec: spec["model"].update(attention_per_visit=True, visit_capacity=2),
               "one-visit": lambda spec: spec["model"].update(num_visits=1) or spec["training"].update(num_visits=1)}
    for name, change in changes.items():
        checkpoint = fake_source(lab.root, f"source-{name}", change)
        with pytest.raises(ValueError, match="fully shared stack-order"):
            runner.untie(checkpoint, lab.root / "u0.json", lab.root / f"untie-{name}", gpus=1)
        assert not (lab.root / f"untie-{name}").exists()


def write_target(root, name, change):
    spec = json.loads((root / "u0.json").read_text())
    change(spec)
    (root / f"{name}.json").write_text(json.dumps(spec))
    return root / f"{name}.json"


def test_targets_that_are_not_the_sources_untied_form_are_refused(lab):
    changes = {"three-cells": lambda spec: spec["model"].update(num_cells=3),
               "half-experts": lambda spec: spec["model"].update(core=spec["model"]["core"] | {"num_experts": 1, "top_k": 1})}
    for name, change in changes.items():
        with pytest.raises(ValueError, match="not the source's untied form"):
            runner.untie(lab.checkpoint, write_target(lab.root, name, change), lab.root / f"untie-{name}", gpus=1)
        assert not (lab.root / f"untie-{name}").exists()


def test_changed_data_topology_optimizer_or_schedule_is_refused(lab):
    changes = {"lr": lambda spec: spec["optimizer"].update(base_lr=1e-3),
               "weight-decay": lambda spec: spec["optimizer"].update(weight_decay=.05),
               "warmup": lambda spec: spec["run"]["schedule"].update(warmup_steps=3),
               "seed": lambda spec: spec["run"].update(seed=8)}
    for name, change in changes.items():
        with pytest.raises(ValueError, match="not the source's untied form"):
            runner.untie(lab.checkpoint, write_target(lab.root, name, change), lab.root / f"untie-{name}", gpus=1)
    # The trainer refuses the converted state at another step or under another identity.
    target = lab.trainers["untie"][0]
    source = lab.trainers["source"][0].identity  # as the checkpoint payload holds it, tuples included
    target.check_untie(source, S)
    with pytest.raises(ValueError, match="only from step"):
        target.check_untie(source, S + 2)
    for key, value in {"world_size": 2, "manifest": "other-data", "debug_seed": 8,
                       "lr_scheduler": source["lr_scheduler"] | {"warmup_steps": 3},
                       "execution": source["execution"] | {"native_experts": False}}.items():
        with pytest.raises(ValueError, match="differs beyond the untied model"):
            target.check_untie(source | {key: value}, S)
    # An untie spec launched without its converted checkpoint would claim a source it never loaded.
    with pytest.raises(ValueError, match="launch it with"):
        runner.run(lab.root / "untie/spec.json", lab.manifest, lab.root / "untie-unloaded", gpus=1)
    assert not (lab.root / "untie-unloaded").exists()  # refused before anything is written
    assert set(UNTIED_FIELDS) <= set(source)


def source_recording(lab, name, change):
    """A copy of the source run whose step-S checkpoint records ``change(identity)`` as its identity."""
    import pickle
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    run = lab.root / name
    (run / "checkpoint").mkdir(parents=True)
    for file in ("spec.json", "request.json"):
        (run / file).write_text((lab.root / "source" / file).read_text())
    reader = dcp.FileSystemReader(lab.checkpoint)
    state = {key: torch.empty(entry.size, dtype=entry.properties.dtype) if isinstance(entry, TensorStorageMetadata)
             else None for key, entry in reader.read_metadata().state_dict_metadata.items()}
    dcp.load(state, storage_reader=reader, no_dist=True,
             planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False))
    payload = pickle.loads(state["train_state.rank_0"])
    payload["identity"] = change(payload["identity"])
    payload["fingerprint"] = continuation_fingerprint(payload["identity"])
    state["train_state.rank_0"] = pickle.dumps(payload, protocol=5)
    checkpoint = run / "checkpoint" / f"step-{S}"
    dcp.save(state, checkpoint_id=checkpoint, no_dist=True)
    return checkpoint


def test_a_source_from_older_code_is_accepted_but_not_other_data_or_schedule(lab):
    older = source_recording(lab, "source-older-code",
                             lambda identity: identity | {"moeloop_source": {"titan.py": "0" * 64}})
    runner.untie(older, lab.root / "u0.json", lab.root / "untie-older-code", gpus=1)
    untied_run = lab.trainers["untie-older-code"][0]
    assert_same_state(untied_run.loaded, lab.trainers["untie"][0].loaded)
    assert json.loads((lab.root / "untie-older-code/outcome.json").read_text())["completed_updates"] == 12
    changes = {"data": lambda identity: identity | {"manifest": "other-data"},
               "schedule": lambda identity: identity | {"lr_scheduler": identity["lr_scheduler"] | {"warmup_steps": 3}}}
    for name, change in changes.items():
        with pytest.raises(ValueError, match="differs beyond the untied model"):
            runner.untie(source_recording(lab, f"source-{name}", change), lab.root / "u0.json",
                         lab.root / f"untie-{name}", gpus=1)


def test_u0_specs_are_m3_untied_with_m3_flops_and_kv():
    files = {path.stem: json.loads(path.read_text()) for path in (ROOT / "recipes/untie").glob("*.json")}
    assert files == untie_specs.specs() and sorted(files) == ["u0-adamw", "u0-muon"]
    subprocess.run([sys.executable, str(ROOT / "scripts/untie_specs.py"), "--check"], check=True)
    for optimizer in ("adamw", "muon"):
        m3 = json.loads((ROOT / f"recipes/phase1/m3-{optimizer}.json").read_text())
        u0 = files[f"u0-{optimizer}"]
        assert u0["name"] == f"untie-u0-{optimizer}" and u0["optimizer"] == m3["optimizer"]
        assert u0["run"] == m3["run"] | {"untie": None}  # config.json records U0's untie as null
        assert (u0["model"]["num_cells"], u0["model"]["num_visits"], u0["model"]["core"]) == (12, 1, m3["model"]["core"])
        check_untied(m3, u0)
        comparison = compare_recipes(RecipeSpec.from_dict(m3), RecipeSpec.from_dict(u0), seq_len=4096)
        for metric in ("training_flops_per_token", "logical_kv_bytes"):
            assert abs(comparison["matched"][metric]["relative_difference"]) < 1e-6
        growth = comparison["measured-different"]["unique_non_embedding_parameters"]["relative_difference"]
        assert abs(growth - .5) < 1e-3  # the 6 core cells' parameters, twice


def trainer_identity(spec):
    """The continuation-identity fields LoopTrainer takes from ``trainer_config(spec)``; the rest are the launch's."""
    config = recipes.trainer_config(spec)
    return {**spec_identity(model=spec.model, recipe=spec.training, recipe_metadata=spec.to_dict(),
                            expert_lr_multiplier=config.expert_lr_multiplier, fork={}, ema_decay=None),
            "optimizer": config.optimizer.to_dict(), "lr_scheduler": config.lr_scheduler.to_dict(),
            "loss": config.loss.to_dict(), "dataloader": config.dataloader.to_dict(),
            "precision": [config.training.dtype, config.training.mixed_precision_param,
                          config.training.mixed_precision_reduce], "max_norm": config.training.max_norm}


def test_the_muon_m3_identity_unties_with_only_its_cell_layouts_changed():
    m3 = RecipeSpec.from_dict(json.loads((ROOT / "recipes/phase1/m3-muon.json").read_text()))
    u0 = json.loads((ROOT / "recipes/untie/u0-muon.json").read_text())
    u0["run"]["untie"] = {"source": m3.name, "step": S, "checkpoint": "m3/checkpoint/step-4", "metadata_sha256": "0" * 64}
    source, identity = trainer_identity(m3), trainer_identity(RecipeSpec.from_dict(u0))
    assert {key for key in identity if identity[key] != source[key]} == {
        "model", "recipe", "loss", "recipe_metadata", "optimizer"}
    assert ({key for key in identity["optimizer"] if identity["optimizer"][key] != source["optimizer"][key]}
            == {"optimizer_factory_kwargs_by_name"})  # the Muon layouts and buckets name every core cell
    target = LoopTrainer.__new__(LoopTrainer)
    target.identity = identity
    target.check_untie(source, S)
    groups = source["optimizer"]["param_groups"]
    changed = source | {"optimizer": source["optimizer"] | {"param_groups": groups[:-1]}}
    with pytest.raises(ValueError, match="differs beyond the untied model"):
        target.check_untie(changed, S)
