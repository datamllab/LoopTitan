"""P1 shape, nominal token rounding and the actual native WSD boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_p1_pair_uses_bounded_cost_screen_and_preserves_engineering_example(monkeypatch):
    from moeloop.comparison import smelt_budget_pair, smelt_p1_pair
    from moeloop.model import RecursiveLM
    from moeloop.recipe_spec import RecipeSpec

    def reject_allocation(*args, **kwargs):
        raise AssertionError("P1 arithmetic must not instantiate model weights")

    monkeypatch.setattr(RecursiveLM, "__init__", reject_allocation)
    pair = smelt_p1_pair()
    assert pair["candidate_count"] == 12
    comparison = pair["comparison"]
    assert set(comparison["matched"]) == {
        "unique_non_embedding_parameters", "training_flops_per_token", "logical_kv_bytes"}
    costs = comparison["costs"]
    assert costs["reference"]["parameters"]["unique_non_embedding"] == 712_656_672
    assert costs["candidate"]["parameters"]["unique_non_embedding"] == 711_302_976
    # Includes the independently owned 151669-token head in both arms.
    assert costs["reference"]["flops"]["forward_components"]["head"] == 203_843_136
    assert costs["candidate"]["flops"]["forward_components"]["head"] == 174_722_688
    assert max(abs(value["relative_difference"]) for value in comparison["matched"].values()) < .01
    assert pair["active_parameters"]["reference"]["physical_non_embedding"] == 98_502_432
    assert pair["active_parameters"]["candidate"]["physical_non_embedding"] == 66_551_616
    for arm in ("reference", "candidate"):
        spec = RecipeSpec.from_dict(json.loads(json.dumps(comparison["arms"][arm])))
        schedule = spec.run["schedule"]
        assert schedule["total_steps"] == spec.run["successful_updates"] == 19_537
        assert schedule["target_label_tokens"] == 20_485_760_000
        assert schedule["rounded_nominal_positions"] == 20_486_029_312
        assert 0 <= schedule["rounded_nominal_positions"] - schedule["target_label_tokens"] < 1_048_576
        assert schedule["warmup_steps"] == 20
        assert schedule["stable_end_steps"] == 10_000
        assert schedule["decay_steps"] == 9_537
        assert spec.run["seed"] == 42
    assert smelt_budget_pair()["selected_shape"]["dim"] == 128
    document_pair = smelt_p1_pair(attention_pairs_per_sequence=8_390_656)
    document = document_pair["comparison"]
    assert document_pair["candidate_count"] == 12
    assert document["costs"]["reference"]["formula_version"] == "fixed_terminal_document_matrix_v1"
    assert document["arms"]["candidate"]["model"]["hidden_dim"] == 432
    assert set(document["matched"]) == set(comparison["matched"])


@pytest.mark.parametrize("policy", ["none", "full_cells"])
@pytest.mark.parametrize("arm", ["reference", "candidate"])
def test_public_pair_roundtrip_preserves_native_activation_checkpointing(policy, arm):
    pytest.importorskip("torchtitan")
    from torchtitan.distributed.activation_checkpoint import FullAC
    from moeloop.comparison import smelt_p1_pair
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.recipes import trainer_config

    pair = smelt_p1_pair(activation_checkpointing=policy)
    spec = RecipeSpec.from_dict(json.loads(json.dumps(pair["comparison"]["arms"][arm])))
    config = trainer_config(spec)
    assert config.recipe_metadata["run"]["activation_checkpointing"] == policy
    if policy == "full_cells":
        assert isinstance(config.activation_checkpoint, FullAC.Config)
    else:
        assert config.activation_checkpoint is None


@pytest.mark.parametrize("policy", ["selective", None])
def test_native_adapter_rejects_explicit_unsupported_checkpoint_policy(policy):
    pytest.importorskip("torchtitan")
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.recipes import p1_moe_spec, trainer_config

    spec = p1_moe_spec()
    spec = replace(spec, run=spec.run | {"activation_checkpointing": policy})
    spec = RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    with pytest.raises(ValueError, match="activation_checkpointing"):
        trainer_config(spec)


def test_native_adapter_defaults_to_no_checkpointing_without_policy():
    pytest.importorskip("torchtitan")
    from moeloop.recipes import tiny_spec, trainer_config

    spec = tiny_spec()
    assert trainer_config(spec).activation_checkpoint is None
    # A recipe JSON frozen without the key resolves it from RUN_DEFAULTS.
    archived = replace(spec, run={key: value for key, value in spec.run.items() if key != "activation_checkpointing"})
    assert trainer_config(archived).activation_checkpoint is None


def test_every_constructor_writes_every_run_setting():
    import inspect
    from moeloop import recipes

    for name, make in inspect.getmembers(recipes, inspect.isfunction):
        if name.endswith("_spec") and not name.startswith("_") and make.__module__ == recipes.__name__:
            specs = make().values() if name == "sharing_panel_specs" else [make()]
            for spec in specs:
                assert set(recipes.RUN_DEFAULTS) | {"seq_len"} <= set(spec.run), name


def test_trainer_takes_every_run_setting_from_the_spec(monkeypatch):
    pytest.importorskip("torchtitan")
    from moeloop.recipes import moe_spec, tiny_spec, trainer_config

    spec = tiny_spec()
    spec = replace(spec, run=spec.run | {"num_chunks": 4, "local_batch_size": 2, "keep_latest_k": 0,
        "checkpoint_interval": 7, "validation_global_batches": 3, "seed": 5, "data_infinite": False})
    config = trainer_config(spec)
    assert (config.loss.num_chunks, config.training.local_batch_size, config.checkpoint.keep_latest_k,
            config.checkpoint.interval, config.validator.global_batches, config.debug.seed,
            config.dataloader.seed, config.dataloader.infinite) == (4, 2, 0, 7, 3, 5, 5, False)
    # local_batch_size None: one microbatch per data-parallel rank.
    spec = moe_spec()
    assert spec.run["local_batch_size"] is None and spec.run["global_batch_positions"] // spec.run["seq_len"] == 16
    for world, local in (("1", 16), ("4", 4), ("32", 1)):
        monkeypatch.setenv("WORLD_SIZE", world)
        assert trainer_config(spec).training.local_batch_size == local
        assert trainer_config(replace(spec, run=spec.run | {"local_batch_size": 2})).training.local_batch_size == 2
    # An archived JSON without either key resolves both from RUN_DEFAULTS.
    archived = replace(spec, run={key: value for key, value in spec.run.items()
                                  if key not in ("global_batch_positions", "local_batch_size")})
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert trainer_config(archived).training.local_batch_size == 4


@pytest.mark.parametrize("field", ["num_visits", "depth_seed", "balance_coeff", "z_loss_coeff",
                                   "depth_mean", "depth_log_std", "entropy_coeff"])
def test_training_recipe_numbers_reject_bool(field):
    from moeloop.training import TrainingRecipe

    with pytest.raises(ValueError, match=field):
        TrainingRecipe(**{"num_visits": 2, field: True})


@pytest.mark.parametrize("roles,embedding_decay,norm_decay,group_count", [
    (["norm", "embedding", "lm_head", "router"], 0., 0., 2),
    (["norm", "router"], .2, 0., 2),
    ([], .2, .2, 1),
])
def test_native_optimizer_applies_declared_zero_decay_roles(roles, embedding_decay, norm_decay, group_count):
    pytest.importorskip("torchtitan")
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.recipes import moe_spec, trainer_config
    from moeloop.titan import TitanModel

    spec = moe_spec(layers=1, vocab_size=19, dim=16, hidden_dim=24, num_heads=2,
                    num_kv_heads=1, num_experts=2, top_k=2, max_seq_len=8)
    spec = replace(spec, optimizer=spec.optimizer | {"zero_weight_decay_roles": roles,
                   "weight_decay": .2, "base_lr": .01})
    config = trainer_config(RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))))
    assert config.recipe_metadata["optimizer"]["zero_weight_decay_roles"] == roles
    model = TitanModel(replace(config.model_spec.model, native_experts=False, flex_attention=False))
    optimizer = replace(config.optimizer, implementation="for-loop").build(model_parts=[model])
    groups = optimizer.optimizers[0].param_groups
    assert len(groups) == group_count
    decay_by_parameter = {id(parameter): group["weight_decay"] for group in groups for parameter in group["params"]}
    parameters = {"embedding": model.core.embedding.weight, "lm_head": model.core.lm_head.weight,
                  "norm": model.core.final_norm.weight, "router": model.core.core[0].routers[0].gate.weight}
    assert {name: decay_by_parameter[id(parameter)] for name, parameter in parameters.items()} == {
        "embedding": embedding_decay, "lm_head": embedding_decay, "norm": norm_decay, "router": norm_decay}
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for name, parameter in parameters.items():
        decay = embedding_decay if name in ("embedding", "lm_head") else norm_decay
        torch.testing.assert_close(parameter, before[name] * (1 - .01 * decay))


@pytest.mark.parametrize("section,key,value", [
    ("optimizer", "zero_weight_decay_roles", ["attention"]),
    ("optimizer", "name", "SGD"),
])
def test_native_adapter_rejects_unsupported_common_stack_controls(section, key, value):
    pytest.importorskip("torchtitan")
    from moeloop.recipe_spec import RecipeSpec
    from moeloop.recipes import p1_moe_spec, trainer_config

    spec = p1_moe_spec()
    spec = replace(spec, **{section: getattr(spec, section) | {key: value}})
    spec = RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    with pytest.raises(ValueError, match=key):
        trainer_config(spec)


@pytest.mark.parametrize("attention_pairs, hidden_dim", [(None, 424), (4_180_216.9229844734, 440)])
def test_anchor_generator_resolves_portable_json_without_importing_torchtitan(tmp_path, attention_pairs, hidden_dim):
    program = """
import importlib.abc, runpy, sys
class NoTitan(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('torchtitan'):
            raise AssertionError('offline anchor generator imported TorchTitan')
sys.meta_path.insert(0, NoTitan())
sys.argv = ['recipe_anchor.py', '--output', sys.argv[1], '--peak-lr', '0.0001', *sys.argv[2:]]
runpy.run_path('scripts/recipe_anchor.py', run_name='__main__')
"""
    root = Path(__file__).resolve().parents[2]
    attention_args = ([] if attention_pairs is None else
                      ["--attention-pairs-per-sequence", str(attention_pairs)])
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path), *attention_args], cwd=root,
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert result.returncode == 0, result.stderr
    pair = json.loads((tmp_path / "pair.json").read_text())
    for arm, filename in (("reference", "p1_moe.json"), ("candidate", "p1_smelt.json")):
        spec = json.loads((tmp_path / filename).read_text())
        assert spec == pair["comparison"]["arms"][arm]
        assert spec["optimizer"]["base_lr"] == .0001
    assert pair["comparison"]["arms"]["candidate"]["model"]["hidden_dim"] == hidden_dim

    from moeloop.comparison import compare_recipes
    from moeloop.costs import estimate_cost
    from moeloop.recipe_spec import RecipeSpec

    reference, candidate = (RecipeSpec.from_dict(pair["comparison"]["arms"][arm])
                            for arm in ("reference", "candidate"))
    options = {"seq_len": 4096, "attention_pairs_per_sequence": attention_pairs}
    for action, filenames in (("compare", ["p1_moe.json", "p1_smelt.json"]),
                              ("inspect", ["p1_smelt.json"])):
        compared = subprocess.run([sys.executable, "scripts/compare_recipes.py", action,
            *(str(tmp_path / name) for name in filenames), "--seq-len", "4096", *attention_args],
            cwd=root, capture_output=True, text=True)
        assert compared.returncode == 0, compared.stderr
        record = json.loads(compared.stdout)
        if action == "compare":
            assert record == compare_recipes(reference, candidate, **options)
        else:
            assert record["cost"] == estimate_cost(candidate, **options)


def test_native_anchor_stop_keeps_full_wsd_horizon_and_resume_lr(tmp_path, monkeypatch):
    pytest.importorskip("torchtitan")
    from moeloop.recipes import p1_moe_spec
    from recipe_anchor import anchor

    spec = p1_moe_spec()
    path = tmp_path / "p1.json"
    path.write_text(json.dumps(spec.to_dict()))
    monkeypatch.setenv("MOELOOP_ANCHOR_SPEC", str(path))
    monkeypatch.setenv("MOELOOP_ANCHOR_STOP", "96")
    config = anchor()
    assert config.training.steps == 96
    assert config.lr_scheduler.total_steps == 19_537
    assert config.training.global_batch_size * config.training.seq_len == 1_048_576
    assert config.training.mixed_precision_reduce == "float32"
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.0003)
    schedulers = config.lr_scheduler.build(optimizers=[optimizer], training_steps=96)
    factor = schedulers.schedulers[0].lr_lambdas[0]
    assert factor(0) == pytest.approx(.05)
    assert factor(19) == factor(20) == factor(10_000) == 1
    assert factor(10_001) < 1
    assert factor(19_537) == pytest.approx(.1)
    # Strict continuation restores the scheduler's coordinate, without a new
    # warmup or a shortened cosine schedule for this stopped prefix.
    schedulers.load_state_dict({"last_epoch": 15_000})
    saved = schedulers.state_dict()
    other = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.0003)
    resumed = config.lr_scheduler.build(optimizers=[other], training_steps=96)
    resumed.load_state_dict(saved)
    assert resumed.schedulers[0].get_last_lr() == schedulers.schedulers[0].get_last_lr()
    spec.run["schedule_total_steps"] = 96
    path.write_text(json.dumps(spec.to_dict()))
    with pytest.raises(ValueError, match="frozen"):
        anchor()


def make_cpu_validator(tmp_path, config, *, world=1, rank=0, batch=1, seq_len=8):
    from moeloop.titan import LoopValidator
    from torchtitan.config import ParallelismConfig

    metrics = SimpleNamespace(output_directory=tmp_path, last_validation={}, ntokens_since_last_log=0)
    metrics.log_validation = lambda loss, step: metrics.last_validation.update(ce=float(loss), step=step)
    def loss(logits, labels):
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum", ignore_index=-100), {}
    validator = LoopValidator(config, parallelism=ParallelismConfig(spmd_backend="partial_dtensor"),
        dp_world_size=world, dp_rank=rank, tokenizer=None,
        parallel_dims=SimpleNamespace(pp_enabled=False, dp_enabled=False, dp_cp_enabled=False),
        loss_fn=loss, validation_context=nullcontext, metrics_processor=metrics,
        seq_len=seq_len, local_batch_size=batch)
    return validator


def test_validation_token_budget_uses_runtime_rank_batch_and_context(tmp_path, monkeypatch):
    pytest.importorskip("torchtitan")
    from moeloop.recipes import p1_moe_spec
    from moeloop.titan import LoopValidator, PackedDataLoader
    from recipe_anchor import anchor

    spec = p1_moe_spec()
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    monkeypatch.setenv("MOELOOP_ANCHOR_SPEC", str(path))
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = anchor()
    assert config.validator.target_label_tokens == 1_000_000
    # These dimensions are passed by Trainer after CLI overrides, not guessed
    # in the recipe factory. Infinite is explicit here to isolate arithmetic.
    budget = LoopValidator.Config(target_label_tokens=1_000_000,
        dataloader=PackedDataLoader.Config(infinite=True))
    validator = make_cpu_validator(tmp_path, budget, world=8, batch=16, seq_len=4096)
    assert validator.config.steps == 2
    assert validator.config.steps * 8 * 16 * 4096 == 1_048_576
    # P1's global batch is 256 blocks: the budget rounds up to one, on any rank count.
    p1 = replace(budget, dataloader=PackedDataLoader.Config(infinite=True, global_batch_size=256))
    for world, steps in ((8, 2), (16, 1), (2, 8)):
        validator = make_cpu_validator(tmp_path, p1, world=world, batch=16, seq_len=4096)
        assert validator.global_batches == 1 and validator.config.steps == steps
    # Without a budget, a fixed number of global batches; accumulation multiplies the steps.
    fixed = make_cpu_validator(tmp_path, replace(p1, target_label_tokens=None, global_batches=3), world=8, batch=16, seq_len=4096)
    assert fixed.config.steps == 3 * 2 and fixed.dl_config.infinite is True
    # Native steps is derived; an explicit --validator.steps is refused, not silently replaced.
    with pytest.raises(ValueError, match="global_batches"):
        make_cpu_validator(tmp_path, replace(p1, steps=1), world=8, batch=16, seq_len=4096)


def test_finite_validation_honors_loader_and_records_actual_labels_and_steps(tmp_path, monkeypatch):
    pytest.importorskip("torchtitan")
    from moeloop.prepare import prepare_dataset
    from moeloop.titan import LoopValidator, PackedDataLoader

    monkeypatch.setattr("torchtitan.components.validate.utils.device_type", "cpu")
    manifest = prepare_dataset(tmp_path / "data", {"public": {"validation": ["10"] * 16}},
        encode=lambda text: [3] * int(text),
        tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 8, "bos_token_id": 1, "eos_token_id": 2})
    config = LoopValidator.Config(target_label_tokens=33, dataloader=PackedDataLoader.Config(
        manifest_path=str(manifest.path), split="validation", shuffle=False, infinite=False))
    validator = make_cpu_validator(tmp_path, config, batch=4)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(flex_attention=False)
            self.core = SimpleNamespace(config=SimpleNamespace(num_visits=1))
        def forward(self, inputs, **kwargs):
            return torch.zeros(*inputs.shape, 8)
    model = Model().eval()
    validator.validate([model], step=7)
    validator.validate([model], step=8)
    records = [json.loads(line) for line in (tmp_path / "validation.jsonl").read_text().splitlines()]
    assert [record["valid_tokens"] for record in records] == [64, 64]
    record = records[-1]
    assert record["valid_tokens"] == 64
    assert record["target_label_tokens"] == 33
    assert record["steps"] == record["configured_steps"] == 2
    assert record["infinite"] is False and validator.dl_config.infinite is False
    assert model.training is False


def test_finite_validation_rejects_insufficient_capacity_before_collectives(tmp_path):
    pytest.importorskip("torchtitan")
    from moeloop.prepare import prepare_dataset
    from moeloop.titan import LoopValidator, PackedDataLoader

    manifest = prepare_dataset(tmp_path / "data", {"public": {"validation": ["7", "7", "7"]}},
        encode=lambda text: [3] * int(text),
        tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 8, "bos_token_id": 1, "eos_token_id": 2})
    data = PackedDataLoader.Config(manifest_path=str(manifest.path), split="validation", shuffle=False, infinite=False)
    with pytest.raises(ValueError, match="finite.*capacity|capacity.*finite"):
        make_cpu_validator(tmp_path, LoopValidator.Config(global_batches=2, dataloader=data), world=2)


@pytest.mark.parametrize("stop,enabled,completed", [(19_537, True, 19_537), (954, True, 954),
                                                    (95, False, 95), (95, True, 94)])
def test_native_train_validates_only_complete_uncovered_endpoints(tmp_path, monkeypatch, stop, enabled, completed):
    pytest.importorskip("torchtitan")
    from torchtitan.trainer import Trainer
    from moeloop.model import ModelConfig
    from moeloop.prepare import prepare_dataset
    from moeloop.titan import LoopTrainer, LoopValidator, PackedDataLoader, TitanModel
    from moeloop.training import CommittedProgress

    monkeypatch.setattr("torchtitan.components.validate.utils.device_type", "cpu")
    manifest = prepare_dataset(tmp_path / "data", {"public": {"validation": ["document"] * 4}},
        encode=lambda _: [3] * 7,
        tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 8, "bos_token_id": 1, "eos_token_id": 2})
    validator = make_cpu_validator(tmp_path, LoopValidator.Config(enable=enabled, freq=477, global_batches=1,
        dataloader=PackedDataLoader.Config(manifest_path=str(manifest.path), split="validation", shuffle=False, infinite=False)))
    validator.loss_fn = lambda output, labels: (torch.nn.functional.cross_entropy(
        output.logits.flatten(0, 1), labels.flatten(), reduction="sum", ignore_index=-100), {})
    model = TitanModel(TitanModel.Config(core=ModelConfig(vocab_size=8, dim=8, hidden_dim=16,
        num_heads=2, num_kv_heads=1, num_experts=0, num_cells=1, num_visits=1, max_seq_len=8),
        native_experts=False, flex_attention=False))

    # Replace only native training/CUDA setup; keep the real endpoint adapter,
    # should_validate predicate and CPU heldout evaluation/file evidence.
    trainer = LoopTrainer.__new__(LoopTrainer)
    trainer.config = SimpleNamespace(training=SimpleNamespace(steps=stop),
        checkpoint=SimpleNamespace(enable=False), validator=SimpleNamespace(enable=enabled))
    trainer.progress, trainer.step = CommittedProgress(completed_updates=completed), stop
    trainer.validator, trainer.model_parts = validator if enabled else None, [model]
    trainer.checkpointer = SimpleNamespace(maybe_wait_for_saving=lambda: None)
    trainer.rank, trainer.identity_digest = 0, "unit"
    trainer._global_observations = lambda: ({"completed_updates": completed}, {})
    outcomes = []
    trainer._write_json = lambda _, value: outcomes.append(value)

    def native_train(self):
        if self.config.validator.enable and self.validator.should_validate(self.step):
            self.validator.validate(self.model_parts, self.step)
    monkeypatch.setattr(Trainer, "train", native_train)
    if completed != stop:
        with pytest.raises(RuntimeError, match="completed updates"):
            trainer.train()
    else:
        trainer.train()
    path = tmp_path / "validation.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    assert [record["step"] for record in records] == ([stop] if enabled and completed == stop else [])
    assert all(record["valid_tokens"] == 8 for record in records)
    assert outcomes[-1]["status"] == ("completed" if completed == stop else "failed")
