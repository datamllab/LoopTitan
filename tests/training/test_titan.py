import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from moeloop.model import ModelConfig, RecursiveLM
from moeloop.titan import (
    LoopChunkedLoss,
    LoopTrainer,
    TitanModel,
    canonical_model_state,
)
from moeloop.training import CommittedProgress, TrainingRecipe, continuation_fingerprint, objective


def config():
    return ModelConfig(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, num_experts=2, top_k=2, num_cells=1, num_visits=2)


def test_core_expert_lr_uses_owner_identity_and_preserves_named_optimizer_state():
    from dataclasses import replace
    from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
    from moeloop.titan import scale_core_expert_lr

    cfg = replace(config(), prefix_cells=1, order="layer", attention_per_visit=True,
                  router_per_visit=True, norm_per_visit=True)
    model = TitanModel(TitanModel.Config(core=cfg, native_experts=False, flex_attention=False))
    optimizer = OptimizersContainer.Config(implementation="for-loop", param_groups=[
        ParamGroupConfig(pattern=r".*", optimizer_name="AdamW", optimizer_kwargs={"lr": .01, "weight_decay": .1}),
    ]).build(model_parts=[model])
    experts = {id(p) for p in model.core.core[0].experts.parameters()}
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    scale_core_expert_lr(optimizer, [model], multiplier=2 ** -.5)
    groups = optimizer.optimizers[0].param_groups
    assigned = [(name, parameter) for group in groups for name, parameter in zip(group["param_names"], group["params"])]
    assert len(assigned) == len({id(p) for _, p in model.named_parameters()})
    assert {name for name, _ in assigned} == set(before)
    for group in groups:
        assert group["weight_decay"] == .1
        for parameter in group["params"]:
            expected_lr = .01 * (2 ** -.5 if id(parameter) in experts else 1)
            assert group["lr"] == pytest.approx(expected_lr)
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        lr = .01 * (2 ** -.5 if id(parameter) in experts else 1)
        expected = before[name] * (1 - lr * .1) - lr / (1 + 1e-8)
        torch.testing.assert_close(parameter, expected)
    saved = optimizer.state_dict()
    optimizer.load_state_dict(saved)
    assert saved


def test_meta_materialize_initialization_and_native_weight_keys_are_complete():
    with torch.device("meta"):
        model = TitanModel(TitanModel.Config(core=config(), flex_attention=False))
    model.verify_module_protocol()
    model.to_empty(device="cpu")
    model.init_weights()
    state = canonical_model_state(model.state_dict())
    reference = RecursiveLM(config())
    reference.load_state_dict(state, strict=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert reference.core[0].experts.w1_EFD.std() > 0
    torch.testing.assert_close(reference.final_norm.weight, torch.ones(16))


def test_native_chunked_head_and_aux_have_same_gradients_as_explicit_logits():
    torch.manual_seed(37)
    bridge = TitanModel(TitanModel.Config(core=config(), native_experts=False, flex_attention=False))
    bridge._skip_lm_head = True
    reference = RecursiveLM(config())
    reference.load_state_dict(canonical_model_state(bridge.state_dict()))
    tokens = torch.randint(0, 19, (2, 4))
    labels = torch.randint(0, 19, (2, 4))
    labels[0, 0] = -100
    denominator = (labels != -100).sum()
    recipe = TrainingRecipe(num_visits=2, balance_coeff=0.17, z_loss_coeff=0.03)
    loss_fn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=2, recipe=recipe))
    loss_fn.set_lm_head(bridge.lm_head)
    loss, metrics = loss_fn(bridge(tokens), labels, denominator)
    expected, _ = objective(reference(tokens), labels, denominator, recipe)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    for (name, parameter), (_, other) in zip(bridge.core.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=2e-7, msg=name)
    with torch.no_grad():
        raw, values = loss_fn(bridge(tokens), labels)
    torch.testing.assert_close(raw, metrics["ce"] * denominator, rtol=1e-5, atol=1e-6)
    assert "aux" not in values


def test_expected_ce_training_path_is_the_reference_objective_bit_for_bit():
    from dataclasses import replace
    cfg = replace(config(), num_visits=3, exit_distribution=True)
    torch.manual_seed(37)
    bridge = TitanModel(TitanModel.Config(core=cfg, native_experts=False, flex_attention=False))
    bridge._skip_lm_head = True
    reference = RecursiveLM(cfg)
    reference.load_state_dict(canonical_model_state(bridge.state_dict()))
    tokens = torch.randint(0, 19, (2, 4))
    labels = torch.randint(0, 19, (2, 4))
    labels[0, 0] = -100
    denominator = torch.tensor(14)
    recipe = TrainingRecipe(num_visits=3, objective="expected_ce", entropy_coeff=0.1, balance_coeff=0.17, z_loss_coeff=0.03)
    loss_fn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=2, recipe=recipe))
    loss_fn.set_lm_head(bridge.lm_head)
    loss_fn.dp_degree, loss_fn.accumulation = 2, 3
    loss, metrics = loss_fn(bridge(tokens), labels, denominator)
    expected, expected_metrics = objective(reference(tokens), labels, denominator, recipe, dp_degree=2,
                                           accumulation=3, head=reference.lm_head, num_chunks=2)
    loss.backward()
    expected.backward()
    assert torch.equal(loss, expected)
    assert metrics.keys() == expected_metrics.keys() and all(torch.equal(metrics[k], expected_metrics[k]) for k in metrics)
    for (name, parameter), (_, other) in zip(bridge.core.named_parameters(), reference.named_parameters()):
        assert torch.equal(parameter.grad, other.grad), name


def trainer_state_stub(rank=0):
    trainer = LoopTrainer.__new__(LoopTrainer)
    trainer.rank = rank
    trainer.identity = {"schema": 1, "manifest": "abc", "world_size": 1, "fork": {}}
    trainer.identity_digest = continuation_fingerprint(trainer.identity)
    trainer.progress = CommittedProgress(completed_updates=7, input_tokens=91, label_tokens=85)
    trainer.step = 7
    trainer.ntokens_seen = 112
    trainer.ema = None
    return trainer


def test_rank_rng_and_completed_progress_roundtrip_and_changed_identity_rejects():
    source = trainer_state_stub()
    torch.manual_seed(71)
    state = source.state_dict()
    expected = torch.rand(5)
    target = trainer_state_stub()
    target.load_state_dict(state)
    assert target.progress == source.progress
    assert torch.equal(torch.rand(5), expected)
    other_rank = trainer_state_stub(rank=1)
    with pytest.raises(ValueError, match="rank-qualified"):
        other_rank.load_state_dict(state)
    target.identity_digest = "changed"
    with pytest.raises(ValueError, match="identity"):
        target.load_state_dict(state)


def test_builtin_python_recipe_has_native_owner_and_explicit_defaults():
    from moeloop.recipes import tiny, injected, expert_tying
    from torchtitan.config import ConfigManager
    recipe = tiny()
    assert recipe._owner is LoopTrainer
    assert recipe.parallelism.spmd_backend == "partial_dtensor"
    assert not recipe.checkpoint.last_save_model_only
    assert recipe.lr_scheduler.total_steps == 1000
    recipe.recipe.validate_model(recipe.model_spec.model.core)
    injected().recipe.validate_model(injected().model_spec.model.core)
    expert_tying().recipe.validate_model(expert_tying().model_spec.model.core)
    parsed = ConfigManager().parse_args(["--module", "moeloop.recipes", "--config", "tiny", "--training.steps", "3"])
    assert parsed.training.steps == 3


@pytest.mark.parametrize("manifest_path", ["data/prepared", "data/prepared/manifest.json"])
def test_training_paths_keep_launch_directory_semantics_in_saved_config(tmp_path, monkeypatch, manifest_path):
    from moeloop.recipes import tiny
    from torchtitan.trainer import Trainer

    recipe = tiny()
    recipe.dataloader.manifest_path = manifest_path
    recipe.hf_assets_path = "data/prepared/tokenizer"
    recipe.dump_folder = "outputs/moeloop"
    (tmp_path / "data/prepared").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    saved = {}

    class ReachedNativeTrainer(Exception):
        pass

    def capture_config(self, config):
        saved.update(config.to_dict())
        raise ReachedNativeTrainer

    # Check the config passed to the native trainer without initializing CUDA.
    monkeypatch.setattr(Trainer, "__init__", capture_config)
    with pytest.raises(ReachedNativeTrainer):
        LoopTrainer(recipe)
    assert saved["dataloader"]["manifest_path"] == str(tmp_path / "data/prepared/manifest.json")
    assert saved["hf_assets_path"] == str(tmp_path / "data/prepared/tokenizer")
    assert saved["dump_folder"] == str(tmp_path / "outputs/moeloop")


@pytest.mark.parametrize("log_freq", [1, 3])
def test_step_records_omit_unsampled_gradient_throughput_and_memory(tmp_path, log_freq):
    import json
    import math
    import time
    from contextlib import nullcontext
    from types import SimpleNamespace

    from torchtitan.components.metrics import BaseLogger
    from torchtitan.components.optimizer import LRSchedulersContainer
    from moeloop.titan import LoopMetrics

    torch.manual_seed(83)
    model_config = TitanModel.Config(core=config(), native_experts=False, flex_attention=False)
    model = TitanModel(model_config)
    model._skip_lm_head = True
    recipe = TrainingRecipe(num_visits=2)
    trainer = LoopTrainer.__new__(LoopTrainer)
    trainer.config = SimpleNamespace(recipe=recipe, dump_folder=str(tmp_path),
        training=SimpleNamespace(disable_cuda_graphs=True, max_norm=1.0))
    trainer.model_config, trainer.model_parts = model_config, [model]
    trainer.optimizers = torch.optim.SGD(model.parameters(), lr=.01)
    trainer.lr_schedulers = LRSchedulersContainer.Config(total_steps=4, warmup_steps=0).build(
        optimizers=[trainer.optimizers], training_steps=4)
    trainer.loss_fn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=1, recipe=recipe))
    trainer.loss_fn.set_lm_head(model.lm_head)
    trainer.parallel_dims = SimpleNamespace(dp_enabled=False, pp_enabled=False,
        dp_cp_enabled=False, ep_enabled=False, get_optional_mesh=lambda _: None)
    trainer.gradient_accumulation_steps = trainer.num_pipeline_parallel_microbatches = 1
    trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    trainer.train_context = nullcontext
    trainer.fwd_bwd_fn = trainer._forward_backward_body
    trainer.rank, trainer.device, trainer.ntokens_seen = 0, torch.device("cpu"), 0
    trainer.progress, trainer.ema = CommittedProgress(), None
    trainer.pending_counts = dict(padded_positions=0, input_tokens=0, label_tokens=0, recursive_input_tokens=0)
    trainer.dataloader = SimpleNamespace(dataset=None)

    # Only device-memory polling is a hardware boundary. Keep the real native
    # train_step, clipping, optimizer, should_log, LoopMetrics.log and JSON sink.
    metrics = LoopMetrics.__new__(LoopMetrics)
    metrics.config, metrics.logger = LoopMetrics.Config(log_freq=log_freq), BaseLogger()
    metrics.last_training, metrics.data_loading_times = {}, []
    metrics.ntokens_since_last_log, metrics.time_last_log = 0, time.perf_counter()
    metrics.device_memory_monitor = SimpleNamespace(
        get_peak_stats=lambda: SimpleNamespace(max_active_gib=13.0, max_reserved_gib=17.0),
        reset_peak_stats=lambda: None)
    trainer.metrics_processor = metrics
    batches = [({"input": torch.tensor([[2, 3, 4, 5]]), "positions": torch.arange(4).unsqueeze(0),
                 "document_ids": torch.zeros(1, 4, dtype=torch.int64),
                 "input_mask": torch.ones(1, 4, dtype=torch.bool)},
                torch.full((1, 4), label, dtype=torch.int64)) for label in (1, 8, 13, 4)]
    data_iterator = trainer.batch_generator(batches)
    for step in range(1, 5):
        trainer.step = step
        trainer.train_step(data_iterator)

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    sampled = {"grad_norm", "padded_positions_per_second_per_rank", "peak_active_gib", "peak_reserved_gib"}
    assert [record["global_progress"]["label_tokens"] for record in records] == [4, 8, 12, 16]
    for step, record in enumerate(records, 1):
        assert math.isfinite(record["ce"]) and math.isfinite(record["aux"])
        if step == 1 or step % log_freq == 0:
            assert sampled <= record.keys()
            assert record["grad_norm"] > 0
        else:
            assert sampled.isdisjoint(record)
