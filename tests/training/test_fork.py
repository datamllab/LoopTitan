"""WSD cooldown branch: a fork starts a new lineage from a parent's checkpoint at step S
with a new decay tail. The tail starts at or after both S and the end of the parent's
warmup, and the fork's LRs through S equal the parent's.

Every run goes through the real launcher (recipe_experiments.py), config factory
(experiment.py), LoopTrainer continuation code and native CheckpointManager on one
CPU gloo rank. Only the CUDA parts of the pinned Trainer are replaced: its
__init__ builds the same components on CPU without FSDP or meta initialization,
and the torchrun child runs in this process.

Parent: 12-update horizon, warmup 2, stable through index 9, then cosine to 0.1;
it stops at update 6 and keeps full checkpoints at 4 and 6. Forks start at S = 4.
"""

from contextlib import nullcontext
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

import recipe_experiments as runner
from recipe_qualification import digest
from torchtitan.trainer import Trainer

from moeloop.experiment import experiment
from moeloop.prepare import prepare_dataset
from moeloop.recipes import smelt_spec
from moeloop.titan import LoopTrainer, lineage_free
from moeloop.training import continuation_fingerprint

S = 4
RUN = {"seq_len": 8, "global_batch_positions": 16, "local_batch_size": 2, "seed": 7,
       "successful_updates": 12, "schedule_total_steps": 12, "checkpoint_interval": 2,
       "schedule": {"total_steps": 12, "warmup_steps": 2, "decay_ratio": .25,
                    "decay_type": "cosine", "min_lr_factor": .1}}


def cpu_native_init(self, config):
    """Pinned Trainer.__init__ for one CPU rank: the same parts, no CUDA, FSDP or meta init.

    Mirrors torchtitan/trainer.py lines 288-617 at the pin; re-check on a pin bump.
    """
    self.config, self.model_config = config, config.model_spec.model
    self.parallel_dims = SimpleNamespace(dp_enabled=False, pp_enabled=False, cp_enabled=False,
        dp_cp_enabled=False, ep_enabled=False, get_optional_mesh=lambda name: None)
    self.device, self.train_context = torch.device("cpu"), nullcontext
    self.gradient_accumulation_steps = config.training.global_batch_size // config.training.local_batch_size
    self.num_pipeline_parallel_microbatches = 1
    torch.manual_seed(config.debug.seed)
    model = self.model_config.build()
    model.init_weights()
    self.model_parts = [model.train()]
    self.loss_fn = config.loss.build(compile_config=config.compile)
    self.loss_fn.set_lm_head(model.lm_head)
    model._skip_lm_head = True
    self.fwd_bwd_fn = self._forward_backward_body
    self.metrics_processor = SimpleNamespace(should_log=lambda step: False,
                                             ntokens_since_last_log=0, data_loading_times=[])
    self.optimizers = config.optimizer.build(model_parts=self.model_parts)
    config.model_spec.post_optimizer_build_fn(self.optimizers, self.model_parts, self.parallel_dims)
    self.lr_schedulers = config.lr_scheduler.build(optimizers=self.optimizers, training_steps=config.training.steps)
    self.step = self.ntokens_seen = 0
    self.dataloader = config.dataloader.build(dp_world_size=1, dp_rank=0, tokenizer=None,
        seq_len=config.training.seq_len, local_batch_size=config.training.local_batch_size)
    self.checkpointer = config.checkpoint.build(dataloader=self.dataloader, model_parts=self.model_parts,
        optimizers=self.optimizers, lr_schedulers=self.lr_schedulers, states={"train_state": self},
        sd_adapter=None, base_folder=config.dump_folder)


def full_state(trainer):
    """Model and FQN-keyed optimizer state, copied."""
    copy = lambda value: value.detach().clone() if torch.is_tensor(value) else value
    return {"model": {k: copy(v) for k, v in trainer.model_parts[0].state_dict().items()},
            "optimizer": {k: copy(v) for k, v in trainer.optimizers.state_dict().items()}}


def native_train(self):
    """Pinned Trainer.train without profiler, GC and timeouts. Records the state as
    loaded and after each update, and the LR factor each update ran at.

    Mirrors torchtitan/trainer.py lines 1013-1067 at the pin; re-check on a pin bump.
    """
    self.loaded = full_state(self) if self.checkpointer.load(step=self.config.checkpoint.load_step) else None
    self.factors, self.states = {}, {}
    data = self.batch_generator(self.dataloader)
    while self.should_continue_training():
        self.step += 1
        self.factors[self.step] = [group["lr"] / group["initial_lr"] for optimizer in self.optimizers
                                   for group in optimizer.param_groups]
        self.train_step(data)
        self.checkpointer.save(self.step, last_step=self.step == self.config.training.steps)
        self.states[self.step] = full_state(self)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """One gloo rank, prepared data, the parent run and forks A and B from its step S.

    ``trainers`` maps each output directory name to its launches, in order.
    """
    import torch.distributed as dist
    root = tmp_path_factory.mktemp("fork")
    lengths = [str(n) for n in (9, 4, 13, 6, 17, 5, 8, 11, 3, 12)]
    manifest = prepare_dataset(root / "data", {"a": {"train": lengths, "validation": lengths}},
        encode=lambda text: [3 + (5 * i + int(text)) % 16 for i in range(int(text))],
        tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 19, "bos_token_id": 1, "eos_token_id": 2})
    spec = smelt_spec(physical_layers=4, vocab_size=19, dim=16, hidden_dim=24, num_heads=2,
                      num_kv_heads=1, num_experts=2, top_k=1, max_seq_len=8)
    spec = replace(spec, run=spec.run | RUN)
    (root / "parent.json").write_text(json.dumps(spec.to_dict()))
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
            runner.run(root / "parent.json", manifest.path, root / "parent", gpus=1, stop=6)
            lab.checkpoint = root / "parent/checkpoint" / f"step-{S}"
            lab.parent_metadata = digest(lab.checkpoint / ".metadata")
            runner.fork(lab.checkpoint, root / "fork-a", decay_steps=4, decay_type="linear", gpus=1)
            runner.fork(lab.checkpoint, root / "fork-b", decay_steps=6, decay_type="cosine",
                        min_lr_factor=0.0, gpus=1)
            yield lab
        finally:
            dist.destroy_process_group()


def assert_same_state(actual, expected, *, param_groups=True):
    """Bitwise model and optimizer state; without ``param_groups``, only tensors
    (the groups hold the LR stored for the next update)."""
    assert actual["model"].keys() == expected["model"].keys()
    assert actual["optimizer"].keys() == expected["optimizer"].keys()
    for part in ("model", "optimizer"):
        for key, value in expected[part].items():
            if torch.is_tensor(value):
                assert torch.equal(actual[part][key], value), key
            elif param_groups:
                assert actual[part][key] == value, key


def tail(update, *, length, kind, floor):
    """Oracle: factor of a 1-based update after a tail from S (native LambdaLR indexing).

    Update u runs at index u-1; indices up to S stay at peak, then progress (u-1-S)/L
    goes from 1 toward ``floor``, which index S+L (one past the last update) reaches.
    """
    progress = max(update - 1 - S, 0) / length
    shape = 1 - progress if kind == "linear" else .5 * (1 + math.cos(math.pi * progress))
    return floor + (1 - floor) * shape


def test_fork_loads_the_parent_state_exactly_then_follows_its_new_tail(lab):
    parent, fork = lab.trainers["parent"][0], lab.trainers["fork-a"][0]
    assert_same_state(fork.loaded, parent.states[S])  # bit-identical model and optimizer at S
    # Parent: warmup over 2 updates, then peak through update 6.
    for update, expected in zip(range(1, 7), [.5, 1, 1, 1, 1, 1]):
        assert parent.factors[update] == pytest.approx([expected] * len(parent.factors[update]), rel=1e-12)
    # Fork A: linear tail of 4 updates to the parent's floor .1 -> 1, .775, .55, .325.
    assert list(fork.factors) == [5, 6, 7, 8]
    for update, expected in zip(fork.factors, [1, .775, .55, .325]):
        assert expected == pytest.approx(tail(update, length=4, kind="linear", floor=.1))
        assert fork.factors[update] == pytest.approx([expected] * len(fork.factors[update]), rel=1e-12)
    # Update S+1 still runs at peak on the parent's data cursor and RNG: the same update.
    assert_same_state(fork.states[S + 1], parent.states[S + 1], param_groups=False)
    assert not torch.equal(fork.states[S + 2]["model"]["core.embedding.weight"],
                           parent.states[S + 2]["model"]["core.embedding.weight"])


def test_fork_records_name_the_parent_and_its_lineage_identity_differs(lab):
    parent_dir, fork_dir = lab.root / "parent", lab.root / "fork-a"
    record = {"checkpoint": str(lab.checkpoint), "metadata_sha256": lab.parent_metadata, "step": S}
    spec = json.loads((fork_dir / "spec.json").read_text())
    assert spec["run"]["fork"] == record
    assert spec["run"]["successful_updates"] == spec["run"]["schedule_total_steps"] == S + 4
    assert spec["run"]["schedule"] == {"total_steps": 8, "warmup_steps": 2, "stable_end_steps": S,
        "decay_steps": 4, "decay_ratio": .5, "decay_type": "linear", "min_lr_factor": .1}
    assert json.loads((fork_dir / "request.json").read_text())["spec_sha256"] == digest(fork_dir / "spec.json")
    identity = json.loads((fork_dir / "continuation_identity.json").read_text())
    parent_identity = json.loads((parent_dir / "continuation_identity.json").read_text())
    assert identity["fork"] == identity["recipe_metadata"]["run"]["fork"] == record
    assert parent_identity["fork"] == {}
    # The fork and its parent differ only in the schedule and the lineage record.
    assert lineage_free(identity) == lineage_free(parent_identity)
    assert set(identity) - set(lineage_free(identity)) == {"lr_scheduler", "fork"}
    assert (set(identity["recipe_metadata"]["run"]) - set(lineage_free(identity)["recipe_metadata"]["run"])
            == {"successful_updates", "schedule_total_steps", "schedule", "fork"})
    outcome = json.loads((fork_dir / "outcome.json").read_text())
    assert outcome["completed_updates"] == S + 4
    assert outcome["continuation_fingerprint"] == continuation_fingerprint(identity)
    assert outcome["continuation_fingerprint"] != continuation_fingerprint(parent_identity)
    # The parent checkpoint was retained unchanged in the fork's own directory.
    protected = fork_dir / "protected" / f"step-{S}"
    assert digest(protected / ".metadata") == digest(lab.checkpoint / ".metadata") == lab.parent_metadata
    assert json.loads((protected / "retention.json").read_text())["original_path"] == str(lab.checkpoint)
    assert len((parent_dir / "metrics.jsonl").read_text().splitlines()) == 6
    # The same command continues the same lineage from the fork's own last checkpoint.
    runner.fork(lab.checkpoint, fork_dir, decay_steps=4, decay_type="linear", gpus=1)
    rerun = lab.trainers["fork-a"][-1]
    assert rerun.step == S + 4 and rerun.factors == {}
    assert_same_state(rerun.loaded, lab.trainers["fork-a"][0].states[S + 4])


def write_spec(path, schedule):
    spec = json.loads((path.parent / "parent.json").read_text())
    spec["run"]["schedule"] |= schedule
    path.write_text(json.dumps(spec))
    return path


def test_strict_resume_with_a_changed_schedule_is_still_refused(lab):
    changed = write_spec(lab.root / "changed.json", {"decay_type": "linear"})
    # The launcher refuses a changed spec in the parent's directory ...
    with pytest.raises(ValueError, match="identity differs"):
        runner.run(changed, lab.manifest, lab.root / "parent", gpus=1, resume=lab.checkpoint)
    # ... and the trainer refuses the parent's state under the changed schedule elsewhere.
    with pytest.raises(ValueError, match="strict continuation identity changed"):
        runner.run(changed, lab.manifest, lab.root / "resume-changed", gpus=1, stop=6, resume=lab.checkpoint)
    # A fork whose optimizer differs from the parent's is refused: only the schedule may change.
    changed_optimizer = json.loads((lab.root / "fork-a/spec.json").read_text())
    changed_optimizer["optimizer"]["weight_decay"] = .05
    (lab.root / "fork-weight-decay.json").write_text(json.dumps(changed_optimizer))
    with pytest.raises(ValueError, match="differs beyond the LR schedule"):
        runner.run(lab.root / "fork-weight-decay.json", lab.manifest, lab.root / "fork-weight-decay",
                   gpus=1, resume=lab.checkpoint)
    # A fork spec launched without its parent checkpoint would claim a parent it never loaded.
    with pytest.raises(ValueError, match="launch it with"):
        runner.run(lab.root / "fork-a/spec.json", lab.manifest, lab.root / "fork-unloaded", gpus=1)
    # The unchanged schedule still resumes the parent strictly.
    runner.run(lab.root / "parent.json", lab.manifest, lab.root / "parent", gpus=1, stop=8,
               resume=lab.root / "parent/checkpoint/step-6")
    assert lab.trainers["parent"][-1].step == 8


def test_tail_starts_at_or_after_the_fork_step_and_warmup(lab):
    """Parent W: warmup 2, horizon 6, decay over indices 4..5, a checkpoint every update."""
    spec = json.loads((lab.root / "parent.json").read_text())
    spec["run"] |= {"successful_updates": 6, "schedule_total_steps": 6, "checkpoint_interval": 1,
        "schedule": {"total_steps": 6, "warmup_steps": 2, "decay_ratio": .5, "decay_type": "linear",
                     "min_lr_factor": .1}}
    (lab.root / "parent-w.json").write_text(json.dumps(spec))
    runner.run(lab.root / "parent-w.json", lab.manifest, lab.root / "parent-w", gpus=1, stop=2)
    warmup = lab.root / "parent-w/checkpoint/step-1"
    # Launcher only: from step 1 with the default start 1 < W, the tail would overlap warmup
    # and the native builder would shorten it, though the LRs through S still match.
    with pytest.raises(ValueError, match="end of warmup"):
        runner.fork(warmup, lab.root / "fork-w-default", decay_steps=3, decay_type="linear", gpus=1)
    assert not (lab.root / "fork-w-default").exists()
    # A tail after warmup from the same warmup checkpoint is accepted and runs as recorded.
    # Here the tail starts exactly at the end of warmup.
    runner.fork(warmup, lab.root / "fork-w-late", decay_steps=3, decay_type="linear",
                decay_start=2, gpus=1)
    fork = lab.trainers["fork-w-late"][0]
    assert_same_state(fork.loaded, lab.trainers["parent-w"][0].states[1])
    assert list(fork.factors) == [2, 3, 4, 5]
    for update, expected in zip(fork.factors, [1, 1, .7, .4]):
        assert fork.factors[update] == pytest.approx([expected] * len(fork.factors[update]), rel=1e-12)
    # A tail before the fork step is refused before anything is written.
    with pytest.raises(ValueError, match="fork step"):
        runner.fork(lab.checkpoint, lab.root / "fork-early", decay_steps=4, decay_type="linear",
                    decay_start=S - 1, gpus=1)
    assert not (lab.root / "fork-early").exists()
    # From a decay-phase checkpoint the LRs through S differ; the trainer refuses at load.
    runner.run(lab.root / "parent-w.json", lab.manifest, lab.root / "parent-w", gpus=1, stop=5,
               resume=lab.root / "parent-w/checkpoint/step-2")
    # Step 4: updates 1..4 ran at peak, but the restored optimizer carries index 4's decayed
    # LR (.7) for update 5, where the fork's schedule would record peak.
    with pytest.raises(ValueError, match="changes LRs the parent used"):
        runner.fork(lab.root / "parent-w/checkpoint/step-4", lab.root / "fork-w-decay-start",
                    decay_steps=3, decay_type="linear", gpus=1)
    with pytest.raises(ValueError, match="changes LRs the parent used"):
        runner.fork(lab.root / "parent-w/checkpoint/step-5", lab.root / "fork-w-decay", decay_steps=3,
                    decay_type="linear", gpus=1)


def test_two_forks_with_different_tails_from_one_parent_are_independent(lab):
    parent, a, b = (lab.trainers[name][0] for name in ("parent", "fork-a", "fork-b"))
    assert_same_state(b.loaded, parent.states[S])  # fork A did not touch the parent state
    assert list(b.factors) == list(range(S + 1, S + 7))
    for update in b.factors:
        expected = tail(update, length=6, kind="cosine", floor=0.)
        assert b.factors[update] == pytest.approx([expected] * len(b.factors[update]), rel=1e-12)
    assert b.factors[S + 2] != a.factors[S + 2]
    assert not torch.equal(a.states[S + 4]["model"]["core.embedding.weight"],
                           b.states[S + 4]["model"]["core.embedding.weight"])
    identities = [json.loads((lab.root / name / "continuation_identity.json").read_text())
                  for name in ("parent", "fork-a", "fork-b")]
    assert len({continuation_fingerprint(identity) for identity in identities}) == 3
    assert identities[1]["fork"] == identities[2]["fork"]
    assert digest(lab.checkpoint / ".metadata") == lab.parent_metadata
