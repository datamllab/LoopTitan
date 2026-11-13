"""EMA of the weights: off by default; with ``run.ema_decay`` it is updated after every
optimizer update, saved with every checkpoint and restored on resume, validated beside the
model and exportable.

The single-rank runs go through the real launcher (recipe_experiments.py), experiment(),
LoopTrainer, LoopValidator and CheckpointManager on one CPU gloo rank, with test_fork's CPU
replacement of the pinned Trainer.__init__ plus the validator it builds. The FSDP test runs
this file as a script on 1, 2 and 4 gloo ranks.
"""

from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("torchtitan")

import recipe_experiments as runner
from test_fork import cpu_native_init as fork_init, full_state
from torchtitan.tools.utils import NoColor
from torchtitan.trainer import Trainer

from moeloop.export import export_checkpoint
from moeloop.model import ModelConfig
from moeloop.prepare import prepare_dataset
from moeloop.recipes import smelt_spec, tiny_spec, trainer_config
from moeloop.titan import LoopMetrics, LoopTrainer, canonical_model_state, ema_weights, update_ema
from moeloop.training import TrainingRecipe

# The byte-level BPE fixture of tests/data/test_bpb.py, padded to 384 IDs, so validation has bytes.
FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "bpb"
TEXTS = ["hello world", "naïve café — 你好，世界 🙂", "tabs\tand\nnewlines  and   spaces",
         "quick 🦊 jumps over the lazy dog", "validation must not depend on the number of ranks"]
TOKENIZER = {"id": "bpb-fixture", "revision": "fixed-v1", "vocab_size": 384, "bos_token_id": 0, "eos_token_id": 0,
             "pad_token_id": 0}
DECAY = .75
RUN = {"seq_len": 8, "global_batch_positions": 16, "local_batch_size": 2, "seed": 7, "successful_updates": 6,
       "schedule_total_steps": 6, "checkpoint_interval": 2, "validation_interval": 2, "validation_global_batches": 1,
       "schedule": {"total_steps": 6, "warmup_steps": 1, "decay_ratio": None, "decay_type": "cosine",
                    "min_lr_factor": .1}}
BPB_FIELDS = ("label_bytes", "ce_sum_text", "bpb", "byte_table_sha256", "bpb_unavailable")


def prepare(path):
    """Prepared text with the fixture's tokenizer assets, an HF config added so export can load them."""
    assets = path.parent / f"{path.name}-tokenizer"
    assets.mkdir()
    (assets / "tokenizer.json").write_bytes((FIXTURE / "tokenizer.json").read_bytes())
    eos = "<|endoftext|>"
    (assets / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": eos, "eos_token": eos, "pad_token": eos}))
    encoder = tokenizers.Tokenizer.from_file(str(assets / "tokenizer.json"))
    return prepare_dataset(path, {"text": {"train": TEXTS * 4, "validation": TEXTS * 2}},
        encode=lambda text: encoder.encode(text, add_special_tokens=False).ids, tokenizer=TOKENIZER,
        tokenizer_directory=assets)


def metrics(output):
    """A LoopMetrics writing to ``output`` without its CUDA memory monitor (as in test_titan);
    ``logged`` holds every (step, values) it logs."""
    processor = LoopMetrics.__new__(LoopMetrics)
    processor.output_directory, processor.last_validation, processor.ema_pass = Path(output), {}, False
    processor.ntokens_since_last_log, processor.time_last_log = 0, time.perf_counter()
    processor.parallel_dims, processor.color = SimpleNamespace(non_data_parallel_size=1), NoColor()
    processor.device_memory_monitor = SimpleNamespace(get_peak_stats=lambda: SimpleNamespace(
        max_active_gib=0., max_active_pct=0., max_reserved_gib=0., max_reserved_pct=0.), reset_peak_stats=lambda: None)
    processor.logged = []
    processor.logger = SimpleNamespace(log=lambda values, step: processor.logged.append((step, values)))
    return processor


def logged(processor, key):
    return [(step, values[key]) for step, values in processor.logged if key in values]


def full(tensors):
    return {name: (value.full_tensor() if hasattr(value, "full_tensor") else value).detach().clone()
            for name, value in tensors.items()}


def reference_ema(trajectory, decay):
    """float64 oracle: e_0 = the initial weights, e_t = decay * e_(t-1) + (1 - decay) * w_t."""
    ema = {name: value.double() for name, value in trajectory[0].items()}
    for weights in trajectory[1:]:
        ema = {name: decay * ema[name] + (1 - decay) * weights[name].double() for name in ema}
    return ema


def cpu_native_init(self, config):
    """test_fork's CPU Trainer.__init__, plus the validator the pinned one builds."""
    fork_init(self, config)
    self.validator = config.validator.build(parallelism=config.parallelism, dp_world_size=1, dp_rank=0,
        tokenizer=None, parallel_dims=SimpleNamespace(pp_enabled=False, dp_enabled=False, dp_cp_enabled=False),
        loss_fn=self.loss_fn, validation_context=nullcontext, metrics_processor=metrics(config.dump_folder),
        seq_len=config.training.seq_len, local_batch_size=config.training.local_batch_size)


def native_train(self):
    """Pinned Trainer.train without profiler, GC and timeouts: update, save, then validate.
    Records the state as loaded and after each update, with the EMA.

    Mirrors torchtitan/trainer.py lines 1013-1067 at the pin; re-check on a pin bump.
    """
    snapshot = lambda: {**full_state(self), "ema": None if self.ema is None else full(self.ema)}
    self.loaded = snapshot() if self.checkpointer.load(step=self.config.checkpoint.load_step) else None
    self.initial, self.states = snapshot(), {}
    data = self.batch_generator(self.dataloader)
    while self.should_continue_training():
        self.step += 1
        self.train_step(data)
        self.checkpointer.save(self.step, last_step=self.step == self.config.training.steps)
        if self.validator.should_validate(self.step):
            self.validator.validate(self.model_parts, self.step)
        self.states[self.step] = snapshot()


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """One gloo rank and these runs of a small SMELT spec, each in its own directory:

    ema: 6 updates with ema_decay; resumed: the same, stopped at 4 and resumed from its
    step-4 checkpoint; plain: the same spec without ema_decay. ``trainers`` maps each
    directory name to its launches, in order.
    """
    import torch.distributed as dist
    root = tmp_path_factory.mktemp("ema")
    manifest = prepare(root / "data")
    spec = smelt_spec(physical_layers=4, vocab_size=384, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1,
                      num_experts=2, top_k=1, max_seq_len=8)
    spec = replace(spec, run=spec.run | RUN)
    (root / "plain.json").write_text(json.dumps(spec.to_dict()))
    (root / "ema.json").write_text(json.dumps(replace(spec, run=spec.run | {"ema_decay": DECAY}).to_dict()))
    trainers = {}

    def launch(argv, directory, *, env, **kwargs):
        """The torchrun child of recipe_experiments.launch, run here on the gloo rank."""
        from moeloop.experiment import experiment
        with pytest.MonkeyPatch.context() as patch:
            for name in ("SPEC", "MANIFEST", "OUTPUT", "STOP", "RESUME"):
                key = "MOELOOP_EXPERIMENT_" + name
                if key in env:
                    patch.setenv(key, env[key])
                else:
                    patch.delenv(key, raising=False)
            config = experiment()
            config.model_spec.model.flex_attention = False  # flex attention binds a GPU kernel
            trainer = LoopTrainer(config)
            trainers.setdefault(Path(directory).name, []).append(trainer)
            trainer.train()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Trainer, "__init__", cpu_native_init)
        patch.setattr(Trainer, "train", native_train)
        patch.setattr(runner, "run_command", launch)
        patch.setattr(runner, "runtime_identity", lambda python: {"torch": torch.__version__})
        patch.setattr("torchtitan.components.validate.utils.device_type", "cpu")
        dist.init_process_group("gloo", init_method=f"file://{root}/init", rank=0, world_size=1)
        try:
            lab = SimpleNamespace(root=root, manifest=manifest.path, trainers=trainers)
            runner.run(root / "ema.json", manifest.path, root / "ema", gpus=1)
            runner.run(root / "ema.json", manifest.path, root / "resumed", gpus=1, stop=4)
            runner.run(root / "ema.json", manifest.path, root / "resumed", gpus=1,
                       resume=root / "resumed/checkpoint/step-4")
            runner.run(root / "plain.json", manifest.path, root / "plain", gpus=1)
            yield lab
        finally:
            dist.destroy_process_group()


def assert_equal_tensors(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(actual[key], value), key


def test_ema_is_off_by_default_and_the_decay_enters_the_identity(lab):
    assert trainer_config(tiny_spec()).ema_decay is None
    with pytest.raises(ValueError, match="ema_decay"):
        replace(trainer_config(tiny_spec()), ema_decay=1.0)
    plain, ema = lab.trainers["plain"][0], lab.trainers["ema"][0]
    assert plain.ema is None and "ema_decay" not in plain.identity
    assert ema.identity["ema_decay"] == ema.identity["recipe_metadata"]["run"]["ema_decay"] == DECAY
    import torch.distributed.checkpoint as dcp
    keys = {name: set(dcp.FileSystemReader(lab.root / name / "checkpoint/step-6").read_metadata().state_dict_metadata)
            for name in ("plain", "ema")}
    assert not any(key.startswith("train_state.ema.") for key in keys["plain"])
    ema_keys = {key.removeprefix("train_state.ema.") for key in keys["ema"] if key.startswith("train_state.ema.")}
    assert ema_keys == {name for name, _ in ema.model_parts[0].named_parameters()}
    assert keys["ema"] - {f"train_state.ema.{key}" for key in ema_keys} == keys["plain"]


def test_ema_follows_the_hand_computed_average_and_leaves_training_unchanged(lab):
    plain, ema = lab.trainers["plain"][0], lab.trainers["ema"][0]
    trajectory = [ema.initial["model"]] + [ema.states[step]["model"] for step in range(1, 7)]
    assert_equal_tensors(ema.initial["ema"], trajectory[0])  # e_0: the initialized weights
    for step in range(1, 7):
        expected = reference_ema(trajectory[:step + 1], DECAY)
        for name, value in ema.states[step]["ema"].items():
            assert value.dtype == torch.float32
            torch.testing.assert_close(value.double(), expected[name], rtol=1e-6, atol=1e-8, msg=name)
        # Keeping, validating and saving the EMA changes no model or optimizer tensor.
        assert_equal_tensors(ema.states[step]["model"], plain.states[step]["model"])
        assert_equal_tensors(ema.states[step]["optimizer"], plain.states[step]["optimizer"])
    assert not torch.equal(ema.states[6]["ema"]["core.embedding.weight"], ema.states[6]["model"]["core.embedding.weight"])


def test_resume_with_ema_equals_the_uninterrupted_run_bitwise(lab):
    uninterrupted, (first, resumed) = lab.trainers["ema"][0], lab.trainers["resumed"]
    assert first.step == 4 and resumed.loaded is not None
    for part in ("model", "optimizer", "ema"):
        assert_equal_tensors(resumed.loaded[part], uninterrupted.states[4][part])
        for step in (5, 6):
            assert_equal_tensors(resumed.states[step][part], uninterrupted.states[step][part])
    ema_records = [json.loads(line) for line in (lab.root / "ema/validation.jsonl").read_text().splitlines()]
    resumed_records = [json.loads(line) for line in (lab.root / "resumed/validation.jsonl").read_text().splitlines()]
    assert resumed_records == ema_records  # steps 1, 2, 4 before the stop; 6 after the resume


def test_resuming_with_another_decay_or_toggled_ema_is_refused(lab):
    spec = json.loads((lab.root / "ema.json").read_text())
    spec["run"]["ema_decay"] = .5
    (lab.root / "other.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="strict continuation identity changed"):
        runner.run(lab.root / "other.json", lab.manifest, lab.root / "other", gpus=1,
                   resume=lab.root / "ema/checkpoint/step-4")
    with pytest.raises(ValueError, match="strict continuation identity changed"):  # EMA turned off
        runner.run(lab.root / "plain.json", lab.manifest, lab.root / "ema-off", gpus=1,
                   resume=lab.root / "ema/checkpoint/step-4")
    # EMA turned on: DCP refuses before the identity check, since the checkpoint holds no EMA.
    from torch.distributed.checkpoint.api import CheckpointException
    with pytest.raises(CheckpointException) as refused:
        runner.run(lab.root / "ema.json", lab.manifest, lab.root / "ema-on", gpus=1,
                   resume=lab.root / "plain/checkpoint/step-4")
    assert "Missing key in checkpoint state_dict: train_state.ema." in str(refused.value.failures[0])


def test_validation_reports_the_model_and_its_ema(lab):
    plain, ema = lab.trainers["plain"][0], lab.trainers["ema"][0]
    plain_records, ema_records = ([json.loads(line) for line in (lab.root / name / "validation.jsonl").read_text().splitlines()]
                                  for name in ("plain", "ema"))
    assert [record["step"] for record in ema_records] == [1, 2, 4, 6]
    for plain_record, record in zip(plain_records, ema_records, strict=True):
        assert {key: value for key, value in record.items() if key != "ema"} == plain_record
        assert set(record["ema"]) == {"ce", *BPB_FIELDS}
        assert record["ema"]["label_bytes"] == record["label_bytes"] > 0
        assert record["ema"]["byte_table_sha256"] == record["byte_table_sha256"]
        assert record["ema"]["ce"] != record["ce"] and record["ema"]["bpb"] != record["bpb"]
    # The model's log is unchanged; the EMA pass logs its own names.
    for key in ("validation_metrics/loss", "validation_metrics/bpb"):
        assert logged(ema.validator.metrics_processor, key) == logged(plain.validator.metrics_processor, key)
    for key, field in (("validation_metrics/ema_loss", "ce"), ("validation_metrics/ema_bpb", "bpb")):
        assert logged(ema.validator.metrics_processor, key) == [(record["step"], record["ema"][field])
                                                                for record in ema_records]
    # Oracle: the model's own validation on the EMA weights of step 6 gives the EMA fields.
    model = ema.model_parts[0]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(ema.states[6]["ema"][name])
    oracle = ema.validator
    oracle.ema, oracle.metrics_processor = None, metrics(lab.root / "oracle")
    oracle.validate([model], 6)
    record = json.loads((lab.root / "oracle/validation.jsonl").read_text())
    assert {key: record[key] for key in ("ce", *BPB_FIELDS)} == ema_records[-1]["ema"]


def test_export_writes_the_ema_or_the_model_weights(lab):
    from moeloop.hf import MoeloopForCausalLM
    ema = lab.trainers["ema"][0]
    checkpoint = lab.root / "ema/checkpoint/step-6"
    for weights in ("ema", "model"):
        output = export_checkpoint(checkpoint, lab.root / f"export-{weights}", weights=weights)
        provenance = json.loads((output / "export_provenance.json").read_text())
        assert provenance["checkpoint"]["weights"] == weights
        exported = MoeloopForCausalLM.from_pretrained(output, local_files_only=True).model.state_dict()
        assert_equal_tensors(exported, canonical_model_state(ema.states[6][weights]))
    with pytest.raises(ValueError, match="no EMA weights"):
        export_checkpoint(lab.root / "plain/checkpoint/step-6", lab.root / "export-plain", weights="ema")


# The FSDP check. Root-owned input-adapter weights stay gathered after an eval forward.
SEQ_LEN, GLOBAL_BATCH, LOCAL_BATCH, UPDATES = 8, 4, 1, 3
MODEL = ModelConfig(vocab_size=384, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, num_experts=2, top_k=1,
                    num_cells=1, num_visits=2, prefix_cells=1, max_seq_len=SEQ_LEN, input_injection="linear")
RECIPE = TrainingRecipe(num_visits=2, balance_coeff=0.3, z_loss_coeff=0.05)


def run_rank(rank, world, data, output):
    """Train UPDATES steps twice from the same weights, keeping the EMA, once validating
    the model and its EMA after every update and once without validation."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.utils import set_spmd_backend
    from torchtitan.tools import utils
    from moeloop.titan import LoopChunkedLoss, LoopValidator, PackedDataLoader, TitanModel, parallelize_model, prepare_batch

    torch.set_num_threads(1)
    utils.device_type = "cpu"  # this process has no accelerator
    dist.init_process_group("gloo", init_method=f"file://{output}/init", rank=rank, world_size=world)
    set_spmd_backend("partial_dtensor")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    parallel_dims = SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, pp_enabled=False,
        dp_enabled=True, dp_cp_enabled=True, get_mesh=lambda names: mesh, get_optional_mesh=lambda names: mesh)

    def build(directory):
        torch.manual_seed(5)
        model = parallelize_model(TitanModel(TitanModel.Config(core=MODEL, native_experts=False, flex_attention=False)),
            parallel_dims=parallel_dims, training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
            parallelism=ParallelismConfig(spmd_backend="partial_dtensor"), compile_config=CompileConfig(enable=False),
            ac_config=None, dump_folder=str(output))
        for module in model.modules():
            if isinstance(module, FSDPModule):  # gloo lacks PREMUL_SUM; native divide factor is 1
                module.set_force_sum_reduction_for_comms(True)
        model._skip_lm_head = True
        loss_fn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=2, recipe=RECIPE))
        loss_fn.set_lm_head(model.lm_head)
        loss_fn.dp_degree, loss_fn.accumulation = world, GLOBAL_BATCH // (world * LOCAL_BATCH)
        validator = LoopValidator(LoopValidator.Config(global_batches=1, dataloader=PackedDataLoader.Config(
                manifest_path=str(data), split="validation", shuffle=False, infinite=False, global_batch_size=GLOBAL_BATCH)),
            parallelism=None, dp_world_size=world, dp_rank=rank, tokenizer=None, parallel_dims=parallel_dims,
            loss_fn=loss_fn, validation_context=nullcontext, metrics_processor=metrics(directory),
            seq_len=SEQ_LEN, local_batch_size=LOCAL_BATCH)
        return model, loss_fn, validator

    def train(validate):
        directory = output / ("validated" if validate else "quiet")
        model, loss_fn, validator = build(directory)
        ema = validator.ema = ema_weights(model)
        placements = all(ema[name].placements == parameter.placements for name, parameter in model.named_parameters())
        loader = iter(PackedDataLoader(PackedDataLoader.Config(manifest_path=str(data), global_batch_size=GLOBAL_BATCH),
            dp_rank=rank, dp_world_size=world, tokenizer=None, seq_len=SEQ_LEN, local_batch_size=LOCAL_BATCH))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        record = {"losses": [], "weights": [full(dict(model.named_parameters()))], "ema": [], "placements": placements}
        for step in range(1, UPDATES + 1):
            optimizer.zero_grad()  # TorchTitan's train_step: one global label count, SUM gradients
            batches = [next(loader) for _ in range(loss_fn.accumulation)]
            valid = sum((labels != -100).sum() for _, labels in batches)
            dist.all_reduce(valid)
            objective = torch.zeros(())
            for input_dict, labels in batches:
                inputs, labels, kwargs = prepare_batch(input_dict, labels, flex_attention=False)
                loss, _ = loss_fn(model(inputs, **kwargs), labels, valid)
                loss.backward()
                objective += loss.detach()
            dist.all_reduce(objective)
            optimizer.step()
            update_ema(ema, model, DECAY)
            if validate:
                validator.validate([model], step)
            record["losses"].append(objective)
            record["weights"].append(full(dict(model.named_parameters())))
            record["ema"].append(full(ema))
        if validate:  # Oracle: a model with the last EMA weights, validated without an EMA.
            oracle, _, oracle_validator = build(output / "oracle")
            with torch.no_grad():
                for name, parameter in oracle.named_parameters():
                    parameter.copy_(ema[name])
            oracle_validator.validate([oracle], UPDATES)
            record["logged"] = logged(validator.metrics_processor, "validation_metrics/ema_loss")
        return record

    validated, quiet = train(True), train(False)
    if rank == 0:
        torch.save({"validated": validated, "quiet": quiet}, output / "records.pt")
    dist.destroy_process_group()


def run(world, data, output):
    output.mkdir()
    root = Path(__file__).resolve().parents[2]  # the script imports test_fork, which imports scripts/
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "scripts")]), "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, __file__, str(world), str(data), str(output)], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    read = lambda name: [json.loads(line) for line in (output / name / "validation.jsonl").read_text().splitlines()]
    return torch.load(output / "records.pt"), read("validated"), read("oracle")[0]


@pytest.mark.skipif(sys.platform != "linux", reason="the CPU FSDP probe uses Linux fork")
def test_one_two_four_fsdp_ranks_keep_the_same_ema_and_validate_it(tmp_path):
    data = prepare(tmp_path / "data").path
    runs = {world: run(world, data, tmp_path / f"world-{world}") for world in (1, 2, 4)}
    for world, (records, validation, oracle) in runs.items():
        validated, quiet = records["validated"], records["quiet"]
        assert validated["placements"]  # each EMA tensor is sharded like its parameter
        # Validating the model and its EMA leaves training and the EMA unchanged.
        for update in range(UPDATES):
            assert torch.equal(validated["losses"][update], quiet["losses"][update]), (world, update)
            assert_equal_tensors(validated["weights"][update + 1], quiet["weights"][update + 1])
            assert_equal_tensors(validated["ema"][update], quiet["ema"][update])
            expected = reference_ema(validated["weights"][:update + 2], DECAY)
            for name, value in validated["ema"][update].items():
                torch.testing.assert_close(value.double(), expected[name], rtol=1e-6, atol=1e-8, msg=name)
        # The EMA fields are the EMA weights' own validation, through the same BPB helpers.
        assert validation[-1]["ema"] == {key: oracle[key] for key in ("ce", *BPB_FIELDS)}
        assert validation[-1]["ema"]["label_bytes"] > 0
        assert validated["logged"] == [(record["step"], record["ema"]["ce"]) for record in validation]
        # The same EMA on any rank count, up to the summation order of the gradients.
        for update in range(UPDATES):
            for name, value in runs[1][0]["validated"]["ema"][update].items():
                torch.testing.assert_close(validated["ema"][update][name], value, rtol=1e-5, atol=1e-6,
                                           msg=f"{name} world={world} update={update}")
        for record, first in zip(validation, runs[1][1], strict=True):
            assert record["ema"]["label_bytes"] == first["ema"]["label_bytes"]
            assert record["ema"]["ce"] == pytest.approx(first["ema"]["ce"], rel=1e-5)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    world, data, output = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mp.start_processes(run_rank, args=(world, data, output), nprocs=world, start_method="fork")
