"""The same global batch gives the same updates and validation on 1, 2 and 4 CPU FSDP ranks.

The tests run this file as a script: one forked gloo process per rank loads its
microbatches with the real PackedDataLoader, sums gradients over accumulation
steps through FSDP and LoopChunkedLoss, and rank 0 records each update. In
``validate`` mode each rank instead initializes the model through init_states and
runs the real LoopValidator at step 0, and rank 0 records the blocks and sums.
"""

from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from moeloop.model import ModelConfig
from moeloop.prepare import prepare_dataset
from moeloop.training import TrainingRecipe

GLOBAL_BATCH, LOCAL_BATCH, SEQ_LEN, UPDATES = 8, 2, 8, 3
MODEL = ModelConfig(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, num_experts=2, top_k=1,
                    num_cells=1, num_visits=2, prefix_cells=1, max_seq_len=SEQ_LEN)
RECIPE = TrainingRecipe(num_visits=2, balance_coeff=0.3, z_loss_coeff=0.05)
# The byte-level BPE fixture of tests/data/test_bpb.py (321 IDs), padded to 384, gives validation bytes.
FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "bpb"
TEXT_MODEL = replace(MODEL, vocab_size=384)
SEED = 7


def prepare(path):
    # Validation labels: a 92 + 31, b 92, i.e. 16 + 12 blocks of SEQ_LEN.
    lengths = [str(n) for n in (1, 9, 4, 13, 2, 6, 3, 17, 5, 8, 1, 11)]
    sources = {"a": {"train": lengths, "validation": lengths + ["30"]},
               "b": {"train": lengths[::-1], "validation": lengths[::-1]}}
    return prepare_dataset(path, sources,
        encode=lambda text: [3 + (5 * i + int(text)) % 16 for i in range(int(text))],
        tokenizer={"id": "unit", "revision": "fixed", "vocab_size": 19, "bos_token_id": 1, "eos_token_id": 2})


def run_rank(rank, world, data, output):
    from types import SimpleNamespace
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.utils import set_spmd_backend
    from moeloop.titan import LoopChunkedLoss, PackedDataLoader, TitanModel, parallelize_model, prepare_batch

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{output}/init", rank=rank, world_size=world)
    set_spmd_backend("partial_dtensor")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    torch.manual_seed(3)
    model = parallelize_model(TitanModel(TitanModel.Config(core=MODEL, native_experts=False, flex_attention=False)),
        parallel_dims=SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, get_mesh=lambda names: mesh),
        training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor"), compile_config=CompileConfig(enable=False),
        ac_config=None, dump_folder=str(output))
    for module in model.modules():
        if isinstance(module, FSDPModule):  # gloo lacks PREMUL_SUM; native divide factor is 1
            module.set_force_sum_reduction_for_comms(True)
    model._skip_lm_head = True
    accumulation = GLOBAL_BATCH // (world * LOCAL_BATCH)
    loss_fn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=2, recipe=RECIPE))
    loss_fn.set_lm_head(model.lm_head)
    loss_fn.dp_degree, loss_fn.accumulation = world, accumulation
    loader = iter(PackedDataLoader(PackedDataLoader.Config(manifest_path=str(data), global_batch_size=GLOBAL_BATCH),
        dp_rank=rank, dp_world_size=world, tokenizer=None, seq_len=SEQ_LEN, local_batch_size=LOCAL_BATCH))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    records = []
    for _ in range(UPDATES):  # TorchTitan's train_step: one global label count, SUM gradients
        optimizer.zero_grad()
        batches = [next(loader) for _ in range(accumulation)]
        valid = sum((labels != -100).sum() for _, labels in batches)
        dist.all_reduce(valid)
        objective, aux = torch.zeros(()), torch.zeros(())
        for input_dict, labels in batches:
            inputs, labels, kwargs = prepare_batch(input_dict, labels, flex_attention=False)
            loss, metrics = loss_fn(model(inputs, **kwargs), labels, valid)
            loss.backward()
            objective, aux = objective + loss.detach(), aux + metrics["aux"]
        dist.all_reduce(objective)
        dist.all_reduce(aux)
        tokens = [None] * world
        dist.all_gather_object(tokens, [(batch[0]["input"], batch[1]) for batch in batches])
        gradients = {name: parameter.grad.full_tensor() for name, parameter in model.named_parameters()}
        optimizer.step()
        records.append({"inputs": torch.cat([x for rank_batches in tokens for x, _ in rank_batches]),
                        "labels": torch.cat([y for rank_batches in tokens for _, y in rank_batches]),
                        "objective": objective, "aux": aux, "gradients": gradients})
    if rank == 0:
        torch.save(records, output / "records.pt")
    dist.destroy_process_group()


def validate_rank(rank, world, data, output):
    from types import SimpleNamespace
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.utils import set_spmd_backend
    from torchtitan.tools import utils
    from moeloop.titan import LoopChunkedLoss, LoopValidator, PackedDataLoader, TitanModel, parallelize_model

    torch.set_num_threads(1)
    utils.device_type = "cpu"  # this process has no accelerator
    dist.init_process_group("gloo", init_method=f"file://{output}/init", rank=rank, world_size=world)
    set_spmd_backend("partial_dtensor")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    parallel_dims = SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, pp_enabled=False,
        dp_enabled=True, dp_cp_enabled=True, get_mesh=lambda names: mesh, get_optional_mesh=lambda names: mesh)
    with torch.device("meta"):  # LoopTrainer's path: meta -> parallelize -> to_empty -> init_states
        model = TitanModel(TitanModel.Config(core=TEXT_MODEL, native_experts=False, flex_attention=False))
    model = parallelize_model(model, parallel_dims=parallel_dims,
        training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor"), compile_config=CompileConfig(enable=False),
        ac_config=None, dump_folder=str(output))
    model.to_empty(device="cpu")
    torch.manual_seed(SEED)  # set_determinism's seeding from debug.seed
    model.init_states()
    model._skip_lm_head = True
    loss_fn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=2, recipe=RECIPE))
    loss_fn.set_lm_head(model.lm_head)
    blocks = []

    class RecordingValidator(LoopValidator):
        def post_dataloading_process(self, input_dict, labels, model_parts):
            blocks.append(input_dict["input"].clone())
            return super().post_dataloading_process(input_dict, labels, model_parts)

    metrics = SimpleNamespace(output_directory=output, last_validation={}, ntokens_since_last_log=0,
                              logger=SimpleNamespace(log=lambda values, step: None))
    metrics.log_validation = lambda loss, step: metrics.last_validation.update(ce=float(loss), step=step)
    # LoopTrainer hands the training global batch to the validation loader.
    data_config = PackedDataLoader.Config(manifest_path=str(data), split="validation", shuffle=False,
                                          infinite=False, global_batch_size=GLOBAL_BATCH)
    validator = RecordingValidator(LoopValidator.Config(dataloader=data_config), parallelism=None,
        dp_world_size=world, dp_rank=rank, tokenizer=None, parallel_dims=parallel_dims, loss_fn=loss_fn,
        validation_context=nullcontext, metrics_processor=metrics, seq_len=SEQ_LEN, local_batch_size=LOCAL_BATCH)
    validator.validate([model], step=0)
    ranks = [None] * world
    dist.all_gather_object(ranks, blocks)
    if rank == 0:
        # Logical order: global batch g is rank 0's, then rank 1's, ... accumulation-step slice.
        accumulation = GLOBAL_BATCH // (world * LOCAL_BATCH)
        logical = [batch for g in range(len(blocks) // accumulation) for rank_blocks in ranks
                   for batch in rank_blocks[g * accumulation:(g + 1) * accumulation]]
        record = json.loads((output / "validation.jsonl").read_text())
        torch.save({"blocks": torch.cat(logical), "record": record}, output / "validate.pt")
    dist.destroy_process_group()


def prepare_text(path):
    tokenizers = pytest.importorskip("tokenizers")
    encoder = tokenizers.Tokenizer.from_file(str(FIXTURE / "tokenizer.json"))
    texts = ["hello world", "naïve café — 你好，世界 🙂", "tabs\tand\nnewlines  and   spaces",
             "quick 🦊 jumps over the lazy dog", "validation must not depend on the number of ranks"] * 4
    return prepare_dataset(path, {"text": {"validation": texts}},
        encode=lambda text: encoder.encode(text, add_special_tokens=False).ids, tokenizer_directory=FIXTURE,
        tokenizer={"id": "bpb-fixture", "revision": "fixed-v1", "vocab_size": 384,
                   "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0})


def run(world, data, output, mode="train"):
    output.mkdir()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"), "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, __file__, str(world), str(data), str(output), mode], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return torch.load(output / ("records.pt" if mode == "train" else "validate.pt"))


@pytest.mark.skipif(sys.platform != "linux", reason="the CPU FSDP probe uses Linux fork")
def test_one_two_four_ranks_see_the_same_updates_and_gradients(tmp_path):
    data = prepare(tmp_path / "data").path
    runs = {world: run(world, data, tmp_path / f"world-{world}") for world in (1, 2, 4)}
    for world in (2, 4):
        for update, (expected, actual) in enumerate(zip(runs[1], runs[world])):
            assert torch.equal(expected["inputs"], actual["inputs"]), (world, update)
            assert torch.equal(expected["labels"], actual["labels"]), (world, update)
            assert expected["aux"] > 0
            torch.testing.assert_close(actual["aux"], expected["aux"], rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(actual["objective"], expected["objective"], rtol=1e-5, atol=1e-6)
            for name, gradient in expected["gradients"].items():
                torch.testing.assert_close(actual["gradients"][name], gradient, rtol=1e-5, atol=1e-6,
                                           msg=f"{name} world={world} update={update}")
    # Each update is GLOBAL_BATCH new blocks of one stream, not the same batch again.
    assert not torch.equal(runs[1][0]["inputs"], runs[1][1]["inputs"])
    assert runs[1][0]["inputs"].shape == (GLOBAL_BATCH, SEQ_LEN)


@pytest.mark.skipif(sys.platform != "linux", reason="the CPU FSDP probe uses Linux fork")
def test_one_two_four_ranks_validate_the_same_blocks_from_the_same_weights(tmp_path):
    """Two global batches of 8 blocks: 8, 4 and 2 steps per rank on 1, 2 and 4 ranks.

    The step-0 CE also checks that init_states gives every rank count the same weights.
    """
    data = prepare_text(tmp_path / "text").path
    runs = {world: run(world, data, tmp_path / f"validate-{world}", "validate") for world in (1, 2, 4)}
    assert runs[1]["blocks"].shape == (2 * GLOBAL_BATCH, SEQ_LEN)
    assert runs[1]["record"]["label_bytes"] > 0
    for world, result in runs.items():
        record = result["record"]
        assert record["global_batches"] == 2 and record["global_batch_size"] == GLOBAL_BATCH
        assert record["steps"] == record["configured_steps"] == 2 * GLOBAL_BATCH // (world * LOCAL_BATCH)
        assert torch.equal(result["blocks"], runs[1]["blocks"]), world
        for key in ("valid_tokens", "label_bytes", "ce_sum_text", "bpb"):
            assert record[key] == runs[1]["record"][key], (world, key)
        assert record["ce"] == pytest.approx(runs[1]["record"]["ce"], rel=1e-6)


def test_validation_refuses_a_global_batch_that_ranks_cannot_split(tmp_path):
    from moeloop.titan import LoopValidator, PackedDataLoader
    data = PackedDataLoader.Config(manifest_path=str(prepare(tmp_path / "data").path), split="validation",
                                   shuffle=False, infinite=False, global_batch_size=GLOBAL_BATCH)
    with pytest.raises(ValueError, match="multiple of world size 3 x local batch 2"):
        LoopValidator(LoopValidator.Config(dataloader=data), parallelism=None, dp_world_size=3, dp_rank=0,
            tokenizer=None, parallel_dims=None, loss_fn=None, validation_context=None,
            metrics_processor=None, seq_len=SEQ_LEN, local_batch_size=LOCAL_BATCH)


def test_trainer_hands_the_global_batch_to_the_loader(tmp_path, monkeypatch):
    """Without it the loaders assume one microbatch per rank and update."""
    from moeloop.recipes import tiny
    from moeloop.titan import LoopTrainer
    from torchtitan.trainer import Trainer

    recipe = tiny()
    recipe.training.global_batch_size = 8
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
    assert saved["dataloader"]["global_batch_size"] == saved["training"]["global_batch_size"] == 8
    assert saved["validator"]["dataloader"]["global_batch_size"] == 8


@pytest.mark.parametrize("world,global_batch,batches", [(1, -1, [14]), (2, -1, [7, 7]), (3, -1, [5, 5, 4]),
                                                        (4, -1, [4, 4, 3, 3]), (2, 8, [8, 6])])
def test_finite_validation_capacity_equals_each_rank_loader(tmp_path, world, global_batch, batches):
    """28 blocks in rounds of world * rank slice; a partial last batch is dropped."""
    from moeloop.titan import LoopValidator, PackedDataLoader
    data = PackedDataLoader.Config(manifest_path=str(prepare(tmp_path / "data").path), split="validation",
                                   shuffle=False, infinite=False, global_batch_size=global_batch)
    validator = LoopValidator(LoopValidator.Config(dataloader=data), parallelism=None, dp_world_size=world,
        dp_rank=0, tokenizer=None, parallel_dims=None, loss_fn=None, validation_context=None,
        metrics_processor=None, seq_len=SEQ_LEN, local_batch_size=LOCAL_BATCH)
    actual = [sum(1 for _ in PackedDataLoader(data, dp_rank=rank, dp_world_size=world, tokenizer=None,
                                               seq_len=SEQ_LEN, local_batch_size=LOCAL_BATCH)) for rank in range(world)]
    assert validator.finite_batch_capacity == actual == batches


if __name__ == "__main__":
    import torch.multiprocessing as mp
    world, data, output, mode = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    mp.start_processes(run_rank if mode == "train" else validate_rank, args=(world, data, output),
                       nprocs=world, start_method="fork")
