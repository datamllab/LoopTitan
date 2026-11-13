"""Validation BPB equals evaluate_heldout's on 1, 2 and 4 CPU FSDP ranks and leaves training unchanged.

The distributed test runs this file as a script: one forked gloo process per rank
validates with the real LoopValidator and LoopChunkedLoss one global batch of the first
BLOCKS validation blocks, rank r its r-th contiguous slice. Each rank then trains UPDATES
steps twice from the same weights, once validating before every update and once
without validation, and rank 0 records both.
"""

import json
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("torchtitan")

from moeloop.evaluate import evaluate_heldout
from moeloop.model import ModelConfig, RecursiveLM
from moeloop.prepare import prepare_dataset
from moeloop.titan import (LoopChunkedLoss, LoopValidator, PackedDataLoader, TitanModel, canonical_model_state,
                           parallelize_model, prepare_batch)
from moeloop.tokens import byte_table_sha256, token_byte_lengths
from moeloop.training import TrainingRecipe

# The byte-level BPE fixture of tests/data/test_bpb.py: 321 IDs, EOS (ID 0) has 0 bytes.
# Manifest and model pad the vocabulary to 384; the recorded hash is the unpadded table's.
FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "bpb"
TEXTS = ["hello world", "naïve café — 你好，世界 🙂", "tabs\tand\nnewlines  and   spaces", "quick 🦊 jumps over the lazy dog"]
TOKENIZER = {"id": "bpb-fixture", "revision": "fixed-v1", "vocab_size": 384, "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0}
SEQ_LEN, BLOCKS, UPDATES, SEED = 8, 8, 3, 5
MODEL = ModelConfig(vocab_size=384, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, num_experts=2, top_k=1,
                    num_cells=1, num_visits=2, prefix_cells=1, max_seq_len=SEQ_LEN)
RECIPE = TrainingRecipe(num_visits=2, balance_coeff=0.3, z_loss_coeff=0.05)
BYTE_FIELDS = ("label_bytes", "ce_sum_text", "bpb", "byte_table_sha256")
UNPADDED_SHA256 = byte_table_sha256(token_byte_lengths(FIXTURE))


def prepare(path, *, assets=True):
    # Two copies of the texts pack into 14 validation blocks of SEQ_LEN.
    encoder = tokenizers.Tokenizer.from_file(str(FIXTURE / "tokenizer.json"))
    return prepare_dataset(path, {"text": {"train": TEXTS * 2, "validation": TEXTS * 2}},
        encode=lambda text: encoder.encode(text, add_special_tokens=False).ids, tokenizer=TOKENIZER,
        tokenizer_directory=FIXTURE if assets else None)


def model_and_loss():
    torch.manual_seed(SEED)
    model = TitanModel(TitanModel.Config(core=MODEL, native_experts=False, flex_attention=False))
    loss_fn = LoopChunkedLoss(LoopChunkedLoss.Config(num_chunks=2, recipe=RECIPE))
    return model, loss_fn


def validator_for(manifest_path, output, loss_fn, parallel_dims, *, world=1, rank=0, blocks=BLOCKS):
    """One validation global batch of ``blocks`` blocks."""
    logged = []
    metrics = SimpleNamespace(output_directory=output, last_validation={}, ntokens_since_last_log=0,
                              logger=SimpleNamespace(log=lambda values, step: logged.append((step, values))))
    metrics.log_validation = lambda loss, step: metrics.last_validation.update(ce=float(loss), step=step)
    validator = LoopValidator(LoopValidator.Config(global_batches=1, dataloader=PackedDataLoader.Config(
            manifest_path=str(manifest_path), split="validation", shuffle=False, infinite=False, global_batch_size=blocks)),
        parallelism=None, dp_world_size=world, dp_rank=rank, tokenizer=None, parallel_dims=parallel_dims,
        loss_fn=loss_fn, validation_context=nullcontext, metrics_processor=metrics, seq_len=SEQ_LEN, local_batch_size=1)
    return validator, logged


def run_rank(rank, world, data, output):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.utils import set_spmd_backend
    from torchtitan.tools import utils

    torch.set_num_threads(1)
    utils.device_type = "cpu"  # this process has no accelerator
    dist.init_process_group("gloo", init_method=f"file://{output}/init", rank=rank, world_size=world)
    set_spmd_backend("partial_dtensor")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    parallel_dims = SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, pp_enabled=False,
        dp_enabled=True, dp_cp_enabled=True, get_mesh=lambda names: mesh, get_optional_mesh=lambda names: mesh)

    def build():
        model, loss_fn = model_and_loss()
        model = parallelize_model(model, parallel_dims=parallel_dims,
            training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
            parallelism=ParallelismConfig(spmd_backend="partial_dtensor"), compile_config=CompileConfig(enable=False),
            ac_config=None, dump_folder=str(output))
        for module in model.modules():
            if isinstance(module, FSDPModule):  # gloo lacks PREMUL_SUM; native divide factor is 1
                module.set_force_sum_reduction_for_comms(True)
        model._skip_lm_head = True
        loss_fn.set_lm_head(model.lm_head)
        loss_fn.dp_degree = world
        return model, loss_fn

    def train(validate):
        model, loss_fn = build()
        validator, logged = validator_for(data, output, loss_fn, parallel_dims, world=world, rank=rank)
        loader = iter(PackedDataLoader(PackedDataLoader.Config(manifest_path=str(data)), dp_rank=rank, dp_world_size=world,
                                       tokenizer=None, seq_len=SEQ_LEN, local_batch_size=1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        losses, gradients = [], []
        for step in range(UPDATES):
            if validate:
                validator.validate([model], step)
            optimizer.zero_grad()
            input_dict, labels = next(loader)
            valid = (labels != -100).sum()
            dist.all_reduce(valid)
            inputs, labels, kwargs = prepare_batch(input_dict, labels, flex_attention=False)
            loss, _ = loss_fn(model(inputs, **kwargs), labels, valid)
            loss.backward()
            losses.append(loss.detach())
            gradients.append({name: parameter.grad.full_tensor() for name, parameter in model.named_parameters()})
            optimizer.step()
        return {"losses": losses, "gradients": gradients, "logged": logged}

    with_validation, without_validation = train(True), train(False)
    if rank == 0:
        torch.save({"with": with_validation, "without": without_validation}, output / "records.pt")
    dist.destroy_process_group()


def run(world, data, output):
    output.mkdir()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"), "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, __file__, str(world), str(data), str(output)], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    return records, torch.load(output / "records.pt")


@pytest.mark.skipif(sys.platform != "linux", reason="the CPU FSDP probe uses Linux fork")
def test_validation_bpb_equals_heldout_oracle_on_any_rank_count_and_training_is_unchanged(tmp_path):
    manifest = prepare(tmp_path / "data")
    model, _ = model_and_loss()
    reference = RecursiveLM(MODEL)
    reference.load_state_dict(canonical_model_state(model.state_dict()))
    oracle = evaluate_heldout(reference, manifest, seq_len=SEQ_LEN, batch_size=1, max_batches=BLOCKS)
    assert oracle["label_bytes"] > 0 and oracle["ce_sum_text"] < oracle["ce_sum"]
    for world in (1, 2, 4):
        records, runs = run(world, manifest.path, tmp_path / f"world-{world}")
        first = records[0]  # step 0: the initial weights, as in the oracle
        assert first["world_size"] == world and first["steps"] == BLOCKS // world
        assert first["valid_tokens"] == oracle["valid_tokens"]
        assert first["label_bytes"] == oracle["label_bytes"]
        assert first["byte_table_sha256"] == oracle["byte_table_sha256"] == UNPADDED_SHA256
        assert first["bpb_unavailable"] is None
        assert first["ce"] == pytest.approx(oracle["ce"], rel=1e-5)
        assert first["ce_sum_text"] == pytest.approx(oracle["ce_sum_text"], rel=1e-5)
        assert first["bpb"] == pytest.approx(oracle["bpb"], rel=1e-5)
        assert [record["step"] for record in records] == list(range(UPDATES))
        assert [values for _, values in runs["with"]["logged"]] == [{"validation_metrics/bpb": record["bpb"]} for record in records]
        for step in range(UPDATES):
            assert torch.equal(runs["with"]["losses"][step], runs["without"]["losses"][step]), (world, step)
            for name, gradient in runs["without"]["gradients"][step].items():
                assert torch.equal(runs["with"]["gradients"][step][name], gradient), (world, step, name)


def test_single_process_validation_matches_oracle_and_reports_missing_byte_table(tmp_path, monkeypatch):
    monkeypatch.setattr("torchtitan.components.validate.utils.device_type", "cpu")
    parallel_dims = SimpleNamespace(pp_enabled=False, dp_enabled=False, dp_cp_enabled=False)
    for assets in (True, False):
        model, loss_fn = model_and_loss()
        model._skip_lm_head = True
        loss_fn.set_lm_head(model.lm_head)
        manifest = prepare(tmp_path / f"data-{assets}", assets=assets)
        output = tmp_path / f"run-{assets}"
        validator, logged = validator_for(manifest.path, output, loss_fn, parallel_dims)
        validator.validate([model], step=3)
        record = json.loads((output / "validation.jsonl").read_text())
        oracle = evaluate_heldout(model.core, manifest, seq_len=SEQ_LEN, batch_size=1, max_batches=BLOCKS)
        assert record["valid_tokens"] == oracle["valid_tokens"]
        assert record["ce"] == pytest.approx(oracle["ce"], rel=1e-5)
        assert record["bpb_unavailable"] == oracle["bpb_unavailable"]
        if assets:
            assert record["label_bytes"] == oracle["label_bytes"]
            assert record["byte_table_sha256"] == oracle["byte_table_sha256"] == UNPADDED_SHA256
            assert record["bpb"] == pytest.approx(oracle["bpb"], rel=1e-5)
            assert logged == [(3, {"validation_metrics/bpb": record["bpb"]})]
        else:
            assert [record[key] for key in BYTE_FIELDS] == [None] * 4
            assert record["bpb_unavailable"] == "manifest has no tokenizer assets" and logged == []


def test_validation_without_byte_bearing_labels_reports_why(tmp_path, monkeypatch):
    monkeypatch.setattr("torchtitan.components.validate.utils.device_type", "cpu")
    encoder = tokenizers.Tokenizer.from_file(str(FIXTURE / "tokenizer.json"))
    # "<think>" is one added token: its labels are that token and EOS, both 0 bytes.
    manifest = prepare_dataset(tmp_path / "data", {"text": {"validation": ["<think>"]}},
        encode=lambda text: encoder.encode(text, add_special_tokens=False).ids, tokenizer=TOKENIZER, tokenizer_directory=FIXTURE)
    model, loss_fn = model_and_loss()
    model._skip_lm_head = True
    loss_fn.set_lm_head(model.lm_head)
    validator, logged = validator_for(manifest.path, tmp_path, loss_fn,
                                      SimpleNamespace(pp_enabled=False, dp_enabled=False, dp_cp_enabled=False), blocks=1)
    validator.validate([model], step=1)
    record = json.loads((tmp_path / "validation.jsonl").read_text())
    assert record["valid_tokens"] == 2 and record["label_bytes"] == 0 and record["ce_sum_text"] == 0.0
    assert record["bpb"] is None and record["byte_table_sha256"] is not None and logged == []
    assert record["bpb_unavailable"] == "no byte-bearing labels evaluated"


if __name__ == "__main__":
    import torch.multiprocessing as mp
    world, data, output = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mp.start_processes(run_rank, args=(world, data, output), nprocs=world, start_method="fork")
