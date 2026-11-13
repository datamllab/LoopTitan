"""TitanModel.init_states is RecursiveLM.initialize_weights, seeded by the run seed.

The decay injections take the reference model's declared initialization and every
other parameter keeps the generic rule bit for bit. The FSDP tests run this file as
a script: one forked gloo process per rank shards the model as training does,
initializes it, and rank 0 saves every gathered full parameter. On 1, 2 and 4 ranks
the step-0 weights are bit-identical to one process's initialize_weights.
"""

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

import torch.nn.functional as F

from moeloop import recipes
from moeloop.model import ModelConfig, RecursiveLM
from moeloop.titan import TitanModel

GEOMETRY = dict(vocab_size=31, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
                num_experts=4, top_k=2, max_seq_len=32)
LOOP = dict(prefix_cells=1, num_cells=2, num_visits=3)
DECAY = {
    "diag_decay": ModelConfig(**GEOMETRY, **LOOP, input_injection="diag_decay"),
    "orthogonal": ModelConfig(**GEOMETRY, **LOOP, input_injection="orthogonal"),
    "orthogonal_dt0.3_a2": ModelConfig(**GEOMETRY, **LOOP, input_injection="orthogonal",
                                       injection_dt=0.3, injection_decay=2.0),
    "sandwich": recipes.sandwich_spec(**GEOMETRY).model,  # injection_dt = 1
}
OTHER = {
    "recursive_moe": recipes.recursive_moe_spec(**GEOMETRY).model,
    "dense": recipes.dense_spec(layers=2, **GEOMETRY).model,
    "moe": recipes.moe_spec(layers=2, **GEOMETRY).model,
    "smelt": recipes.smelt_spec(**GEOMETRY).model,
    "latentloop_residual": recipes.latentloop_residual_spec(**GEOMETRY).model,
    "meta": recipes.meta_spec(**GEOMETRY).model,
    "foil_base": recipes.foil_base_spec(**GEOMETRY).model,
    "foil": recipes.foil_spec(**GEOMETRY).model,
    "expert_tying": recipes.expert_tying_spec(**GEOMETRY).model,
    "per_visit_router": recipes.per_visit_router_spec(**GEOMETRY).model,
    "input_anchor": recipes.input_anchor_spec(**GEOMETRY).model,
    "huginn": recipes.huginn_spec(**GEOMETRY).model,
    "loom": recipes.loom_spec(**GEOMETRY).model,
    "ouro": recipes.ouro_spec(**GEOMETRY).model,
}
INJECTION = ("core.input_adapter.input_proj.weight", "core.input_adapter.A_log", "core.input_adapter.dt_bias")


def titan(config):
    return TitanModel(TitanModel.Config(core=config, native_experts=False, flex_attention=False))


def init_states(model, seed=0):
    """The trainer's call on materialized storage; NaN stands in for to_empty's arbitrary bytes."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(math.nan)
        model.init_states()
    return model


def generic_init(model, seed=0):
    """init_states before #90: unit vectors, normal(init_std) matrices in parameter order, zero exit gate."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0)
            else:
                parameter.normal_(std=model.config.core.init_std)
        if model.core.exit_gate is not None:
            model.core.exit_gate.weight.zero_()
            model.core.exit_gate.bias.zero_()
    return model


def assert_reference_injection(config, actual):
    """``actual`` maps INJECTION names to full tensors; the oracle is RecursiveLM.initialize_weights."""
    reference = dict(RecursiveLM(config).named_parameters())
    for name in INJECTION:
        assert torch.equal(actual[name], reference[name.removeprefix("core.")]), name
    # The declared values come from this config, not from the ModelConfig defaults.
    assert torch.equal(actual[INJECTION[0]], torch.eye(config.dim))
    torch.testing.assert_close(actual[INJECTION[1]].exp(), torch.full((config.dim,), config.injection_decay))
    torch.testing.assert_close(F.softplus(actual[INJECTION[2]]), torch.full((config.dim,), config.injection_dt))


@pytest.mark.parametrize("name", DECAY)
def test_decay_injection_takes_the_reference_initialization(name):
    model = init_states(titan(DECAY[name]))
    assert_reference_injection(DECAY[name], dict(model.named_parameters()))


@pytest.mark.parametrize("name", [*OTHER, *DECAY])
def test_every_other_parameter_keeps_the_generic_initialization(name):
    config = {**OTHER, **DECAY}[name]
    actual, expected = init_states(titan(config)), generic_init(titan(config))
    for n, parameter in actual.named_parameters():
        if name in OTHER or n not in INJECTION:
            assert torch.equal(parameter, expected.get_parameter(n)), n


RELAXED = {
    "relaxed": recipes.relaxed_spec(**GEOMETRY).model,
    "relaxed_dense": recipes.relaxed_spec(dense=True, **GEOMETRY).model,
}
# Uneven Shard(0) rows (vocab 31 over 2 or 4 ranks), LoRA, the exit gate, and with 2
# experts on 3 or 4 ranks the training path's Shard(1) expert banks (24 rows split evenly).
WORLD = {
    "relaxed": RELAXED["relaxed"],
    "orthogonal": DECAY["orthogonal"],
    "ouro": recipes.ouro_spec(**GEOMETRY).model,
    "two_experts": recipes.smelt_spec(**{**GEOMETRY, "dim": 24, "num_experts": 2, "top_k": 1}).model,
}
SEED = 11


@pytest.mark.parametrize("name", RELAXED)
def test_lora_takes_the_reference_initialization(name):
    """B = 0, and every parameter equals RecursiveLM.initialize_weights from the same seed.

    Both skip LoRA in the generic pass and draw A last, so the shared weights
    also equal those of the plain model from that seed.
    """
    config = RELAXED[name]
    actual = dict(init_states(titan(config)).named_parameters())
    reference = RecursiveLM(config)
    torch.manual_seed(0)
    reference.initialize_weights()
    expected = {f"core.{key}": value for key, value in reference.named_parameters()}
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key
    lora = [value for key, value in actual.items() if "_loras." in key and key.endswith(".B")]
    assert lora and all(torch.equal(value, torch.zeros_like(value)) for value in lora)


def run_rank(rank, world, output):
    from types import SimpleNamespace
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.utils import set_spmd_backend
    from moeloop.titan import parallelize_model

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{output}/init", rank=rank, world_size=world)
    set_spmd_backend("partial_dtensor")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    gathered = {}
    for name, config in {**DECAY, **WORLD}.items():
        with torch.device("meta"):  # LoopTrainer's path: meta -> parallelize -> to_empty -> init
            model = titan(config)
        model = parallelize_model(model,
            parallel_dims=SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, get_mesh=lambda names: mesh),
            training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
            parallelism=ParallelismConfig(spmd_backend="partial_dtensor"), compile_config=CompileConfig(enable=False),
            ac_config=None, dump_folder=str(output))
        model.to_empty(device="cpu")
        init_states(model, SEED)  # set_determinism's torch.manual_seed(debug.seed)
        parameters = dict(model.named_parameters())
        rows = len(torch.arange(config.dim).chunk(world)[rank])
        for key in INJECTION if name in DECAY else ():  # each rank holds only its own rows of W, a and dt
            assert isinstance(parameters[key], DTensor) and parameters[key].placements == (Shard(0),), key
            assert parameters[key].to_local().shape[0] == rows, key
        if name == "two_experts" and world > 2:
            assert parameters["core.core.0.experts.w1_EFD"].placements == (Shard(1),)
        gathered[name] = {key: value.full_tensor() for key, value in parameters.items()}
    if rank == 0:
        torch.save(gathered, output / "parameters.pt")
    dist.destroy_process_group()


def gathered_parameters(world, output):
    output.mkdir()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"), "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, __file__, str(world), str(output)], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return torch.load(output / "parameters.pt")


@pytest.mark.skipif(sys.platform != "linux", reason="the CPU FSDP probe uses Linux fork")
@pytest.mark.parametrize("world", [2, 3])  # 3 ranks shard d = 16 unevenly: 6, 6 and 4 rows
def test_fsdp_shards_gather_to_the_reference_initialization(tmp_path, world):
    gathered = gathered_parameters(world, tmp_path / "run")
    for name, config in DECAY.items():
        assert_reference_injection(config, gathered[name])


@pytest.mark.skipif(sys.platform != "linux", reason="the CPU FSDP probe uses Linux fork")
def test_step_zero_weights_do_not_depend_on_the_rank_count(tmp_path):
    """Oracle: one process's initialize_weights from torch.Generator().manual_seed(SEED)."""
    runs = {world: gathered_parameters(world, tmp_path / f"world-{world}") for world in (1, 2, 4)}
    for name, config in WORLD.items():
        reference = RecursiveLM(config)
        reference.initialize_weights(torch.Generator().manual_seed(SEED))
        expected = {f"core.{key}": value for key, value in reference.named_parameters()}
        for world, gathered in runs.items():
            assert gathered[name].keys() == expected.keys(), (name, world)
            for key, value in expected.items():
                assert torch.equal(gathered[name][key], value), (name, world, key)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    world, output = int(sys.argv[1]), Path(sys.argv[2])
    mp.start_processes(run_rank, args=(world, output), nprocs=world, start_method="fork")
