"""Muon grouping by physical owner, and one real DistMuon step on one CPU rank."""

import fnmatch
import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from torchtitan.components.optimizer import OptimizersContainer

from moeloop import recipes
from moeloop.model import RMSNorm
from moeloop.moe import Router
from moeloop.optim import CORE_EXPERT_MUON
from moeloop.recipe_spec import RecipeSpec
from moeloop.titan import TitanModel, configure_optimizer
from moeloop.training import continuation_fingerprint

GEOMETRY = dict(vocab_size=19, dim=16, hidden_dim=24, num_heads=2, num_kv_heads=1, top_k=2, max_seq_len=64)
MUON = {"lr": .02, "momentum": .95, "nesterov": True, "ns_steps": 5,
        "ns_coefficients": [3.4445, -4.7750, 2.0315], "eps": 1e-7, "weight_decay": .1,
        "adjust_lr_fn": "match_rms_adamw"}
SPECS = {
    "smelt": lambda: recipes.smelt_spec(physical_layers=4, num_experts=4, **GEOMETRY),
    "foil": lambda: recipes.foil_spec(num_experts=2, core_layers=2, **GEOMETRY),
    "expert_tying": lambda: recipes.expert_tying_spec(groups=2, group_size=2, num_experts=4, **GEOMETRY),
    "loom": lambda: recipes.loom_spec(num_experts=4, **GEOMETRY),
    "ouro": lambda: recipes.ouro_spec(num_experts=4, **GEOMETRY),
    "huginn": lambda: recipes.huginn_spec(num_experts=4, **GEOMETRY),
    "dense": lambda: recipes.dense_spec(layers=2, **GEOMETRY),
}
# Every recipe constructor the repo defines, with its defaults: a new owner type
# must be named by MUON_PATTERN or the coverage test fails on CPU.
CONSTRUCTED = {}
for _name, _build in vars(recipes).items():
    if _name.endswith(("_spec", "_specs")) and not _name.startswith("_"):
        _built = _build()
        CONSTRUCTED.update(_built if isinstance(_built, dict) else {_name: _built})
CONSTRUCTED["relaxed_dense"] = recipes.relaxed_spec(dense=True, **GEOMETRY)  # the FFN LoRA matrices


def muon_spec(spec, *, muon=None, **optimizer):
    spec = replace(spec, optimizer=spec.optimizer | {"name": "Muon", "muon": MUON | (muon or {})} | optimizer)
    return RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))


def hidden_matrices(model):
    """Oracle: every >=2D parameter outside embedding, lm_head, exit gate and routers."""
    core = model.core
    excluded = [core.embedding, core.lm_head, core.exit_gate, *(m for m in core.modules() if isinstance(m, Router))]
    excluded = {id(p) for module in excluded if module is not None for p in module.parameters()}
    return {id(p) for p in core.parameters() if p.ndim >= 2 and id(p) not in excluded}


def core_experts(model):
    return {id(p) for cell in model.core.core if cell.experts is not None for p in cell.experts.parameters()}


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("name", list(CONSTRUCTED))
def test_every_physical_owner_is_in_exactly_one_group(name, checkpointed, tmp_path):
    from torchtitan.distributed.activation_checkpoint import FullAC
    config = recipes.trainer_config(muon_spec(CONSTRUCTED[name], expert_lr_multiplier=.5))
    with torch.device("meta"):
        model = TitanModel(config.model_spec.model)
    if checkpointed:  # as parallelize_model wraps cells: raw FQNs gain a wrapper segment
        policy = FullAC.Config().build(dump_folder=str(tmp_path))
        for region in (model.core.prefix, model.core.core, model.core.suffix):
            for index, cell in enumerate(region):
                region[index] = policy._wrap_block(cell)
    groups, _ = OptimizersContainer._build_param_groups(model, config.optimizer.param_groups, {})
    assigned = {name: [id(p) for group in entries for p in group["params"]] for name, entries in groups.items()}
    every = [i for ids in assigned.values() for i in ids]
    assert len(every) == len(set(every)) and set(every) == {id(p) for p in model.parameters()}
    muon = set(assigned.get("DistMuon", [])) | set(assigned.get(CORE_EXPERT_MUON, []))
    assert muon == hidden_matrices(model)
    assert set(assigned.get(CORE_EXPERT_MUON, [])) == core_experts(model)
    decay = {id(p): group["weight_decay"] for group in groups["AdamW"] for p in group["params"]}
    core = model.core
    zero = [core.embedding.weight, core.lm_head.weight, core.final_norm.weight]
    zero += [p for module in core.modules() if isinstance(module, (Router, RMSNorm)) for p in module.parameters()]
    assert {decay[id(p)] for p in zero} == {0.0}
    if core.exit_gate is not None:
        assert decay[id(core.exit_gate.weight)] == decay[id(core.exit_gate.bias)] == .1
    for optimizer_name, kwargs in config.optimizer.optimizer_factory_kwargs_by_name.items():
        names = [fqn for group in groups[optimizer_name] for fqn in group["param_names"]]
        assert set(kwargs["compute_sharding_by_fqn"]) == set(names)
        for fqn in names:
            buckets = kwargs["bucket_configs"]
            assert sum(any(fnmatch.fnmatchcase(fqn, p) for p in bucket.patterns) for bucket in buckets) == 1
            placement = kwargs["compute_sharding_by_fqn"][fqn].shardings_by_mesh_axis["fsdp"]
            assert type(placement).__name__ == ("Shard" if "experts" in fqn else "Owned")


def newton_schulz(matrix, *, steps, coefficients, eps):
    """Reference quintic iteration X <- aX + (bG + cG^2)X, G = XX^T, for one matrix.

    It runs in bf16 with fused addmm, as DistMuon does: the iteration expands
    small singular directions ~a^steps, so unfused rounding is not comparable.
    """
    a, b, c = coefficients
    x = matrix.bfloat16()
    transposed = x.size(0) > x.size(1)
    x = x.T if transposed else x
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return (x.T if transposed else x).float()


def reference_muon(parameter, gradient, *, lr):
    """First Muon update: zero momentum, Nesterov, per-matrix NS, Moonlight RMS scale."""
    momentum = torch.zeros_like(gradient).lerp_(gradient, 1 - MUON["momentum"])
    update = torch.lerp(gradient, momentum, MUON["momentum"])
    matrices = update if update.ndim == 3 else update[None]
    direction = torch.stack([newton_schulz(m, steps=MUON["ns_steps"], coefficients=MUON["ns_coefficients"],
                                           eps=MUON["eps"]) for m in matrices]).view_as(update)
    scale = .2 * math.sqrt(max(update.shape[-2:]))
    return parameter * (1 - lr * MUON["weight_decay"]) - lr * scale * direction, lr * scale


@pytest.fixture
def one_cpu_rank(tmp_path, monkeypatch):
    """One gloo rank. Pinned DistMuon rejects non-CUDA devices; bypass only that check."""
    import torch.distributed as dist
    from torchtitan.distributed.utils import get_spmd_backend, set_spmd_backend
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from torchtitan.distributed.flex_shard.dist_muon import DistMuon

    def cpu_storage(self):
        assert all(isinstance(p, DTensor) for group in self.param_groups for p in group["params"])
        return torch.device("cpu")

    # The only CPU pin of the Muon math: one rank needs no CUDA stream, only this device check.
    monkeypatch.setattr(DistMuon, "_validate_parameter_storage", cpu_storage)
    backend = get_spmd_backend()
    set_spmd_backend("partial_dtensor")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/init", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
    finally:
        dist.destroy_process_group()
        set_spmd_backend(backend)


def fsdp_model(config, mesh, tmp_path):
    from types import SimpleNamespace
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.activation_checkpoint import FullAC
    from moeloop.titan import parallelize_model
    torch.manual_seed(5)
    # LoopTrainer copies its expert_lr_multiplier into the model config.
    model = TitanModel(replace(config.model_spec.model, flex_attention=False,
                               expert_lr_multiplier=config.expert_lr_multiplier))
    dims = SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, get_mesh=lambda names: mesh)
    return parallelize_model(model, parallel_dims=dims,
        training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor"), compile_config=CompileConfig(enable=False),
        ac_config=FullAC.Config(), dump_folder=str(tmp_path))


def test_one_dist_muon_step_matches_reference_and_updates_shared_owners_once(one_cpu_rank, tmp_path):
    spec = muon_spec(SPECS["smelt"](), expert_lr_multiplier=.5, base_lr=.01)
    config = recipes.trainer_config(spec)
    model = fsdp_model(config, one_cpu_rank, tmp_path)
    optimizers = replace(config.optimizer, implementation="for-loop").build(model_parts=[model])
    configure_optimizer(optimizers, [model], None)
    assert sorted(type(o).__name__ for o in optimizers) == ["AdamW", "DistMuon", "DistMuon"]
    owners = [id(p) for o in optimizers for group in o.param_groups for p in group["params"]]
    assert len(owners) == len(set(owners)) == len(list(model.parameters()))
    tokens = torch.randint(0, 19, (2, 8))
    output = model(tokens)
    assert model.core.config.num_visits == 2  # every core owner is called twice
    torch.nn.functional.cross_entropy(output.logits.flatten(0, 1), tokens.flatten()).backward()
    before = {n: (p.detach().full_tensor().clone(), p.grad.full_tensor().clone()) for n, p in model.named_parameters()}
    optimizers.step()
    muon, core = hidden_matrices(model), core_experts(model)
    decay = {id(p): group["weight_decay"] for o in optimizers if isinstance(o, torch.optim.AdamW)
             for group in o.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        value, gradient = before[name]
        actual = parameter.detach().full_tensor()
        if id(parameter) in muon:  # one update from the gradient summed over both calls
            expected, step = reference_muon(value, gradient, lr=MUON["lr"] * (.5 if id(parameter) in core else 1))
            torch.testing.assert_close(actual, expected, rtol=0, atol=step * 1e-4, msg=name)
            assert (actual - value).abs().max() > step * .05, name
        else:
            expected = value * (1 - .01 * decay[id(parameter)]) - .01 * gradient / (gradient.abs() + 1e-8)
            torch.testing.assert_close(actual, expected, msg=name)
    state = optimizers.state_dict()  # FQN-keyed momentum for both DistMuon instances
    for kwargs in config.optimizer.optimizer_factory_kwargs_by_name.values():
        assert {f"state.{fqn}.momentum_buffer" for fqn in kwargs["compute_sharding_by_fqn"]} <= set(state)
    optimizers.load_state_dict(state)


def test_only_muon_keeps_expert_storage_shard0():
    spec = SPECS["smelt"]()
    assert recipes.trainer_config(muon_spec(spec)).model_spec.model.dist_muon
    assert not recipes.trainer_config(spec).model_spec.model.dist_muon


def test_muon_settings_enter_the_continuation_identity():
    def digest(**muon):
        config = recipes.trainer_config(muon_spec(SPECS["expert_tying"](), muon=muon))
        # LoopTrainer needs a process group and data to build its identity; these
        # are its two fields that carry optimizer settings.
        return continuation_fingerprint({"optimizer": config.optimizer.to_dict(),
                                         "recipe_metadata": config.recipe_metadata})

    base = digest()
    assert digest() == base
    changes = {"lr": .01, "momentum": .9, "nesterov": False, "ns_steps": 6, "ns_coefficients": [3., -4., 2.],
               "eps": 1e-6, "weight_decay": 0., "adjust_lr_fn": "original"}
    for key, value in changes.items():
        assert digest(**{key: value}) != base, key
    adamw = recipes.trainer_config(SPECS["expert_tying"]())
    assert continuation_fingerprint({"optimizer": adamw.optimizer.to_dict(),
                                     "recipe_metadata": adamw.recipe_metadata}) != base


@pytest.mark.parametrize("optimizer,match", [
    ({"name": "SGD"}, "optimizer.name"),
    ({"name": "Muon"}, "optimizer.muon"),
    ({"name": "Muon", "muon": {k: v for k, v in MUON.items() if k != "ns_steps"}}, "optimizer.muon"),
    ({"name": "Muon", "muon": MUON | {"backend": "x"}}, "optimizer.muon"),
    ({"name": "AdamW", "muon": MUON}, "optimizer.muon"),
])
def test_trainer_config_rejects_incomplete_or_unknown_optimizers(optimizer, match):
    spec = SPECS["smelt"]()
    with pytest.raises(ValueError, match=match):
        recipes.trainer_config(replace(spec, optimizer=spec.optimizer | optimizer))

