"""Core residual x a is the same training as residual 1 with the core output projections at init x a,
LR x a and weight decay / a (ModelConfig.output_lr_scale, output_init_scale), under AdamW and Muon.

With h <- h + a * W phi(h), put V = a * W. Adam's and Muon's updates do not change when a gradient
is scaled, so the update of W, times a, is the update V gets at LR x a; decoupled decay keeps
V's lr * weight_decay at decay / a. With a = 0.5 every scaling is exact in binary floating point,
so only Adam's eps (eps / a in the residual form) and Muon's norm clamp (inactive here) can make
the two differ: with eps 1e-30 AdamW's trajectories are equal, Muon's are equal at P2's settings,
and at P2's eps 1e-8 AdamW's differ by about 3e-4 nats in 30 steps (rarely routed experts have
second moments near eps), against about 0.1 for the negative controls, which change only the LR
or only the init. Global gradient-norm clipping (training's max_norm) also breaks the equivalence:
it sees W's gradient, a times V's. These runs therefore do not clip; on this model with P2's
max_norm 1.0 (active every step) the matched forms drifted to about 8e-3 (AdamW) and 2e-2 (Muon)
nats in 30 steps, still 10-20x below the controls (#216).

The models and optimizers are built as training builds them: trainer_config, TitanModel with
native experts, parallelize_model (FSDP2 on one CPU gloo rank, which DistMuon needs), the
optimizer container, configure_optimizer and the LR scheduler. Parameters are float64.
"""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan")

from test_muon import muon_spec, one_cpu_rank  # noqa: F401  (one_cpu_rank is the fixture)

from moeloop import recipes
from moeloop.config import ModelConfig, RegionConfig
from moeloop.optim import CORE_OUTPUT_MUON, optimizer_config
from moeloop.recipe_spec import RecipeSpec
from moeloop.titan import TitanModel, configure_optimizer

STEPS = 30
ALPHA = 0.5
# 1 prefix, 1 core and 1 suffix cell, width 64, 4 experts of width 32, top-2.
MODEL = ModelConfig(vocab_size=64, dim=64, hidden_dim=32, num_heads=4, num_kv_heads=1, num_experts=4, top_k=2,
                    prefix_cells=1, num_cells=1, suffix_cells=1, num_visits=1, max_seq_len=16)
RUN = {"successful_updates": STEPS, "schedule_total_steps": STEPS,
       "schedule": {"warmup_steps": 3, "decay_ratio": None, "decay_type": "cosine", "min_lr_factor": 0.1}}
ARMS = {  # arm: core region
    "residual": RegionConfig(residual_scale=ALPHA),
    "scales": RegionConfig(output_lr_scale=ALPHA, output_init_scale=ALPHA),
    "lr_only": RegionConfig(output_lr_scale=ALPHA),  # negative controls
    "init_only": RegionConfig(output_init_scale=ALPHA),
}
OUTPUTS = ("core.core.0.attentions.0.out_proj.weight", "core.core.0.experts.routed.inner_experts.w2_EDF")


def spec_for(arm, optimizer, eps=1e-8):
    base = recipes.moe_spec(layers=3)
    spec = replace(base, model=replace(MODEL, core=ARMS[arm]), run=base.run | RUN,
                   optimizer=base.optimizer | {"base_lr": 3e-3, "eps": eps, "expert_lr_multiplier": 1.0})
    spec = RecipeSpec.from_dict(spec.to_dict())
    return muon_spec(spec, base_lr=3e-3) if optimizer == "muon" else spec


def build(spec, mesh, tmp_path):
    """The model, optimizers and schedulers as LoopTrainer builds them, in float64 on one CPU rank."""
    from types import SimpleNamespace
    from torchtitan.config import TORCH_DTYPE_MAP, CompileConfig, ParallelismConfig
    from moeloop.titan import parallelize_model
    config = recipes.trainer_config(spec)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    try:
        model = TitanModel(replace(config.model_spec.model, flex_attention=False))
    finally:
        torch.set_default_dtype(torch.float32)
    dims = SimpleNamespace(tp=1, ep=1, cp=1, pp=1, dp_replicate_enabled=False, get_mesh=lambda names: mesh)
    with pytest.MonkeyPatch.context() as patch:  # the pinned map has no float64; FSDP2 keeps float64 with it
        patch.setitem(TORCH_DTYPE_MAP, "float64", torch.float64)
        training = SimpleNamespace(mixed_precision_param="float64", mixed_precision_reduce="float64",
                                   enable_cpu_offload=False)
        model = parallelize_model(model, parallel_dims=dims, training=training,
                                  parallelism=ParallelismConfig(spmd_backend="partial_dtensor"),
                                  compile_config=CompileConfig(enable=False), ac_config=None, dump_folder=str(tmp_path))
    optimizers = replace(config.optimizer, implementation="for-loop").build(model_parts=[model])
    configure_optimizer(optimizers, [model], None)
    schedulers = config.lr_scheduler.build(optimizers=optimizers, training_steps=STEPS)
    return model, optimizers, schedulers


def losses(spec, mesh, tmp_path):
    """Initial weights and the loss of every step: CE plus the router balance term, no clipping."""
    model, optimizers, schedulers = build(spec, mesh, tmp_path)
    initial = {name: parameter.detach().full_tensor().clone() for name, parameter in model.named_parameters()}
    data = torch.Generator().manual_seed(1)
    trajectory = []
    for _ in range(STEPS):
        start = torch.randint(0, MODEL.vocab_size, (4, 1), generator=data)  # learnable: next = token + 7
        tokens = (start + 7 * torch.arange(16)) % MODEL.vocab_size
        output = model(tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(output.logits.flatten(0, 1).double(), tokens[:, 1:].flatten())
        loss = loss + 0.01 * output.balance_sum / output.aux_calls
        loss.backward()
        optimizers.step()
        schedulers.step()
        optimizers.zero_grad()
        trajectory.append(loss.item())
    return initial, torch.tensor(trajectory, dtype=torch.float64), optimizers


@pytest.mark.parametrize("optimizer,eps,tolerance", [
    ("adamw", 1e-8, 2e-3),  # P2's eps: observed 3.5e-4
    ("adamw", 1e-30, 1e-12),  # eps out of the way: observed 0
    ("muon", 1e-8, 1e-12),  # observed 0
])
def test_residual_scale_equals_scaled_output_init_lr_and_decay(optimizer, eps, tolerance, one_cpu_rank, tmp_path):
    runs = {arm: losses(spec_for(arm, optimizer, eps), one_cpu_rank, tmp_path) for arm in ARMS}
    initial, reference, _ = runs["residual"]
    scaled, matched, optimizers = runs["scales"]
    for name, value in initial.items():  # init x a on the output projections only, from the same draws
        assert value.dtype == torch.float64
        assert torch.equal(scaled[name], ALPHA * value if name in OUTPUTS else value), name
    groups = {name: (type(o).__name__, group) for o in optimizers for group in o.param_groups
              for name in group["param_names"]}
    other = groups["core.core.0.attentions.0.q_proj.weight"][1]
    for name in OUTPUTS:  # one group of exactly these, at LR x a and decay / a: lr * weight_decay is unchanged
        kind, group = groups[name]
        assert kind == ("DistMuon" if optimizer == "muon" else "AdamW")
        assert sorted(group["param_names"]) == sorted(OUTPUTS)
        assert group["initial_lr"] == ALPHA * other["initial_lr"]
        assert group["weight_decay"] == other["weight_decay"] / ALPHA
    assert (matched - reference).abs().max() <= tolerance
    for arm in ("lr_only", "init_only"):
        assert (runs[arm][1] - reference).abs().max() > 2e-2, arm


def test_muon_gives_the_core_outputs_their_own_instance():
    spec = spec_for("scales", "muon")
    config = recipes.trainer_config(spec)
    kwargs = config.optimizer.optimizer_factory_kwargs_by_name
    assert sorted(kwargs[CORE_OUTPUT_MUON]["compute_sharding_by_fqn"]) == sorted(OUTPUTS)
    group = next(group for group in config.optimizer.param_groups if group.optimizer_name == CORE_OUTPUT_MUON)
    muon = spec.optimizer["muon"]
    assert group.optimizer_kwargs["lr"] == ALPHA * muon["lr"]
    assert group.optimizer_kwargs["weight_decay"] == muon["weight_decay"] / ALPHA
    with pytest.raises(ValueError, match="expert_lr_multiplier"):
        optimizer_config(spec.optimizer | {"expert_lr_multiplier": 0.5}, config.model_spec.model)


def test_output_scales_are_core_only_and_written_only_when_set():
    with pytest.raises(ValueError, match="only to the core"):
        replace(MODEL, prefix=RegionConfig(output_lr_scale=ALPHA))
    with pytest.raises(ValueError, match="positive"):
        replace(MODEL, output_init_scale=0.0)
    assert not {"output_lr_scale", "output_init_scale"} & set(MODEL.to_dict())
    record = replace(MODEL, core=ARMS["lr_only"]).to_dict()
    assert record["core"]["output_lr_scale"] == ALPHA and "output_init_scale" not in record["core"]
    assert ModelConfig.from_dict(record).to_dict() == record
