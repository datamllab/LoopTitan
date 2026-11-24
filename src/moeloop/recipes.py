"""Ordinary method specs and finite native-training adapters.

``smelt_spec()`` (and the other ``*_spec`` functions) returns portable Python
configuration without importing TorchTitan. ``smelt()`` constructs its native
trainer, preserving ``--module moeloop.recipes --config smelt``. Set
MOELOOP_DATA to a prepared manifest/tokenizer directory for native training.
"""

from __future__ import annotations

import copy
import os
import math
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import ModelConfig, RegionConfig, TrainingRecipe
from .recipe_spec import RecipeSpec


# Every run setting the trainer reads, written out by every constructor (``_spec``).
# A recipe JSON frozen before a key existed resolves that key from here.
# ``local_batch_size`` None gives each data-parallel rank its whole share of the
# global batch in one microbatch (``local_batch_size()``).
RUN_DEFAULTS = {
    "global_batch_positions": 32768, "local_batch_size": None,
    "successful_updates": 512, "schedule_total_steps": 512,
    "schedule": {"warmup_steps": 10, "decay_ratio": None, "decay_type": "cosine", "min_lr_factor": 0.1},
    "seed": 42, "data_infinite": True, "mixed_precision_param": "bfloat16",
    "activation_checkpointing": "none", "num_chunks": 8,
    "checkpoint_interval": 50, "keep_latest_k": 2,  # 0 keeps every checkpoint (TorchTitan)
    "validation_interval": 64, "validation_label_tokens": None,
    "validation_global_batches": 2, "validation_infinite": True,
}
# ``ema_decay`` is the one run setting outside RUN_DEFAULTS: absent or None keeps no EMA of
# the weights (``LoopTrainer.Config.ema_decay``), so no constructor writes it and frozen
# specs that predate it, Phase 1's among them, are unchanged.


# Single-valued settings that recipe JSONs frozen before #155 still carry; no code reads them.
REMOVED_RUN_KEYS = ("master_dtype", "reduce_dtype", "objective")
REMOVED_OPTIMIZER_KEYS = ("expert_lr_scope", "weight_decay_compensation")


def resolve(spec: RecipeSpec) -> RecipeSpec:
    """``spec`` as the trainer reads it: ``run`` over ``RUN_DEFAULTS``, without the removed keys."""
    return replace(spec,
        run={key: value for key, value in (RUN_DEFAULTS | spec.run).items() if key not in REMOVED_RUN_KEYS},
        optimizer={key: value for key, value in spec.optimizer.items() if key not in REMOVED_OPTIMIZER_KEYS})


def local_batch_size(spec: RecipeSpec, world: int) -> int:
    """Sequences per rank per microbatch on ``world`` data-parallel ranks."""
    batch = spec.run.get("local_batch_size")
    return batch or max(1, spec.run["global_batch_positions"] // spec.run["seq_len"] // world)


def trainer_config(spec: RecipeSpec):
    """Only this finite boundary imports the optional native training stack.

    Every value comes from ``spec`` (``run`` over ``RUN_DEFAULTS``); a launcher
    sets only what its command decides: output, data paths, stop and topology.
    The local batch reads the launched WORLD_SIZE when the spec leaves it None.
    """
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.optimizer import LRSchedulersContainer
    from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.activation_checkpoint import FullAC
    from .titan import LoopChunkedLoss, LoopMetrics, LoopTrainer, LoopValidator, PackedDataLoader, model_spec
    from .optim import optimizer_config

    prepared = Path(os.environ.get("MOELOOP_DATA", "data/prepared"))
    resolved = resolve(spec)
    run, optimizer = resolved.run, spec.optimizer
    if run["activation_checkpointing"] not in ("none", "full_cells"):
        raise ValueError("activation_checkpointing must be none or full_cells")
    schedule = run["schedule"]
    if schedule.get("total_steps", run["schedule_total_steps"]) != run["schedule_total_steps"]:
        raise ValueError("schedule total_steps must match the frozen schedule_total_steps")
    seq_len, positions = run["seq_len"], run["global_batch_positions"]
    if positions is not None and (positions < seq_len or positions % seq_len):
        raise ValueError("global_batch_positions must be divisible by seq_len")
    native = model_spec(spec.model, dist_muon=optimizer.get("name") == "Muon")
    return LoopTrainer.Config(
        model_spec=native,
        recipe=spec.training,
        recipe_metadata=spec.to_dict(),
        expert_lr_multiplier=optimizer.get("expert_lr_multiplier", 1.0),
        ema_decay=run.get("ema_decay"),
        hf_assets_path=str(prepared / "tokenizer"),
        dump_folder="outputs/moeloop",
        dataloader=PackedDataLoader.Config(manifest_path=str(prepared / "manifest.json"),
            seed=run["seed"], infinite=run["data_infinite"]),
        training=TrainingConfig(local_batch_size=local_batch_size(resolved, int(os.environ.get("WORLD_SIZE", "1"))),
            global_batch_size=-1 if positions is None else positions // seq_len,
            seq_len=seq_len, steps=run["successful_updates"], disable_cuda_graphs=True, dtype="float32",
            mixed_precision_param=run["mixed_precision_param"],
            mixed_precision_reduce="float32", max_norm=optimizer.get("max_norm", 1.0)),
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor", enable_sequence_parallel=False),
        optimizer=optimizer_config(optimizer, native.model),
        lr_scheduler=LRSchedulersContainer.Config(
            total_steps=run["schedule_total_steps"],
            warmup_steps=schedule["warmup_steps"],
            decay_ratio=schedule["decay_ratio"],
            decay_type=schedule["decay_type"],
            min_lr_factor=schedule["min_lr_factor"]),
        loss=LoopChunkedLoss.Config(num_chunks=run["num_chunks"], recipe=spec.training),
        metrics=LoopMetrics.Config(log_freq=1, disable_color_printing=True),
        checkpoint=CheckpointManager.Config(enable=True, interval=run["checkpoint_interval"],
            keep_latest_k=run["keep_latest_k"], initial_load_model_only=False,
            last_save_model_only=False, last_save_in_hf=False),
        validator=LoopValidator.Config(enable=True, freq=run["validation_interval"],
            target_label_tokens=run["validation_label_tokens"],
            global_batches=run["validation_global_batches"], dataloader=PackedDataLoader.Config(
                manifest_path=str(prepared / "manifest.json"), split="validation", shuffle=False,
                infinite=run["validation_infinite"])),
        activation_checkpoint=None if run["activation_checkpointing"] == "none" else FullAC.Config(),
        compile=CompileConfig(enable=False),
        debug=DebugConfig(seed=run["seed"]),
    )


_GEOMETRY = {
    "vocab_size": 50257, "dim": 128, "hidden_dim": 256,
    "num_heads": 4, "num_kv_heads": 2, "num_experts": 8,
    "top_k": 2, "max_seq_len": 2048,
}
_GEOMETRY_KEYS = set(_GEOMETRY) | {"qk_norm", "rms_eps", "rope_theta", "init_std", "always_on_hidden_dim"}


def _model(geometry: dict[str, Any], **topology: Any) -> ModelConfig:
    unknown = set(geometry) - _GEOMETRY_KEYS
    if unknown:
        raise ValueError(f"unsupported geometry fields: {sorted(unknown)}")
    return ModelConfig(**(_GEOMETRY | geometry | topology))


_COMMON_STACK = (
    "Common RMSNorm/RoPE/SwiGLU cells, softmax top-k with selected-weight normalization and dropless dispatch.",
    "Common AdamW and independently owned input embedding/output head; no original-paper quality or scaling claim.",
)


def _spec(name: str, model: ModelConfig, *, source: dict[str, Any] | None = None,
          preserved: tuple[str, ...] = (), adaptations: tuple[str, ...] = (),
          training: TrainingRecipe | None = None, group_size: int = 1, dense: bool = False) -> RecipeSpec:
    """``dense`` marks a source path that is itself dense; other named methods need sparse cells."""
    if source is not None and not dense and any(model.region_config(region).num_experts < 1
                                                for region in ("prefix", "core", "suffix")):
        raise ValueError("named methods require sparse cells; use dense_spec for a dense control")
    return RecipeSpec(
        name=name, model=model, training=training or TrainingRecipe(num_visits=model.num_visits),
        source=source or {"kind": "component", "version": 1},
        preserved=preserved,
        adaptations=adaptations + (
            "Public prepared-text fixture replaces the original data/training stack; its manifest is a separate run identity.",
            *_COMMON_STACK,
        ),
        optimizer={"name": "AdamW", "base_lr": 3e-4, "betas": [0.9, 0.95],
                   "eps": 1e-8, "weight_decay": 0.1,
                   "zero_weight_decay_roles": ["norm", "embedding", "lm_head", "router"],
                   "expert_group_size": group_size, "expert_lr_multiplier": 1 / math.sqrt(group_size)},
        inference={"num_visits": model.num_visits, "readout": "terminal",
                   "generation": "fixed_depth", "cache": "logical_visit_slots",
                   "qualification_evidence": []},
        run={"seq_len": min(2048, model.max_seq_len), **copy.deepcopy(RUN_DEFAULTS),
             "manifest_identity": None},
    )


def tiny_spec() -> RecipeSpec:
    """Small installation/development component, separate from method qualification."""
    model = _model({"num_experts": 4, "max_seq_len": 1024}, num_cells=1, num_visits=2)
    spec = _spec("tiny", model)
    # A short run with no fixed global batch: TorchTitan derives it from the ranks.
    spec.run.update(global_batch_positions=None, local_batch_size=1, successful_updates=100,
                    schedule_total_steps=1000, validation_interval=50)
    return spec


def recursive_moe_spec(**geometry: Any) -> RecipeSpec:
    """A shared-cell component; paper names are reserved for selected paths."""
    return _spec("recursive_moe", _model(geometry, num_cells=2, num_visits=4))


def dense_spec(*, layers: int = 8, **geometry: Any) -> RecipeSpec:
    return _spec("dense", _model(geometry, num_experts=0, num_cells=layers, num_visits=1),
                 training=TrainingRecipe(num_visits=1, balance_coeff=0.0))


def moe_spec(*, layers: int = 8, **geometry: Any) -> RecipeSpec:
    """Standard MoE: independent sparse layers; matching is an explicit contract."""
    return _spec("moe", _model(geometry, num_cells=layers, num_visits=1),
                 preserved=("Every logical layer has independently owned attention, router, norms and experts.",))


def smelt_spec(*, physical_layers: int = 8, **geometry: Any) -> RecipeSpec:
    """Middle half twice; an odd outside remainder puts its extra cell last.

    Ten physical layers are 2-prefix/5-core/3-suffix. Existing multiples of
    four retain their symmetric split and serialized parameter ownership.
    """
    if type(physical_layers) is not int or physical_layers < 4 or physical_layers % 2:
        raise ValueError("SMELT physical_layers must be an even integer (at least four)")
    model = _model(geometry, prefix_cells=physical_layers // 4,
                   num_cells=physical_layers // 2,
                   suffix_cells=physical_layers - physical_layers // 4 - physical_layers // 2,
                   num_visits=2, core=RegionConfig(residual_scale=0.5))
    return _spec("smelt", model,
                 source={"method": "SMELT", "version": "v3", "url": "https://arxiv.org/html/2609.01343v3",
                         "sections": ["3", "Appendix A"], "selected_path": "middle_half_R2_core_only_scale_0.5"},
                 preserved=("Middle half of physical layers is executed twice, with shared core owners.",
                            "Both core attention/FFN residuals use 1/2; independent prelude/coda use 1.",
                            "Fixed depth, full BPTT, terminal cross entropy."))


def latentloop_residual_spec(*, physical_layers: int = 8, **geometry: Any) -> RecipeSpec:
    """LatentLoop Section 3.1's fixed per-visit residual: SMELT's split, core visits scaled by 0.25 then 1.

    Not the paper's final LatentLoop method (2x pooled first visit, scaled decoder,
    wider model), which needs sequence pooling; the notes say so.
    """
    model = replace(smelt_spec(physical_layers=physical_layers, **geometry).model,
                    core=None, visit_residual_scales=(.25, 1.))
    return _spec("latentloop_residual", model,
                 source={"method": "LatentLoop", "version": "v1", "url": "https://arxiv.org/html/2610.19975v1",
                         "sections": ["2.1", "3.1", "Figure 2", "Table 5"],
                         "selected_path": "middle_half_R2_shared_fixed_asymmetric_residual_0.25_1"},
                 preserved=("Middle half of physical layers is executed twice with every core owner shared "
                            "(attention, router, experts, norms): the paper's MoE Loop (Section 2.1).",
                            "Every core attention/FFN residual branch is scaled by 0.25 in the first visit and by 1 "
                            "in the second; independent prelude/coda use 1. (0.25, 1) is the paper's best fixed pair "
                            "at 1.3B and 3B, below (1, 1) and (0.5, 0.5) (Section 3.1, Figure 2b).",
                            "Fixed depth, full BPTT, terminal cross entropy."),
                 adaptations=("This is LatentLoop Section 3.1's fixed (0.25, 1) per-visit residual finding on "
                              "SMELT's loop, not the paper's final LatentLoop method. That method runs the first "
                              "visit on a 2x mean-pooled sequence, writes it back through a learned scaled decoder "
                              "and spends the saved compute on a wider model (Sections 3.2-3.4); it is not "
                              "reproduced here because it needs sequence pooling, which the common model lacks.",
                              "Fixed (0.25, 1) instead of learned per-visit coefficients: the learned ones end near "
                              "(0.27, 0.87) at 1.3B and (0.36, 1.09) at 3B, but the paper reports validation NLL "
                              "only for fixed pairs (Section 3.1).",
                              "Attention stays shared across visits: at matched parameters, untying it in the "
                              "full-resolution MoE Loop raised validation NLL from 2.0293 to 2.0319 (Table 5).",
                              "The common AdamW, schedule and run defaults replace the paper's training settings "
                              "(AdamW, peak LR 1e-3 annealed to 1e-4 by 1-sqrt over the last 20%, 500 warmup "
                              "updates, router aux 1e-3, FineWeb-Edu with the 32K TinyLlama tokenizer, 8 query and "
                              "2 KV heads with QK norm)."))


_P1_GEOMETRY = ("P1 public-stack expert intermediate/head geometry and independent 151669-token embedding/head are explicit project choices, not disclosed paper hyperparameters.",)


def _p1_spec(spec: RecipeSpec, *, base_lr: float, geometry_notes: tuple[str, ...] = _P1_GEOMETRY) -> RecipeSpec:
    """Resolve one common P1 launch identity; no runtime measurements implied.

    ``spec`` is a plain moe/smelt template, whose only adaptations are the
    common-stack notes; DCLM takes the place of the fixture-data note and
    ``geometry_notes`` say how the widths were chosen. Study policy (the peak-LR
    trials, diagnostic updates, final evaluation sizes) is recorded in the
    frozen ``recipes/p1_*.json`` and ``recipes/p2_*.json``, not here.
    """
    positions = 256 * 4096
    stable_labels, decay_labels, warmup_labels = 10_000 * positions, 10_000_000_000, 20_000_000
    total_steps = math.ceil((stable_labels + decay_labels) / positions)
    stable_end_steps = math.ceil(stable_labels / positions)
    decay_steps = total_steps - stable_end_steps
    schedule = {
        "name": "warmup_stable_cosine_decay",
        "coordinate": "successful_optimizer_updates",
        "requested_coordinate": "global_valid_label_tokens",
        "target_label_tokens": stable_labels + decay_labels,
        "warmup_label_tokens": warmup_labels,
        "stable_end_label_tokens": stable_labels,
        "decay_label_tokens": decay_labels,
        "total_steps": total_steps,
        "warmup_steps": math.ceil(warmup_labels / positions),
        "stable_end_steps": stable_end_steps,
        "decay_steps": decay_steps,
        "decay_ratio": decay_steps / total_steps,
        "decay_type": "cosine", "min_lr_factor": .1,
        "rounded_nominal_positions": total_steps * positions,
        "rounding": "ceil(requested labels / nominal global positions per update); actual labels must be reported separately",
        "native_step_convention": "Pinned TorchTitan LambdaLR: first update uses index 0; virtual final stable step keeps index 10000 at peak and index 19537 at min_lr_factor.",
        "label_position_policy": "Fresh document packing normally fills every position; padding is not a label. Verify actual endpoint labels and report any shortfall, never infer labels from step count.",
        "maximum_endpoint_rounding_positions": positions,
    }
    return replace(spec, name=f"p1_{spec.name}",
        optimizer=spec.optimizer | {"base_lr": base_lr, "max_norm": 1.0},
        source=spec.source | {"anchor": "P1", "paper_reference": {
            "url": "https://arxiv.org/html/2609.01343v3", "sections": ["3.1", "Appendix A", "Appendix B"],
            "baseline_active_tier": "100M", "non_embedding_unique_tier": ".71B",
            "private_stack_reproduction": False}},
        adaptations=_COMMON_STACK + (
            "Publicly reconstructible DCLM replaces the original private corpus; the frozen manifest and tokenizer are separate comparison identities.",
            *geometry_notes,
            "Normal matrix initialization std=.02, unit norm scales, aux coefficient .01, 20M-label linear warmup and final LR factor .1 are public-stack adaptations requiring the bounded stability pilot.",
            "Physical active counts name one traversal of unique cells; repeated core calls still incur compute.",
        ),
        run=spec.run | {
            "anchor": "P1", "seq_len": 4096, "global_batch_positions": positions,
            "successful_updates": total_steps, "schedule_total_steps": total_steps,
            "schedule": schedule, "seed": 42, "data_infinite": False,
            "mixed_precision_param": "bfloat16",
            "validation_interval": math.ceil(500_000_000 / positions),
            "validation_label_tokens": 1_000_000, "validation_infinite": False,
            "checkpoint_interval": math.ceil(500_000_000 / positions),
            "training_seeds": [42], "activation_checkpointing": "none",
        })


def p1_moe_spec(*, base_lr: float = 3e-4) -> RecipeSpec:
    """P1 standard MoE: public-stack geometry near the 100M-active tier."""
    return _p1_spec(moe_spec(layers=10, vocab_size=151669, dim=672,
        hidden_dim=544, num_heads=7, num_kv_heads=1, num_experts=64,
        top_k=8, max_seq_len=4096), base_lr=base_lr)


def p1_smelt_spec(*, base_lr: float = 3e-4, hidden_dim: int = 440) -> RecipeSpec:
    """P1 middle-half SMELT; F440 is the trained arm, ``recipes/p1_smelt.json``.

    ``smelt_p1_pair`` selects F440 when it counts the document-causal attention
    pairs measured on the frozen P1 training prefix, the convention P2 also
    uses. Its dense-S² default counts every pair of a 4096 window and selects
    F424, a shape that was never trained.
    """
    return _p1_spec(smelt_spec(physical_layers=10, vocab_size=151669, dim=576,
        hidden_dim=hidden_dim, num_heads=9, num_kv_heads=1, num_experts=96,
        top_k=8, max_seq_len=4096), base_lr=base_lr)


def _p2_spec(spec: RecipeSpec, *, base_lr: float) -> RecipeSpec:
    """Reuse the P1 training controls, with a fresh second-scale identity."""
    resolved = _p1_spec(spec, base_lr=base_lr, geometry_notes=())
    return replace(resolved, name=f"p2_{spec.name}",
        source=resolved.source | {"anchor": "P2", "paper_reference":
            resolved.source["paper_reference"] | {
                "baseline_active_tier": "200M", "non_embedding_unique_tier": "1.32B"}},
        adaptations=resolved.adaptations + (
            "P2 public-stack intermediate/head geometry is selected by a predeclared 32-shape analytical budget screen, without quality observations.",
            "SMELT expert count (default E88 instead of the paper's E96) is an explicit resource-matching adaptation; expert pool and head dimensions differ, so this is a recipe comparison, not a pure loop ablation.",
            "P1 initialization, AdamW, auxiliary reduction and full WSD controls are inherited; formal P2 source/data/evaluation identities must be frozen independently.",
        ),
        run=resolved.run | {"anchor": "P2", "panel": "P2", "local_batch_size": 16})


def p2_moe_spec(*, base_lr: float = 3e-4) -> RecipeSpec:
    """Second-scale standard MoE, near the 200M-active / 1.32B-unique tier."""
    return _p2_spec(moe_spec(layers=12, vocab_size=151669, dim=1280,
        hidden_dim=432, num_heads=10, num_kv_heads=1, num_experts=64,
        top_k=8, max_seq_len=4096), base_lr=base_lr)


def p2_smelt_spec(*, base_lr: float = 3e-4, hidden_dim: int = 384,
                  num_experts: int = 88) -> RecipeSpec:
    """P2 middle-half R2 path; E88/F384 is the screen's choice under measured document pairs."""
    return _p2_spec(smelt_spec(physical_layers=12, vocab_size=151669, dim=1056,
        hidden_dim=hidden_dim, num_heads=12, num_kv_heads=1, num_experts=num_experts,
        top_k=8, max_seq_len=4096), base_lr=base_lr)


def sharing_panel_specs(*, horizon: int = 2048) -> dict[str, RecipeSpec]:
    """C: a same-shape attention x router owner grid, not iso-parameter arms."""
    if type(horizon) is not int or horizon <= 64:
        raise ValueError("C horizon must be an integer greater than the 64-update warmup")
    panel = {}
    for attention in (False, True):
        for router in (False, True):
            name = f"a{int(attention)}_r{int(router)}"
            model = _model({"vocab_size": 151669, "dim": 512, "hidden_dim": 1024,
                "num_heads": 8, "num_kv_heads": 2, "num_experts": 8, "top_k": 2},
                num_cells=2, num_visits=4, attention_per_visit=attention, router_per_visit=router)
            spec = _spec(name, model, source={"kind": "attention_router_ownership_grid", "version": 1},
                preserved=("Two core banks in stack order, four visits, shared experts and norms; only attention/router owners vary.",
                           "Fixed-depth full BPTT, terminal CE and identical initialization/optimizer rules; independent owners increase unique parameters."))
            panel[name] = replace(spec, run=spec.run | {
                "panel": "C", "successful_updates": horizon, "schedule_total_steps": horizon,
                "schedule": {"name": "warmup_cosine_decay", "coordinate": "successful_optimizer_updates",
                    "total_steps": horizon, "warmup_steps": 64, "decay_type": "cosine",
                    "decay_ratio": (horizon - 64) / horizon, "min_lr_factor": .1},
                "seed": 42, "training_seeds": [42], "data_infinite": False,
                "validation_infinite": False, "activation_checkpointing": "none"})
    return panel


def meta_spec(*, core_layers: int = 2, passes: int = 4,
              prelude_layers: int = 1, coda_layers: int = 1, **geometry: Any) -> RecipeSpec:
    """Selected Meta middle-loop path, with explicit common-stack replacements."""
    if min(prelude_layers, coda_layers) < 1:
        raise ValueError("Meta middle-loop path requires independent prelude and coda")
    model = _model(geometry, prefix_cells=prelude_layers, num_cells=core_layers,
                   suffix_cells=coda_layers, num_visits=passes)
    return _spec("meta", model,
                 source={"method": "Meta Looped MoE", "version": "v1", "url": "https://arxiv.org/html/2609.40316v1",
                         "sections": ["C.1", "C.2"], "selected_path": "middle_loop_fixed_depth_full_BPTT_terminal"},
                 preserved=("Independent prelude/coda surround a shared sparse middle stack.",
                            "Fixed loop depth, full BPTT and terminal CE; parameter/sparsity/data facts are reported separately."),
                 adaptations=("Original routing is replaced by common softmax top-k routing.",
                              "Original dispatch/capacity handling is replaced by dropless dispatch.",
                              "Original optimizer background is replaced by AdamW.",
                              "Embedding and output head are independently owned in the common stack."))


def foil_base_spec(*, core_layers: int = 4, passes: int = 1,
                   prelude_layers: int = 1, coda_layers: int = 1, **geometry: Any) -> RecipeSpec:
    """The flattening control uses independent attention at every core visit.

    E is the geometry's ``num_experts``, 4 if it is omitted (other constructors: 8).
    """
    if min(prelude_layers, coda_layers) < 1:
        raise ValueError("Foil requires independent prelude and coda")
    geometry = {"num_experts": 4} | geometry
    num_experts = geometry["num_experts"]
    model = _model(geometry, num_cells=core_layers,
                   num_visits=passes, prefix_cells=prelude_layers, suffix_cells=coda_layers,
                   attention_per_visit=True,
                   prefix=RegionConfig(num_experts=num_experts),
                   core=RegionConfig(num_experts=num_experts), suffix=RegionConfig(num_experts=num_experts))
    return _spec("foil_base", model,
                 source={"method": "Foil Base", "version": "v1", "url": "https://arxiv.org/html/2609.35751v1",
                         "sections": ["2.1", "2.2"], "selected_path": "unflattened_per_visit_attention",
                         "E_D_L": [num_experts, core_layers, passes]},
                 preserved=("Base has independent per-visit attention, shared experts/router/norms within each core bank.",
                            "Prelude/coda are independent external banks; Base-tied attention is a separate ablation."))


def foil_spec(*, core_layers: int = 4, passes: int = 1,
              prelude_layers: int = 1, coda_layers: int = 1, **geometry: Any) -> RecipeSpec:
    """Foil (E,D,L)->(2E,D/2,2L); D counts physical core banks.

    E is the geometry's ``num_experts``, 4 if it is omitted (other constructors: 8).
    """
    if type(core_layers) is not int or core_layers < 2 or core_layers % 2:
        raise ValueError("Foil core_layers D must be positive and even")
    if min(prelude_layers, coda_layers) < 1:
        raise ValueError("Foil requires independent prelude and coda")
    geometry = {"num_experts": 4} | geometry
    num_experts = geometry["num_experts"]
    model = _model(geometry, num_cells=core_layers // 2,
                   num_visits=2 * passes, prefix_cells=prelude_layers, suffix_cells=coda_layers,
                   attention_per_visit=True,
                   prefix=RegionConfig(num_experts=num_experts),
                   core=RegionConfig(num_experts=2 * num_experts), suffix=RegionConfig(num_experts=num_experts))
    return _spec("foil", model,
                 source={"method": "Foil", "version": "v1", "url": "https://arxiv.org/html/2609.35751v1",
                         "sections": ["2.1", "2.2"], "selected_path": "flatten_core_per_visit_attention",
                         "base_E_D_L": [num_experts, core_layers, passes],
                         "selected_E_D_L": [2 * num_experts, core_layers // 2, 2 * passes]},
                 preserved=("Flatten only core: (E,D,L)->(2E,D/2,2L); D is physical bank count.",
                            "Residual and single-expert widths stay fixed; experts/router/norms are shared within each bank.",
                            "Base and Foil both have per-visit attention: DL attention owners are preserved.",
                            "Independent prelude/coda retain the external Base expert count E."))


def expert_tying_spec(*, groups: int = 2, group_size: int = 4, **geometry: Any) -> RecipeSpec:
    """Consecutive groups share experts only, with the expert-only LR rule."""
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    model = _model(geometry, num_cells=groups, num_visits=group_size, order="layer",
                   attention_per_visit=True, router_per_visit=True, norm_per_visit=True)
    return _spec("expert_tying", model, group_size=group_size,
                 source={"method": "Tying", "version": "v1", "url": "https://arxiv.org/html/2606.16825v1",
                         "sections": ["3", "Appendix B"], "selected_path": "consecutive_expert_only_groups",
                         "group_topology": [[i * group_size + j for j in range(group_size)] for i in range(groups)]},
                 preserved=("Each consecutive group of g logical layers owns one expert bank.",
                            "Every attention/router/norm is independent per logical layer.",
                            "Canonical core expert LR is base_lr/sqrt(g), other LRs are unchanged; WD coefficient is uncompensated."),
                 adaptations=("Common AdamW replaces the paper's Muon optimizer background.",))


def per_visit_router_spec(**geometry: Any) -> RecipeSpec:
    """Router-sharing component only; not a complete LOOM recipe."""
    return _spec("per_visit_router", _model(geometry, num_cells=2, num_visits=4, router_per_visit=True))


def input_anchor_spec(**geometry: Any) -> RecipeSpec:
    """Live input-anchor/sampled-tail component; not a Huginn reproduction."""
    model = _model(geometry, num_cells=2, num_visits=4, input_injection="linear", initial_state="zero")
    return _spec("input_anchor", model,
                 training=TrainingRecipe(num_visits=4, sampled_visits=(2, 4, 6), grad_visits=2, depth_seed=42))


def huginn_spec(
    *, core_layers: int = 2, eval_passes: int = 4, grad_visits: int = 2,
    depth_mean: float = 4.0, depth_log_std: float = 0.5,
    latent_init_std: float = math.sqrt(2 / 5), latent_truncation: float = 3.0,
    **geometry: Any,
) -> RecipeSpec:
    """Random latent, live prelude anchor and sampled tail-gradient training."""
    model = _model(geometry, prefix_cells=1, num_cells=core_layers, suffix_cells=1,
                   num_visits=eval_passes, input_injection="linear", initial_state="random",
                   latent_init_std=latent_init_std, latent_truncation=latent_truncation)
    training = TrainingRecipe(num_visits=eval_passes, grad_visits=grad_visits,
        depth_seed=42, depth_rule="poisson_lognormal_micro_v1",
        depth_mean=depth_mean, depth_log_std=depth_log_std)
    spec = _spec("huginn", model, training=training,
        source={"method": "Huginn-inspired MoE", "version": "v2",
                "url": "https://arxiv.org/html/2502.05171v2", "sections": ["3", "4.1"],
                "selected_path": "random_latent_live_anchor_sampled_tail_gradients"},
        preserved=("Independent prelude supplies a live anchor to every recurrent stack visit; independent coda reads the state.",
                   "Training starts from an independent truncated-normal latent, separate from parameter initialization.",
                   "Shifted Poisson-lognormal depth is sampled per microbatch without a hard cap; only the last min(k,r) visits differentiate."),
        adaptations=("Common pre-norm sparse cells replace the original dense sandwich/core-boundary normalization; original Dense performance is not claimed.",
                     "Mean depth, lognormal spread and gradient-tail length are explicit small-model adaptations.",
                     "Fixed-depth evaluation starts from zero deterministically; this is an evaluation adaptation, not the random training initial-state distribution.",
                     "The first delivery uses uncached fixed-depth generation."))
    spec.inference.update(initial_state="zero", cache="uncached")
    return spec


def loom_spec(
    *, core_layers: int = 2, passes: int = 3, attention_per_visit: bool = False,
    memory_scale: float = 0.5, memory_beta: float = 0.5, **geometry: Any,
) -> RecipeSpec:
    """Full-BPTT two-memory updates, optionally with independent attention."""
    model = _model(geometry, num_cells=core_layers, num_visits=passes,
                   router_per_visit=True, attention_per_visit=attention_per_visit,
                   state_update="ema", memory_scale=memory_scale, memory_beta=memory_beta)
    adaptations = ("The first delivery uses uncached fixed-depth generation; forward-local memory resets for every new forward.",)
    if attention_per_visit:
        adaptations += ("Independent per-visit attention is an explicit structure-state hybrid; all memory and reinjection equations are retained.",)
    spec = _spec("loom_attention" if attention_per_visit else "loom", model,
        source={"method": "LOOM", "version": "v1", "url": "https://arxiv.org/html/2610.01153v1",
                "sections": ["3", "4.2.1"], "selected_path": "full_BPTT_dual_EMA_per_visit_router",
                "attention_per_visit": attention_per_visit},
        preserved=("Gamma=lambda/(H*sqrt(M)); later visits reinject embedding input with g_t=lambda/(t*sqrt(M)).",
                   "Global EMA spans layers and visits; local EMA resets per visit. Both read after their current attention write.",
                   "Attention write enters the two EMA numerators once; their normalized reads are added without division by two.",
                   "Routers are independent per visit; experts and norms are shared. Full BPTT retains both memory numerator gradients.",
                   "Every new forward starts with empty memory; no segment-wise detach or optimizer step."),
        adaptations=adaptations)
    spec.inference.update(cache="uncached")
    return spec


def ouro_spec(
    *, core_layers: int = 2, passes: int = 3, entropy_coeff: float = 0.1,
    **geometry: Any,
) -> RecipeSpec:
    """Stage-I full-depth training with tokenwise expected CE and entropy."""
    model = _model(geometry, num_cells=core_layers, num_visits=passes,
                   exit_distribution=True)
    spec = _spec("ouro", model,
        training=TrainingRecipe(num_visits=passes, objective="expected_ce", entropy_coeff=entropy_coeff),
        source={"method": "Ouro", "version": "v5", "url": "https://arxiv.org/html/2510.25741v5",
                "sections": ["3.2", "3.3", "3.4"],
                "selected_path": "stage_I_full_depth_tokenwise_expected_CE_entropy"},
        preserved=("A shared linear gate defines tokenwise conditional halting hazards; the final exit receives all remaining survival probability.",
                   "Training minimizes sum_r p_r*CE_r minus beta*H(p), with full gradients through the gate, backbone and unique head.",
                   "Every visit is executed; terminal logits provide common fixed-depth evaluation and generation."),
        adaptations=("Shared gate weights and bias start at zero; the entropy coefficient is an explicit public-stack choice.",
                     "Each visit reads through the common final norm and unique output head, with no independent coda.",
                     "Stage II, token-level early exit and missing-KV strategies are outside this selected path.",))
    spec.inference.update(labels_loss="terminal_ce_only")
    return spec


def sandwich_spec(
    *, dense: bool = False, prelude_layers: int = 2, core_layers: int = 3, passes: int = 8,
    coda_layers: int = 2, **geometry: Any,
) -> RecipeSpec:
    """Part I middle loop with prelude injection; the default P2-R3x8-C2 keeps the 2:3:2 ratio of P8-R12x8-C8.

    ``dense=True`` is the Q2 path of Section 3.4.2 (``dense_middle_prelude_injection``
    in the reference code). The default MoE path replaces every FFN as in
    Section 4.1 and keeps z_0 = e (the released ``moe_middle_input_injection``);
    the paper's MoE tables report the random-state Huginn MoE instead.
    Select the dense path with ``dense=True``; ``num_experts=0`` alone is
    rejected like any named sparse method.
    """
    if min(prelude_layers, coda_layers) < 1:
        raise ValueError("the sandwich requires an independent prelude and coda")
    model = _model(geometry | ({"num_experts": 0} if dense else {}), prefix_cells=prelude_layers,
                   num_cells=core_layers, suffix_cells=coda_layers, num_visits=passes,
                   input_injection="diag_decay", injection_dt=1.0, injection_decay=1.0)
    ffn = (("Every physical FFN is a dense SwiGLU, with no auxiliary loss.",) if dense else
           ("Every physical FFN is a top-2 expert bank without shared experts; the balance loss "
            "E*sum(f_i*P_i), averaged over calls and microbatches, has coefficient 0.001.",))
    ffn_adaptations = (("The dense SwiGLU width is the common hidden_dim instead of 4352.",) if dense else
                       ("Softmax top-k scores replace the paper's sigmoid top-2 scores; the paper's 25 experts "
                        "of width 2432 become the common expert count and width.",
                        "The Section 4.1 MoE tables report the random-state Huginn MoE; this recipe is the released "
                        "input-initialized path (moe_middle_input_injection)."))
    return _spec("sandwich_dense" if dense else "sandwich", model, dense=dense,
        training=TrainingRecipe(num_visits=passes, balance_coeff=0.0 if dense else 0.001),
        source={"method": "Looped Models Done Right, Part I", "version": "v1",
                "url": "https://arxiv.org/html/2610.23036v1",
                "sections": ["2.1", "3.1", "3.4.2", "4.1", "Appendix A"],
                "selected_path": "dense_middle_prelude_injection" if dense else "moe_middle_input_injection",
                "reference_code": "https://github.com/ifm-ai/xllm-loop/blob/755c7b3/xllm/paper_part1/recipes.py",
                "P_R_T_C": [prelude_layers, core_layers, passes, coda_layers]},
        preserved=("Independent prelude and coda stacks surround one tied core stack run `passes` times in stack order.",
                   "Before every core pass the state becomes exp(-dt*a)*z + dt*(W e) with the fixed prelude output e: "
                   "per-channel dt = softplus(.), a = exp(.), full W; W = I, a = 1 and dt = 1 at initialization, "
                   "and no RMSNorm on e.",
                   "The recurrent state starts at the prelude output (z_0 = e); fixed depth, full BPTT and terminal CE.")
                  + ffn,
        adaptations=("P2-R3x8-C2 keeps the paper's 2:3:2 prelude/core/coda ratio and eight passes at the common small "
                     "geometry; the 1536-wide, 24/6-head, RoPE-base-1e6 paper cells are not reproduced.",
                     *ffn_adaptations,
                     "The paper's sequence length 8192 and global batch of 512 sequences, and the reference code's "
                     "250,624-token vocabulary, are replaced by the common run defaults and vocabulary.",
                     "The common AdamW replaces the paper's AdamW (peak LR swept over 4e-4 to 8e-4, 200 warmup steps, "
                     "cosine decay, weight decay 0.1) and the reference code's settings (peak LR 6e-4, betas "
                     "0.9/0.98, decay to 0.01x rather than the common 0.1x, weight decay on every parameter rather "
                     "than exempting norms, embedding, head and router).",
                     "Common normal(init_std) matrices replace the reference truncated-normal init "
                     "(std 0.02 cut at 3 std, embedding and head std 1/sqrt(d))."))


def relaxed_spec(
    *, dense: bool = False, core_layers: int = 4, passes: int = 2, rank: int = 16, **geometry: Any,
) -> RecipeSpec:
    """Relaxed Recursive Transformer: one shared block looped, with a LoRA per layer and loop.

    ``dense=True`` is the paper's path: LoRA on every shared linear weight
    (q, k, v, out and SwiGLU w1, w3, w2). The default MoE path puts LoRA on
    the attention projections only; routers and routed experts stay shared.
    Rank 16 at width 128 has the ratio of rank 256 at Gemma's width 2048, one
    of the paper's 64-512 settings.
    """
    model = _model(geometry | ({"num_experts": 0} if dense else {}), num_cells=core_layers,
                   num_visits=passes, lora_attention_rank=rank, lora_ffn_rank=rank if dense else 0)
    ffn = (f"dense FFN, LoRA rank {rank}" if dense else
           f"{model.num_experts} experts top-{model.top_k}, attention LoRA rank {rank}")
    lora = (("Every visit of every shared layer owns a LoRA on q, k, v, out, w1, w3 and w2.",) if dense else
            ("Every visit of every shared layer owns a LoRA on q, k, v and out.",))
    ffn_adaptations = (("The FFN is the common SwiGLU (w1, w3, w2): TinyLlama's (Llama) FFN form, while Gemma uses "
                        "GeGLU (Appendix A) and Pythia a two-matrix GeLU MLP with LayerNorm, which are not reproduced.",) if dense else
                       ("Every FFN is a shared top-k expert bank; routers and routed experts take no LoRA, "
                        "so only attention is relaxed.",
                        "MoE path: balance loss coefficient 0.01, the repo default; the paper has no MoE."))
    return _spec("relaxed_dense" if dense else "relaxed", model, dense=dense,
        training=TrainingRecipe(num_visits=passes, balance_coeff=0.0) if dense else None,
        source={"method": "Relaxed Recursive Transformers", "version": "v3",
                "url": "https://arxiv.org/html/2410.20672v3",
                "sections": ["2.2", "2.3", "3.5", "Appendix C"],
                "selected_path": "cycle_layerwise_lora_zero_init" + ("" if dense else "_attention"),
                "layers_loops_rank": [core_layers, passes, rank]},
        preserved=("CYCLE sharing: one block of shared layers, norms included, runs `passes` times in stack order "
                   "with no independent prelude or coda.",
                   *lora,
                   "Each LoRA adds B A x to W' x with no alpha scaling; the loop-l LoRA is used only at loop l.",
                   "Zero initialization: B = 0 and Gaussian A, so training starts at the plain Recursive "
                   "Transformer; every parameter, shared or LoRA, trains.",
                   "Fixed depth, full BPTT and terminal CE."),
        adaptations=(f"{core_layers} shared layers x {passes} loops at the common small geometry (width {model.dim}, "
                     f"{model.num_heads} heads, {model.num_kv_heads} KV heads, "
                     f"{ffn}); the paper's "
                     "Gemma 2B (18 layers as 9 shared x 2 loops), TinyLlama 1.1B (22 layers) and Pythia 1B, each with "
                     "two looping blocks (Gemma also three), are not reproduced.",
                     "Training starts from random weights instead of uptraining a pretrained Gemma, TinyLlama or "
                     "Pythia, so Stepwise/Average/Lower looped-layer and truncated-SVD LoRA initializations do not "
                     "apply; LoRA.svd_init provides eq. 4 for converted full-size weights.",
                     "A draws from the common normal(init_std) rather than an unstated Gaussian.",
                     *ffn_adaptations,
                     "One rank for all attention projections and one for the FFN replace the paper's per-component "
                     "rank ablation (Table G.1).",
                     "The common AdamW, schedule and run defaults replace the paper's uptraining settings "
                     "(AdamW, LR 2e-4, cosine, SlimPajama, context 2048, about 2M-token batches).",
                     "Early exit and continuous depth-wise batching (Section 2.4) are outside this selected path."))


def tiny():
    return trainer_config(tiny_spec())


def dense():
    return trainer_config(dense_spec())


def moe():
    return trainer_config(moe_spec())


def recursive_moe():
    return trainer_config(recursive_moe_spec())


def smelt():
    return trainer_config(smelt_spec())


def meta():
    return trainer_config(meta_spec())


def latentloop_residual():
    return trainer_config(latentloop_residual_spec())


def foil_base():
    return trainer_config(foil_base_spec())


def foil():
    return trainer_config(foil_spec())


def loom():
    """The full memory-based method; per_visit_router() remains a component."""
    return trainer_config(loom_spec())


def huginn():
    return trainer_config(huginn_spec())


def ouro():
    return trainer_config(ouro_spec())


def sandwich():
    return trainer_config(sandwich_spec())


def relaxed():
    return trainer_config(relaxed_spec())


def expert_tying():
    return trainer_config(expert_tying_spec())


def per_visit_router():
    return trainer_config(per_visit_router_spec())


def input_anchor():
    return trainer_config(input_anchor_spec())


def injected():
    """Compatibility alias for the input-anchor component, not full Huginn."""
    warnings.warn("injected() is a component alias; use input_anchor() (not full Huginn)", DeprecationWarning, stacklevel=2)
    return input_anchor()
