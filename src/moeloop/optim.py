"""Native optimizer groups by physical parameter owner (TorchTitan 086bf6c).

``optimizer_config`` turns a recipe's ``optimizer`` mapping into native
``ParamGroupConfig`` entries; first match wins, so each physical parameter is
in exactly one group. A shared owner's gradient is already the sum over its
calls, so it receives one update per optimizer step.

- AdamW: declared zero-decay roles, then every other parameter.
- Muon: TorchTitan's DistMuon updates every hidden matrix that
  ``MUON_PATTERN`` names (attention projections, expert banks,
  dense/always-on FFNs, input-adapter projections, per-visit LoRA A and B);
  AdamW, configured as
  above, updates embedding, lm_head, routers, the exit gate, norms and every
  1D parameter (Moonlight, arXiv 2502.16982, Sec. 2.2). Each expert matrix is
  orthogonalized separately; an attention projection as one matrix. A new
  matrix owner must be added to ``MUON_PATTERN``; tests/training/test_muon.py
  checks the pattern against every recipe constructor.
  This is nanochat's ``setup_optimizer`` partition (block matrices on Muon;
  embedding, lm_head and scalars on AdamW) and Moonlight's, except that routers
  stay on AdamW, unlike Moonlight's generic >=2D rule, for Infra E1 comparability.

The pinned DistMuon holds exactly one parameter group. Core experts, which
carry ``expert_lr_multiplier``, therefore use a second DistMuon instance
(``CORE_EXPERT_MUON``) whose single LR ``scale_core_expert_lr`` multiplies; the
weight-decay coefficient stays uncompensated, as for AdamW. DistMuon applies
decay as ``1 - lr * weight_decay`` with the unadjusted group LR.

``output_lr_scale`` s (``ModelConfig``, core region) gives the core output projections
(``CORE_OUTPUT_PATTERN``) LR x s and weight decay / s, so lr * weight_decay per step is kept:
under AdamW one more AdamW group, under Muon a third DistMuon instance (``CORE_OUTPUT_MUON``).
Under Muon it cannot be combined with ``expert_lr_multiplier``, which would split that instance.

Expert banks keep Shard(0) storage under Muon at every FSDP size
(``TitanModel.Config.dist_muon``): the pinned DistMuon cannot change S(1)
storage to S(0) compute, so FSDP2 pads the expert dimension instead.

Limits, verified only on GPU by #49 (CPU tests run DistMuon on one rank):
- DistMuon accepts only DTensor parameters on CUDA, so a Muon recipe trains
  only under the native FSDP path on GPU.
- Multi-rank redistribution (owner gathers, expert shards with fewer experts
  than ranks, HSDP replicas) runs on CUDA streams.
- Every Muon parameter needs a gradient at every step; an owner unused by a
  step fails loudly instead of being skipped as AdamW would.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned, build_dist_muon

from .model import CORE_OUTPUT_PATTERN

ZERO_DECAY_ROLES = ("norm", "embedding", "lm_head", "router")
MUON_FIELDS = ("lr", "momentum", "nesterov", "ns_steps", "ns_coefficients", "eps", "weight_decay", "adjust_lr_fn")
CORE_EXPERT_MUON = "DistMuonCoreExperts"
CORE_OUTPUT_MUON = "DistMuonCoreOutputs"
# The one grouping rule, matched against raw FQNs (activation checkpointing
# inserts a wrapper segment).
MUON_PATTERN = (r"(?:^|\.)(?:attentions\.\d+\.(?:q|k|v|out)_proj\.weight"
                r"|experts\.(?:routed\.inner_experts\.)?w[123]_E(?:FD|DF)"
                r"|(?:dense_ffn|always_on_ffn)\.w[123]\.weight"
                r"|input_adapter\.(?:state|input)_proj\.weight"
                r"|attention_loras\.\d+\.(?:q|k|v|out)_proj\.[AB]|ffn_loras\.\d+\.w[123]\.[AB])$")
CORE_EXPERT_PATTERN = r"(?:^|\.)core\.\d+\.(?:_checkpoint_wrapped_module\.)?experts\."


class LoopOptimizers(OptimizersContainer):
    """Native container plus two more DistMuon names: core experts and core output projections."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_factory(name):
        if name in (CORE_EXPERT_MUON, CORE_OUTPUT_MUON):
            return build_dist_muon
        return OptimizersContainer._resolve_optimizer_factory(name)


def _muon_kwargs(optimizer: dict[str, Any]) -> dict[str, Any]:
    muon = optimizer.get("muon")
    if not isinstance(muon, dict) or set(muon) != set(MUON_FIELDS):
        raise ValueError(f"optimizer.muon must set exactly {list(MUON_FIELDS)}")
    # DistMuon validates the values; it rejects the container's fused/foreach flags.
    return {**muon, "ns_coefficients": tuple(muon["ns_coefficients"]), "fused": False, "foreach": False}


def _owner_cell(fqn: str) -> str:
    """``core.core.0`` for a cell parameter, ``core.input_adapter`` otherwise."""
    parts = fqn.split(".")
    return ".".join(parts[:3] if parts[1] in ("prefix", "core", "suffix") else parts[:2])


def _dist_muon_kwargs(names: list[str], shapes: dict[str, torch.Size]) -> dict[str, Any]:
    """Compute layouts on the FSDP axis and one ordered bucket per physical cell.

    A 3D expert bank uses Shard(0): complete expert matrices are distributed
    and orthogonalized separately. A 2D matrix is computed whole on one owner.
    """
    layouts = {fqn: ComputeLayout(shardings_by_mesh_axis={"fsdp": Shard(0) if len(shapes[fqn]) == 3 else Owned()})
               for fqn in names}
    cells = dict.fromkeys(_owner_cell(fqn) for fqn in names)
    return {"compute_sharding_by_fqn": layouts,
            "bucket_configs": tuple(BucketConfig(name=cell, patterns=(f"{cell}.*",)) for cell in cells)}


def optimizer_config(optimizer: dict[str, Any], model) -> LoopOptimizers.Config:
    """Resolve a recipe optimizer mapping for one ``TitanModel.Config``."""
    name = optimizer.get("name", "AdamW")
    if name not in ("AdamW", "Muon"):
        raise ValueError("optimizer.name must be AdamW or Muon")
    if name == "AdamW" and "muon" in optimizer:
        raise ValueError("optimizer.muon requires optimizer.name Muon")
    roles = optimizer.get("zero_weight_decay_roles", list(ZERO_DECAY_ROLES))
    if not isinstance(roles, (list, tuple)) or any(role not in ZERO_DECAY_ROLES for role in roles):
        raise ValueError("zero_weight_decay_roles must contain only norm, embedding, lm_head or router")
    adamw = {"lr": optimizer.get("base_lr", 3e-4), "betas": tuple(optimizer.get("betas", (0.9, 0.95))),
             "eps": optimizer.get("eps", 1e-8)}
    decay = optimizer.get("weight_decay", 0.1)
    scale = model.core.region_config("core").output_lr_scale
    # LR x scale and decay / scale keep lr * weight_decay; first match wins, so this group comes first.
    groups = [ParamGroupConfig(pattern=CORE_OUTPUT_PATTERN, optimizer_name="AdamW",
                               optimizer_kwargs=adamw | {"lr": adamw["lr"] * scale, "weight_decay": decay / scale})
              ] if scale != 1 else []
    if roles:
        groups.append(ParamGroupConfig(pattern=f"(?:{'|'.join(roles)})", optimizer_name="AdamW",
                                       optimizer_kwargs=adamw | {"weight_decay": 0.0}))
    groups.append(ParamGroupConfig(pattern=r".*", optimizer_name="AdamW",
                                   optimizer_kwargs=adamw | {"weight_decay": decay}))
    if name == "AdamW":
        return LoopOptimizers.Config(param_groups=groups)
    from .titan import TitanModel
    muon = _muon_kwargs(optimizer)
    with torch.device("meta"):
        titan = TitanModel(model)
    shapes = {canonical_fqn(fqn): parameter.shape for fqn, parameter in titan.named_parameters()}
    names = [fqn for fqn in shapes if re.search(MUON_PATTERN, fqn)]
    rest = [fqn for fqn in shapes if fqn not in names]
    outputs = [fqn for fqn in names if re.search(CORE_OUTPUT_PATTERN, fqn)] if scale != 1 else []
    if outputs and optimizer.get("expert_lr_multiplier", 1.0) != 1:
        raise ValueError("under Muon, output_lr_scale cannot be combined with expert_lr_multiplier")
    core = [fqn for fqn in names if re.search(CORE_EXPERT_PATTERN, fqn) and fqn not in outputs]
    output_muon = muon | {"lr": muon["lr"] * scale, "weight_decay": muon["weight_decay"] / scale}
    resolved, factory_kwargs = [], {}
    for optimizer_name, pattern, selected, kwargs in (
            (CORE_OUTPUT_MUON, CORE_OUTPUT_PATTERN, outputs, output_muon),
            (CORE_EXPERT_MUON, CORE_EXPERT_PATTERN, core, muon),
            ("DistMuon", MUON_PATTERN, [fqn for fqn in names if fqn not in core + outputs], muon)):
        if selected:
            resolved.append(ParamGroupConfig(pattern=pattern, optimizer_name=optimizer_name, optimizer_kwargs=kwargs))
            factory_kwargs[optimizer_name] = _dist_muon_kwargs(selected, shapes)
    for group in groups:  # TorchTitan rejects a group that matches nothing; under Muon an AdamW group may.
        claimed = {fqn for fqn in rest if re.search(group.pattern, fqn)}
        if claimed:
            resolved.append(group)
            rest = [fqn for fqn in rest if fqn not in claimed]
    return LoopOptimizers.Config(param_groups=resolved, optimizer_factory_kwargs_by_name=factory_kwargs)
