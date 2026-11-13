"""Model and training configurations as plain dataclasses.

This module imports no torch, so recipe specs, cost estimates and comparisons
load and run without it. ``moeloop.model`` and ``moeloop.training`` re-export
these classes under their original names.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class RegionConfig:
    """Local geometry, residual scale and output-projection scales; None inherits the model default.

    Regions keep a common residual width and attention geometry. Reusing an
    override never shares modules or parameters between physical cells. The
    resolved :class:`ModelConfig` validates these values together.
    """

    hidden_dim: int | None = None
    num_experts: int | None = None
    top_k: int | None = None
    always_on_hidden_dim: int | None = None
    residual_scale: float | None = None
    output_lr_scale: float | None = None
    output_init_scale: float | None = None


# Written by to_dict only when on, so records from before these fields existed keep their bytes.
WRITTEN_WHEN_ON = ("sandwich_norm", "visit_final_norm", "qk_bias", "residual_norm", "core_final_norm")
# Written by to_dict only when not 1 (in a region: when set), for the same reason.
WRITTEN_WHEN_SCALED = ("output_lr_scale", "output_init_scale")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 512
    hidden_dim: int = 1536
    num_heads: int = 8
    num_kv_heads: int | None = None
    num_experts: int = 8
    top_k: int = 2
    num_cells: int = 2
    num_visits: int = 2
    visit_capacity: int | None = None
    prefix_cells: int = 0
    suffix_cells: int = 0
    order: Literal["stack", "layer"] = "stack"
    attention_per_visit: bool = False
    router_per_visit: bool = False
    norm_per_visit: bool = False
    # Per-visit LoRA on the shared core (Relaxed Recursive Transformers); rank 0 is none. See Cell.
    lora_attention_rank: int = 0
    lora_ffn_rank: int = 0
    residual_scale: float = 1.0
    # One residual scale per core visit (LatentLoop's alpha_v): core visit v scales its attention and FFN
    # branches by visit_residual_scales[v] in place of the core residual_scale, which must stay 1. Empty: none.
    visit_residual_scales: tuple[float, ...] = ()
    # Only the core cells' branch output projections use these (CORE_OUTPUT_PATTERN in model.py):
    # attention out_proj and the down projection w2 of routed experts and of the dense or always-on
    # FFN. The LR scale multiplies their learning rate and divides their weight decay, so
    # lr * weight_decay per step is kept (moeloop.optim); the init scale multiplies their init std
    # (LoopedLM.initialize_weights). Under Adam or Muon, core residual_scale a is the same training
    # as residual 1 with both scales a, up to Adam's eps and global gradient-norm clipping, when
    # sandwich_norm is off and there is no LoRA (an output norm or LoRA B is not scaled by these).
    output_lr_scale: float = 1.0
    output_init_scale: float = 1.0
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    qk_norm: bool = False
    # A bias on the query and key projections, zero at initialization. See Attention.
    qk_bias: bool = False
    # Sandwich norm: an RMSNorm also on each attention/FFN output before its residual add. See Cell.
    sandwich_norm: bool = False
    # Residual norm: an RMSNorm also on the residual stream after each residual add. See Cell.
    residual_norm: bool = False
    # The final norm also closes every core visit: visit v > 0 starts from final_norm(state). See LoopedLM.forward.
    visit_final_norm: bool = False
    # The final norm also closes the core: readouts apply it before the suffix cells too. See LoopedLM.readout.
    core_final_norm: bool = False
    always_on_hidden_dim: int = 0
    input_injection: Literal["none", "add", "linear", "diag_decay", "orthogonal"] = "none"
    injection_scale: float = 1.0
    # diag_decay/orthogonal only: initial dt and a, and the OrthoInj eps; see InputAdapter.
    injection_dt: float = 0.5 * math.log(5)
    injection_decay: float = 1.0
    injection_eps: float = 1e-6
    initial_state: Literal["input", "zero", "random"] = "input"
    latent_init_std: float = math.sqrt(2 / 5)
    latent_truncation: float = 3.0
    state_update: Literal["residual", "ema"] = "residual"
    memory_scale: float = 0.5
    memory_beta: float = 0.5
    exit_distribution: bool = False
    init_std: float = 0.02
    schema_version: int = 4
    prefix: RegionConfig | None = None
    core: RegionConfig | None = None
    suffix: RegionConfig | None = None

    def __post_init__(self) -> None:
        positive = ("vocab_size", "dim", "hidden_dim", "num_heads", "num_cells", "num_visits", "max_seq_len")
        nonnegative = ("num_experts", "prefix_cells", "suffix_cells", "always_on_hidden_dim",
                       "lora_attention_rank", "lora_ffn_rank")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in positive else 0):
                raise ValueError(f"{name} must be {'positive' if name in positive else 'nonnegative'} integer")
        for name in ("attention_per_visit", "router_per_visit", "norm_per_visit", "qk_norm", "exit_distribution",
                     "sandwich_norm", "visit_final_norm", "qk_bias", "residual_norm", "core_final_norm"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        kv = self.num_heads if self.num_kv_heads is None else self.num_kv_heads
        if type(kv) is not int or kv < 1 or self.num_heads % kv:
            raise ValueError("num_kv_heads must be positive and divide num_heads")
        object.__setattr__(self, "num_kv_heads", kv)
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 2:
            raise ValueError("dim / num_heads must be a positive even head dimension")
        if type(self.top_k) is not int or self.top_k < 1 or (self.num_experts and self.top_k > self.num_experts):
            raise ValueError("top_k must be positive and no larger than num_experts for sparse models")
        if self.num_experts == 0 and (self.router_per_visit or self.always_on_hidden_dim):
            raise ValueError("dense cells have no router or separate always-on sparse branch")
        if self.order not in ("stack", "layer") or self.initial_state not in ("input", "zero", "random"):
            raise ValueError("unknown order or initial_state")
        if self.state_update not in ("residual", "ema"):
            raise ValueError("unknown state_update")
        if self.state_update == "ema" and (self.order != "stack" or self.input_injection != "none"
                or self.initial_state != "input" or self.norm_per_visit):
            raise ValueError("EMA state uses stack order, input initial state and shared norms without input adapter")
        if self.state_update == "ema" and (self.lora_attention_rank or self.lora_ffn_rank):
            raise ValueError("per-visit LoRA is defined for the residual cell, not the EMA memory cell")
        if self.state_update == "ema" and self.num_experts and not self.router_per_visit:
            raise ValueError("EMA state requires an independently owned router per visit")
        if self.state_update == "ema" and self.sandwich_norm:
            raise ValueError("sandwich norm is defined for the residual cell, not the EMA memory cell")
        if self.state_update == "ema" and self.residual_norm:
            raise ValueError("residual norm is defined for the residual cell, not the EMA memory cell")
        if self.visit_final_norm and self.order != "stack":
            raise ValueError("visit_final_norm closes complete stack visits")
        if self.exit_distribution and self.order != "stack":
            raise ValueError("exit distributions require complete stack visits")
        if self.input_injection not in ("none", "add", "linear", "diag_decay", "orthogonal"):
            raise ValueError("input_injection must be none, add, linear, diag_decay or orthogonal")
        if self.order == "layer" and self.input_injection != "none":
            raise ValueError("stage input injection is defined only for stack order")
        for name in ("residual_scale", "output_lr_scale", "output_init_scale", "rms_eps", "rope_theta",
                     "injection_scale", "injection_dt", "injection_decay", "injection_eps", "init_std",
                     "latent_init_std", "latent_truncation", "memory_scale", "memory_beta"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.rms_eps, self.rope_theta, self.init_std, self.output_lr_scale, self.output_init_scale) <= 0:
            raise ValueError("rms_eps, rope_theta, init_std and the output scales must be positive")
        for name in ("injection_dt", "injection_decay", "injection_eps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.input_injection in ("diag_decay", "orthogonal") and self.injection_scale != 1:
            raise ValueError("injection_scale applies only to add and linear injection")
        if min(self.latent_init_std, self.latent_truncation) <= 0 or self.memory_scale < 0 or not 0 <= self.memory_beta < 1:
            raise ValueError("invalid latent initialization or EMA coefficients")
        if type(self.schema_version) is not int or self.schema_version not in (1, 2, 3, 4):
            raise ValueError(f"unsupported model schema version: {self.schema_version}")
        if self.schema_version == 1 and any(getattr(self, region) is not None for region in ("prefix", "core", "suffix")):
            raise ValueError("model schema version 1 does not support region overrides; use schema version 2")
        if self.schema_version < 4 and self.input_injection in ("diag_decay", "orthogonal"):
            raise ValueError(f"model schema version {self.schema_version} has no {self.input_injection} injection; use schema version 4")
        # Schemas 1-3 acquire only inactive defaults for the new state and
        # injection paths. Existing state-dict names and physical parameter
        # ownership are unchanged.
        object.__setattr__(self, "schema_version", 4)
        if self.independent_visits:
            capacity = self.num_visits if self.visit_capacity is None else self.visit_capacity
            if type(capacity) is not int or capacity < self.num_visits:
                raise ValueError("visit_capacity must cover num_visits for independent owners")
            object.__setattr__(self, "visit_capacity", capacity)
        elif self.visit_capacity is not None:
            raise ValueError("fully shared owners have no visit_capacity; use num_visits for default depth")
        for region in ("prefix", "core", "suffix"):
            override = getattr(self, region)
            if override is not None:
                if not isinstance(override, RegionConfig):
                    raise ValueError(f"{region} region must be RegionConfig or None")
                try:
                    self.region_config(region)
                except ValueError as error:
                    raise ValueError(f"invalid {region} region configuration: {error}") from error
        for region in ("prefix", "suffix"):
            override = getattr(self, region)
            if override is not None and (override.output_lr_scale, override.output_init_scale) != (None, None):
                raise ValueError("output_lr_scale and output_init_scale apply only to the core cells")
        core_scale = self.residual_scale if self.core is None or self.core.residual_scale is None else self.core.residual_scale
        scales = self.visit_residual_scales
        if not isinstance(scales, (tuple, list)) or any(
                type(value) not in (float, int) or not math.isfinite(value) for value in scales):
            raise ValueError("visit_residual_scales must be a sequence of finite numbers")
        object.__setattr__(self, "visit_residual_scales", tuple(scales))
        if scales and (len(scales) != self.num_visits or core_scale != 1 or self.state_update != "residual"):
            raise ValueError("visit_residual_scales needs one scale per visit, core residual_scale 1 and the "
                             "residual state update")
        if self.state_update == "ema" and core_scale != 1:
            raise ValueError("EMA core residual_scale must be 1; memory_scale defines its stabilization")
        if self.lora_ffn_rank:
            core = self if self.core is None else self.region_config("core")
            if core.num_experts and not core.always_on_hidden_dim:
                raise ValueError("FFN LoRA needs a dense or always-on core FFN; routed experts take no LoRA")

    @property
    def independent_visits(self) -> bool:
        return (self.attention_per_visit or self.router_per_visit or self.norm_per_visit
                or self.lora_attention_rank > 0 or self.lora_ffn_rank > 0)

    def region_config(self, region: Literal["prefix", "core", "suffix"]) -> "ModelConfig":
        """Resolve the actual config passed to cells and native factories.

        LoRA relaxes the shared core only: prefix and suffix cells own their
        matrices, so their configs carry no LoRA ranks.
        """
        if region not in ("prefix", "core", "suffix"):
            raise ValueError(f"unknown model region: {region}")
        override = getattr(self, region)
        values = {} if override is None else {
            field.name: getattr(override, field.name) for field in fields(RegionConfig)
            if getattr(override, field.name) is not None
        }
        if self.state_update == "ema" and region != "core":
            values["state_update"] = "residual"
        if region != "core":
            values.update(lora_attention_rank=0, lora_ffn_rank=0, visit_residual_scales=())
            if not (self.attention_per_visit or self.router_per_visit or self.norm_per_visit):
                values["visit_capacity"] = None  # LoRA was the only per-visit owner.
        return replace(self, **values, prefix=None, core=None, suffix=None)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in WRITTEN_WHEN_ON:
            if not value[name]:
                del value[name]
        for name in WRITTEN_WHEN_SCALED:
            if value[name] == 1:
                del value[name]
            for region in ("prefix", "core", "suffix"):
                if value[region] is not None and value[region][name] is None:
                    del value[region][name]
        # Written only when set, for the same reason; as a JSON array, as RecipeSpec writes tuples.
        if self.visit_residual_scales:
            value["visit_residual_scales"] = list(self.visit_residual_scales)
        else:
            del value["visit_residual_scales"]
        return value

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model configuration fields: {sorted(unknown)}")
        values = dict(values)
        for region in ("prefix", "core", "suffix"):
            override = values.get(region)
            if override is not None and not isinstance(override, RegionConfig):
                if not isinstance(override, Mapping):
                    raise ValueError(f"{region} region must be a configuration mapping")
                unknown = set(override) - {field.name for field in fields(RegionConfig)}
                if unknown:
                    raise ValueError(f"unknown {region} region configuration fields: {sorted(unknown)}")
                values[region] = RegionConfig(**dict(override))
        return cls(**values)


@dataclass(frozen=True)
class TrainingRecipe:
    """Explicit fixed/update/microbatch depth policies, outside model replay."""

    num_visits: int = 1
    sampled_visits: tuple[int, ...] = ()
    grad_visits: int | None = None
    balance_coeff: float = 0.01
    z_loss_coeff: float = 0.0
    depth_seed: int = 0
    depth_rule: str = "uniform_update_v1"
    # Recorded by archived recipes; the one implemented rule (titan.py) averages over local microbatch calls.
    aux_reduction: str = "local_micro_call_mean"
    depth_mean: float = 4.0
    depth_log_std: float = 0.5
    objective: str = "terminal_ce"
    entropy_coeff: float = 0.0

    def __post_init__(self) -> None:
        for name in ("num_visits", "depth_seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("balance_coeff", "z_loss_coeff", "depth_mean", "depth_log_std", "entropy_coeff"):
            if type(getattr(self, name)) not in (float, int):
                raise ValueError(f"{name} must be a number")
        depths = self.sampled_visits or (self.num_visits,)
        if self.num_visits < 1 or any(type(d) is not int or d < 1 for d in depths):
            raise ValueError("num_visits and sampled_visits must be positive integers")
        if len(set(depths)) != len(depths):
            raise ValueError("sampled_visits must contain distinct depths")
        sampled_micro = self.depth_rule == "poisson_lognormal_micro_v1"
        if self.grad_visits is not None and (type(self.grad_visits) is not int or self.grad_visits < 1
                or (not sampled_micro and self.grad_visits > min(depths))):
            raise ValueError("grad_visits must be positive and no larger than fixed/finite training depths")
        for name in ("balance_coeff", "z_loss_coeff", "entropy_coeff"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.depth_rule not in ("uniform_update_v1", "poisson_lognormal_micro_v1"):
            raise ValueError("unsupported depth sampling rule")
        if sampled_micro and self.sampled_visits:
            raise ValueError("Poisson-lognormal depths cannot also specify sampled_visits")
        if not math.isfinite(self.depth_mean) or self.depth_mean <= 1 or not math.isfinite(self.depth_log_std) or self.depth_log_std < 0:
            raise ValueError("depth_mean must exceed one and depth_log_std must be nonnegative")
        if self.objective not in ("terminal_ce", "expected_ce"):
            raise ValueError("unknown training objective")
        if self.aux_reduction != "local_micro_call_mean":
            raise ValueError("unsupported auxiliary reduction")

    def depth_at(self, completed_updates: int, microbatch: int = 0) -> int:
        """Training depth of one microbatch of an update.

        ``uniform_update_v1`` draws once per update, so its depth sequence does
        not depend on the number of ranks. ``poisson_lognormal_micro_v1``
        draws per microbatch. FSDP needs one depth per accumulation step on
        every rank, so the trainer uses rank 0's microbatch index; the depth
        sequence then depends on the accumulation steps, global batch /
        (local batch x ranks).
        """
        if completed_updates < 0:
            raise ValueError("completed_updates cannot be negative")
        if self.depth_rule == "poisson_lognormal_micro_v1":
            from .latent import sampled_depth
            return sampled_depth(seed=self.depth_seed, completed_updates=completed_updates,
                                 microbatch=microbatch, mean=self.depth_mean, log_std=self.depth_log_std)
        if not self.sampled_visits:
            return self.num_visits
        key = f"{self.depth_rule}:{self.depth_seed}:{completed_updates}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest(), "big")
        return random.Random(seed).choice(self.sampled_visits)

    def validate_model(self, config: Any) -> None:
        depths = self.sampled_visits or (self.num_visits,)
        independent = config.independent_visits
        stochastic = bool(self.sampled_visits) or self.depth_rule == "poisson_lognormal_micro_v1"
        if independent and (stochastic or self.grad_visits is not None):
            raise ValueError("per-visit owners support fixed depth and full BPTT in v1")
        if independent and max(depths) > config.visit_capacity:
            raise ValueError("training depth exceeds per-visit owner capacity")
        if independent and self.num_visits != config.visit_capacity:
            raise ValueError("fixed training depth must use all allocated per-visit owners")
        if self.grad_visits is not None and config.order != "stack":
            raise ValueError("grad_visits means complete stack visits; layer order uses full BPTT")
        if config.state_update == "ema" and (stochastic or self.grad_visits is not None):
            raise ValueError("EMA state currently supports fixed depth and full BPTT")
        if (self.objective == "expected_ce") != config.exit_distribution:
            raise ValueError("expected CE and model exit_distribution must be selected together")
        if self.objective == "expected_ce" and (stochastic or self.grad_visits is not None):
            raise ValueError("expected CE requires fixed depth and full BPTT")
