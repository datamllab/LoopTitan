"""Plain PyTorch recursive language models with explicit physical owners.

Read ``LoopedLM.forward`` for execution/gradient order and ``Cell.forward``
for the state update. Paper names belong in recipes rather than this module.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
import re
from typing import Callable, Iterator, Literal, NamedTuple

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.experimental import implicit_replication
import torch.nn.functional as F

from .cache import CacheSlot, DecodeCache
from .config import ModelConfig, RegionConfig  # RegionConfig: documented import path
from .moe import DenseFFN, ExpertBank, Router
from .latent import initial_latent
from .memory import EMACarry, ema_update, residual_gain, reinjection_gain


# The core cells' branch output projections, the last matrix before each residual add, which
# ModelConfig.output_lr_scale and output_init_scale act on. It matches LoopedLM names (core.0...)
# and TitanModel names (core.core.0...), with or without the activation-checkpoint wrapper segment.
CORE_OUTPUT_PATTERN = (r"(?:^|\.)core\.\d+\.(?:_checkpoint_wrapped_module\.)?"
                       r"(?:attentions\.\d+\.out_proj\.weight|experts\.(?:routed\.inner_experts\.)?w2_EDF"
                       r"|(?:dense_ffn|always_on_ffn)\.w2\.weight)$")


class ModelOutput(NamedTuple):
    """Counts retain physical aggregation and logical-call identity.

    ``router_counts`` has one row per physical prefix/core/suffix cell. Both
    tensors pad to the largest expert count; per-call widths distinguish real
    experts from padding, including dense cells with width zero. Call rows
    follow the actual stack/layer execution order.
    """

    logits: Tensor | None
    hidden_states: Tensor
    balance_sum: Tensor
    z_sum: Tensor
    aux_calls: int
    router_counts: Tensor
    router_call_counts: Tensor | None = None
    router_call_slots: tuple[CacheSlot, ...] = ()
    router_call_widths: tuple[int, ...] = ()
    visit_hidden_states: tuple[Tensor, ...] = ()
    hazard_logits: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps)


@torch.no_grad()
def init_normal(parameter: Tensor, std: float, generator: torch.Generator | None) -> None:
    """Draw the full tensor from normal(0, std) on the default device and write it into ``parameter``.

    The default device is CPU (meta inside a meta build, which draws nothing). A DTensor
    keeps only this rank's part, cut locally with no communication, so the values depend
    only on the generator, never on the rank count, sharding or GPU model. A plain CPU
    parameter gets exactly the values of an in-place ``normal_``.
    """
    full = torch.empty(parameter.shape, dtype=parameter.dtype).normal_(0.0, std, generator=generator)
    if isinstance(parameter, DTensor):
        full = distribute_tensor(full, parameter.device_mesh, parameter.placements, src_data_rank=None)
    parameter.copy_(full)


def apply_rotary(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Split-half RoPE on [batch, heads, tokens, head_dim] using [batch,tokens]."""
    half = x.size(-1) // 2
    frequencies = theta ** (-torch.arange(half, dtype=torch.float32, device=x.device) / half)
    angles = positions.float().unsqueeze(-1) * frequencies
    cosine, sine = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1).to(x.dtype)


class LoRA(nn.Module):
    """One visit's low-rank delta ``B A x`` on one shared Linear ``W' x``.

    Relaxed Recursive Transformers (arXiv 2410.20672v3, Section 2.3, after eq. 3):
    ``h = W' x + B A x`` with ``A`` of shape [rank, in] and ``B`` [out, rank],
    and no alpha/rank scaling. ``reset_parameters`` draws ``A`` from
    normal(init_std) and zeroes ``B`` (the paper's "zero initialization",
    Appendix C), so a new model computes exactly its shared-weight model.
    ``LoopedLM.initialize_weights`` calls it after every other draw.
    """

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.A), self.B)

    @torch.no_grad()
    def reset_parameters(self, std: float, generator: torch.Generator | None = None) -> None:
        """Zero initialization: A from normal(std), B = 0."""
        init_normal(self.A, std, generator)
        nn.init.zeros_(self.B)

    @torch.no_grad()
    def svd_init(self, residual: Tensor) -> None:
        """The paper's optional SVD initialization (eq. 4): ``B = U_r S_r``, ``A = V_r^T``.

        ``residual`` is ``W_l - W'``: a full-size model's weight at this
        layer and visit minus the shared weight, so ``W' + B A`` is its best
        rank-r approximation. The SVD runs in FP64.
        """
        u, s, vh = torch.linalg.svd(residual.double(), full_matrices=False)
        rank = self.A.size(0)
        self.B.copy_(u[:, :rank] * s[:rank])
        self.A.copy_(vh[:rank])


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.dim // config.num_heads
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=config.qk_bias)
        self.k_proj = nn.Linear(config.dim, self.num_kv_heads * self.head_dim, bias=config.qk_bias)
        self.v_proj = nn.Linear(config.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_eps) if config.qk_norm else nn.Identity()

    def _attention(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask: object | None, *, is_causal: bool,
    ) -> Tensor:
        """Finite kernel seam: native bindings can override only attention math."""
        if attention_mask is not None and not isinstance(attention_mask, Tensor):
            raise TypeError("reference attention accepts bool tensor masks; use a native attention binding for prepared masks")
        repetitions = self.num_heads // self.num_kv_heads
        if repetitions > 1:
            key = key.repeat_interleave(repetitions, dim=1)
            value = value.repeat_interleave(repetitions, dim=1)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=is_causal)

    def forward(
        self, x: Tensor, *, positions: Tensor, attention_mask: object | None = None,
        cache: DecodeCache | None = None, slot: CacheSlot = ("core", 0, 0),
        lora: nn.ModuleDict | None = None,
    ) -> Tensor:
        """``lora`` holds this visit's LoRA for q_proj, k_proj, v_proj and out_proj."""
        batch, tokens, dim = x.shape
        query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if lora is not None:
            query, key, value = query + lora.q_proj(x), key + lora.k_proj(x), value + lora.v_proj(x)
        query = query.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query = apply_rotary(self.q_norm(query), positions, self.rope_theta)
        key = apply_rotary(self.k_norm(key), positions, self.rope_theta)
        if cache is not None:
            key, value = cache.append(slot, key, value)
        result = self._attention(query, key, value, attention_mask, is_causal=attention_mask is None)
        mixed = result.transpose(1, 2).contiguous().view(batch, tokens, dim)
        return self.out_proj(mixed) if lora is None else self.out_proj(mixed) + lora.out_proj(mixed)


def dense_ffn(ffn: DenseFFN, x: Tensor, lora: nn.ModuleDict | None) -> Tensor:
    """``DenseFFN`` (SwiGLU) with this visit's LoRA on w1, w3 and w2, if any."""
    if lora is None:
        return ffn(x)
    hidden = F.silu(ffn.w1(x) + lora.w1(x)) * (ffn.w3(x) + lora.w3(x))
    return ffn.w2(hidden) + lora.w2(hidden)


class InputAdapter(nn.Module):
    """Stage input injection: the state entering each core visit, from the state and the anchor.

    - ``add``: ``h + s e``; ``linear``: ``A h + s W e``.
    - ``diag_decay``: ``exp(-dt * a) * h + dt * (W e)`` with per-channel
      ``dt = softplus(dt_bias)`` and ``a = exp(A_log)``; arXiv 2610.06833
      Appendix B.4 equation (24) ("Parcae w/o norm", Table 7) and xllm-loop
      (commit 755c7b3) ``xllm/models/looped.py`` ``InputInjection`` (lines 69-138).
    - ``orthogonal`` (OrthoInj): ``Q_q exp(-dt * a) * h + q`` with ``q = dt * (W e)``
      and ``Q_q = I - q q^T / (|q|^2 + eps)``; arXiv 2610.06833 Section 4.2
      equation (8) and xllm-loop ``OrthogonalInputInjection`` (lines 162-178).
      ``Q_q`` is a per-token projector off ``q``, not an orthogonal weight
      matrix; the paper has no orthogonality constraint on ``W``. Its inner
      products run in FP32 (FP64 stays FP64), as in the paper's recipe.

    Both new kinds start at ``W = I``, ``a = injection_decay`` and
    ``dt = injection_dt``. The defaults are the OrthoInj recipe of Appendix B.4
    (a = 1, dt = log(5)/2, retention 1/sqrt(5)); its "Parcae w/o norm" row uses
    dt = 1. The anchor ``e`` is the prefix output with no prelude RMSNorm, so
    Parcae-Decay's normalized input is not expressed here. ``q`` is recomputed
    at every visit, as in the reference code. ``LoopedLM.initialize_weights``
    applies this initialization, also on the TorchTitan path (#90).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.kind, self.scale, self.eps = config.input_injection, config.injection_scale, config.injection_eps
        self.dt_init, self.decay_init = config.injection_dt, config.injection_decay
        if self.kind == "linear":
            self.state_proj = nn.Linear(config.dim, config.dim, bias=False)
        if self.kind != "add":
            self.input_proj = nn.Linear(config.dim, config.dim, bias=False)
        if self.kind in ("diag_decay", "orthogonal"):
            self.A_log = nn.Parameter(torch.empty(config.dim))
            self.dt_bias = nn.Parameter(torch.empty(config.dim))

    @torch.no_grad()
    def reset_decay(self) -> None:
        """Declared initialization of the decay kinds: W = I, a and dt by inverse exp/softplus."""
        weight = self.input_proj.weight
        weight.copy_(torch.eye(weight.size(0), dtype=weight.dtype, device=weight.device))
        self.A_log.fill_(math.log(self.decay_init))
        self.dt_bias.fill_(self.dt_init + math.log(-math.expm1(-self.dt_init)))

    def forward(self, state: Tensor, anchor: Tensor) -> Tensor:
        if self.kind == "add":
            return state + self.scale * anchor
        if self.kind == "linear":
            return self.state_proj(state) + self.scale * self.input_proj(anchor)
        dt = F.softplus(self.dt_bias).to(state.dtype)
        retained = torch.exp(-dt * self.A_log.exp().to(state.dtype)) * state
        injected = dt * self.input_proj(anchor)
        if self.kind == "diag_decay":
            return retained + injected
        compute = torch.float64 if retained.dtype == torch.float64 else torch.float32
        value, q = retained.to(compute), injected.to(compute)
        coefficient = (value * q).sum(-1, keepdim=True) / (q.square().sum(-1, keepdim=True) + self.eps)
        return (value - coefficient * q).to(retained.dtype) + injected


def per_visit(owners: nn.ModuleList, visit: int) -> nn.Module:
    """The visit's own module when the cell keeps one per visit, else the shared one."""
    return owners[visit if len(owners) > 1 else 0]


class Cell(nn.Module):
    """One physical bank, with singleton or per-visit attention/router/norm owners.

    Every visit can also own a LoRA on each shared attention projection (rank
    ``lora_attention_rank``) and on w1, w3, w2 of the dense or always-on FFN
    (rank ``lora_ffn_rank``); routers and routed experts take no LoRA. A rank
    of 0 builds none; ``region_config`` gives prefix and suffix cells rank 0.
    """

    def __init__(
        self, config: ModelConfig, visits: int = 1, *,
        expert_factory: Callable[[ModelConfig], nn.Module] | None = None,
        attention_factory: Callable[[ModelConfig], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        attention_factory = attention_factory or Attention
        expert_factory = expert_factory or ExpertBank
        self.attentions = nn.ModuleList(attention_factory(config) for _ in range(visits if config.attention_per_visit else 1))
        self.attention_norms = nn.ModuleList(RMSNorm(config.dim, config.rms_eps) for _ in range(visits if config.norm_per_visit else 1))
        self.ffn_norms = nn.ModuleList(RMSNorm(config.dim, config.rms_eps) for _ in range(visits if config.norm_per_visit else 1))
        norms = visits if config.norm_per_visit else 1  # sandwich output and residual norms, identities when off
        self.attention_output_norms = nn.ModuleList(RMSNorm(config.dim, config.rms_eps) if config.sandwich_norm else nn.Identity() for _ in range(norms))
        self.ffn_output_norms = nn.ModuleList(RMSNorm(config.dim, config.rms_eps) if config.sandwich_norm else nn.Identity() for _ in range(norms))
        self.attention_residual_norms = nn.ModuleList(RMSNorm(config.dim, config.rms_eps) if config.residual_norm else nn.Identity() for _ in range(norms))
        self.ffn_residual_norms = nn.ModuleList(RMSNorm(config.dim, config.rms_eps) if config.residual_norm else nn.Identity() for _ in range(norms))
        self.routers = nn.ModuleList(Router(config) for _ in range(visits if config.router_per_visit else 1)) if config.num_experts else nn.ModuleList()
        self.experts = expert_factory(config) if config.num_experts else None
        self.dense_ffn = DenseFFN(config.dim, config.hidden_dim) if not config.num_experts else None
        self.always_on_ffn = DenseFFN(config.dim, config.always_on_hidden_dim) if config.always_on_hidden_dim else None
        self.residual_scale = config.residual_scale
        self.visit_residual_scales = config.visit_residual_scales  # core cells only; see ModelConfig
        d, kv = config.dim, config.num_kv_heads * (config.dim // config.num_heads)
        ffn = config.always_on_hidden_dim if config.num_experts else config.hidden_dim

        def visit_loras(rank: int, shapes: dict[str, tuple[int, int]]) -> nn.ModuleList | None:
            if not rank:
                return None
            return nn.ModuleList(nn.ModuleDict({name: LoRA(i, o, rank) for name, (i, o) in shapes.items()})
                                 for _ in range(visits))
        self.attention_loras = visit_loras(config.lora_attention_rank, {
            "q_proj": (d, d), "k_proj": (d, kv), "v_proj": (d, kv), "out_proj": (d, d)})
        self.ffn_loras = visit_loras(config.lora_ffn_rank, {"w1": (d, ffn), "w3": (d, ffn), "w2": (ffn, d)})

    def feed_forward(self, x: Tensor, *, visit: int, router_mask: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """FFN update of normalized ``x`` and its routing statistics ``(balance, z_loss, counts)``.

        The visit's FFN LoRA sits on the dense or always-on FFN; a dense cell has zero statistics.
        """
        lora = None if self.ffn_loras is None else self.ffn_loras[visit]
        if self.experts is None:
            zero = x.new_zeros((), dtype=torch.float32)
            return dense_ffn(self.dense_ffn, x, lora), (zero, zero, x.new_empty((0,), dtype=torch.long))
        routing = per_visit(self.routers, visit)(x, router_mask)
        update = self.experts(x, routing.ids, routing.weights, routing.dispatch_counts)
        if self.always_on_ffn is not None:
            update = update + dense_ffn(self.always_on_ffn, x, lora)
        return update, (routing.balance, routing.z_loss, routing.counts.detach())

    def forward(
        self, x: Tensor, *, visit: int, positions: Tensor, attention_mask: object | None,
        router_mask: Tensor, cache: DecodeCache | None, slot: CacheSlot,
        memory: object = None, depth: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], object]:
        """``x + s attention(norm(x))``, then ``x + s ffn(norm(x))`` with ``s = residual_scale``,
        or ``s = visit_residual_scales[visit]`` when the cell has per-visit scales.

        With ``sandwich_norm`` each sublayer output is normalized too:
        ``x + s out_norm(attention(norm(x)))``; with ``residual_norm`` the sum is normalized:
        ``residual_norm(x + s attention(norm(x)))``. Each extra norm is the identity when its flag is off.

        Every cell takes these keywords and returns ``(x, (balance, z_loss, counts), memory)``.
        ``memory`` and ``depth`` are for ``MemoryCell``; a ``Cell`` returns ``memory`` unchanged.
        """
        # Only attention LoRA passes ``lora=``, so other attention factories keep their signature.
        lora = {} if self.attention_loras is None else {"lora": self.attention_loras[visit]}
        scale = self.visit_residual_scales[visit] if self.visit_residual_scales else self.residual_scale
        update = per_visit(self.attentions, visit)(
            per_visit(self.attention_norms, visit)(x), positions=positions,
            attention_mask=attention_mask, cache=cache, slot=slot, **lora,
        )
        x = per_visit(self.attention_residual_norms, visit)(
            x + scale * per_visit(self.attention_output_norms, visit)(update))
        update, routing = self.feed_forward(per_visit(self.ffn_norms, visit)(x), visit=visit, router_mask=router_mask)
        x = per_visit(self.ffn_residual_norms, visit)(x + scale * per_visit(self.ffn_output_norms, visit)(update))
        return x, routing, memory


class MemoryCell(Cell):
    """LOOM: the attention write enters the ``memory = (global, local)`` EMA streams before the FFN.

    Both residual writes are scaled by ``residual_gain(num_cells, depth)``.
    """

    def __init__(self, config, visits=1, **kwargs):
        super().__init__(config, visits, **kwargs)
        self.ffn_output_norm = RMSNorm(config.dim, config.rms_eps)

    def forward(self, x, *, visit, positions, attention_mask, router_mask, cache, slot, memory, depth):
        gamma = residual_gain(self.config.num_cells, depth, scale=self.config.memory_scale)
        write = gamma * per_visit(self.attentions, visit)(
            self.attention_norms[0](x), positions=positions, attention_mask=attention_mask, cache=cache, slot=slot)
        global_memory, global_read = ema_update(memory[0], write, beta=self.config.memory_beta)
        local_memory, local_read = ema_update(memory[1], write, beta=self.config.memory_beta)
        x = x + global_read + local_read
        update, routing = self.feed_forward(self.ffn_norms[0](x), visit=visit, router_mask=router_mask)
        return x + gamma * self.ffn_output_norm(update), routing, (global_memory, local_memory)


class CellCall(NamedTuple):
    """One logical call's routing statistics; ``width`` is 0 for a dense cell."""

    slot: CacheSlot
    width: int
    differentiated: bool  # made with gradients enabled: only these enter the auxiliary sums
    balance: Tensor
    z_loss: Tensor
    counts: Tensor


def call_cell(
    cell: Cell, state: Tensor, slot: CacheSlot, inputs: dict[str, object], calls: list[CellCall],
    memory: object = None,
) -> tuple[Tensor, object]:
    """Run one logical call (``slot`` = region, cell index, visit) and append its statistics to ``calls``."""
    state, (balance, z_loss, counts), memory = cell(state, visit=slot[2], slot=slot, memory=memory, **inputs)
    calls.append(CellCall(slot, cell.config.num_experts, torch.is_grad_enabled(), balance, z_loss, counts))
    return state, memory


class LoopedLM(nn.Module):
    def __init__(
        self, config: ModelConfig, *, expert_factory: Callable[[ModelConfig], nn.Module] | None = None,
        attention_factory: Callable[[ModelConfig], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        def cell(region: Literal["prefix", "core", "suffix"], visits: int) -> Cell:
            kind = MemoryCell if region == "core" and config.state_update == "ema" else Cell
            return kind(config.region_config(region), visits, expert_factory=expert_factory, attention_factory=attention_factory)
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.prefix = nn.ModuleList(cell("prefix", 1) for _ in range(config.prefix_cells))
        self.core = nn.ModuleList(cell("core", config.visit_capacity or 1) for _ in range(config.num_cells))
        self.suffix = nn.ModuleList(cell("suffix", 1) for _ in range(config.suffix_cells))
        self.router_count_width = max(cell.config.num_experts for group in (self.prefix, self.core, self.suffix) for cell in group)
        self.input_adapter = InputAdapter(config) if config.input_injection != "none" else None
        self.final_norm = RMSNorm(config.dim, config.rms_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.exit_gate = nn.Linear(config.dim, 1) if config.exit_distribution else None
        parameters = list(self.named_parameters(remove_duplicate=False))
        if len(parameters) != len({id(parameter) for _, parameter in parameters}):
            raise ValueError("each parameter needs one canonical owner; factories must return independent modules")
        self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self, generator: torch.Generator | None = None) -> None:
        """Initialize every unique owner once; also call after meta->to_empty.

        V1 uses explicit normal(init_std) matrices, unit norm scales and zero qk biases; the core
        output projections (``CORE_OUTPUT_PATTERN``) draw normal(init_std * output_init_scale). There
        is no automatic reuse-count rescaling or copy-initialized router alias.
        LoRA draws last (A normal, B zero), so the same seed gives a relaxed
        model the shared weights of its plain counterpart.
        Every matrix is drawn whole on CPU from ``generator`` (the global CPU
        generator when None) in ``canonical_parameters`` order (``init_normal``),
        also when parameters are sharded DTensors, so the values depend only on
        the generator. ``TitanModel.init_states`` calls this method.
        Checkpoint loads must not call this after restoring saved weights.
        """
        loras = [module for module in self.modules() if isinstance(module, LoRA)]
        lora_parameters = {id(parameter) for module in loras for parameter in module.parameters()}
        output_scale = self.config.region_config("core").output_init_scale
        for name, parameter in self.canonical_parameters():
            if id(parameter) in lora_parameters:
                continue
            if parameter.ndim == 1:
                (nn.init.zeros_ if name.endswith(("q_proj.bias", "k_proj.bias")) else nn.init.ones_)(parameter)
            else:
                scale = output_scale if re.search(CORE_OUTPUT_PATTERN, name) else 1
                init_normal(parameter, self.config.init_std * scale, generator)
        if self.exit_gate is not None:
            nn.init.zeros_(self.exit_gate.weight)
            nn.init.zeros_(self.exit_gate.bias)
        if self.input_adapter is not None and self.input_adapter.kind in ("diag_decay", "orthogonal"):
            # A sharded W takes copy_ of the plain identity as Replicate: each rank keeps its own rows.
            with implicit_replication():
                self.input_adapter.reset_decay()
        for module in loras:
            module.reset_parameters(self.config.init_std, generator)

    def canonical_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Each unique parameter once: exactly ``named_parameters(remove_duplicate=True)``."""
        return self.named_parameters(remove_duplicate=True)

    def cache_slots(self, num_visits: int | None = None) -> tuple[CacheSlot, ...]:
        depth = self.config.num_visits if num_visits is None else num_visits
        return tuple(
            [("prefix", i, 0) for i in range(len(self.prefix))]
            + [("core", i, visit) for i in range(len(self.core)) for visit in range(depth)]
            + [("suffix", i, 0) for i in range(len(self.suffix))]
        )

    def parse_inputs(
        self, token_ids: Tensor, *, positions: Tensor | None = None,
        attention_mask: object | None = None, router_mask: Tensor | None = None,
        num_visits: int | None = None, grad_visits: int | None = None,
        cache: DecodeCache | None = None, collect_readouts: bool | None = None,
    ) -> tuple[dict[str, object], Tensor, int, bool]:
        """Check one forward's arguments; return ``(inputs, query_valid, grad_visits, collect_readouts)``.

        ``inputs`` are the keywords of every cell call: ``positions``, the prepared
        ``attention_mask``, the routing-valid ``router_mask``, ``cache`` and ``depth``.
        The prepared mask is None (SDPA's causal path) for an unpadded full prefix,
        a bool [B,1|H,Q,K] tensor (True allows; causality enforced) built from a bool
        [B,K], [B,Q,K] or [B,1|H,Q,K] mask, or a native kernel's prepared non-Tensor mask.
        """
        config = self.config
        depth = config.num_visits if num_visits is None else num_visits
        if type(depth) is not int or depth < 1:
            raise ValueError("num_visits must be a positive integer")
        if config.visit_capacity is not None and depth > config.visit_capacity:
            raise ValueError("num_visits exceeds the allocated independent owner visit_capacity")
        if config.visit_residual_scales and depth > len(config.visit_residual_scales):
            raise ValueError("num_visits exceeds the per-visit residual scales")
        differentiated = depth if grad_visits is None else grad_visits
        if type(differentiated) is not int or not 1 <= differentiated <= depth:
            raise ValueError("grad_visits must be in [1, num_visits]")
        if differentiated < depth and (config.order != "stack" or config.independent_visits):
            raise ValueError("tail-only gradients currently require fully shared stack order")
        if differentiated < depth and (config.state_update != "residual" or config.exit_distribution):
            raise ValueError("EMA state and expected exits require full BPTT")
        collect = config.exit_distribution and self.training if collect_readouts is None else collect_readouts
        if collect and (not config.exit_distribution or cache is not None):
            raise ValueError("intermediate readouts require exit_distribution and no inference cache")
        if token_ids.ndim != 2 or token_ids.dtype != torch.long or min(token_ids.shape) < 1:
            raise ValueError("token_ids must be nonempty int64 [batch, tokens]")
        batch, tokens = token_ids.shape
        past = 0 if cache is None else cache.length
        if cache is not None:
            if self.training or torch.is_grad_enabled() or grad_visits is not None:
                raise ValueError("decode cache requires eval(), no_grad() and no gradient truncation")
            if cache.num_visits != depth:
                raise ValueError("cannot change num_visits during a cache session")
        tensor_mask = isinstance(attention_mask, Tensor)
        if tensor_mask:
            if attention_mask.dtype != torch.bool or attention_mask.device != token_ids.device:
                raise ValueError("attention_mask must be bool on the token device")
            if attention_mask.ndim not in (2, 3, 4):
                raise ValueError("attention_mask must have two, three or four dimensions")
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)  # [B,Q,K] -> [B,1,Q,K]
        if router_mask is not None and (router_mask.shape != token_ids.shape or router_mask.dtype != torch.bool or router_mask.device != token_ids.device):
            raise ValueError("router_mask must be bool [batch,tokens] on the token device")
        if tensor_mask and attention_mask.ndim == 2:
            if attention_mask.size(0) != batch or attention_mask.size(1) not in (tokens, past + tokens):
                raise ValueError("2D attention_mask must cover query columns or the full cached prefix")
            if past and attention_mask.size(1) == past + tokens and not torch.equal(attention_mask[:, :past], cache.valid_mask):
                raise ValueError("a cache session cannot rewrite past padding metadata")
            query_valid = attention_mask[:, -tokens:]
        elif tensor_mask:
            if attention_mask.size(0) not in (1, batch) or attention_mask.shape[-2:] != (tokens, past + tokens):
                raise ValueError("attention_mask query/key dimensions differ from this forward")
            query_valid = attention_mask.any(dim=-1).any(dim=1).expand(batch, -1) if router_mask is None else router_mask
        elif attention_mask is not None and router_mask is None:
            raise ValueError("prepared native masks require explicit router_mask")
        else:
            query_valid = torch.ones_like(token_ids, dtype=torch.bool) if router_mask is None else router_mask
        if positions is None:
            positions = query_valid.long().cumsum(-1).sub(1).clamp_min(0) if cache is None else cache.next_positions(query_valid)
        if positions.shape != token_ids.shape or positions.dtype != torch.long or positions.device != token_ids.device:
            raise ValueError("positions must be int64 [batch,tokens] on the token device")
        if not torch.compiler.is_compiling() and bool(((positions < 0) | (positions >= config.max_seq_len)).logical_and(query_valid).any()):
            raise ValueError("valid positions must be in [0, max_seq_len)")
        if attention_mask is not None and not tensor_mask:
            if past:
                raise ValueError("prepared native masks do not support decode cache")
            mask = attention_mask
        elif attention_mask is None and router_mask is None and not past:
            mask = None
        else:
            key_valid = query_valid if not past else torch.cat((cache.valid_mask, query_valid), dim=1)
            causal = torch.arange(past + tokens, device=token_ids.device)[None, :] <= (
                past + torch.arange(tokens, device=token_ids.device)[:, None])
            mask = causal[None, None] & key_valid[:, None, None, :] & query_valid[:, None, :, None]
            if tensor_mask and attention_mask.ndim == 4:
                mask = mask & attention_mask
        inputs = dict(positions=positions, attention_mask=mask, cache=cache, depth=depth,
                      router_mask=query_valid if router_mask is None else router_mask)
        return inputs, query_valid, differentiated, collect

    def readout(
        self, state: Tensor, visit: int, inputs: dict[str, object], calls: list[CellCall] | None = None,
    ) -> Tensor:
        """The hidden state the head reads after ``visit``: the suffix cells, then the final norm.

        With ``core_final_norm`` the final norm also comes first, before the suffix cells.
        The suffix calls' statistics go to ``calls`` when it is given.
        """
        calls = [] if calls is None else calls
        if self.config.core_final_norm:
            state = self.final_norm(state)
        for index, cell in enumerate(self.suffix):
            state, _ = call_cell(cell, state, ("suffix", index, visit), inputs, calls)
        return self.final_norm(state)

    def forward(
        self, token_ids: Tensor, *, positions: Tensor | None = None,
        attention_mask: object | None = None, router_mask: Tensor | None = None,
        num_visits: int | None = None, grad_visits: int | None = None,
        cache: DecodeCache | None = None, return_hidden: bool = False,
        collect_readouts: bool | None = None,
    ) -> ModelOutput:
        """Prefix, ``num_visits`` core visits (each read out when collecting), suffix and head.

        Stack order runs all core cells per visit, layer order all visits per core cell.
        Visits before the last ``grad_visits`` run under no_grad. With
        ``visit_final_norm`` every visit after the first starts from
        ``final_norm(state)``; without suffix cells each visit's readout is then the next visit's input.
        """
        config = self.config
        inputs, query_valid, grad_visits, collect = self.parse_inputs(
            token_ids, positions=positions, attention_mask=attention_mask, router_mask=router_mask,
            num_visits=num_visits, grad_visits=grad_visits, cache=cache, collect_readouts=collect_readouts)
        depth = inputs["depth"]
        calls: list[CellCall] = []

        anchor = self.embedding(token_ids)
        for index, cell in enumerate(self.prefix):
            anchor, _ = call_cell(cell, anchor, ("prefix", index, 0), inputs, calls)
        if config.initial_state == "random":
            state = initial_latent(anchor, training=self.training, std=config.latent_init_std,
                                   truncation=config.latent_truncation)
        else:
            state = anchor if config.initial_state == "input" else torch.zeros_like(anchor)
        readouts: list[Tensor] = []
        if config.order == "stack":
            memory = (EMACarry(), EMACarry())  # MemoryCell's global and local EMA; other cells pass it on
            for visit in range(depth):
                memory = (memory[0], EMACarry())  # the local stream restarts every visit
                with torch.no_grad() if visit < depth - grad_visits else nullcontext():
                    if config.visit_final_norm and visit:
                        state = self.final_norm(state)
                    if config.state_update == "ema" and visit:
                        gain = reinjection_gain(config.num_cells, visit + 1, scale=config.memory_scale)
                        state = (1 - gain) * state + gain * anchor
                    if self.input_adapter is not None:
                        state = self.input_adapter(state, anchor)
                    for index, cell in enumerate(self.core):
                        state, memory = call_cell(cell, state, ("core", index, visit), inputs, calls, memory)
                if collect:
                    readouts.append(self.readout(state, visit, inputs, calls))
        else:
            for index, cell in enumerate(self.core):
                for visit in range(depth):
                    state, _ = call_cell(cell, state, ("core", index, visit), inputs, calls)
        hidden = readouts[-1] if collect else self.readout(state, 0, inputs, calls)
        if cache is not None:
            cache.advance(inputs["positions"], query_valid, self.cache_slots(depth))
        return ModelOutput(
            logits=None if return_hidden else self.lm_head(hidden), hidden_states=hidden,
            visit_hidden_states=tuple(readouts),
            hazard_logits=torch.stack([self.exit_gate(h).squeeze(-1) for h in readouts]) if collect else None,
            **self.routing_statistics(calls, token_ids.device),
        )

    def routing_statistics(self, calls: list[CellCall], device: torch.device) -> dict[str, object]:
        """ModelOutput's routing fields; the auxiliary sums add the differentiated sparse calls in order."""
        width = self.router_count_width
        padded = [F.pad(call.counts, (0, width - call.width)) for call in calls]
        first = {"prefix": 0, "core": len(self.prefix), "suffix": len(self.prefix) + len(self.core)}
        per_cell = [torch.zeros(width, dtype=torch.long, device=device)
                    for _ in range(len(self.prefix) + len(self.core) + len(self.suffix))]
        for row, call in zip(padded, calls):  # one row per physical cell
            owner = first[call.slot[0]] + call.slot[1]
            per_cell[owner] = per_cell[owner] + row
        auxiliary = [call for call in calls if call.width and call.differentiated]
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return dict(balance_sum=sum((call.balance for call in auxiliary), zero),
                    z_sum=sum((call.z_loss for call in auxiliary), zero), aux_calls=len(auxiliary),
                    router_counts=torch.stack(per_cell), router_call_counts=torch.stack(padded),
                    router_call_slots=tuple(call.slot for call in calls),
                    router_call_widths=tuple(call.width for call in calls))


# The same class under its former name: old scripts, pickles and checkpoints name it this way.
RecursiveLM = LoopedLM
