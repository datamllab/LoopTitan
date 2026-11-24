#!/usr/bin/env python3
"""Matched single-GPU full-step comparison with public TorchTitan or Qwen3-MoE.

The primary reference executes the pinned TorchTitan Qwen3Model unchanged.
Only its finite config, physical weights and parameter aliases are adapted.
The optional Transformers reference reuses the existing FP32 math oracle.
No launcher, allocation, downloads, training dataset or task database is used.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from moeloop.model import ModelConfig, RecursiveLM


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER = {"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8,
             "weight_decay": 0.1, "foreach": False, "fused": False}
# Match the existing native-kernel gradient gate. Independent BF16 reductions
# are judged as vectors and relative to a tensor peak, not cancelled coordinates.
TOLERANCES = {
    "float32": {"relative_l2": 5e-4, "peak_factor": 0.0005, "floor": 3e-6,
                "loss_atol": 3e-6, "loss_rtol": 3e-5},
    "bfloat16": {"relative_l2": 0.03, "peak_factor": 2 * 2**-7, "floor": 1e-6,
                 "loss_atol": 0.005, "loss_rtol": 0.01},
}
TT_REFERENCE_HASHES = {
    "models/qwen3/__init__.py": "b10e47dc694d084f09ccdc7c8e8396f28213b5dc0a51dec19018b7f78af4642a",
    "models/qwen3/model.py": "473a7c36901e60bdd27f80ec2030cb870cd69d318f6663526bb9a678a47bf979",
    "models/common/decoder.py": "66d8067c4a3b2e0f430791345581d6514aecfada0e3b6f4f001dac68b1b0d230",
    "models/common/rope.py": "c81b4bf4ee9bfc13a97924e1ada97204879d8065f7cd4ccc2d272eabc4ad3027",
    "models/common/linear.py": "1ab1b342ac9f3c557904040c8459c082f9ba5ed8fd9c96f7bc6e175b89078cf3",
    "models/common/nn_modules.py": "f39d67269400fb31c4b599a917e6a519efeadc06eeed3af204a244d5ac6ab056",
    "models/common/config_utils.py": "c05af268ef1e289e53ebbd63bad0a63b4d97db214935a8624a214734116781b9",
}
COMPONENT_SCOPE = "component_equivalence_with_observed_end_to_end_bf16_route_divergence"


@dataclass(frozen=True)
class Settings:
    mode: str = "describe"
    reference: str = "torchtitan"
    candidate: str = "native"
    attention: str = "flex"
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 2
    seq_len: int = 2048
    head_chunks: int = 8
    activation_checkpoint: str = "none"
    warmup: int = 5
    steps: int = 20
    component_math: bool = False


@dataclass
class Pair:
    name: str
    actual: torch.nn.Parameter
    reference: torch.nn.Parameter

    def reference_value(self, *, gradient=False):
        return self.reference.grad if gradient else self.reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(config: ModelConfig, settings: Settings) -> dict:
    if settings.mode not in ("describe", "sanity", "benchmark"):
        raise ValueError("mode must be describe, sanity or benchmark")
    if settings.reference not in ("torchtitan", "transformers") or settings.candidate not in ("native", "plain"):
        raise ValueError("unknown public reference or candidate")
    if settings.dtype not in TOLERANCES or settings.attention not in ("sdpa", "flex"):
        raise ValueError("unsupported precision or attention backend")
    if settings.activation_checkpoint not in ("none", "full"):
        raise ValueError("only no AC or full per-logical-block AC is supported")
    if min(settings.batch_size, settings.seq_len, settings.head_chunks, settings.steps) < 1 or settings.warmup < 0:
        raise ValueError("batch, sequence, chunks and steps must be positive; warmup nonnegative")
    if settings.seq_len > config.max_seq_len or settings.seq_len % settings.head_chunks:
        raise ValueError("sequence must fit the model and divide evenly into head chunks")
    if config.input_injection != "none" or config.initial_state != "input":
        raise ValueError("injection and non-input initial states are outside the public reference")
    for region in ("prefix", "core", "suffix"):
        local = config.region_config(region)
        if local.residual_scale != 1.0:
            raise ValueError("scaled residual branches are outside the unchanged public reference")
        if local.always_on_hidden_dim or local.num_experts < 1:
            raise ValueError("only routed sparse cells without a shared FFN are compared")
        if (local.hidden_dim, local.num_experts, local.top_k) != (config.hidden_dim, config.num_experts, config.top_k):
            raise ValueError("heterogeneous expert regions are outside this finite mapping")
    if config.visit_capacity not in (None, config.num_visits):
        raise ValueError("unused per-visit parameter capacity cannot be matched")
    if settings.reference == "transformers" and (settings.attention != "sdpa" or not config.qk_norm):
        raise ValueError("Transformers Qwen3-MoE requires QK norm and matched SDPA here")
    if settings.component_math and (settings.reference != "torchtitan" or settings.dtype != "bfloat16"
            or settings.candidate != "native" or config.num_visits != 1):
        raise ValueError("component math qualification is limited to the standard MoE in the native BF16 TorchTitan comparison")
    versions = {}
    for name in ("torch", "torchtitan", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": 1, "status": "not_run", "settings": asdict(settings),
        "model_config": config.to_dict(), "runtime": {"python": platform.python_version(),
            "packages": versions, "torch": torch.__version__, "cuda": torch.version.cuda},
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / "src/moeloop/model.py", ROOT / "src/moeloop/moe.py", ROOT / "src/moeloop/titan.py")},
        "conditions": {"seed": 42, "parameter_dtype": "float32", "compute_dtype": settings.dtype,
            "optimizer": {"class": "torch.optim.AdamW", **OPTIMIZER},
            "balance_coeff": 0, "z_loss_coeff": 0, "expert_bias_update": False,
            "gradient_policy": "full BPTT; fixed terminal readout; one update per batch",
            "head_loss": "same pinned TorchTitan ChunkedLossWrapper and sum CE / all valid labels",
            "mask": "one unpadded causal document per row; all input/label positions valid",
            "attention": settings.attention, "activation_checkpoint": settings.activation_checkpoint,
            "attention_dropout": 0.0, "projection_bias": False, "norm_topk_prob": True,
            "router_scores": "FP32 softmax, normalized top-k, dropless", "tied_vocabulary_weights": False,
            "timed_routing": "natural_unmodified_router_outputs",
            "model_compile": False, "world_size": 1, "parameter_sharing": "same unique physical owners and logical order",
            "reference_adaptations": ["exact geometry/weights/parameter aliases", "auxiliary bias and losses disabled",
                "QK norm configurable in TorchTitan GQAttention; strict HF Qwen3 requires it",
                "HF router forward scoped to FP32 to match both native routers"],
            "timed_scope": "zero_grad + backbone forward + chunked head/CE/backward + decoder backward + AdamW",
            "excluded_from_timing": ["construction", "correctness checks", "compile/warmup", "data I/O", "checkpoint/eval"],
        },
        "tolerances_predeclared": TOLERANCES[settings.dtype],
        "lag_policy": {"max_step_ratio": 1.2, "action": "investigate a >20% median full-step gap; no universal performance guarantee"},
        "limitations": ["single GPU; no multi-node/FSDP/EP/TP qualification",
            "synthetic resident tokens measure implementation cost, not pretraining quality",
            "standard residual scale=1 only; selected SMELT scale=0.5 is not represented",
            "HF expert loop is a public math reference, not optimized TorchTitan throughput",
            "matched SDPA is a compact causal adaptation of the TT config; Flex is the primary native comparison"],
        "correctness": None, "performance": {"status": "not_measured"},
    }


@torch.no_grad()
def make_torchtitan_reference(actual: RecursiveLM, *, attention="flex"):
    """Unchanged public decoder/operators; map and deduplicate every owner."""
    from moeloop.titan import require_torchtitan_pin
    require_torchtitan_pin()
    import torchtitan
    from torchtitan.models.common import CosSinRoPE, Embedding, Linear
    from torchtitan.models.common.attention import ScaledDotProductAttention
    from torchtitan.models.common.nn_modules import RMSNorm
    from torchtitan.models.qwen3 import _build_qwen3_moe_layers
    from torchtitan.models.qwen3.model import Qwen3Model

    package = Path(torchtitan.__file__).parent
    if any(digest(package / name) != expected for name, expected in TT_REFERENCE_HASHES.items()):
        raise RuntimeError("public Qwen3 reference source differs from the pinned TorchTitan commit")
    cfg = actual.config
    count = cfg.prefix_cells + cfg.num_cells * cfg.num_visits + cfg.suffix_cells
    layers = _build_qwen3_moe_layers(n_layers=count, dim=cfg.dim, n_heads=cfg.num_heads,
        n_kv_heads=cfg.num_kv_heads, head_dim=cfg.dim // cfg.num_heads,
        moe_hidden_dim=cfg.hidden_dim, num_experts=cfg.num_experts, top_k=cfg.top_k,
        fuse_qkv=False, attn_backend="flex", moe_comm_backend="standard",
        rope=CosSinRoPE.Config(dim=cfg.dim // cfg.num_heads, max_seq_len=cfg.max_seq_len, theta=cfg.rope_theta))
    for layer in layers:
        layer.moe.load_balance_coeff = None
        layer.attention_norm.eps = layer.ffn_norm.eps = cfg.rms_eps
        layer.attention.qk_norm.eps = cfg.rms_eps
        if not cfg.qk_norm:
            layer.attention.qk_norm = None
        if attention == "sdpa":
            layer.attention.inner_attention = ScaledDotProductAttention.Config()
    reference = Qwen3Model.Config(dim=cfg.dim, vocab_size=cfg.vocab_size, layers=layers,
        tok_embeddings=Embedding.Config(num_embeddings=cfg.vocab_size, embedding_dim=cfg.dim),
        lm_head=Linear.Config(in_features=cfg.dim, out_features=cfg.vocab_size),
        norm=RMSNorm.Config(normalized_shape=cfg.dim, eps=cfg.rms_eps), enable_weight_tying=False).build()
    reference.init_states()
    names = {id(param): name for name, param in actual.canonical_parameters()}
    owners, pairs = {}, []

    def weight(parent, name, source):
        if id(source) not in owners:
            target = getattr(parent, name)
            target.copy_(source)
            owners[id(source)] = target
            pairs.append(Pair(names[id(source)], source, target))
        else:
            setattr(parent, name, owners[id(source)])

    weight(reference.tok_embeddings, "weight", actual.embedding.weight)
    weight(reference.lm_head, "weight", actual.lm_head.weight)
    weight(reference.norm, "weight", actual.final_norm.weight)
    calls = [(cell, 0) for cell in actual.prefix]
    calls += ([(cell, visit) for visit in range(cfg.num_visits) for cell in actual.core] if cfg.order == "stack"
              else [(cell, visit) for cell in actual.core for visit in range(cfg.num_visits)])
    calls += [(cell, 0) for cell in actual.suffix]
    for layer, (cell, visit) in zip(reference.layers.values(), calls, strict=True):
        source = cell.attentions[visit if len(cell.attentions) > 1 else 0]
        for source_name, parent, target_name in (
            ("q_proj", layer.attention.qkv_linear, "wq"), ("k_proj", layer.attention.qkv_linear, "wk"),
            ("v_proj", layer.attention.qkv_linear, "wv"), ("out_proj", layer.attention, "wo")):
            weight(getattr(parent, target_name), "weight", getattr(source, source_name).weight)
        if cfg.qk_norm:
            for name in ("q_norm", "k_norm"):
                weight(getattr(layer.attention, name), "weight", getattr(source, name).weight)
        weight(layer.attention_norm, "weight", cell.attention_norms[visit if len(cell.attention_norms) > 1 else 0].weight)
        weight(layer.ffn_norm, "weight", cell.ffn_norms[visit if len(cell.ffn_norms) > 1 else 0].weight)
        weight(layer.moe.router.gate, "weight", cell.routers[visit if len(cell.routers) > 1 else 0].gate.weight)
        for name in ("w1_EFD", "w3_EFD", "w2_EDF"):
            weight(layer.moe.routed_experts.inner_experts, name, getattr(cell.experts, name))
    if set(names) != set(owners) or {id(p.reference) for p in pairs} != {id(p) for p in reference.parameters()}:
        raise AssertionError("public reference does not cover every unique physical parameter")
    reference._skip_lm_head = True
    return reference, pairs


def make_models(config, settings):
    torch.manual_seed(42)
    actual = RecursiveLM(config)
    if settings.reference == "torchtitan":
        reference, pairs = make_torchtitan_reference(actual, attention=settings.attention)
    else:
        from compare_baselines import make_reference
        reference, pairs = make_reference(actual)
        reference.set_attn_implementation("sdpa")
        # Keep the official router forward and its FP32 softmax/top-k, while
        # preventing outer BF16 autocast from rounding its gate matmul.
        for gate in {id(layer.mlp.gate): layer.mlp.gate for layer in reference.model.layers}.values():
            original = gate.forward

            def fp32_gate(hidden, forward=original):
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    return forward(hidden.float())
            gate.forward = fp32_gate
    if settings.candidate == "native":
        from moeloop.titan import NativeAttention, NativeExperts, canonical_model_state, require_torchtitan_pin
        require_torchtitan_pin()
        native = RecursiveLM(config, expert_factory=NativeExperts,
            attention_factory=NativeAttention if settings.attention == "flex" else None)
        # The adapter consumes TitanModel's outer ``core.`` namespace. This
        # candidate is a bare RecursiveLM, whose own ``core.0`` is a real region
        # and must survive the adapter's removal of exactly one outer prefix.
        parameters = canonical_model_state({"core." + name: parameter for name, parameter in native.named_parameters()})
        if set(parameters) != {pair.name for pair in pairs}:
            raise AssertionError("native replacement must preserve every canonical physical parameter key")
        with torch.no_grad():
            for pair in pairs:
                parameters[pair.name].copy_(pair.actual)
                pair.actual = parameters[pair.name]
        actual = native
    if settings.activation_checkpoint == "full":
        from torchtitan.distributed.activation_checkpoint import FullAC
        policy = FullAC.Config().build()
        for region in (actual.prefix, actual.core, actual.suffix):
            for index, cell in enumerate(region):
                region[index] = policy._wrap_block(cell)
        layers = reference.layers if settings.reference == "torchtitan" else reference.model.layers
        for key, layer in list(layers.named_children()):
            layers[int(key) if isinstance(layers, torch.nn.ModuleList) else key] = policy._wrap_block(layer)
    return actual, reference, pairs


def make_batch(config, settings):
    # S+1 tokens give exactly S valid next-token labels, including the final
    # input column. Both arms receive these already shifted labels unchanged.
    generator = torch.Generator().manual_seed(42)
    tokens = torch.randint(config.vocab_size, (settings.batch_size, settings.seq_len + 1), generator=generator)
    return {"input": tokens[:, :-1].contiguous().to(settings.device),
            "labels": tokens[:, 1:].contiguous().to(settings.device),
            "positions": torch.arange(settings.seq_len).expand(settings.batch_size, -1).to(settings.device)}


def autocast(settings):
    return (torch.autocast(device_type=torch.device(settings.device).type, dtype=torch.bfloat16)
            if settings.dtype == "bfloat16" else nullcontext())


def hidden_forward(model, batch, settings, *, reference=False):
    if reference and settings.reference == "transformers":
        return model.model(input_ids=batch["input"], position_ids=batch["positions"], use_cache=False).last_hidden_state
    kwargs = {"positions": batch["positions"]}
    if "block_mask" in batch:
        kwargs["attention_masks" if reference else "attention_mask"] = batch["block_mask"]
    if reference:
        return model(batch["input"], **kwargs)
    if "block_mask" in batch:
        kwargs["router_mask"] = torch.ones_like(batch["input"], dtype=torch.bool)
    return model(batch["input"], return_hidden=True, **kwargs).hidden_states


def make_loss(model, settings):
    from torchtitan.components.loss import ChunkedLossWrapper
    loss = ChunkedLossWrapper.Config(num_chunks=settings.head_chunks).build()
    loss.set_lm_head(model.lm_head)
    return loss


def tensor_error(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    diff = actual - expected
    return {"max_absolute": float(diff.abs().max()),
            "relative_l2": float(diff.norm()) / max(float(expected.norm()), 1e-20),
            "reference_peak": float(expected.abs().max())}


def full_step_safe(result):
    return (all(result.get(key) is True for key in ("initial_parameters_equal", "losses_finite", "ce_close",
        "all_gradients_finite", "all_updated_parameters_finite", "all_parameter_gradients_checked"))
        and result.get("optimizer_updates") == 1)


def check_component_math(actual, batch, settings):
    """Same-input operator derivatives, independent of whole-network routing.

    Only the representative homogeneous cell is exercised. Its expert operator
    receives fixed matching routes solely for this derivative oracle. The timed
    full model always executes its original router forwards.
    """
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen
    cfg, tolerance = actual.config, TOLERANCES[settings.dtype]
    cell = getattr(actual.core[0], "_checkpoint_wrapped_module", actual.core[0])
    attention, router = cell.attentions[0], cell.routers[0]
    bank = cell.experts.routed.inner_experts if hasattr(cell.experts, "routed") else cell.experts
    checks = {}
    hf_source = {"version": importlib.metadata.version("transformers"), "file_sha256": digest(qwen.__file__)}

    def new_check(name, reference, inputs, parameters):
        result = {"status": "passed", "reference": reference,
            "coverage": {"input_gradients": inputs, "parameter_gradients": parameters}, "errors": {}, "failures": []}
        checks[name] = result
        return result

    def record(check, name, observed, expected):
        if (observed is None or expected is None or not torch.isfinite(observed).all()
                or not torch.isfinite(expected).all()):
            check["errors"][name] = {key: None for key in ("max_absolute", "relative_l2", "reference_peak", "max_absolute_allowed")}
            check["failures"].append("missing or nonfinite " + name)
        else:
            error = tensor_error(observed, expected)
            bound = tolerance["peak_factor"] * error["reference_peak"] + tolerance["floor"]
            check["errors"][name] = {**error, "max_absolute_allowed": bound}
            if error["relative_l2"] > tolerance["relative_l2"] or error["max_absolute"] > bound:
                check["failures"].append(name + " exceeds the predeclared vector or peak-scale bound")
        if check["failures"]:
            check["status"] = "failed"

    def upstream(value):
        return torch.sin(torch.arange(value.numel(), device=value.device, dtype=torch.float32)).reshape_as(value)

    with torch.no_grad(), autocast(settings):
        x = cell.attention_norms[0](actual.embedding(batch["input"]))
        q = attention.q_proj(x).reshape(*x.shape[:2], cfg.num_heads, -1).transpose(1, 2)
        k = attention.k_proj(x).reshape(*x.shape[:2], cfg.num_kv_heads, -1).transpose(1, 2)
        v = attention.v_proj(x).reshape(*x.shape[:2], cfg.num_kv_heads, -1).transpose(1, 2)
        from moeloop.model import apply_rotary
        q = apply_rotary(attention.q_norm(q), batch["positions"], cfg.rope_theta)
        k = apply_rotary(attention.k_norm(k), batch["positions"], cfg.rope_theta)
    check = new_check("attention", {"name": "torch.nn.functional.scaled_dot_product_attention",
        "version": torch.__version__, "file_sha256": digest(torch.nn.functional.__file__),
        "mask": "same causal predicate; SDPA is_causal=True; enable_gqa=True"}, ["query", "key", "value"], [])
    qa, ka, va = [value.detach().clone().requires_grad_() for value in (q, k, v)]
    qr, kr, vr = [value.detach().clone().requires_grad_() for value in (q, k, v)]
    with autocast(settings):
        observed = attention._attention(qa, ka, va, batch.get("block_mask"), is_causal=True)
        expected = torch.nn.functional.scaled_dot_product_attention(qr, kr, vr, is_causal=True, enable_gqa=True)
    record(check, "forward", observed, expected)
    grad = upstream(observed)
    (observed.float() * grad).mean().backward()
    (expected.float() * grad).mean().backward()
    for name, a, r in (("query", qa, qr), ("key", ka, kr), ("value", va, vr)):
        record(check, "gradient." + name, a.grad, r.grad)
    del qa, ka, va, qr, kr, vr, observed, expected, q, k, v

    hf_config = Qwen3MoeConfig(hidden_size=cfg.dim, moe_intermediate_size=cfg.hidden_dim,
        num_experts=cfg.num_experts, num_experts_per_tok=cfg.top_k, norm_topk_prob=True)
    hf_router = qwen.Qwen3MoeTopKRouter(hf_config).to(settings.device)
    with torch.no_grad():
        hf_router.weight.copy_(router.gate.weight)
    check = new_check("router", {"name": "transformers.Qwen3MoeTopKRouter", **hf_source}, ["input"], ["gate.weight"])
    xa, xr = [x.detach().float().clone().requires_grad_() for _ in range(2)]
    actual.zero_grad(set_to_none=True)
    with torch.autocast(device_type=torch.device(settings.device).type, enabled=False):
        routed = router(xa)
        actual_logits = torch.nn.functional.linear(xa, router.gate.weight.float())
        ref_logits, ref_weights, ref_ids = hf_router(xr)
        ref_logits, ref_weights, ref_ids = [value.reshape(*xa.shape[:2], -1) for value in (ref_logits, ref_weights, ref_ids)]
        actual_ids, order = routed.ids.sort(-1)
        actual_weights = routed.weights.gather(-1, order)
        ref_ids, order = ref_ids.sort(-1)
        ref_weights = ref_weights.gather(-1, order)
        check["expert_sets_equal"] = torch.equal(actual_ids, ref_ids)
        if not check["expert_sets_equal"]:
            check["failures"].append("same-input FP32 router expert sets differ")
            check["status"] = "failed"
        record(check, "logits", actual_logits, ref_logits)
        record(check, "weights", actual_weights, ref_weights)
        logit_grad, weight_grad = upstream(actual_logits), upstream(actual_weights)
        ((actual_logits * logit_grad).mean() + (actual_weights * weight_grad).mean()).backward()
        ((ref_logits * logit_grad).mean() + (ref_weights * weight_grad).mean()).backward()
    record(check, "gradient.input", xa.grad, xr.grad)
    record(check, "gradient.gate.weight", router.gate.weight.grad, hf_router.weight.grad)
    ids, weights = routed.ids.detach(), routed.weights.detach()
    del routed, xa, xr, actual_logits, ref_logits, actual_weights, ref_weights, hf_router
    actual.zero_grad(set_to_none=True)

    hf_experts = qwen.Qwen3MoeExperts(hf_config).to(settings.device)
    with torch.no_grad():
        hf_experts.gate_up_proj[:, :cfg.hidden_dim].copy_(bank.w1_EFD)
        hf_experts.gate_up_proj[:, cfg.hidden_dim:].copy_(bank.w3_EFD)
        hf_experts.down_proj.copy_(bank.w2_EDF)
    check = new_check("experts", {"name": "transformers.Qwen3MoeExperts", **hf_source},
        ["input", "route_weights"], ["w1_EFD", "w3_EFD", "w2_EDF"])
    check["coverage"].update(experts=cfg.num_experts, top_k=cfg.top_k)
    check["routing"] = "fixed same-input FP32 router outputs for this operator derivative check only"
    xa, xr = [x.detach().float().clone().requires_grad_() for _ in range(2)]
    wa, wr = [weights.clone().requires_grad_() for _ in range(2)]
    counts = torch.bincount(ids.reshape(-1), minlength=cfg.num_experts)
    with autocast(settings):
        observed = cell.experts(xa, ids, wa, counts)
        expected = hf_experts(xr.reshape(-1, cfg.dim), ids.reshape(-1, cfg.top_k), wr.reshape(-1, cfg.top_k)).reshape_as(observed)
    record(check, "forward", observed, expected)
    grad = upstream(observed)
    (observed.float() * grad).mean().backward()
    (expected.float() * grad).mean().backward()
    record(check, "gradient.input", xa.grad, xr.grad)
    record(check, "gradient.route_weights", wa.grad, wr.grad)
    for name, expected_gradient in (("w1_EFD", hf_experts.gate_up_proj.grad[:, :cfg.hidden_dim]),
            ("w3_EFD", hf_experts.gate_up_proj.grad[:, cfg.hidden_dim:]), ("w2_EDF", hf_experts.down_proj.grad)):
        record(check, "gradient." + name, getattr(bank, name).grad, expected_gradient)
    actual.zero_grad(set_to_none=True)
    return {"status": "passed" if all(check["status"] == "passed" for check in checks.values()) else "failed",
        "scope": COMPONENT_SCOPE, "tolerances": tolerance, "checks": checks,
        "timed_routing": "natural_unmodified_router_outputs",
        "coverage": {"representative_physical_cell": "core.0", "homogeneous_geometry": True,
            "all_model_parameter_mapping_verified": False}}


def check_correctness(actual, reference, pairs, batch, settings):
    """One genuine full update, all mapped gradients, no performance claim."""
    actual.to(settings.device).train()
    reference.to(settings.device).train()
    failures = []
    if any(not torch.equal(pair.actual, pair.reference_value()) for pair in pairs):
        return {"status": "failed", "failures": ["initial parameters differ"], "optimizer_updates": 0}
    tolerance = TOLERANCES[settings.dtype]

    def check(name, observed, expected):
        error = tensor_error(observed, expected)
        bound = tolerance["peak_factor"] * error["reference_peak"] + tolerance["floor"]
        if (not torch.isfinite(observed).all() or not torch.isfinite(expected).all()
                or error["relative_l2"] > tolerance["relative_l2"] or error["max_absolute"] > bound):
            failures.append(name + " exceeds the predeclared vector or peak-scale bound")
        return {**error, "max_absolute_allowed": bound}

    optimizers = [torch.optim.AdamW(model.parameters(), **OPTIMIZER) for model in (actual, reference)]
    losses, hidden_states, sampled_logits = [], [], []
    route_calls = [[], []]
    denominator = torch.tensor(batch["labels"].numel(), device=settings.device)
    for index, model in enumerate((actual, reference)):
        optimizers[index].zero_grad(set_to_none=True)
        handles = []
        if settings.reference == "torchtitan":
            from moeloop.moe import Router
            from torchtitan.models.common.moe import TokenChoiceTopKRouter
            router_type = Router if index == 0 else TokenChoiceTopKRouter

            def capture(module, inputs, output, index=index):
                ids = output.ids if index == 0 else output[1]
                route_calls[index].append(ids.detach().sort(-1).values.cpu())
            handles = [module.register_forward_hook(capture) for module in model.modules() if isinstance(module, router_type)]
        with autocast(settings):
            try:
                hidden = hidden_forward(model, batch, settings, reference=bool(index))
            finally:
                for handle in handles:
                    handle.remove()
            with torch.no_grad():
                sampled_logits.append(model.lm_head(hidden[:, :min(16, settings.seq_len)].detach()))
            loss, _ = make_loss(model, settings)(hidden, batch["labels"], denominator)
        loss.backward()
        hidden_states.append(hidden.detach())
        losses.append(float(loss.detach()))
    if (not all(math.isfinite(value) for value in losses)
            or any(not torch.isfinite(value).all() for value in hidden_states + sampled_logits)):
        return {"status": "failed", "failures": ["nonfinite forward or CE"], "optimizer_updates": 0}
    hidden_error = check("hidden states", *hidden_states)
    logits_error = check("sampled logits", *sampled_logits)
    ce_close = math.isclose(*losses, abs_tol=tolerance["loss_atol"], rel_tol=tolerance["loss_rtol"])
    if not ce_close:
        failures.append("CE exceeds the predeclared absolute/relative bound")
    gradients = {}
    for pair in pairs:
        expected = pair.reference_value(gradient=True)
        if pair.actual.grad is None or expected is None:
            failures.append("missing gradient: " + pair.name)
            continue
        if not torch.isfinite(pair.actual.grad).all() or not torch.isfinite(expected).all():
            return {"status": "failed", "failures": ["nonfinite gradient: " + pair.name], "optimizer_updates": 0}
        gradients[pair.name] = check("gradient " + pair.name, pair.actual.grad, expected)
    for optimizer in optimizers:
        optimizer.step()
    optimizer_updates = min((int(state["step"]) for optimizer in optimizers for state in optimizer.state.values()), default=0)
    if optimizer_updates != 1:
        failures.append("optimizer did not complete one full update")
    if any(not torch.isfinite(pair.actual).all() or not torch.isfinite(pair.reference_value()).all() for pair in pairs):
        return {"status": "failed", "failures": ["nonfinite updated parameters"], "optimizer_updates": optimizer_updates}
    parameter_errors = [tensor_error(pair.actual, pair.reference_value()) for pair in pairs]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    routing = None
    if route_calls[0] and len(route_calls[0]) == len(route_calls[1]):
        calls = []
        for index, (a, r) in enumerate(zip(*route_calls, strict=True)):
            calls.append({"logical_call": index, "tokens": a.numel() // a.shape[-1],
                "changed_tokens": int((a != r).any(-1).sum()),
                "candidate_expert_ids_sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                "reference_expert_ids_sha256": hashlib.sha256(r.numpy().tobytes()).hexdigest()})
        tokens, changed = sum(row["tokens"] for row in calls), sum(row["changed_tokens"] for row in calls)
        routing = {"logical_calls": len(calls), "compared_tokens": tokens, "changed_tokens": changed,
            "changed_token_fraction": changed / tokens, "calls": calls}
    return {"status": "failed" if failures else "passed", "failures": failures,
            "initial_parameters_equal": True, "losses_finite": True, "ce_close": ce_close,
            "all_gradients_finite": True, "all_updated_parameters_finite": True, "routing": routing,
            "physical_parameter_tensors": len(pairs), "physical_parameter_elements": sum(pair.actual.numel() for pair in pairs),
            "all_parameter_gradients_checked": len(gradients) == len(pairs), "optimizer_updates": optimizer_updates,
            "candidate_ce": losses[0], "reference_ce": losses[1], "hidden_states": hidden_error,
            "sampled_logits": logits_error, "sampled_logit_positions_per_row": min(16, settings.seq_len),
            "gradient_errors": gradients,
            "updated_parameters_max_absolute": max(error["max_absolute"] for error in parameter_errors),
            "updated_parameters_max_relative_l2": max(error["relative_l2"] for error in parameter_errors)}


def summary(values):
    median = statistics.median(values)
    return {"median": median, "min": min(values), "max": max(values),
            "mad": statistics.median(abs(value - median) for value in values),
            "p10": float(torch.tensor(values, dtype=torch.float64).quantile(0.1)),
            "p90": float(torch.tensor(values, dtype=torch.float64).quantile(0.9))}


def profile(model, batch, settings, *, reference=False):
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
    loss_fn = make_loss(model, settings)
    denominator = torch.tensor(batch["labels"].numel(), device=settings.device)
    phases = {key: [] for key in ("backbone_forward", "head_loss_and_head_backward", "decoder_backward", "optimizer")}

    def step(events=None):
        optimizer.zero_grad(set_to_none=True)
        with autocast(settings):
            hidden = hidden_forward(model, batch, settings, reference=reference)
            if events:
                events[1].record()
            loss, _ = loss_fn(hidden, batch["labels"], denominator)
            if events:
                events[2].record()
        loss.backward()
        if events:
            events[3].record()
        optimizer.step()
        if events:
            events[4].record()
        return loss.detach()

    started = time.perf_counter()
    for _ in range(settings.warmup):
        step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    durations, losses = [], []
    for _ in range(settings.steps):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        torch.cuda.synchronize()
        started = time.perf_counter()
        events[0].record()
        loss = step(events)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        losses.append(float(loss))
        if not math.isfinite(losses[-1]):
            raise RuntimeError("nonfinite CE during measured updates")
        for index, name in enumerate(phases):
            phases[name].append(events[index].elapsed_time(events[index + 1]) / 1000)
    measured = summary(durations)
    return {"full_step_seconds": measured, "full_step_samples_seconds": durations,
            "tokens_per_second": batch["labels"].numel() / measured["median"],
            "phase_seconds_cuda_events": {name: summary(values) for name, values in phases.items()},
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "warmup_seconds_excluded": warmup_seconds, "losses": losses}


def performance_verdict(candidate_seconds, reference_seconds):
    return "passed" if candidate_seconds / reference_seconds <= 1.2 else "needs_investigation"


def run(config, settings):
    report = describe(config, settings)
    if settings.mode == "describe":
        return report
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("public-reference benchmark is single GPU; use the native launcher for distributed scaling")
    if settings.mode == "benchmark" and (torch.device(settings.device).type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("performance qualification requires an already visible CUDA GPU")
    if settings.mode == "benchmark" and (settings.dtype != "bfloat16" or settings.candidate != "native" or settings.steps < 5 or settings.warmup < 2):
        raise ValueError("performance qualification requires native BF16, at least 2 warmup and 5 measured updates")
    if settings.attention == "flex" and torch.device(settings.device).type != "cuda":
        raise ValueError("Flex/grouped execution requires CUDA; CPU supports describe or plain HF/SDPA sanity")
    if torch.device(settings.device).type == "cuda":
        device_index = torch.device(settings.device).index
        torch.cuda.set_device(torch.cuda.current_device() if device_index is None else device_index)
    torch.manual_seed(42)
    actual, reference, pairs = make_models(config, settings)
    initial = {pair.name: pair.actual.detach().cpu().clone() for pair in pairs}
    weight_hash = hashlib.sha256()
    for name, tensor in initial.items():
        weight_hash.update(name.encode())
        weight_hash.update(tensor.numpy().tobytes())
    report["initial_weights_sha256"] = weight_hash.hexdigest()
    report["reference_class"] = type(reference).__module__ + "." + type(reference).__name__
    report["source_sha256"]["public_reference"] = digest(inspect.getfile(type(reference)))
    if settings.reference == "torchtitan":
        from moeloop.titan import TORCHTITAN_COMMIT
        report["torchtitan_commit"] = TORCHTITAN_COMMIT
        report["public_reference_source_sha256"] = TT_REFERENCE_HASHES
    else:
        report["source_sha256"]["hf_weight_mapping"] = digest(ROOT / "scripts/compare_baselines.py")
    batch = make_batch(config, settings)
    if settings.attention == "flex":
        from torch.nn.attention.flex_attention import create_block_mask
        batch["block_mask"] = create_block_mask(lambda b, h, q, k: q >= k,
            B=settings.batch_size, H=None, Q_LEN=settings.seq_len, KV_LEN=settings.seq_len,
            device=settings.device, _compile=True)
    report["batch"] = {"input_positions": batch["input"].numel(), "valid_inputs": batch["input"].numel(),
        "valid_labels": batch["labels"].numel(), "documents": settings.batch_size, "padding_positions": 0,
        "input_sha256": hashlib.sha256(batch["input"].cpu().numpy().tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(batch["labels"].cpu().numpy().tobytes()).hexdigest()}
    report["correctness"] = check_correctness(actual, reference, pairs, batch, settings)
    report["status"] = report["correctness"]["status"]
    allow_profile = report["status"] == "passed"
    if settings.component_math and full_step_safe(report["correctness"]):
        with torch.no_grad():
            for pair in pairs:
                pair.actual.copy_(initial[pair.name])
                pair.reference_value().copy_(initial[pair.name])
        try:
            report["component_math"] = check_component_math(actual, batch, settings)
        except (RuntimeError, ImportError) as error:
            # Preserve the original strict result even when an optional public
            # component dependency or GPU kernel fails before returning metrics.
            report["component_math"] = {"status": "failed", "scope": COMPONENT_SCOPE,
                "tolerances": TOLERANCES[settings.dtype], "checks": {}, "coverage": {},
                "timed_routing": "natural_unmodified_router_outputs",
                "error": type(error).__name__ + ": " + str(error)}
        report["component_math"]["coverage"].update(
            all_model_parameter_mapping_verified=all(torch.equal(pair.actual, pair.reference_value()) for pair in pairs),
            model_physical_parameter_tensors=len(pairs),
            model_physical_parameter_elements=sum(pair.actual.numel() for pair in pairs))
        report["mathematics_scope"] = COMPONENT_SCOPE
        routing = report["correctness"].get("routing")
        allow_profile = (report["component_math"]["status"] == "passed"
            and report["component_math"]["coverage"]["all_model_parameter_mapping_verified"]
            and (report["correctness"]["status"] == "passed" or routing is not None and routing["changed_tokens"] > 0))
    if not allow_profile:
        report["status"] = "failed"
        return report
    if settings.mode != "benchmark":
        return report
    with torch.no_grad():
        for pair in pairs:
            pair.actual.copy_(initial[pair.name])
            pair.reference_value().copy_(initial[pair.name])
    del initial
    arms = {}
    for name, model, inactive in (("candidate", actual, reference), ("reference", reference, actual)):
        inactive.cpu()
        model.to(settings.device)
        gc.collect()
        torch.cuda.empty_cache()
        arms[name] = profile(model, batch, settings, reference=name == "reference")
        model.zero_grad(set_to_none=True)
    candidate_seconds = arms["candidate"]["full_step_seconds"]["median"]
    reference_seconds = arms["reference"]["full_step_seconds"]["median"]
    report["status"] = performance_verdict(candidate_seconds, reference_seconds)
    report["performance"] = {"status": report["status"], "arms": arms,
        "candidate_reference_step_ratio": candidate_seconds / reference_seconds,
        "measurement_order": ["candidate", "reference"],
        "gpu": torch.cuda.get_device_name(), "device_total_memory_bytes": torch.cuda.get_device_properties(settings.device).total_memory,
        "memory_scope": "each arm measured in isolation; inactive model and optimizer states reside off GPU"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True, help="ModelConfig JSON, or RecipeSpec JSON with a model field")
    parser.add_argument("--mode", choices=("describe", "sanity", "benchmark"), default="describe")
    parser.add_argument("--reference", choices=("torchtitan", "transformers"), default="torchtitan")
    parser.add_argument("--candidate", choices=("native", "plain"), default="native")
    parser.add_argument("--attention", choices=("sdpa", "flex"), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--activation-checkpoint", choices=("none", "full"), default="none")
    parser.add_argument("--component-math", action="store_true", help="explicitly qualify the standard MoE in TT BF16 by independent same-input component derivatives; retain strict whole-model result")
    for name, default in (("batch-size", 2), ("seq-len", 2048), ("head-chunks", 8), ("warmup", 5), ("steps", 20)):
        parser.add_argument("--" + name, type=int, default=default)
    args = vars(parser.parse_args())
    output, model_path = args.pop("output"), args.pop("model_config")
    payload = json.loads(model_path.read_text())
    config = ModelConfig.from_dict(payload.get("model", payload))
    args["attention"] = args["attention"] or ("flex" if args["reference"] == "torchtitan" else "sdpa")
    try:
        report = run(config, Settings(**args))
    except (ValueError, RuntimeError, ImportError) as error:
        report = {"schema_version": 1, "status": "error", "settings": args,
                  "error": type(error).__name__ + ": " + str(error), "performance": {"status": "not_measured"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "performance_status": report["performance"]["status"]}))
    if report["status"] in ("failed", "error", "needs_investigation"):
        raise SystemExit(2 if report["status"] == "needs_investigation" else 1)


if __name__ == "__main__":
    main()
