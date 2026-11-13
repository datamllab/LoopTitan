#!/usr/bin/env python3
"""Bounded numerical comparison against the official Transformers Qwen3-MoE.

No pretrained weights or data downloads are needed. The reference executes
Qwen3MoeForCausalLM.forward unchanged, with official decoder modules placed in
the desired logical order. Its eager attention is independent of our SDPA.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import transformers
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen

from moeloop.cache import DecodeCache
from moeloop.model import ModelConfig, RecursiveLM


@dataclass
class ParameterPair:
    name: str
    actual: nn.Parameter
    reference: nn.Parameter
    part: slice | None = None

    def reference_value(self, *, gradient: bool = False) -> Tensor | None:
        value = self.reference.grad if gradient else self.reference
        if value is None or self.part is None:
            return value
        return value[:, self.part, :]


def reference_config(config: ModelConfig) -> Qwen3MoeConfig:
    """Only the intersection with standard Qwen3-MoE is compared here."""
    if not config.qk_norm or config.num_experts < 1:
        raise ValueError("Qwen3-MoE comparison requires sparse cells and QK normalization")
    if (config.input_injection != "none" or config.initial_state != "input"
            or config.always_on_hidden_dim or config.residual_scale != 1.0):
        raise ValueError("injection, zero state, shared FFN and scaled residuals are outside this reference")
    count = config.prefix_cells + config.num_cells * config.num_visits + config.suffix_cells
    result = Qwen3MoeConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.dim,
        intermediate_size=config.hidden_dim,
        moe_intermediate_size=config.hidden_dim,
        num_hidden_layers=count,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_kv_heads,
        num_experts=config.num_experts,
        num_experts_per_tok=config.top_k,
        norm_topk_prob=True,
        max_position_embeddings=config.max_seq_len,
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        rms_norm_eps=config.rms_eps,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        tie_word_embeddings=False,
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        use_cache=False,
    )
    result._attn_implementation = "eager"
    return result


@torch.no_grad()
def make_reference(actual: RecursiveLM) -> tuple[Qwen3MoeForCausalLM, list[ParameterPair]]:
    """Map every physical parameter to official modules, preserving sharing.

    A packed Qwen gate_up_proj holds our w1/w3 in its two halves. Multiple
    logical decoder layers refer to the same official parameter owners when
    shared. Attention objects have distinct logical cache indices, even with
    shared weights. No moeloop forward/operator is used by the reference.
    """
    config = actual.config
    hf_config = reference_config(config)
    device = next(actual.parameters()).device
    reference = Qwen3MoeForCausalLM(hf_config).to(device=device, dtype=torch.float32)
    pairs: list[ParameterPair] = []
    owners: dict[int, nn.Module] = {}
    names = {id(parameter): name for name, parameter in actual.canonical_parameters()}

    def copy(parameter: nn.Parameter, target: nn.Parameter, part: slice | None = None) -> None:
        value = target if part is None else target[:, part, :]
        value.copy_(parameter)
        pairs.append(ParameterPair(names[id(parameter)], parameter, target, part))

    def attention(source: nn.Module) -> nn.Module:
        if id(source) not in owners:
            target = qwen.Qwen3MoeAttention(hf_config, layer_idx=0).to(device)
            for source_name, target_name in (("q_proj", "q_proj"), ("k_proj", "k_proj"),
                                             ("v_proj", "v_proj"), ("out_proj", "o_proj"),
                                             ("q_norm", "q_norm"), ("k_norm", "k_norm")):
                copy(getattr(source, source_name).weight, getattr(target, target_name).weight)
            owners[id(source)] = target
        return owners[id(source)]

    def norm(source: nn.Module) -> nn.Module:
        if id(source) not in owners:
            target = qwen.Qwen3MoeRMSNorm(source.weight.numel(), eps=config.rms_eps).to(device)
            copy(source.weight, target.weight)
            owners[id(source)] = target
        return owners[id(source)]

    def router(source: nn.Module) -> nn.Module:
        if id(source) not in owners:
            target = qwen.Qwen3MoeTopKRouter(hf_config).to(device)
            copy(source.gate.weight, target.weight)
            owners[id(source)] = target
        return owners[id(source)]

    def experts(source: nn.Module) -> nn.Module:
        if id(source) not in owners:
            target = qwen.Qwen3MoeExperts(hf_config).to(device)
            width = config.hidden_dim
            copy(source.w1_EFD, target.gate_up_proj, slice(0, width))
            copy(source.w3_EFD, target.gate_up_proj, slice(width, 2 * width))
            copy(source.w2_EDF, target.down_proj)
            owners[id(source)] = target
        return owners[id(source)]

    def decoder(cell: nn.Module, visit: int) -> nn.Module:
        layer = qwen.Qwen3MoeDecoderLayer(hf_config, layer_idx=0).to(device)
        physical_attention = attention(cell.attentions[visit if len(cell.attentions) > 1 else 0])
        # Use official logical attention instances with shared Linear/Norm
        # owners. Their distinct layer_idx keeps KV independent across visits.
        for name, child in physical_attention.named_children():
            setattr(layer.self_attn, name, child)
        index = visit if len(cell.attention_norms) > 1 else 0
        layer.input_layernorm = norm(cell.attention_norms[index])
        layer.post_attention_layernorm = norm(cell.ffn_norms[index])
        layer.mlp.gate = router(cell.routers[visit if len(cell.routers) > 1 else 0])
        layer.mlp.experts = experts(cell.experts)
        return layer

    copy(actual.embedding.weight, reference.model.embed_tokens.weight)
    copy(actual.lm_head.weight, reference.lm_head.weight)
    reference.model.norm = norm(actual.final_norm)
    layers = [decoder(cell, 0) for cell in actual.prefix]
    if config.order == "stack":
        layers.extend(decoder(cell, visit) for visit in range(config.num_visits) for cell in actual.core)
    else:
        layers.extend(decoder(cell, visit) for cell in actual.core for visit in range(config.num_visits))
    layers.extend(decoder(cell, 0) for cell in actual.suffix)
    for index, layer in enumerate(layers):
        layer.self_attn.layer_idx = index
    reference.model.layers = nn.ModuleList(layers)
    if len(pairs) != len(names) or {pair.name for pair in pairs} != set(names.values()):
        raise AssertionError("the independent baseline must cover every physical moeloop parameter")
    if {id(pair.reference) for pair in pairs} != {id(p) for p in reference.parameters()}:
        raise AssertionError("reference has unmapped or unused parameters")
    return reference, pairs


def tiny_config(**changes: Any) -> ModelConfig:
    return replace(ModelConfig(
        vocab_size=43, dim=24, hidden_dim=32, num_heads=4, num_kv_heads=2,
        num_experts=4, top_k=2, num_cells=2, num_visits=1,
        max_seq_len=32, rope_theta=10000.0, qk_norm=True, init_std=0.08,
    ), **changes)


def cases() -> dict[str, ModelConfig]:
    return {
        "ordinary_moe": tiny_config(prefix_cells=1, suffix_cells=1),
        "shared_stack": tiny_config(num_visits=3),
        "shared_layer": tiny_config(num_visits=3, order="layer"),
        "partial_attention": tiny_config(num_visits=3, attention_per_visit=True),
        "partial_router": tiny_config(num_visits=3, router_per_visit=True),
        "expert_tying": tiny_config(num_visits=3, order="layer", attention_per_visit=True,
                                   router_per_visit=True, norm_per_visit=True),
    }


def max_error(actual: Tensor, expected: Tensor) -> float:
    return float((actual.detach().float() - expected.detach().float()).abs().max().cpu())


@torch.no_grad()
def compare_cache(actual: RecursiveLM, reference: Qwen3MoeForCausalLM, ids: Tensor) -> dict[str, Any]:
    """Compare true cached chunks to official per-logical-layer DynamicCache."""
    actual.eval()
    reference.eval()
    cache = DecodeCache(actual.config.num_visits)
    reference_cache = None
    actual_chunks, reference_chunks = [], []
    for part in (ids[:, :3], ids[:, 3:5], ids[:, 5:]):
        actual_chunks.append(actual(part, cache=cache).logits)
        expected = reference(input_ids=part, use_cache=True, past_key_values=reference_cache)
        reference_chunks.append(expected.logits)
        reference_cache = expected.past_key_values
    result, expected = torch.cat(actual_chunks, 1), torch.cat(reference_chunks, 1)
    full = actual(ids).logits
    torch.testing.assert_close(result, expected, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(result, full, atol=3e-5, rtol=3e-4)
    if cache.length != ids.size(1) or reference_cache.get_seq_length() != ids.size(1):
        raise AssertionError("both caches must advance exactly once per token column")
    if len(cache.slots) != reference.config.num_hidden_layers:
        raise AssertionError("shared weights still require one cache slot per logical attention call")
    return {"logits_max_absolute_error": max_error(result, expected),
            "full_prefix_max_absolute_error": max_error(result, full),
            "logical_slots": len(cache.slots), "cached_columns": cache.length}


def run_case(config: ModelConfig, *, steps: int = 4, device: str = "cpu", seed: int = 123) -> dict[str, Any]:
    """Compare forward, all physical gradients, and a short AdamW trajectory."""
    if steps < 2:
        raise ValueError("at least two updates are required for the trajectory check")
    torch.manual_seed(seed)
    actual = RecursiveLM(config).to(device).train()
    reference, pairs = make_reference(actual)
    reference.train()
    ids = torch.tensor([[2, 5, 11, 8, 3, 21, 6], [4, 9, 1, 13, 7, 2, 17]], device=device)
    options = {"lr": 0.003, "betas": (0.9, 0.95), "eps": 1e-8,
               "weight_decay": 0.01, "foreach": False, "fused": False}
    actual_optimizer = torch.optim.AdamW(actual.parameters(), **options)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), **options)
    maxima = {"logits": 0.0, "loss": 0.0, "gradient": 0.0, "updated_parameter": 0.0}
    actual_losses: list[float] = []
    reference_losses: list[float] = []
    worst_gradient = ""
    for _ in range(steps):
        actual_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        output = actual(ids)
        hf_output = reference(input_ids=ids, labels=ids, use_cache=False)
        actual_loss = F.cross_entropy(output.logits[:, :-1].reshape(-1, config.vocab_size),
                                      ids[:, 1:].reshape(-1))
        torch.testing.assert_close(output.logits, hf_output.logits, atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(actual_loss, hf_output.loss, atol=3e-6, rtol=3e-5)
        maxima["logits"] = max(maxima["logits"], max_error(output.logits, hf_output.logits))
        maxima["loss"] = max(maxima["loss"], max_error(actual_loss, hf_output.loss))
        actual_losses.append(float(actual_loss.detach().cpu()))
        reference_losses.append(float(hf_output.loss.detach().cpu()))
        actual_loss.backward()
        hf_output.loss.backward()
        for pair in pairs:
            gradient = pair.actual.grad
            expected = pair.reference_value(gradient=True)
            if gradient is None or expected is None:
                raise AssertionError(f"missing gradient for mapped owner {pair.name}")
            torch.testing.assert_close(gradient, expected, atol=3e-6, rtol=5e-4,
                                       msg=lambda msg: f"{pair.name}: {msg}")
            error = max_error(gradient, expected)
            if error > maxima["gradient"]:
                maxima["gradient"], worst_gradient = error, pair.name
        actual_optimizer.step()
        reference_optimizer.step()
        for pair in pairs:
            expected = pair.reference_value()
            # Adam's 1e-8 epsilon amplifies tiny roundoff in near-zero gradients;
            # compare its bounded trajectory as well as the much tighter logits.
            torch.testing.assert_close(pair.actual, expected, atol=1e-4, rtol=5e-4,
                                       msg=lambda msg: f"{pair.name}: {msg}")
            maxima["updated_parameter"] = max(maxima["updated_parameter"], max_error(pair.actual, expected))
    if not actual_losses[-1] < actual_losses[0] or not reference_losses[-1] < reference_losses[0]:
        raise AssertionError("the bounded fitting check must reduce CE in both implementations")
    return {
        "config": config.to_dict(), "updates": steps, "losses": actual_losses,
        "reference_losses": reference_losses, "max_absolute_errors": maxima,
        "worst_gradient_owner": worst_gradient, "mapped_parameter_tensors": len(pairs),
        "physical_parameter_elements": sum(p.numel() for p in actual.parameters()),
        "reference_physical_parameter_elements": sum(p.numel() for p in reference.parameters()),
        "logical_decoder_calls": reference.config.num_hidden_layers,
        "cache": compare_cache(actual, reference, ids),
    }


def run_suite(*, steps: int = 4, device: str = "cpu", selected: list[str] | None = None) -> dict[str, Any]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        configurations = cases()
        selected = list(configurations) if selected is None else selected
        results = {name: run_case(configurations[name], steps=steps, device=device) for name in selected}
        source = Path(qwen.__file__)
        return {
            "schema_version": 1,
            "reference": "Transformers official Qwen3MoeForCausalLM; unmodified forward; eager attention",
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "python_version": platform.python_version(), "device": device, "dtype": "float32",
            "official_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_url": f"https://github.com/huggingface/transformers/blob/v{transformers.__version__}/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
            "scope": "parameter-mapped numerical/gradient/update check, not paper reproduction or pretraining-quality evidence",
            "auxiliary_losses": "disabled in both objectives; aggregation semantics intentionally tested separately",
            "optimizer": {"name": "AdamW", "lr": 0.003, "betas": [0.9, 0.95],
                          "eps": 1e-8, "weight_decay": 0.01, "foreach": False, "fused": False},
            "tolerances": {
                "logits_and_cached_logits": {"atol": 3e-5, "rtol": 3e-4},
                "loss": {"atol": 3e-6, "rtol": 3e-5},
                "gradient": {"atol": 3e-6, "rtol": 5e-4},
                "updated_parameter": {"atol": 1e-4, "rtol": 5e-4},
            },
            "elapsed_seconds": time.perf_counter() - started,
            "cases": results,
        }
    finally:
        torch.set_num_threads(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--case", action="append", choices=tuple(cases()), dest="selected")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_suite(steps=arguments.steps, device=arguments.device, selected=arguments.selected)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
