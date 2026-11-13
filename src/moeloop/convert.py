"""Published looped-model checkpoints as ``LoopedLM`` weights: one plain function per release.

``convert_ouro``: official Ouro checkpoints, checked against the official
``modeling_ouro.py`` of ByteDance/Ouro-1.4B (revision 574fa66cb8bf) and
ByteDance/Ouro-2.6B (revision 1ed04250da1a, the same code). Ouro is a dense
pre-norm Transformer whose ``num_hidden_layers`` layers form one shared core run
``total_ut_steps`` times, with the topology of ``ouro_spec`` (stack order, no
prefix or suffix, one exit gate on every visit's readout) but dense FFNs. Two
pieces of its structure need model flags:

- ``sandwich_norm``: each layer also normalizes the attention and MLP outputs
  (``input_layernorm_2``, ``post_attention_layernorm_2``) before the residual add;
- ``visit_final_norm``: the shared final norm is applied at the end of every
  step and its output is the next step's input.

The official forward exits at the last step by default (``early_exit_threshold``
1.0); ``LoopedLM`` reads out at ``num_visits``, so a shorter depth gives the
official ``exit_at_step=num_visits - 1`` logits. The decode cache keeps one slot
per (layer, visit), as the official ``UniversalTransformerCache`` does.

``convert_huginn``: the official Huginn checkpoint, checked against the official
``raven_modeling_minimal.py`` of tomg-group-umd/huginn-0125 (revision
bb6621b65e90). Huginn is a dense Transformer of ``SandwichBlock`` layers: a
prelude (our prefix cells) embeds the input once, a recurrent block (our core)
runs ``mean_recurrence`` times by default, and a coda (our suffix cells) reads
out. Every core visit starts with ``adapter(cat(state, prelude output))``, our
``linear`` input injection. Three pieces of its structure need model flags:

- ``residual_norm``: each block normalizes the residual stream after each add,
  ``x = norm_2(x + attn(norm_1(x)))`` and ``x = norm_4(x + mlp(norm_3(x)))``;
- ``qk_bias``: a learned bias on queries and keys before RoPE;
- ``core_final_norm``: the final norm ``ln_f`` is applied to the core output
  before the coda, and again after it.

The rest is a change of layout. Huginn's RoPE rotates interleaved pairs
``(2i, 2i+1)``; ``LoopedLM`` rotates split halves ``(i, i + head_dim/2)`` with the
same frequencies, so the query and key rows (and biases) of every head are
permuted, which leaves every attention score unchanged. The embedding is tied
to the head and scaled by ``sqrt(n_embd)``: the head gets ``wte`` and our
embedding gets ``sqrt(n_embd) * wte``. The stored ``lm_head.weight`` (the tied
copy) and ``freqs_cis`` (recomputed from ``rope_base``) are not used.

The official forward draws a random initial state at every call, also at
inference: a truncated normal with std ``sqrt(2/5)`` (``init_values["std"]`` times
the embedding scale) cut at 3 std. That is ``initial_state="random"`` with the
default ``latent_init_std`` and ``latent_truncation``, but ``LoopedLM`` draws it
only in training and starts evaluation from zero, which is the official forward
with ``init_scale=0``. Logits match that call.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor

from .config import ModelConfig

OURO_LAYER_NAMES = {  # official name in model.layers.{i}. -> ours in core.{i}.
    "self_attn.q_proj.weight": "attentions.0.q_proj.weight",
    "self_attn.k_proj.weight": "attentions.0.k_proj.weight",
    "self_attn.v_proj.weight": "attentions.0.v_proj.weight",
    "self_attn.o_proj.weight": "attentions.0.out_proj.weight",
    "mlp.gate_proj.weight": "dense_ffn.w1.weight",
    "mlp.up_proj.weight": "dense_ffn.w3.weight",
    "mlp.down_proj.weight": "dense_ffn.w2.weight",
    "input_layernorm.weight": "attention_norms.0.weight",
    "input_layernorm_2.weight": "attention_output_norms.0.weight",
    "post_attention_layernorm.weight": "ffn_norms.0.weight",
    "post_attention_layernorm_2.weight": "ffn_output_norms.0.weight",
}
OURO_TOP_NAMES = {
    "model.embed_tokens.weight": "embedding.weight",
    "model.norm.weight": "final_norm.weight",
    "model.early_exit_gate.weight": "exit_gate.weight",
    "model.early_exit_gate.bias": "exit_gate.bias",
    "lm_head.weight": "lm_head.weight",
}


def convert_ouro(state: Mapping[str, Tensor], hf_config: Mapping[str, Any]) -> tuple[ModelConfig, dict[str, Tensor]]:
    """The ``ModelConfig`` and ``LoopedLM`` state dict of an official Ouro checkpoint.

    ``hf_config`` is the release's ``config.json``; ``state`` its safetensors
    tensors, renamed and kept in their dtype. Load the result with
    ``LoopedLM(config).load_state_dict(converted, strict=True)``.
    """
    unsupported = [name for name, bad in (
        ("sliding-window attention", hf_config.get("use_sliding_window") or any(
            kind != "full_attention" for kind in hf_config.get("layer_types") or ())),
        ("rope_scaling", hf_config.get("rope_scaling") is not None),
        ("tied embedding and head", hf_config.get("tie_word_embeddings", False)),
        ("hidden_act other than silu", hf_config.get("hidden_act", "silu") != "silu"),
        ("head_dim other than hidden_size / num_attention_heads", hf_config.get("head_dim") not in (
            None, hf_config["hidden_size"] // hf_config["num_attention_heads"])),
    ) if bad]
    if unsupported:
        raise ValueError(f"LoopedLM cannot express this Ouro config: {', '.join(unsupported)}")
    config = ModelConfig(
        vocab_size=hf_config["vocab_size"], dim=hf_config["hidden_size"],
        hidden_dim=hf_config["intermediate_size"], num_heads=hf_config["num_attention_heads"],
        num_kv_heads=hf_config["num_key_value_heads"], num_experts=0,
        num_cells=hf_config["num_hidden_layers"], num_visits=hf_config["total_ut_steps"],
        rms_eps=hf_config["rms_norm_eps"], rope_theta=float(hf_config["rope_theta"]),
        max_seq_len=hf_config["max_position_embeddings"], init_std=hf_config.get("initializer_range", 0.02),
        exit_distribution=True, sandwich_norm=True, visit_final_norm=True,
    )
    names = dict(OURO_TOP_NAMES)
    for i in range(config.num_cells):
        names |= {f"model.layers.{i}.{theirs}": f"core.{i}.{ours}" for theirs, ours in OURO_LAYER_NAMES.items()}
    unused = sorted(set(state) - set(names))
    if unused:
        raise ValueError(f"the converter does not map these official tensors: {unused}")
    return config, {ours: state[theirs] for theirs, ours in names.items()}


HUGINN_BLOCK_NAMES = {  # official name in transformer.{prelude,core_block,coda}.{i}. -> ours in the cell
    "attn.proj.weight": "attentions.0.out_proj.weight",
    "mlp.proj.weight": "dense_ffn.w2.weight",
    "norm_1.weight": "attention_norms.0.weight",
    "norm_2.weight": "attention_residual_norms.0.weight",
    "norm_3.weight": "ffn_norms.0.weight",
    "norm_4.weight": "ffn_residual_norms.0.weight",
}
HUGINN_REGIONS = {  # official module list -> our region and the config key of its length
    "prelude": ("prefix", "n_layers_in_prelude"),
    "core_block": ("core", "n_layers_in_recurrent_block"),
    "coda": ("suffix", "n_layers_in_coda"),
}


def convert_huginn(state: Mapping[str, Tensor], hf_config: Mapping[str, Any]) -> tuple[ModelConfig, dict[str, Tensor]]:
    """The ``ModelConfig`` and ``LoopedLM`` state dict of an official Huginn checkpoint.

    ``hf_config`` is the release's ``config.json``; ``state`` its safetensors
    tensors, renamed, split, permuted or scaled as the module docstring says,
    and kept in their dtype. Load the result with
    ``LoopedLM(config).load_state_dict(converted, strict=True)``.
    """
    unsupported = [name for name, bad in (
        ("block other than SandwichBlock", hf_config.get("block_class_name") != "SandwichBlock"),
        ("norm other than RMSNorm_llama", hf_config.get("norm_class_name") != "RMSNorm_llama"),
        ("MLP other than GatedMLP with SiLU", (hf_config.get("mlp_class_name"), hf_config.get("nonlin_name")) != ("GatedMLP", "SiLU")),
        ("injection other than linear", hf_config.get("injection_type") != "linear"),
        ("adapter bias", hf_config.get("bias", False)),
        ("untied embedding and head", not hf_config.get("tie_embeddings", True)),
    ) if bad]
    if unsupported:
        raise ValueError(f"LoopedLM cannot express this Huginn config: {', '.join(unsupported)}")
    d, heads = hf_config["n_embd"], hf_config["n_heads"]
    config = ModelConfig(
        vocab_size=hf_config["vocab_size"], dim=d, hidden_dim=hf_config["intermediate_size"],
        num_heads=heads, num_kv_heads=heads, num_experts=0,
        prefix_cells=hf_config["n_layers_in_prelude"], num_cells=hf_config["n_layers_in_recurrent_block"],
        suffix_cells=hf_config["n_layers_in_coda"], num_visits=hf_config["mean_recurrence"],
        rms_eps=hf_config["norm_eps"], rope_theta=float(hf_config["rope_base"]), max_seq_len=hf_config["block_size"],
        init_std=math.sqrt(2 / (5 * d)), input_injection="linear", initial_state="random",
        qk_bias=hf_config.get("qk_bias", False), residual_norm=True, core_final_norm=True,
    )
    head_dim, hidden = d // heads, config.hidden_dim
    split_halves = torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))

    def rope_rows(tensor: Tensor) -> Tensor:  # interleaved -> split-half order within every head
        return tensor.reshape(heads, head_dim, -1)[:, split_halves].reshape(tensor.shape)

    wte = state["transformer.wte.weight"]
    adapter = state["transformer.adapter.weight"]
    converted = {"embedding.weight": wte * math.sqrt(d), "lm_head.weight": wte,
                 "final_norm.weight": state["transformer.ln_f.weight"],
                 "input_adapter.state_proj.weight": adapter[:, :d], "input_adapter.input_proj.weight": adapter[:, d:]}
    used = {"transformer.wte.weight", "transformer.adapter.weight", "transformer.ln_f.weight", "lm_head.weight", "freqs_cis"}
    for theirs_region, (ours_region, length) in HUGINN_REGIONS.items():
        for i in range(hf_config[length]):
            theirs, ours = f"transformer.{theirs_region}.{i}.", f"{ours_region}.{i}."
            converted |= {ours + name: state[theirs + official] for official, name in HUGINN_BLOCK_NAMES.items()}
            q, k, v = state[theirs + "attn.Wqkv.weight"].split(d)
            fc = state[theirs + "mlp.fc.weight"]
            converted |= {ours + "attentions.0.q_proj.weight": rope_rows(q), ours + "attentions.0.k_proj.weight": rope_rows(k),
                          ours + "attentions.0.v_proj.weight": v,
                          ours + "dense_ffn.w1.weight": fc[:hidden], ours + "dense_ffn.w3.weight": fc[hidden:]}
            used |= {theirs + name for name in (*HUGINN_BLOCK_NAMES, "attn.Wqkv.weight", "mlp.fc.weight")}
            if config.qk_bias:
                q_bias, k_bias = state[theirs + "attn.qk_bias"].reshape(2, d)
                converted |= {ours + "attentions.0.q_proj.bias": rope_rows(q_bias),
                              ours + "attentions.0.k_proj.bias": rope_rows(k_bias)}
                used.add(theirs + "attn.qk_bias")
    unused = sorted(set(state) - used)
    if unused:
        raise ValueError(f"the converter does not map these official tensors: {unused}")
    return config, converted
