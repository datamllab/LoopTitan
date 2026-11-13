"""Huginn checkpoint conversion: logits of the converted LoopedLM against the official forward.

The tiny test needs no network: it builds a random state dict in the official
layout and compares with ``official_forward``, a plain transcription of
``RavenForCausalLM.forward`` in ``raven_modeling_minimal.py``
(tomg-group-umd/huginn-0125 at bb6621b) with ``init_scale=0``, the zero initial
state. The real-checkpoint test runs the official remote code itself; it needs
MOELOOP_HUGINN_DIR (a snapshot of the official repo) and a transformers version
that loads that code, and is skipped otherwise.
"""

import json
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop.convert import convert_huginn
from moeloop.model import LoopedLM

HF_CONFIG = {
    "n_embd": 32, "n_heads": 4, "head_dim": 8, "num_key_value_heads": 4, "intermediate_size": 48,
    "vocab_size": 64, "padded_vocab_size": 64, "block_size": 128, "norm_eps": 1e-6, "rope_base": 50000,
    "n_layers_in_prelude": 1, "n_layers_in_recurrent_block": 2, "n_layers_in_coda": 1, "n_layers": 4,
    "mean_recurrence": 3, "qk_bias": True, "bias": False, "tie_embeddings": True, "injection_type": "linear",
    "block_class_name": "SandwichBlock", "norm_class_name": "RMSNorm_llama", "mlp_class_name": "GatedMLP",
    "nonlin_name": "SiLU", "state_init": "like-init",
}
BLOCKS = {"prelude": "n_layers_in_prelude", "core_block": "n_layers_in_recurrent_block", "coda": "n_layers_in_coda"}


def official_state(config, generator):
    """Random weights under the official names, with the tied head copy and the RoPE buffer of a release."""
    d, f, h = config["n_embd"], config["intermediate_size"], config["n_heads"]
    shapes = {"transformer.wte.weight": (config["vocab_size"], d), "transformer.adapter.weight": (d, 2 * d),
              "transformer.ln_f.weight": (d,)}
    for block, length in BLOCKS.items():
        for i in range(config[length]):
            p = f"transformer.{block}.{i}."
            shapes |= {p + "attn.Wqkv.weight": (3 * d, d), p + "attn.qk_bias": (2, 1, h, d // h),
                       p + "attn.proj.weight": (d, d), p + "mlp.fc.weight": (2 * f, d), p + "mlp.proj.weight": (d, f)}
            shapes |= {p + f"norm_{n}.weight": (d,) for n in range(1, 5)}
    state = {name: (1 if "norm_" in name or "ln_f" in name else 0)
             + 0.2 * torch.randn(shape, generator=generator) for name, shape in shapes.items()}
    return state | {"lm_head.weight": state["transformer.wte.weight"].clone(),
                    "freqs_cis": torch.randn(1, config["block_size"], 1, d // h // 2, 2, generator=generator)}


def official_forward(state, config, tokens, num_steps):
    """``RavenForCausalLM.forward(tokens, num_steps=num_steps, init_scale=0)`` without cache or mask."""
    d, heads, eps = config["n_embd"], config["n_heads"], config["norm_eps"]
    h, batch, length = d // heads, *tokens.shape

    def norm(x, name):
        return state[name] * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    inv_freq = 1.0 / (config["rope_base"] ** (torch.arange(0, h, 2).float() / h))
    angles = torch.arange(length).float()[:, None] * inv_freq[None]
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]

    def rope(x):  # apply_rotary_emb_complex_like: interleaved pairs (2i, 2i+1)
        pairs = x.unflatten(-1, (-1, 2))
        return torch.stack((pairs[..., 0] * cos - pairs[..., 1] * sin,
                            pairs[..., 1] * cos + pairs[..., 0] * sin), -1).flatten(3)

    causal = torch.full((length, length), float("-inf")).triu(1)

    def block(x, p):  # SandwichBlock
        q, k, v = (norm(x, p + "norm_1.weight") @ state[p + "attn.Wqkv.weight"].T).split(d, dim=-1)
        q, k, v = (t.view(batch, length, heads, h) for t in (q, k, v))
        if config["qk_bias"]:
            q, k = q + state[p + "attn.qk_bias"][0], k + state[p + "attn.qk_bias"][1]
        q, k = rope(q), rope(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        weights = torch.softmax(q @ k.transpose(2, 3) * h ** -0.5 + causal, dim=-1)
        y = (weights @ v).transpose(1, 2).reshape(batch, length, d) @ state[p + "attn.proj.weight"].T
        x = norm(y + x, p + "norm_2.weight")
        gate, up = (norm(x, p + "norm_3.weight") @ state[p + "mlp.fc.weight"].T).chunk(2, dim=-1)
        m = (torch.nn.functional.silu(gate) * up) @ state[p + "mlp.proj.weight"].T
        return norm(m + x, p + "norm_4.weight")

    embeds = state["transformer.wte.weight"][tokens] * math.sqrt(d)
    for i in range(config["n_layers_in_prelude"]):
        embeds = block(embeds, f"transformer.prelude.{i}.")
    x = torch.zeros_like(embeds)  # initialize_state with init_scale=0
    for _ in range(num_steps):
        x = torch.cat([x, embeds], dim=-1) @ state["transformer.adapter.weight"].T
        for i in range(config["n_layers_in_recurrent_block"]):
            x = block(x, f"transformer.core_block.{i}.")
    x = norm(x, "transformer.ln_f.weight")
    for i in range(config["n_layers_in_coda"]):
        x = block(x, f"transformer.coda.{i}.")
    return norm(x, "transformer.ln_f.weight") @ state["transformer.wte.weight"].T


def relative(ours, theirs):
    return ((ours - theirs).abs().max() / theirs.abs().max()).item()


def converted_model(state, hf_config=HF_CONFIG):
    config, converted = convert_huginn(state, hf_config)
    model = LoopedLM(config).eval()
    model.load_state_dict(converted, strict=True)
    return model


def test_tiny_official_layout_converts_to_the_same_logits_at_every_depth():
    generator = torch.Generator().manual_seed(0)
    state = official_state(HF_CONFIG, generator)
    tokens = torch.randint(0, HF_CONFIG["vocab_size"], (2, 11), generator=generator)
    model = converted_model(state)
    config = model.config
    assert (config.prefix_cells, config.num_cells, config.suffix_cells, config.num_visits, config.num_experts) == (1, 2, 1, 3, 0)
    assert config.qk_bias and config.residual_norm and config.core_final_norm and config.input_injection == "linear"
    with torch.no_grad():
        for depth in (1, 2, 3, 6):
            assert relative(model(tokens, num_visits=depth).logits, official_forward(state, HF_CONFIG, tokens, depth)) < 1e-5


def test_without_qk_bias_the_official_layout_has_no_bias_tensor():
    generator = torch.Generator().manual_seed(1)
    hf_config = HF_CONFIG | {"qk_bias": False}
    state = {name: tensor for name, tensor in official_state(hf_config, generator).items() if not name.endswith("qk_bias")}
    tokens = torch.randint(0, HF_CONFIG["vocab_size"], (1, 7), generator=generator)
    model = converted_model(state, hf_config)
    assert not model.config.qk_bias
    with torch.no_grad():
        assert relative(model(tokens).logits, official_forward(state, hf_config, tokens, 3)) < 1e-5


def test_cached_decode_of_a_converted_model_matches_recompute():
    from moeloop.cache import DecodeCache
    generator = torch.Generator().manual_seed(2)
    model = converted_model(official_state(HF_CONFIG, generator))
    tokens = torch.randint(0, HF_CONFIG["vocab_size"], (1, 9), generator=generator)
    with torch.no_grad():
        full = model(tokens).logits
        cache = DecodeCache(model.config.num_visits)
        steps = [model(tokens[:, :5], cache=cache).logits] + [
            model(tokens[:, i:i + 1], cache=cache).logits for i in range(5, 9)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-5)


def test_an_official_tensor_the_converter_does_not_map_is_refused():
    state = official_state(HF_CONFIG, torch.Generator().manual_seed(3))
    with pytest.raises(ValueError, match=r"transformer.adapter.bias"):
        convert_huginn(state | {"transformer.adapter.bias": torch.zeros(32)}, HF_CONFIG)


def test_structure_the_model_cannot_express_is_rejected():
    for change in ({"injection_type": "add"}, {"bias": True}, {"tie_embeddings": False},
                   {"block_class_name": "TransformerPreNormBlock"}, {"nonlin_name": "GELU"}):
        with pytest.raises(ValueError, match="cannot express"):
            convert_huginn({}, HF_CONFIG | change)


@pytest.mark.skipif(not os.environ.get("MOELOOP_HUGINN_DIR"), reason="set MOELOOP_HUGINN_DIR to an official Huginn snapshot")
def test_real_checkpoint_matches_the_official_implementation():
    from safetensors.torch import load_file
    transformers = pytest.importorskip("transformers")
    path = Path(os.environ["MOELOOP_HUGINN_DIR"])
    try:
        official = transformers.AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float32).eval()
    except Exception as error:  # the remote code was written for transformers 4.44-4.47
        pytest.skip(f"the official Huginn code does not load here: {error!r}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    hf_config = json.loads((path / "config.json").read_text())
    state = {name: tensor for file in sorted(path.glob("*.safetensors")) for name, tensor in load_file(file).items()}
    model = converted_model(state, hf_config)
    del state
    for prompt in ("The capital of France is", "def fibonacci(n):"):
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            for depth in (1, 4, 32):
                theirs = official(tokens, num_steps=depth, init_scale=0.0).logits
                ours = model(tokens, num_visits=depth).logits
                assert relative(ours, theirs) < 1e-5, (prompt, depth)
