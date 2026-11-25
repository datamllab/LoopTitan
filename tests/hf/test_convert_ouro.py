"""Ouro checkpoint conversion: logits of the converted LoopedLM against the official forward.

The tiny test needs no network: it builds a random state dict in the official
layout and compares with ``official_forward``, an independent re-implementation of the forward of
``modeling_ouro.py`` (ByteDance/Ouro-1.4B at 574fa66). The real-checkpoint test
runs the official remote code itself; it needs MOELOOP_OURO_DIR (a snapshot of
an official Ouro repo) and a transformers version that loads that code
(the authors recommend 4.54.1), and is skipped otherwise.
"""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop.convert import convert_ouro
from moeloop.model import LoopedLM

HF_CONFIG = {
    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "hidden_act": "silu",
    "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "rope_scaling": None,
    "tie_word_embeddings": False, "use_sliding_window": False, "total_ut_steps": 3,
    "layer_types": ["full_attention"] * 2, "initializer_range": 0.02,
}


def official_state(config, generator):
    """Random weights under the official names; norms and gate are off their initial values."""
    d, f, h = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * h, config["num_key_value_heads"] * h
    shapes = {"model.embed_tokens.weight": (config["vocab_size"], d), "model.norm.weight": (d,),
              "model.early_exit_gate.weight": (1, d), "model.early_exit_gate.bias": (1,),
              "lm_head.weight": (config["vocab_size"], d)}
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        shapes |= {p + "self_attn.q_proj.weight": (q, d), p + "self_attn.k_proj.weight": (kv, d),
                   p + "self_attn.v_proj.weight": (kv, d), p + "self_attn.o_proj.weight": (d, q),
                   p + "mlp.gate_proj.weight": (f, d), p + "mlp.up_proj.weight": (f, d),
                   p + "mlp.down_proj.weight": (d, f)}
        shapes |= {p + norm + ".weight": (d,) for norm in ("input_layernorm", "input_layernorm_2",
                   "post_attention_layernorm", "post_attention_layernorm_2")}
    return {name: (1 if name.endswith("norm.weight") else 0) + 0.2 * torch.randn(shape, generator=generator)
            for name, shape in shapes.items()}


def official_forward(state, config, tokens):
    """``OuroForCausalLM`` without cache or padding: per-step logits and exit-gate logits."""
    eps, h = config["rms_norm_eps"], config["head_dim"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]

    def norm(x, name):
        return state[name] * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    def rotate_half(x):
        return torch.cat((-x[..., h // 2:], x[..., :h // 2]), dim=-1)

    batch, length = tokens.shape
    inv_freq = 1.0 / (config["rope_theta"] ** (torch.arange(0, h, 2).float() / h))
    angles = torch.arange(length).float()[:, None] * inv_freq[None]
    cos, sin = torch.cat((angles, angles), -1).cos(), torch.cat((angles, angles), -1).sin()
    causal = torch.full((length, length), float("-inf")).triu(1)
    x = state["model.embed_tokens.weight"][tokens]
    step_logits, gate_logits = [], []
    for _ in range(config["total_ut_steps"]):
        for i in range(config["num_hidden_layers"]):
            p = f"model.layers.{i}."
            a = norm(x, p + "input_layernorm.weight")
            q, k, v = ((a @ state[p + f"self_attn.{n}_proj.weight"].T).view(batch, length, -1, h).transpose(1, 2)
                       for n in "qkv")
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            k, v = (t.repeat_interleave(heads // kv_heads, dim=1) for t in (k, v))
            weights = torch.softmax(q @ k.transpose(2, 3) * h ** -0.5 + causal, dim=-1)
            mixed = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
            x = x + norm(mixed @ state[p + "self_attn.o_proj.weight"].T, p + "input_layernorm_2.weight")
            m = norm(x, p + "post_attention_layernorm.weight")
            m = (torch.nn.functional.silu(m @ state[p + "mlp.gate_proj.weight"].T)
                 * (m @ state[p + "mlp.up_proj.weight"].T)) @ state[p + "mlp.down_proj.weight"].T
            x = x + norm(m, p + "post_attention_layernorm_2.weight")
        x = norm(x, "model.norm.weight")
        step_logits.append(x @ state["lm_head.weight"].T)
        gate_logits.append((x @ state["model.early_exit_gate.weight"].T + state["model.early_exit_gate.bias"]).squeeze(-1))
    return step_logits, torch.stack(gate_logits)


def relative(ours, theirs):
    return ((ours - theirs).abs().max() / theirs.abs().max()).item()


def test_tiny_official_layout_converts_to_the_same_logits_at_every_depth():
    generator = torch.Generator().manual_seed(0)
    state = official_state(HF_CONFIG, generator)
    tokens = torch.randint(0, HF_CONFIG["vocab_size"], (2, 11), generator=generator)
    expected, gates = official_forward(state, HF_CONFIG, tokens)
    config, converted = convert_ouro(state, HF_CONFIG)
    assert (config.num_cells, config.num_visits, config.num_experts, config.prefix_cells, config.suffix_cells) == (2, 3, 0, 0, 0)
    assert config.sandwich_norm and config.visit_final_norm and config.exit_distribution
    model = LoopedLM(config).eval()
    model.load_state_dict(converted, strict=True)
    with torch.no_grad():
        for depth, reference in enumerate(expected, start=1):
            assert relative(model(tokens, num_visits=depth).logits, reference) < 1e-5
        out = model(tokens, collect_readouts=True)
    assert relative(out.hazard_logits, gates) < 1e-5
    for readout, reference in zip(out.visit_hidden_states, expected):
        assert relative(model.lm_head(readout), reference) < 1e-5


def test_cached_decode_of_a_converted_model_matches_recompute():
    from moeloop.cache import DecodeCache
    generator = torch.Generator().manual_seed(1)
    config, converted = convert_ouro(official_state(HF_CONFIG, generator), HF_CONFIG)
    model = LoopedLM(config).eval()
    model.load_state_dict(converted, strict=True)
    tokens = torch.randint(0, HF_CONFIG["vocab_size"], (1, 9), generator=generator)
    with torch.no_grad():
        full = model(tokens).logits
        cache = DecodeCache(config.num_visits)
        steps = [model(tokens[:, :5], cache=cache).logits] + [
            model(tokens[:, i:i + 1], cache=cache).logits for i in range(5, 9)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-5)


def test_an_official_tensor_the_converter_does_not_map_is_refused():
    state = official_state(HF_CONFIG, torch.Generator().manual_seed(2))
    with pytest.raises(ValueError, match=r"model.layers.0.self_attn.q_proj.bias"):
        convert_ouro(state | {"model.layers.0.self_attn.q_proj.bias": torch.zeros(32)}, HF_CONFIG)


def test_structure_the_model_cannot_express_is_rejected():
    for change in ({"use_sliding_window": True}, {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
                   {"tie_word_embeddings": True}, {"head_dim": 16}):
        with pytest.raises(ValueError, match="cannot express"):
            convert_ouro({}, HF_CONFIG | change)


@pytest.mark.skipif(not os.environ.get("MOELOOP_OURO_DIR"), reason="set MOELOOP_OURO_DIR to an official Ouro snapshot")
def test_real_checkpoint_matches_the_official_implementation():
    from safetensors.torch import load_file
    transformers = pytest.importorskip("transformers")
    path = Path(os.environ["MOELOOP_OURO_DIR"])
    try:
        official = transformers.AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
    except Exception as error:  # the remote code targets transformers < 4.56
        pytest.skip(f"the official Ouro code does not load here: {error!r}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    hf_config = json.loads((path / "config.json").read_text())
    state = {name: tensor for file in sorted(path.glob("*.safetensors")) for name, tensor in load_file(file).items()}
    config, converted = convert_ouro(state, hf_config)
    model = LoopedLM(config).eval()
    model.load_state_dict(converted, strict=True)
    depth = hf_config["total_ut_steps"]
    for prompt in ("The capital of France is", "def fibonacci(n):", "1 + 1 = 2, 2 + 2 ="):
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            for step in range(depth):
                theirs = official(tokens, use_cache=False, exit_at_step=step).logits
                ours = model(tokens, num_visits=step + 1).logits
                assert relative(ours, theirs) < 1e-5, (prompt, step + 1)
