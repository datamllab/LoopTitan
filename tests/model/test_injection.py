"""Diagonal-decay and orthogonal input injection: equations, costs, HF and schema contracts.

Oracles follow arXiv 2610.06833 (Part II of xllm-loop): Appendix B.4 equation (24)
for the diagonal decay and Section 4.2 equation (8) for OrthoInj.
"""

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from moeloop.cache import DecodeCache
from moeloop.costs import estimate_cost
from moeloop.model import InputAdapter, ModelConfig, RecursiveLM

NEW_KINDS = ("diag_decay", "orthogonal")
FIXTURES = Path(__file__).parents[1] / "delivery" / "fixtures"


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    yield
    torch.set_num_threads(previous)


def tiny(**changes):
    return ModelConfig(**(dict(vocab_size=29, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
        num_experts=4, top_k=2, num_cells=2, num_visits=3, prefix_cells=1, suffix_cells=1,
        max_seq_len=16, initial_state="zero", init_std=.2) | changes))


def hand_adapter(kind):
    """d=2: dt_bias=0 gives dt=softplus(0)=ln 2; A_log=(0, ln 2) gives a=(1, 2), decay=(1/2, 1/4)."""
    adapter = InputAdapter(tiny(dim=2, num_heads=1, num_kv_heads=1, input_injection=kind))
    with torch.no_grad():
        adapter.input_proj.weight.copy_(torch.tensor([[0., 1.], [1., 0.]]))
        adapter.A_log.copy_(torch.tensor([0., math.log(2)]))
        adapter.dt_bias.zero_()
    return adapter


HAND_STATE, HAND_ANCHOR = torch.tensor([[[4., 4.]]]), torch.tensor([[[1., 3.]]])
LN2 = math.log(2)


def test_diag_decay_closed_form_on_hand_state():
    # exp(-dt a) h = (4/2, 4/4) = (2, 1); W e = (3, 1); dt W e = ln 2 (3, 1).
    expected = torch.tensor([[[2. + 3 * LN2, 1. + LN2]]])
    torch.testing.assert_close(hand_adapter("diag_decay")(HAND_STATE, HAND_ANCHOR), expected)


def test_orthogonal_closed_form_on_hand_state():
    adapter = hand_adapter("orthogonal")
    # q = ln 2 (3, 1); Q_q (2, 1) = (2, 1) - <(2, 1), q> / (|q|^2 + eps) q, with <(2, 1), q> = 7 ln 2.
    coefficient = 7 * LN2 / (10 * LN2 ** 2 + adapter.eps)
    expected = torch.tensor([[[2. - coefficient * 3 * LN2 + 3 * LN2, 1. - coefficient * LN2 + LN2]]])
    torch.testing.assert_close(adapter(HAND_STATE, HAND_ANCHOR), expected)
    # eps -> 0: the carryover (2, 1) - 0.7 (3, 1) = (-0.1, 0.3) is orthogonal to (3, 1).
    assert adapter.eps == 1e-6
    torch.testing.assert_close(expected, torch.tensor([[[-.1 + 3 * LN2, .3 + LN2]]]), rtol=0, atol=2e-6)


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_initialization_matches_paper_retention(kind):
    model = RecursiveLM(tiny(input_injection=kind))
    adapter = model.input_adapter
    torch.testing.assert_close(adapter.input_proj.weight, torch.eye(16))
    dt = torch.nn.functional.softplus(adapter.dt_bias)
    torch.testing.assert_close(dt, torch.full((16,), .5 * math.log(5)))
    torch.testing.assert_close(adapter.A_log.exp(), torch.ones(16))
    # Appendix B.4: Delta_init = log(5)/2 and alpha_init = 1 give retention 1/sqrt(5).
    torch.testing.assert_close(torch.exp(-dt * adapter.A_log.exp()), torch.full((16,), 5 ** -.5))
    custom = RecursiveLM(tiny(input_injection=kind, injection_dt=.3, injection_decay=2.5)).input_adapter
    torch.testing.assert_close(torch.nn.functional.softplus(custom.dt_bias), torch.full((16,), .3))
    torch.testing.assert_close(custom.A_log.exp(), torch.full((16,), 2.5))


def test_orthogonal_carryover_is_orthogonal_before_and_after_optimizer_steps():
    """Q_q is recomputed from q every call, so orthogonality survives any update of W, a, dt.

    Equation (8) is exact up to eps: <Q_q v, q> = <v, q> eps / (|q|^2 + eps).
    """
    adapter = InputAdapter(tiny(input_injection="orthogonal")).double()
    with torch.no_grad():
        adapter.input_proj.weight.copy_(torch.eye(16) + .3 * torch.randn(16, 16))
        adapter.A_log.normal_(0, .5)
        adapter.dt_bias.normal_(0, .5)
    state, anchor = torch.randn(3, 5, 16, dtype=torch.float64), torch.randn(3, 5, 16, dtype=torch.float64)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.1, weight_decay=.1)
    before = [p.detach().clone() for p in adapter.parameters()]
    for _ in range(3):
        dt = torch.nn.functional.softplus(adapter.dt_bias)
        retained = torch.exp(-dt * adapter.A_log.exp()) * state
        q = dt * adapter.input_proj(anchor)
        carry = adapter(state, anchor) - q
        q2 = q.square().sum(-1)
        along = (carry * q).sum(-1)
        torch.testing.assert_close(along, (retained * q).sum(-1) * adapter.eps / (q2 + adapter.eps), rtol=1e-6, atol=1e-12)
        assert (along.abs() / (carry.norm(dim=-1) * q2.sqrt())).max() < 10 * adapter.eps / q2.min()
        # The carryover equals the explicit projector Q_q = I - q q^T / (|q|^2 + eps).
        projector = torch.eye(16, dtype=q.dtype) - q.unsqueeze(-1) * q.unsqueeze(-2) / (
            q.square().sum(-1)[..., None, None] + adapter.eps)
        torch.testing.assert_close(carry, (projector @ retained.unsqueeze(-1)).squeeze(-1))
        optimizer.zero_grad()
        adapter(state, anchor).square().sum().backward()
        optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, adapter.parameters()))


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_bfloat16_keeps_state_dtype_and_tracks_float32(kind):
    adapter = InputAdapter(tiny(input_injection=kind))
    with torch.no_grad():
        adapter.input_proj.weight.normal_(0, .3)
        adapter.A_log.normal_(0, .3)
        adapter.dt_bias.normal_(0, .3)
    state, anchor = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    reference = adapter(state, anchor)
    low = adapter.to(torch.bfloat16)(state.bfloat16(), anchor.bfloat16())
    assert low.dtype == torch.bfloat16
    torch.testing.assert_close(low.float(), reference, rtol=.02, atol=.02)


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_adapter_gradients_match_finite_differences(kind):
    adapter = InputAdapter(tiny(dim=4, num_heads=1, num_kv_heads=1, input_injection=kind)).double()
    with torch.no_grad():
        adapter.input_proj.weight.normal_()
        adapter.A_log.normal_(0, .5)
        adapter.dt_bias.normal_(0, .5)
    state = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    anchor = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(adapter, (state, anchor))
    names = dict(adapter.named_parameters())
    for name in names:
        def through(value, name=name):
            return torch.func.functional_call(adapter, {**names, name: value}, (state.detach(), anchor.detach()))
        assert torch.autograd.gradcheck(through, (names[name].detach().clone().requires_grad_(),))


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_model_forward_matches_explicit_recurrence_and_gradients_reach_every_owner(kind):
    model = RecursiveLM(tiny(input_injection=kind, initial_state="random"))
    with torch.no_grad():
        model.input_adapter.input_proj.weight.add_(.1 * torch.randn(16, 16))
        model.input_adapter.A_log.normal_(0, .3)
    ids = torch.tensor([[2, 5, 7, 9], [3, 1, 4, 1]])
    positions = torch.arange(4).expand_as(ids)
    rng = torch.get_rng_state()
    output = model(ids)
    torch.set_rng_state(rng)

    def cell_update(cell, x):
        x = x + cell.attentions[0](cell.attention_norms[0](x), positions=positions)
        normalized = cell.ffn_norms[0](x)
        routing = cell.routers[0](normalized, torch.ones_like(ids, dtype=torch.bool))
        return x + cell.experts(normalized, routing.ids, routing.weights, routing.dispatch_counts)

    adapter, cfg = model.input_adapter, model.config
    anchor = cell_update(model.prefix[0], model.embedding(ids))
    state = torch.nn.init.trunc_normal_(torch.empty_like(anchor), std=cfg.latent_init_std,
        a=-cfg.latent_truncation * cfg.latent_init_std, b=cfg.latent_truncation * cfg.latent_init_std)
    dt = torch.nn.functional.softplus(adapter.dt_bias)
    q = dt * (anchor @ adapter.input_proj.weight.T)
    for _ in range(cfg.num_visits):
        state = torch.exp(-dt * adapter.A_log.exp()) * state
        if kind == "orthogonal":
            state = state - (state * q).sum(-1, keepdim=True) / (q.square().sum(-1, keepdim=True) + cfg.injection_eps) * q
        state = state + q
        for cell in model.core:
            state = cell_update(cell, state)
    state = cell_update(model.suffix[0], state)
    expected = model.lm_head(model.final_norm(state))
    torch.testing.assert_close(output.logits, expected, rtol=2e-5, atol=2e-6)
    model.zero_grad()
    model(ids, grad_visits=1).logits.square().sum().backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("input_adapter", "embedding", "prefix")):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_cached_decode_matches_recomputation(kind):
    model = RecursiveLM(tiny(input_injection=kind)).eval()
    ids = torch.tensor([[2, 5, 7, 9, 4], [3, 1, 4, 1, 5]])
    with torch.no_grad():
        full = model(ids).logits
        cache = DecodeCache(model.config.num_visits)
        prefix = model(ids[:, :3], cache=cache).logits
        steps = [model(ids[:, i:i + 1], cache=cache).logits for i in (3, 4)]
    torch.testing.assert_close(torch.cat([prefix, *steps], dim=1), full, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_cost_model_counts_actual_parameters_and_projection_flops(kind):
    config = tiny(input_injection=kind)
    model = RecursiveLM(config)
    cost = estimate_cost(config, seq_len=8)
    base = estimate_cost(replace(config, input_injection="none"), seq_len=8)
    d = config.dim
    assert cost["parameters"]["unique_total"] == sum(p.numel() for p in model.parameters())
    assert cost["parameters"]["by_component"]["input_adapter"] == sum(
        p.numel() for p in model.input_adapter.parameters()) == d * d + 2 * d
    macs = []
    model.input_adapter.input_proj.register_forward_hook(
        lambda module, args, out: macs.append(args[0].numel() // d * module.weight.numel()))
    ids = torch.randint(0, config.vocab_size, (2, 8))
    model(ids)
    per_token = 2 * sum(macs) / ids.numel()
    assert cost["flops"]["forward_components"]["input_adapter"] == per_token == config.num_visits * 2 * d * d
    assert cost["flops"]["forward_per_token"] - base["flops"]["forward_per_token"] == per_token
    assert cost["formula_version"] == base["formula_version"] == "fixed_terminal_matrix_v1"
    assert "input_injection" in cost["unknown"] and "input_injection" not in base["unknown"]
    replayed = estimate_cost(config, seq_len=8, activation_checkpointing="full_cells")
    assert replayed["flops"]["recompute_components"]["input_adapter"] == 0


@pytest.mark.parametrize("kind,parameters,adapter_flops,forward,training", [
    ("none", 22832, 0, 55200, 219872),
    ("add", 22832, 0, 55200, 219872),
    ("linear", 23344, 3072, 58272, 229088),
])
def test_existing_kinds_keep_their_cost_numbers(kind, parameters, adapter_flops, forward, training):
    """Literal numbers produced by the formula before schema 4 (fixed_terminal_matrix_v1)."""
    cost = estimate_cost(tiny(input_injection=kind), seq_len=8, activation_checkpointing="full_cells")
    assert cost["formula_version"] == "fixed_terminal_matrix_v1"
    assert cost["parameters"]["unique_total"] == parameters
    assert cost["flops"]["forward_components"]["input_adapter"] == adapter_flops
    assert cost["flops"]["forward_per_token"] == forward
    assert cost["flops"]["training_per_token"] == training
    assert "input_injection" not in cost["unknown"]


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_hf_export_round_trip_has_identical_weights_logits_and_cached_generation(kind, tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from moeloop.export import export_checkpoint

    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3} | {f"w{i}": i for i in range(4, 29)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>",
                                         eos_token="<eos>", unk_token="<unk>").save_pretrained(tmp_path / "tokenizer")
    config = tiny(input_injection=kind, injection_dt=.6, injection_decay=1.7, injection_eps=1e-5)
    model = RecursiveLM(config).eval()
    with torch.no_grad():
        model.input_adapter.input_proj.weight.add_(.1 * torch.randn(16, 16))
        model.input_adapter.A_log.normal_(0, .3)
        model.input_adapter.dt_bias.normal_(0, .3)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    (tmp_path / "model_config.json").write_text(json.dumps(config.to_dict()))
    output = export_checkpoint(tmp_path / "weights.pt", tmp_path / "artifact", tokenizer_path=tmp_path / "tokenizer")
    saved = json.loads((output / "config.json").read_text())["model_config"]
    assert saved["schema_version"] == 4 and saved["input_injection"] == kind
    assert (saved["injection_dt"], saved["injection_decay"], saved["injection_eps"]) == (.6, 1.7, 1e-5)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True, local_files_only=True).eval()
    assert ModelConfig.from_dict(loaded.config.model_config) == config
    original = model.state_dict()
    assert loaded.model.state_dict().keys() == original.keys()
    assert all(torch.equal(value, original[name]) for name, value in loaded.model.state_dict().items())
    rebuilt = RecursiveLM(config).eval()
    rebuilt.load_state_dict(loaded.model.state_dict())
    ids = torch.tensor([[0, 4, 5, 6], [7, 8, 9, 10]])
    mask = ids != 0
    with torch.no_grad():
        expected = model(ids, attention_mask=mask, router_mask=mask).logits
        assert torch.equal(rebuilt(ids, attention_mask=mask, router_mask=mask).logits, expected)
        # from_pretrained leaves the weights at the safetensors header's byte offset, and CPU
        # expert kernels then round differently (seen in the prefix, before any injection).
        torch.testing.assert_close(loaded(ids, attention_mask=mask, use_cache=False).logits, expected,
                                   rtol=0, atol=5e-6)
    kwargs = dict(attention_mask=mask, max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=0)
    assert torch.equal(loaded.generate(ids, use_cache=True, **kwargs), loaded.generate(ids, use_cache=False, **kwargs))


@pytest.mark.parametrize("schema", [1, 2, 3])
@pytest.mark.parametrize("kind", ["none", "add", "linear"])
def test_old_schemas_load_with_old_semantics(schema, kind):
    old = dict(vocab_size=29, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2, num_experts=4, top_k=2,
               num_cells=2, num_visits=3, prefix_cells=1, suffix_cells=1, max_seq_len=16,
               initial_state="zero", input_injection=kind, injection_scale=.7, schema_version=schema)
    config = ModelConfig.from_dict(old)
    assert config.schema_version == 4
    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    model = RecursiveLM(config)
    keys = {"none": set(), "add": set(),
            "linear": {"input_adapter.state_proj.weight", "input_adapter.input_proj.weight"}}[kind]
    assert {key for key in model.state_dict() if key.startswith("input_adapter")} == keys
    if kind != "none":
        adapter = model.input_adapter
        state, anchor = torch.randn(1, 3, 16), torch.randn(1, 3, 16)
        expected = state + .7 * anchor if kind == "add" else (
            state @ adapter.state_proj.weight.T + .7 * anchor @ adapter.input_proj.weight.T)
        torch.testing.assert_close(adapter(state, anchor), expected)


@pytest.mark.parametrize("fixture", ["p1-reference-hf-config.json", "p1-candidate-hf-config.json"])
def test_released_schema2_hf_configs_still_load(fixture):
    from moeloop.hf import MoeloopConfig

    raw = json.loads((FIXTURES / fixture).read_text())
    config = MoeloopConfig(**{k: v for k, v in raw.items() if k not in ("architectures", "auto_map")}).to_model_config()
    assert config.schema_version == 4 and config.input_injection == "none"
    assert {key: getattr(config, key) for key in raw["model_config"] if key not in ("schema_version", "core")} == {
        key: value for key, value in raw["model_config"].items() if key not in ("schema_version", "core")}


@pytest.mark.parametrize("changes,message", [
    (dict(input_injection="diag_decay", schema_version=3), "schema version 3"),
    (dict(input_injection="orthogonal", schema_version=2), "schema version 2"),
    (dict(input_injection="diag_decay", injection_scale=.5), "injection_scale"),
    (dict(input_injection="orthogonal", injection_dt=0.), "injection_dt"),
    (dict(input_injection="orthogonal", injection_decay=-1.), "injection_decay"),
    (dict(input_injection="orthogonal", injection_eps=0.), "injection_eps"),
    (dict(input_injection="rotation"), "input_injection"),
    (dict(input_injection="orthogonal", order="layer"), "stack order"),
])
def test_invalid_injection_configs_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        tiny(**changes)
