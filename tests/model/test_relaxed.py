"""Relaxed Recursive Transformers: per-visit LoRA on shared weights (arXiv 2410.20672)."""

from dataclasses import replace
import json
from tempfile import TemporaryDirectory

import pytest

torch = pytest.importorskip("torch")

from torch.utils.flop_counter import FlopCounterMode

from moeloop import recipes, testing
from moeloop.cache import DecodeCache
from moeloop.costs import estimate_cost
from moeloop.model import Attention, LoRA, ModelConfig, RecursiveLM
from moeloop.recipe_spec import RecipeSpec
from moeloop.config import RegionConfig, TrainingRecipe


SMALL = dict(vocab_size=31, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
             num_experts=4, top_k=2, max_seq_len=32)
KINDS = {
    "dense": dict(num_experts=0, lora_attention_rank=3, lora_ffn_rank=2),
    "moe": dict(lora_attention_rank=3),
    "moe_always_on": dict(always_on_hidden_dim=8, lora_attention_rank=3, lora_ffn_rank=2),
    "ffn_only": dict(num_experts=0, lora_ffn_rank=2),
    # Only the core has a dense or always-on FFN; the prefix and suffix are routed experts.
    "always_on_core": dict(prefix_cells=1, suffix_cells=1, core=RegionConfig(always_on_hidden_dim=8),
                           lora_attention_rank=3, lora_ffn_rank=2),
    "dense_core": dict(prefix_cells=1, suffix_cells=1, core=RegionConfig(num_experts=0),
                       lora_attention_rank=3, lora_ffn_rank=2),
}


def relaxed(kind="dense", **topology):
    return ModelConfig(**SMALL | dict(num_cells=2, num_visits=3) | KINDS[kind] | topology)


def plain(config):
    """The shared-weight (strict Recursive Transformer) counterpart."""
    return replace(config, lora_attention_rank=0, lora_ffn_rank=0, visit_capacity=None)


def randomize_lora(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRA):
                module.B.normal_(std=0.3, generator=generator)
    return model


def ids(batch=2, tokens=7, seed=2):
    return torch.randint(SMALL["vocab_size"], (batch, tokens), generator=torch.Generator().manual_seed(seed))


def merged(linear, lora):
    """Hand-written W' + B A of one visit."""
    return linear.weight if lora is None else linear.weight + lora.B @ lora.A


def unrolled(model):
    """Independent cells holding W' + B_v A_v: the paper's eq. 3 without sharing or LoRA modules."""
    config = model.config
    cells, visits = config.num_cells, config.num_visits
    flat = RecursiveLM(replace(plain(config), num_cells=cells * visits, num_visits=1, order="stack"))
    with torch.no_grad():
        for name in ("embedding", "prefix", "suffix", "final_norm", "lm_head"):
            getattr(flat, name).load_state_dict(getattr(model, name).state_dict())
        for index, cell in enumerate(model.core):
            for visit in range(visits):
                target = flat.core[visit * cells + index if config.order == "stack" else index * visits + visit]
                shared = {key: value for key, value in cell.state_dict().items() if "loras" not in key}
                target.load_state_dict(shared)
                attention = None if cell.attention_loras is None else cell.attention_loras[visit]
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    getattr(target.attentions[0], name).weight.copy_(
                        merged(getattr(cell.attentions[0], name), None if attention is None else attention[name]))
                ffn = cell.dense_ffn if cell.experts is None else cell.always_on_ffn
                if ffn is not None:
                    lora = None if cell.ffn_loras is None else cell.ffn_loras[visit]
                    target_ffn = target.dense_ffn if cell.experts is None else target.always_on_ffn
                    for name in ("w1", "w3", "w2"):
                        getattr(target_ffn, name).weight.copy_(
                            merged(getattr(ffn, name), None if lora is None else lora[name]))
    return flat


@pytest.mark.parametrize("order", ["stack", "layer"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_lora_forward_equals_the_unrolled_merged_weights(kind, order):
    torch.manual_seed(0)
    model = randomize_lora(RecursiveLM(relaxed(kind, order=order))).eval()
    reference = unrolled(model).eval()
    tokens = ids()
    with torch.no_grad():
        actual, expected = model(tokens).logits, reference(tokens).logits
        torch.testing.assert_close(actual, expected)
        # The LoRA is not a no-op here: without it the logits differ.
        shared = RecursiveLM(plain(model.config)).eval()
        shared.load_state_dict({k: v for k, v in model.state_dict().items() if "loras" not in k})
        assert (shared(tokens).logits - actual).abs().max() > 1e-3


@pytest.mark.parametrize("kind", list(KINDS))
def test_zero_init_is_exactly_the_shared_weight_model(kind):
    config = relaxed(kind)
    torch.manual_seed(0)
    model = RecursiveLM(config)
    torch.manual_seed(0)
    shared = RecursiveLM(plain(config))
    loras = [module for module in model.modules() if isinstance(module, LoRA)]
    assert loras and all(torch.equal(m.B, torch.zeros_like(m.B)) for m in loras)
    assert all(0 < m.A.std() < 2 * config.init_std for m in loras)
    # LoRA initializes last, so the same seed gives the same shared weights.
    state = model.state_dict()
    assert all(torch.equal(state[key], value) for key, value in shared.state_dict().items())
    tokens = ids()
    out, reference = model.train()(tokens), shared.train()(tokens)
    assert torch.equal(out.logits, reference.logits)
    out.logits.square().mean().backward()
    reference.logits.square().mean().backward()
    gradients = dict(model.named_parameters())
    for name, parameter in shared.named_parameters():
        assert torch.equal(gradients[name].grad, parameter.grad), name
    assert all(m.B.grad.abs().sum() > 0 for m in loras)  # B learns from the first step.


def test_svd_init_gives_the_best_rank_r_residual():
    generator = torch.Generator().manual_seed(3)
    shared, full = torch.randn(6, 5, generator=generator), torch.randn(6, 5, generator=generator)
    residual = full - shared
    lora = LoRA(5, 6, 5)
    lora.svd_init(residual)
    torch.testing.assert_close(shared + lora.B @ lora.A, full)  # full rank recovers W exactly
    singular = torch.linalg.svdvals(residual.double())
    lora = LoRA(5, 6, 2)
    lora.svd_init(residual)
    _, s, vh = torch.linalg.svd(residual.double())
    torch.testing.assert_close(lora.A, vh[:2].float())
    error = (residual - lora.B @ lora.A).square().sum()
    torch.testing.assert_close(error.double(), singular[2:].square().sum(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dense", [False, True])
def test_relaxed_parameter_counts_and_costs(dense):
    spec = recipes.relaxed_spec(dense=dense)
    model = spec.model
    assert (model.num_cells, model.num_visits, model.visit_capacity, model.prefix_cells, model.suffix_cells) == (4, 2, 2, 0, 0)
    assert (model.lora_attention_rank, model.lora_ffn_rank) == ((16, 16) if dense else (16, 0))
    # Per cell (d=128, 4 heads, 2 KV heads of width 64): attention 2*128*128 + 2*128*64 = 49152, two
    # norms 256, and either a dense SwiGLU 3*128*256 = 98304 or a router 128*8 = 1024 plus eight
    # experts 8*3*128*256 = 786432. LoRA per visit: attention 16*(2*128 + 2*(128+64) + 2*128)
    # = 16*896 = 14336, FFN 16*3*(128+256) = 18432. Four cells of two visits; untied embedding
    # and head 2*50257*128 = 12865792 and final norm 128.
    lora_visit = 14336 + (18432 if dense else 0)
    cell = 49152 + 256 + (98304 if dense else 1024 + 786432)
    expected = 4 * cell + 4 * 2 * lora_visit + 12865792 + 128
    assert expected == (13718912 if dense else 16328064)
    assert sum(p.numel() for p in RecursiveLM(model).parameters()) == expected
    cost = estimate_cost(spec, seq_len=16, activation_checkpointing="full_cells")
    assert cost["parameters"]["unique_total"] == expected
    assert cost["parameters"]["by_component"]["lora"] == 8 * lora_visit
    core = cost["parameters"]["by_region"]["core"]
    assert core["parameters"]["lora"] == 8 * lora_visit and core["owners"]["lora"] == 8
    flops = cost["flops"]
    assert flops["forward_components"]["lora"] == 8 * 2 * lora_visit
    assert flops["backward_components"]["lora"] == 2 * 8 * 2 * lora_visit
    assert flops["recompute_components"]["lora"] == 8 * 2 * lora_visit  # inside the replayed Cell
    # Everything else is the strict Recursive Transformer's cost; KV is unchanged.
    strict = estimate_cost(replace(spec, model=plain(model)), seq_len=16, activation_checkpointing="full_cells")
    assert flops["forward_per_token"] - strict["flops"]["forward_per_token"] == 8 * 2 * lora_visit
    assert flops["training_per_token"] - strict["flops"]["training_per_token"] == 4 * 8 * 2 * lora_visit
    assert cost["kv"] == strict["kv"] and cost["calls"] == strict["calls"]


@pytest.mark.parametrize("kind", ["dense", "moe_always_on", "always_on_core", "dense_core"])
def test_lora_flops_equal_counted_matmuls(kind):
    """The cost model's LoRA FLOPs are exactly the extra counted matmul FLOPs, forward and backward."""
    config, tokens = relaxed(kind), ids()

    def counted(model):
        with FlopCounterMode(display=False) as counter:
            model.train()(tokens).logits.sum().backward()
        return counter.get_total_flops()

    torch.manual_seed(0)
    extra = counted(RecursiveLM(config)) - counted(RecursiveLM(plain(config)))
    forward = estimate_cost(config, seq_len=tokens.size(1))["flops"]["forward_components"]["lora"]
    assert extra == 3 * forward * tokens.numel()  # forward plus two backward products


@pytest.mark.parametrize("dense", [False, True])
def test_relaxed_conforms(dense):
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    spec = recipes.relaxed_spec(dense=dense, **SMALL)
    report = testing.check_recipe(spec)
    assert {result["status"] for result in report["checks"].values()} == {"pass"}
    assert RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_trained_lora_keeps_cache_and_export_consistent(kind):
    """check_recipe sees B = 0; nonzero LoRA must also decode with the cache and export."""
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from moeloop.hf import MoeloopConfig, MoeloopForCausalLM

    torch.manual_seed(0)
    model = randomize_lora(RecursiveLM(relaxed(kind))).eval()
    tokens = ids(tokens=8)
    cache = DecodeCache(model.config.num_visits)
    with torch.no_grad():
        full = model(tokens).logits
        chunks = [model(tokens[:, a:b], cache=cache).logits for a, b in ((0, 4), (4, 7), (7, 8))]
    torch.testing.assert_close(torch.cat(chunks, 1), full)
    exported = MoeloopForCausalLM(MoeloopConfig(model.config.to_dict()))
    exported.model.load_state_dict(model.state_dict(), strict=True)
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        exported.save_pretrained(directory, safe_serialization=True)
        restored = MoeloopForCausalLM.from_pretrained(directory, local_files_only=True).eval()
    state = restored.model.state_dict()
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert any(".attention_loras.1.q_proj.B" in key for key in state)
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens, use_cache=False).logits, full)
        generated = restored.generate(tokens[:, :4], max_new_tokens=3, do_sample=False, use_cache=True)
        uncached = restored.generate(tokens[:, :4], max_new_tokens=3, do_sample=False, use_cache=False)
    assert torch.equal(generated, uncached)


@pytest.mark.parametrize("kind", ["dense", "always_on_core", "dense_core"])
def test_lora_owners_live_in_core_cells_per_visit(kind):
    config = relaxed(kind, prefix_cells=1, suffix_cells=1, num_visits=2, visit_capacity=3)
    for region in ("prefix", "suffix"):  # resolved prefix/suffix configs carry no LoRA
        local = config.region_config(region)
        assert (local.lora_attention_rank, local.lora_ffn_rank, local.visit_capacity) == (0, 0, None)
    model = RecursiveLM(config)
    assert all(cell.attention_loras is None and cell.ffn_loras is None for cell in (*model.prefix, *model.suffix))
    assert all(len(cell.attention_loras) == len(cell.ffn_loras) == 3 for cell in model.core)
    cost = estimate_cost(model.config, seq_len=8)
    assert cost["parameters"]["unique_total"] == sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError, match="visit_capacity"):
        model(ids(), num_visits=4)


class NoLoRAAttention(Attention):
    """A third-party attention factory with the forward signature from before LoRA."""

    def forward(self, x, *, positions, attention_mask=None, cache=None, slot=("core", 0, 0)):
        return super().forward(x, positions=positions, attention_mask=attention_mask, cache=cache, slot=slot)


def test_attention_factories_without_lora_keep_working_unless_attention_lora_is_on():
    tokens = ids()
    for config in (ModelConfig(**SMALL), relaxed("ffn_only")):  # no attention LoRA
        torch.manual_seed(0)
        custom = RecursiveLM(config, attention_factory=NoLoRAAttention)
        torch.manual_seed(0)
        assert torch.equal(custom(tokens).logits, RecursiveLM(config)(tokens).logits)
    with pytest.raises(TypeError, match="lora"):
        RecursiveLM(relaxed("moe"), attention_factory=NoLoRAAttention)(tokens)


def test_default_config_and_archived_records_have_no_lora():
    config = ModelConfig()
    assert (config.lora_attention_rank, config.lora_ffn_rank, config.independent_visits) == (0, 0, False)
    record = {key: value for key, value in config.to_dict().items() if not key.startswith("lora_")}
    assert ModelConfig.from_dict(record) == config
    assert not any(isinstance(m, LoRA) for m in RecursiveLM(ModelConfig(**SMALL)).modules())


@pytest.mark.parametrize("change, message", [
    (dict(num_experts=4, always_on_hidden_dim=0), "FFN LoRA"),
    (dict(lora_attention_rank=-1), "nonnegative"),
    (dict(state_update="ema", num_experts=0), "EMA"),
])
def test_invalid_lora_configs(change, message):
    with pytest.raises(ValueError, match=message):
        relaxed("dense", **change)


def test_lora_requires_fixed_depth_and_full_bptt():
    config = relaxed("moe", num_visits=2)
    TrainingRecipe(num_visits=2).validate_model(config)
    for recipe in (TrainingRecipe(num_visits=2, sampled_visits=(1, 2)), TrainingRecipe(num_visits=2, grad_visits=1)):
        with pytest.raises(ValueError, match="per-visit owners"):
            recipe.validate_model(config)
    with pytest.raises(ValueError, match="tail-only gradients"):
        RecursiveLM(config)(ids(), grad_visits=1)


def test_relaxed_records_its_source():
    spec, dense = recipes.relaxed_spec(), recipes.relaxed_spec(dense=True)
    assert (spec.name, dense.name) == ("relaxed", "relaxed_dense")
    assert spec.source["url"] == "https://arxiv.org/html/2410.20672v3"
    assert spec.source["layers_loops_rank"] == [4, 2, 16]
    assert spec.preserved and spec.adaptations
    assert any("only attention is relaxed" in line for line in spec.adaptations)
    assert (spec.training.balance_coeff, dense.training.balance_coeff) == (0.01, 0.0)
    with pytest.raises(ValueError, match="sparse cells"):
        recipes.relaxed_spec(num_experts=0)
