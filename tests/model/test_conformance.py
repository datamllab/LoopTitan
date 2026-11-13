"""Every shipped recipe passes check_recipe; each check rejects a broken variant."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from moeloop import recipes, testing
from moeloop.model import RecursiveLM


GEOMETRY = dict(vocab_size=31, dim=16, hidden_dim=24, num_heads=4, num_kv_heads=2,
                num_experts=4, top_k=2, max_seq_len=32)
SPECS = {
    "smelt": lambda: recipes.smelt_spec(**GEOMETRY),
    "latentloop_residual": lambda: recipes.latentloop_residual_spec(**GEOMETRY),
    "meta": lambda: recipes.meta_spec(**GEOMETRY),
    "foil": lambda: recipes.foil_spec(**GEOMETRY),
    "expert_tying": lambda: recipes.expert_tying_spec(**GEOMETRY),
    "huginn": lambda: recipes.huginn_spec(**GEOMETRY),
    "loom": lambda: recipes.loom_spec(**GEOMETRY),
    "ouro": lambda: recipes.ouro_spec(**GEOMETRY),
    "moe": lambda: recipes.moe_spec(layers=2, **GEOMETRY),
    "dense": lambda: recipes.dense_spec(layers=2, **GEOMETRY),
}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", SPECS)
def test_shipped_recipe_conforms(name):
    state = torch.get_rng_state()
    report = testing.check_recipe(SPECS[name]())
    assert torch.equal(torch.get_rng_state(), state)
    checks = report["checks"]
    assert tuple(checks) == testing.CHECKS
    uncached = name in ("huginn", "loom")
    assert {check: result["status"] for check, result in checks.items()} == {
        check: "not_applicable" if check == "cache" and uncached else "pass" for check in testing.CHECKS}
    if uncached:
        assert "uncached fixed-depth generation" in checks["cache"]["detail"]
    # Non-vacuous: looped recipes reuse owners, independent stacks reuse none.
    assert (checks["shared_gradients"]["reused_parameters"] > 0) == (name not in ("moe", "dense"))


def test_reused_parameters_count_only_parameters_with_several_gradient_calls():
    # LOOM with per-visit attention: the 2 core banks each share their 2 norms,
    # FFN output norm and 3 expert tensors; attention and routers are per visit.
    spec = recipes.loom_spec(attention_per_visit=True, **GEOMETRY)
    assert testing.check_recipe(spec)["checks"]["shared_gradients"]["reused_parameters"] == 2 * 6


def test_default_geometry_passes_beyond_float32_noise():
    # Cached decoding at dim 128 / vocab 50257 differs from recomputation by
    # float32 noise above test_cache.py's tiny-model tolerance.
    report = testing.check_recipe(recipes.smelt_spec(), seq_len=16)
    assert all(result["status"] == "pass" for result in report["checks"].values())


class TiedHead(RecursiveLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.lm_head.weight = self.embedding.weight


class ReuseScaledGradients(RecursiveLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for parameter in self.core.parameters():
            parameter.register_hook(lambda grad: grad / config.num_visits)


class CacheForgetsPositions(RecursiveLM):
    def forward(self, token_ids, *, positions=None, cache=None, **kwargs):
        if cache is not None and positions is None:
            positions = torch.arange(token_ids.size(1)).expand_as(token_ids)
        return super().forward(token_ids, positions=positions, cache=cache, **kwargs)


class UnexportedEps(RecursiveLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.final_norm.eps = 1e-2


BROKEN = {
    "ownership": (TiedHead, None),
    "shared_gradients": (ReuseScaledGradients, None),
    "cost_calls": (None, lambda spec: replace(spec, training=replace(spec.training, num_visits=spec.model.num_visits + 1))),
    "cache": (CacheForgetsPositions, None),
    "hf_roundtrip": (UnexportedEps, None),
    "identity": (None, lambda spec: replace(spec, run=spec.run | {"depth_schedule": {1: 4}})),
}


@pytest.mark.parametrize("check", testing.CHECKS)
def test_each_check_rejects_its_broken_variant(check, monkeypatch):
    model, change = BROKEN[check]
    if model is not None:
        monkeypatch.setattr(testing, "RecursiveLM", model)
    spec = recipes.meta_spec(**GEOMETRY)
    with pytest.raises(testing.ConformanceError, match=f"failed conformance:(.|\n)*{check}: ") as error:
        testing.check_recipe(spec if change is None else change(spec))
    assert error.value.report["checks"][check]["status"] == "fail"
