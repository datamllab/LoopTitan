"""Random initial state and stateless per-microbatch depth contracts."""

import math

import pytest

torch = pytest.importorskip("torch")

from moeloop.latent import initial_latent, sampled_depth


@pytest.fixture(autouse=True)
def isolated_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield


def test_random_initial_latent_is_independent_bounded_and_reproducible():
    anchor = torch.full((2, 128, 64), 9.0, requires_grad=True)
    saved = torch.get_rng_state()
    latent = initial_latent(anchor, training=True)
    torch.set_rng_state(saved)
    repeated = initial_latent(anchor, training=True)
    assert torch.equal(latent, repeated)
    assert latent.shape == anchor.shape and latent.dtype == anchor.dtype
    assert latent.device == anchor.device and not latent.requires_grad
    assert latent.data_ptr() != anchor.data_ptr()
    assert latent.abs().max() <= 3 * math.sqrt(2 / 5)
    # Truncation at three standard deviations slightly lowers the variance.
    assert abs(latent.mean().item()) < .03
    assert .60 < latent.std().item() < .65
    assert torch.equal(anchor, torch.full_like(anchor, 9.0))


def test_eval_latent_is_zero_without_consuming_rng():
    anchor = torch.ones((2, 3, 4), dtype=torch.bfloat16, requires_grad=True)
    before = torch.get_rng_state()
    actual = initial_latent(anchor, training=False)
    assert torch.equal(before, torch.get_rng_state())
    assert actual.dtype == anchor.dtype and actual.device == anchor.device
    assert not actual.requires_grad and torch.count_nonzero(actual) == 0


def test_training_latent_consumes_custom_scale_and_preserves_low_precision():
    anchor = torch.ones((64, 64), dtype=torch.bfloat16)
    actual = initial_latent(anchor, training=True, std=.125, truncation=2.0)
    assert actual.dtype == anchor.dtype and actual.shape == anchor.shape
    assert actual.abs().max() <= .25
    assert actual.abs().max() > .125


def test_depth_coordinate_survives_resume_without_consuming_global_rng():
    coordinates = [(update, micro) for update in range(5) for micro in range(4)]
    before = torch.get_rng_state()
    first = [sampled_depth(seed=42, completed_updates=u, microbatch=m) for u, m in coordinates]
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(7)
    resumed = [sampled_depth(seed=42, completed_updates=u, microbatch=m) for u, m in coordinates[7:]]
    assert resumed == first[7:]
    assert all(type(depth) is int and depth >= 1 for depth in first)
    assert len(set(first)) > 1


@pytest.mark.parametrize("sigma, variance", [(0.0, 3.0), (0.5, 5.556228750189673)])
def test_depth_has_shifted_poisson_lognormal_moments_and_a_tail(sigma, variance):
    # One seed, varying microbatch coordinates: no training seed sweep.
    samples = torch.tensor([
        sampled_depth(seed=42, completed_updates=0, microbatch=m, mean=4.0, log_std=sigma)
        for m in range(4096)
    ], dtype=torch.float64)
    # These follow E[1+Pois(exp(tau))]=4 and the mixed-Poisson variance,
    # and catch a missing shift, incorrect lognormal location, or missing Poisson.
    assert abs(samples.mean().item() - 4.0) < .15
    assert abs(samples.var().item() - variance) < .8
    assert samples.max() > 8


@pytest.mark.parametrize("kwargs", [
    {"completed_updates": -1}, {"completed_updates": True}, {"microbatch": -1}, {"seed": 1.5},
])
def test_depth_rejects_invalid_identity(kwargs):
    values = dict(seed=42, completed_updates=0, microbatch=0)
    with pytest.raises(ValueError):
        sampled_depth(**(values | kwargs))


def test_initial_latent_rejects_non_bool_training():
    with pytest.raises(ValueError, match="training"):
        initial_latent(torch.ones(2, 3), training=1)


def test_initial_latent_rejects_integer_reference():
    with pytest.raises(ValueError):
        initial_latent(torch.ones(2, 3, dtype=torch.long), training=True)
