"""Independent latent initialization and stateless microbatch depth sampling."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor, nn


def initial_latent(
    anchor: Tensor, *, training: bool, std: float = math.sqrt(2 / 5),
    truncation: float = 3.0,
) -> Tensor:
    """Draw an independent training state; evaluation explicitly starts at zero.

    Training uses the device's global Torch RNG, which the trainer checkpoints.
    ``std`` describes the normal before truncation, not the resulting variance.
    Neither branch creates a gradient path from the initial state to the anchor.
    The live anchor must still be passed separately to the recurrent update.

    Precondition: finite positive ``std`` and ``truncation``; ``ModelConfig``
    enforces this for ``latent_init_std`` and ``latent_truncation``.
    """
    if not isinstance(anchor, Tensor) or not anchor.is_floating_point():
        raise ValueError("initial latent needs a floating-point anchor tensor")
    if type(training) is not bool:
        raise ValueError("training must be bool")
    if not training:
        return torch.zeros_like(anchor)
    return nn.init.trunc_normal_(torch.empty_like(anchor), mean=0.0, std=std,
                                 a=-truncation * std, b=truncation * std)


def sampled_depth(
    *, seed: int, completed_updates: int, microbatch: int,
    mean: float = 4.0, log_std: float = 0.5,
) -> int:
    """Sample ``1 + Poisson(LogNormal)`` with the requested expected depth.

    The coordinate, rather than a rank's mutable RNG, determines the draw. All
    ranks using the same recipe/update/microbatch therefore agree, including
    after restoration. No upper depth cap or reconstructed global RNG is used.

    Precondition: finite ``mean`` > 1 and finite ``log_std`` >= 0;
    ``TrainingRecipe`` enforces this for ``depth_mean`` and ``depth_log_std``.
    """
    if type(seed) is not int:
        raise ValueError("depth seed must be an integer")
    for name, value in (("completed_updates", completed_updates), ("microbatch", microbatch)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    location = math.log(mean - 1) - 0.5 * log_std * log_std
    if not math.isfinite(location):
        raise ValueError("lognormal depth location must be finite")
    key = f"poisson_lognormal_micro_v1:{seed}:{completed_updates}:{microbatch}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    epsilon = torch.randn((), dtype=torch.float64, generator=generator)
    rate = torch.exp(location + log_std * epsilon)
    if not torch.isfinite(rate).item():
        raise ValueError("sampled depth rate overflowed; change the declared distribution")
    return 1 + int(torch.poisson(rate, generator=generator).item())
