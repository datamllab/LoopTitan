"""Functional normalized memories; callers own forward and visit resets."""

from __future__ import annotations

import math
from typing import NamedTuple

from torch import Tensor


class EMACarry(NamedTuple):
    """A differentiable numerator and its scalar normalization, never a buffer."""

    numerator: Tensor | None = None
    denominator: float = 0.0


def ema_update(carry: EMACarry, write: Tensor, *, beta: float = .5) -> tuple[EMACarry, Tensor]:
    """Return new carry and its updated read without mutating or detaching history.

    Start a new forward with ``EMACarry()`` for global memory; separately start
    each visit with ``EMACarry()`` for local memory. Add the two returned reads
    directly to the residual stream. This function neither combines memories
    nor adds a separate attention residual.

    Precondition: ``beta`` is finite and in [0, 1); ``ModelConfig`` enforces
    this for ``memory_beta``. The carry and write are checked here.
    """
    if not isinstance(write, Tensor) or not write.is_floating_point():
        raise ValueError("write must be a floating-point tensor")
    numerator, denominator = carry
    if type(denominator) not in (int, float) or not math.isfinite(denominator):
        raise ValueError("carry denominator must be finite")
    if numerator is None:
        if denominator != 0:
            raise ValueError("empty carry must have zero denominator")
        numerator = write
    else:
        if (denominator <= 0 or numerator.shape != write.shape
                or numerator.device != write.device or numerator.dtype != write.dtype):
            raise ValueError("carry must have positive normalization and match the write shape/device/dtype")
        numerator = beta * numerator + write
    denominator = beta * denominator + 1.0
    return EMACarry(numerator, denominator), numerator / denominator


def residual_gain(num_cells: int, num_visits: int, *, scale: float = .5) -> float:
    """Scale a residual write by ``scale / (num_visits * sqrt(num_cells))``.

    Precondition: positive integer ``num_cells`` and ``num_visits`` and finite
    nonnegative ``scale``; ``ModelConfig`` and ``RecursiveLM.forward`` enforce them.
    """
    return scale / (num_visits * math.sqrt(num_cells))


def reinjection_gain(num_cells: int, visit: int, *, scale: float = .5) -> float:
    """Gain for a one-based visit >= 2; visit 1 starts from the input itself."""
    if type(visit) is not int or visit < 2:
        raise ValueError("reinjection requires a one-based visit >= 2")
    return residual_gain(num_cells, visit, scale=scale)
