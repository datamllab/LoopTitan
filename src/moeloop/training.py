"""Visible training dynamics and objective reductions, independent of TorchTitan."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainingRecipe


def auxiliary_loss(output: Any, recipe: TrainingRecipe, *, dp_degree: int, accumulation: int) -> torch.Tensor:
    """For native SUM gradients: mean the calls of one microbatch, then mean microbatches.

    ``dp_degree * accumulation`` is the number of logical microbatches of the
    update, global batch / local batch. A microbatch is one forward pass and the
    data stream fixes its blocks, so for a fixed global and local batch neither
    the per-microbatch term nor the divisor depends on the number of ranks.
    """
    if dp_degree < 1 or accumulation < 1:
        raise ValueError("dp_degree and accumulation must be positive")
    calls = output.aux_calls
    divisor = max(calls, 1) * dp_degree * accumulation
    return (recipe.balance_coeff * output.balance_sum + recipe.z_loss_coeff * output.z_sum) / divisor


def objective(
    output: Any,
    labels: torch.Tensor,
    global_valid_labels: torch.Tensor | None,
    recipe: TrainingRecipe,
    *,
    dp_degree: int = 1,
    accumulation: int = 1,
    head: torch.nn.Module | None = None,
    num_chunks: int = 8,
    ce: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The training objective: expected CE or terminal CE, plus router auxiliaries.

    Labels are already shifted/aligned by the data stream. A missing denominator
    is the validator's raw CE sum contract. Training CE divides by the global
    valid labels of the entire optimizer update, without an additional
    rank/accumulation division. Stock averaging DDP is different.

    ``num_chunks`` splits the expected-CE head readout (see ``expected_lm_loss``).
    ``ce`` is terminal CE the caller already computed under this same reduction,
    TorchTitan's native chunked head loss; otherwise it comes from ``output.logits``.
    """
    if recipe.objective == "expected_ce" and global_valid_labels is not None:
        from .halting import expected_lm_loss
        if head is None:
            raise ValueError("expected CE requires the shared head module")
        loss, metrics = expected_lm_loss(output.visit_hidden_states, output.hazard_logits,
            head, labels, global_valid_labels, entropy_coeff=recipe.entropy_coeff, num_chunks=num_chunks)
        aux = auxiliary_loss(output, recipe, dp_degree=dp_degree, accumulation=accumulation)
        return loss + aux, {**metrics, "aux": aux.detach()}
    if ce is None:
        if output.logits is None:
            raise ValueError("objective requires logits or the caller's terminal CE")
        ce = F.cross_entropy(output.logits.flatten(0, 1).float(), labels.flatten(), reduction="sum", ignore_index=-100)
        if global_valid_labels is not None:
            if global_valid_labels.numel() != 1 or global_valid_labels.item() <= 0:
                raise ValueError("an optimizer update needs at least one globally valid label")
            ce = ce / global_valid_labels
    if global_valid_labels is None:
        return ce, {"ce": ce.detach()}
    aux = auxiliary_loss(output, recipe, dp_degree=dp_degree, accumulation=accumulation)
    return ce + aux, {"ce": ce.detach(), "aux": aux.detach()}


@dataclass
class CommittedProgress:
    """Rank-local lifetime counts, committed only after the native update returns."""

    completed_updates: int = 0
    padded_positions: int = 0
    input_tokens: int = 0
    label_tokens: int = 0
    recursive_input_tokens: int = 0

    def commit(self, *, padded_positions: int, input_tokens: int, label_tokens: int, recursive_input_tokens: int) -> None:
        counts = (padded_positions, input_tokens, label_tokens, recursive_input_tokens)
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("committed counts must be nonnegative integers")
        self.completed_updates += 1
        for name, count in zip(("padded_positions", "input_tokens", "label_tokens", "recursive_input_tokens"), counts):
            setattr(self, name, getattr(self, name) + count)


def capture_rng() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    if "numpy" in sys.modules:
        state["numpy"] = sys.modules["numpy"].random.get_state()
    return state


def seed_host_rng(seed: int, *, rank: int = 0) -> int:
    """Seed Python/NumPy explicitly; native Titan seeds only Torch generators."""
    value = (int(seed) + int(rank)) % (2**64)
    random.seed(value)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(value % (2**32))
    return value


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


@contextmanager
def isolated_rng() -> Iterator[None]:
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def spec_identity(*, model: ModelConfig, recipe: TrainingRecipe, recipe_metadata: dict[str, Any],
                  expert_lr_multiplier: float, fork: dict[str, Any], ema_decay: float | None) -> dict[str, Any]:
    """The spec-derived fields of the continuation identity (``LoopTrainer.identity``).

    A field is spec-derived when LoopTrainer.Config carries it from the RecipeSpec in a
    moeloop field rather than a TorchTitan config class; trainer_config and experiment
    fill these from the spec. Values TorchTitan resolves or the launch measures stay in
    LoopTrainer.identity. A new spec-derived field is a new keyword here, which every
    caller must then pass. ``ema_decay`` enters only when set, so runs without an EMA keep
    the identity they had before it existed.
    """
    identity = {"model": model.to_dict(), "recipe": asdict(recipe), "recipe_metadata": recipe_metadata,
                "expert_lr_multiplier": expert_lr_multiplier, "fork": fork}
    if ema_decay is not None:
        identity["ema_decay"] = ema_decay
    return identity


def continuation_fingerprint(identity: dict[str, Any]) -> str:
    """An inspectable identity accompanies the digest in the checkpoint."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def updates_for_token_budget(input_tokens: int, *, seq_len: int, global_batch_size: int) -> int:
    per_update = seq_len * global_batch_size
    if min(input_tokens, seq_len, global_batch_size) <= 0 or input_tokens % per_update:
        raise ValueError("input token budget must be a positive integral number of fixed-shape updates")
    return input_tokens // per_update
