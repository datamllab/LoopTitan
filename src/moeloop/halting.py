"""Tokenwise exit probabilities and a memory-bounded expected LM objective.

Training evaluates every visit. Exit probabilities weight per-token losses;
they do not select a shortened execution path or mix vocabulary logits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _log_probabilities(logits: Tensor) -> Tensor:
    if logits.ndim != 3 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError("hazard logits must be floating [visits,batch,tokens]")
    scores = logits.float()
    # The last visit receives all remaining mass; its gate is never used.
    survival = torch.cat((torch.zeros_like(scores[:1]),
                          F.logsigmoid(-scores[:-1]).cumsum(0)), dim=0)
    return torch.cat((survival[:-1] + F.logsigmoid(scores[:-1]), survival[-1:]), dim=0)


def hazard_probabilities(logits: Tensor) -> Tensor:
    """Return FP32 ``p[r,b,t]`` with residual mass assigned to the last visit.

    The final entry in ``logits`` is ignored, including its gradient. A single
    visit therefore has probability one independently of the gate.
    """
    return _log_probabilities(logits).exp()


def expected_lm_loss(
    visit_hidden_states: Sequence[Tensor],
    hazard_logits: Tensor,
    head: nn.Module,
    labels: Tensor,
    global_valid_labels: Tensor,
    *,
    entropy_coeff: float = 0.1,
    num_chunks: int = 8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum expected token CE minus exit entropy over one global update count.

    ``visit_hidden_states`` already contains the recipe's norm/coda readouts.
    Labels are aligned targets, with ``-100`` ignored in both CE and entropy.
    Every rank/microbatch uses the same whole-update denominator; there is no
    additional rank or accumulation mean. Router auxiliaries remain separate.

    Nonreentrant checkpointing saves hidden chunks and token losses rather
    than every visit's vocabulary logits/softmax. Backward recomputes each head
    and CE chunk through a normal module call, preserving FSDP hooks and all
    hidden, shared-head, and gate gradients. The head is a deterministic LM
    projection; RNG is not consumed by this objective.

    ``ce`` is standard terminal CE, distinct from ``expected_ce``. Detached
    metric contributions use the same global denominator and can be summed
    across ranks/microbatches. No early-exit compute saving is implied.

    Precondition: finite nonnegative ``entropy_coeff``; ``TrainingRecipe``
    enforces it. Tensor shapes, ``num_chunks`` and the denominator are checked here.
    """
    if labels.ndim != 2 or labels.dtype != torch.long or min(labels.shape) < 1:
        raise ValueError("labels must be nonempty int64 [batch,tokens]")
    if not visit_hidden_states:
        raise ValueError("visit hidden states cannot be empty")
    shape = visit_hidden_states[0].shape
    if len(shape) != 3 or shape[:2] != labels.shape or shape[2] < 1:
        raise ValueError("visit hidden states must have shape [batch,tokens,hidden]")
    if any(h.shape != shape or h.device != labels.device or not h.is_floating_point()
           for h in visit_hidden_states):
        raise ValueError("visit hidden shapes/devices must match labels and each other")
    if hazard_logits.shape != (len(visit_hidden_states), *labels.shape) or hazard_logits.device != labels.device:
        raise ValueError("hazard logits must match [visits,batch,tokens] on the label device")
    if type(num_chunks) is not int or num_chunks < 1:
        raise ValueError("num_chunks must be a positive integer")
    if (not isinstance(global_valid_labels, Tensor) or global_valid_labels.numel() != 1
            or not torch.isfinite(global_valid_labels).item() or global_valid_labels.item() <= 0):
        raise ValueError("global valid label count must be a positive finite scalar")
    denominator = global_valid_labels.detach().to(device=labels.device, dtype=torch.float32)
    log_p = _log_probabilities(hazard_logits)
    p = log_p.exp()
    valid = labels != -100
    entropy = (-(p * log_p).sum(0) * valid).sum() / denominator

    flat_labels = labels.flatten()
    flat_hidden = [h.reshape(-1, h.shape[-1]) for h in visit_hidden_states]
    flat_p = p.flatten(1)
    chunk_size = math.ceil(flat_labels.numel() / num_chunks)
    expected_ce = torch.zeros((), dtype=torch.float32, device=labels.device)
    terminal_ce = expected_ce.clone()
    visit_ce = [expected_ce.clone() for _ in flat_hidden]

    def token_ce(hidden: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(head(hidden).float(), targets, reduction="none", ignore_index=-100)

    for start in range(0, flat_labels.numel(), chunk_size):
        stop = min(start + chunk_size, flat_labels.numel())
        for visit, hidden in enumerate(flat_hidden):
            arguments = (hidden[start:stop], flat_labels[start:stop])
            ce = (checkpoint(token_ce, *arguments, use_reentrant=False, preserve_rng_state=False)
                  if torch.is_grad_enabled() else token_ce(*arguments))
            expected_ce = expected_ce + (flat_p[visit, start:stop] * ce).sum() / denominator
            visit_ce[visit] = visit_ce[visit] + ce.detach().sum() / denominator
            if visit == len(flat_hidden) - 1:
                terminal_ce = terminal_ce + ce.sum() / denominator
    loss = expected_ce - entropy_coeff * entropy
    return loss, {
        "ce": terminal_ce.detach(),
        "expected_ce": expected_ce.detach(),
        "entropy": entropy.detach(),
        "visit_ce": torch.stack(visit_ce).detach(),
        "mean_exit_probabilities": ((p * valid).sum((1, 2)) / denominator).detach(),
    }
