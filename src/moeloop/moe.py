"""Small, stateless sparse operators and their transparent PyTorch reference."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model import ModelConfig


class Routing(NamedTuple):
    ids: Tensor
    weights: Tensor
    balance: Tensor
    z_loss: Tensor
    counts: Tensor
    dispatch_counts: Tensor


class Router(nn.Module):
    """FP32 softmax top-k with selected-set normalization and dropless routing.

    Balance/z terms are per-call valid-input means. Dispatch counts include all
    rows, whereas auxiliary counts exclude padding. No persistent counters are
    mutated, including during activation-checkpoint replay.
    """

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.top_k = config.top_k

    def forward(self, x: Tensor, router_mask: Tensor | None = None) -> Routing:
        shape = x.shape[:-1]
        if router_mask is None:
            router_mask = torch.ones(shape, dtype=torch.bool, device=x.device)
        if router_mask.shape != shape or router_mask.dtype != torch.bool:
            raise ValueError("router_mask must be bool and match the input token dimensions")
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.gate.weight.float())
            probs = logits.softmax(-1)
            selected, ids = probs.topk(self.top_k, dim=-1, sorted=False)
            weights = selected / selected.sum(-1, keepdim=True)
            experts = probs.size(-1)
            dispatch_counts = torch.bincount(ids.reshape(-1), minlength=experts)
            assignment_mask = router_mask.unsqueeze(-1).expand_as(ids).reshape(-1)
            counts = torch.zeros(experts, dtype=torch.long, device=x.device)
            counts.scatter_add_(0, ids.reshape(-1), assignment_mask.long())
            valid = router_mask.float()
            denominator = valid.sum().clamp_min(1)
            probability_mean = (probs * valid.unsqueeze(-1)).sum(tuple(range(probs.ndim - 1))) / denominator
            load = counts.float() / (denominator * self.top_k)
            balance = experts * (load * probability_mean).sum()
            z_loss = (logits.logsumexp(-1).square() * valid).sum() / denominator
        return Routing(ids, weights.to(x.dtype), balance, z_loss, counts, dispatch_counts)


class DenseFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ExpertBank(nn.Module):
    """Packed physical expert owners; no token dropping or hidden capacity.

    This deliberately readable reference loops over experts. The finite native
    training binding uses the same tensors with a grouped kernel/dispatcher.
    """

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__()
        self.w1_EFD = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim, config.dim))
        self.w3_EFD = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim, config.dim))
        self.w2_EDF = nn.Parameter(torch.empty(config.num_experts, config.dim, config.hidden_dim))

    def forward(
        self, x: Tensor, ids: Tensor, weights: Tensor, dispatch_counts: Tensor | None = None,
    ) -> Tensor:
        flat = x.reshape(-1, x.size(-1))
        ids, weights = ids.reshape(flat.size(0), -1), weights.reshape(flat.size(0), -1)
        output = torch.zeros_like(flat)
        for expert in range(self.w1_EFD.size(0)):
            rows, choices = torch.where(ids == expert)
            tokens = flat[rows]
            hidden = F.silu(F.linear(tokens, self.w1_EFD[expert])) * F.linear(tokens, self.w3_EFD[expert])
            update = F.linear(hidden, self.w2_EDF[expert])
            output.index_add_(0, rows, update * weights[rows, choices, None])
        return output.view_as(x)
