"""Reference-only FFN-chain update using an already initialized physical Cell.

Attention and FFN normalization run once, followed by shared router/expert
rounds with inner residuals, and a final outer residual. This is a different
update from whole-Cell recurrence. Native training and HF need their own
explicit binding before they can execute this update.
"""

from __future__ import annotations

import torch
from torch import Tensor

from moeloop.model import Cell


def forward_ffn_chain(
    cell: Cell, state: Tensor, *, ffn_steps: int, positions: Tensor,
    attention_mask: Tensor | None = None, router_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return state and auxiliary sums over all FFN calls, with full BPTT.

    The caller owns/initializes the Cell and supplies a prepared causal mask.
    Divide the auxiliary sums by ffn_steps for a per-call mean. No decode cache
    or gradient truncation is provided by this small extension example.
    """
    if type(ffn_steps) is not int or ffn_steps < 1:
        raise ValueError("ffn_steps must be a positive integer")
    if cell.experts is None or any(len(owners) != 1 for owners in (
        cell.attentions, cell.attention_norms, cell.ffn_norms, cell.routers,
    )):
        raise ValueError("this example requires a sparse Cell with shared component owners")
    residual = state + cell.residual_scale * cell.attentions[0](
        cell.attention_norms[0](state), positions=positions,
        attention_mask=attention_mask,
    )
    state = cell.ffn_norms[0](residual)
    balance_sum = state.new_zeros((), dtype=torch.float32)
    z_sum = balance_sum.clone()
    for _ in range(ffn_steps):
        routing = cell.routers[0](state, router_mask)
        update = cell.experts(state, routing.ids, routing.weights, routing.dispatch_counts)
        if cell.always_on_ffn is not None:
            update = update + cell.always_on_ffn(state)
        state = state + cell.residual_scale * update
        balance_sum = balance_sum + routing.balance
        z_sum = z_sum + routing.z_loss
    return residual + state, balance_sum, z_sum
