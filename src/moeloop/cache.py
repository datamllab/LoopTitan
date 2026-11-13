"""Exact, inference-only KV storage indexed by logical attention calls."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


CacheSlot = tuple[str, int, int]


@dataclass
class DecodeCache:
    """Fixed-depth KV cache. Shared attention weights still have separate slots.

    ``length`` counts stored columns (including padding), while ``valid_mask``
    and ``positions`` retain each row's valid-token and RoPE metadata. Use
    :meth:`reorder` for beam/batch reordering; never reorder slots individually.
    During a forward every attention call appends to its slot; ``advance`` then
    records the forward's columns. A forward that raises midway leaves some
    slots longer than ``length``, and the next forward on this cache fails, so
    ``clear`` it or start a new cache.
    """

    num_visits: int
    length: int = field(default=0, init=False)
    slots: dict[CacheSlot, tuple[Tensor, Tensor]] = field(default_factory=dict, init=False)
    positions: Tensor | None = field(default=None, init=False)
    valid_mask: Tensor | None = field(default=None, init=False)

    def next_positions(self, valid_mask: Tensor) -> Tensor:
        """Standard padding-aware positions for the new query columns."""
        offset = valid_mask.new_zeros((valid_mask.size(0), 1), dtype=torch.long)
        if self.valid_mask is not None:
            if self.valid_mask.size(0) != valid_mask.size(0):
                raise ValueError("cache batch size changed without reorder")
            columns = torch.arange(self.length, device=self.valid_mask.device)
            last = torch.where(self.valid_mask, columns, -1).amax(-1, keepdim=True)
            offset = self.positions.gather(1, last.clamp_min(0)) + 1
            offset = torch.where(last >= 0, offset, 0)
        positions = offset + valid_mask.long().cumsum(-1) - 1
        return positions.clamp_min(0)

    def append(self, slot: CacheSlot, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Append this forward's [batch, kv_heads, tokens, head_dim] K/V; return the slot's whole K/V."""
        if slot in self.slots:
            old_key, old_value = self.slots[slot]
            key, value = torch.cat((old_key, key), dim=-2), torch.cat((old_value, value), dim=-2)
        self.slots[slot] = (key, value)
        return key, value

    def advance(self, positions: Tensor, valid_mask: Tensor, expected_slots: tuple[CacheSlot, ...]) -> None:
        """Record a finished forward's columns; every expected slot must hold all of them."""
        if self.positions is not None:
            positions = torch.cat((self.positions, positions), dim=1)
            valid_mask = torch.cat((self.valid_mask, valid_mask), dim=1)
        length = positions.size(1)
        if set(self.slots) != set(expected_slots) or any(key.size(-2) != length for key, _ in self.slots.values()):
            raise ValueError("forward did not update every logical cache slot; "
                             "clear() the cache after a failed forward")
        self.positions, self.valid_mask, self.length = positions, valid_mask, length

    def reorder(self, indices: Tensor) -> "DecodeCache":
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache reorder indices must be integer [new_batch]")
        self.slots = {
            slot: (key.index_select(0, indices.to(key.device, dtype=torch.long)),
                   value.index_select(0, indices.to(value.device, dtype=torch.long)))
            for slot, (key, value) in self.slots.items()
        }
        if self.positions is not None:
            self.positions = self.positions.index_select(0, indices.to(self.positions.device, dtype=torch.long))
            self.valid_mask = self.valid_mask.index_select(0, indices.to(self.valid_mask.device, dtype=torch.long))
        return self

    def clear(self) -> None:
        self.slots.clear()
        self.length = 0
        self.positions = self.valid_mask = None
