"""Batched tensor implementation of minesweeper.

Written from spec.md ONLY — do not look at reference.py while writing this.
State: tensors with leading [N] dim. Branching -> torch.where masking. See
simulacrum.batched docstrings for the idioms and the auto-reset contract.
"""

from __future__ import annotations

import torch

from simulacrum import BatchedEnv, invariant, rng

from minesweeper import Slots


class MinesweeperBatched(BatchedEnv):
    def _reset_instances(self, mask: torch.Tensor) -> None:
        # Allocate state tensors on first call; masked-fill afterwards, e.g.:
        #   pos0 = rng.draw_randint_torch(self.keys, 0, Slots.INIT_POSITION, n)
        #   self.pos = torch.where(mask, pos0, self.pos)
        raise NotImplementedError("TODO: initialize per spec.md #Reset")

    def _step_impl(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Draw for ALL instances (stateless RNG makes discarded draws safe):
        #   slip = rng.draw_bernoulli_torch(self.keys, self.t, Slots.SLIP, p)
        raise NotImplementedError("TODO: transition per spec.md; return (rewards, terminated)")

    def observe(self) -> torch.Tensor:
        raise NotImplementedError("TODO: observation encoding per spec.md")

    def slice_to_json(self, i: int) -> dict:
        raise NotImplementedError("TODO: serialize instance i per schema.json $defs/state")

    # TODO: one @invariant per entry in spec.md #Invariants, e.g.:
    # @invariant("position_in_bounds")
    # def _inv_position(self) -> torch.Tensor:
    #     return (self.pos >= -L) & (self.pos <= L)
