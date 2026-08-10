"""Readable single-instance reference implementation of minesweeper.

Written from spec.md ONLY. Style: dataclass state, explicit ifs, no
vectorization, no premature abstraction, every rule traceable to a spec line.
All randomness via simulacrum.rng scalar draws with slots from minesweeper.Slots.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulacrum import ReferenceEnv, rng

from minesweeper import Slots


@dataclass
class State:
    t: int  # in-episode step counter (RNG draws are keyed on it)
    # TODO: fields per spec.md's state-space table


class MinesweeperReference(ReferenceEnv):
    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)
        # Reset-time draws: step 0, dedicated slots, e.g.
        #   pos = rng.draw_randint(self.key, 0, Slots.INIT_POSITION, n)
        raise NotImplementedError("TODO: build initial State per spec.md #Reset")

    def step(self, action) -> tuple[State, float, bool, dict]:
        # Per-step draws: rng.draw_*(self.key, self.state.t, Slots.X)
        raise NotImplementedError("TODO: transition per spec.md")

    def observe(self, state: State):
        raise NotImplementedError("TODO: observation encoding per spec.md")

    def to_json(self, state: State) -> dict:
        raise NotImplementedError("TODO: serialize per schema.json $defs/state")

    def from_json(self, obj: dict) -> State:
        raise NotImplementedError("TODO: inverse of to_json")
