"""Readable single-instance reference implementation of minesweeper.

Written from spec.md ONLY. Style: dataclass state, explicit ifs, no
vectorization, no premature abstraction, every rule traceable to a spec line.
All randomness via simulacrum.rng scalar draws with slots from minesweeper.Slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulacrum import ReferenceEnv, rng

from minesweeper import Slots

# spec: Parameters — defaults (Expert)
H_DEFAULT = 16
W_DEFAULT = 30
M_DEFAULT = 99
LATENCY_DEFAULT = 200
HORIZON_DEFAULT = 2000

# spec: Actions — kinds
REVEAL, FLAG, CHORD = 0, 1, 2

# spec: Parameters — action costs, in milliseconds, before LATENCY is added
COST = {REVEAL: 30, FLAG: 30, CHORD: 60}

# spec: Rewards — win bonus and death penalty
WIN_BONUS = 1_000_000
DEATH_PENALTY = 1_000_000


@dataclass
class State:
    t: int  # in-episode step counter (RNG draws are keyed on it)
    mines: list[int]
    revealed: list[int]
    flagged: list[int]
    generated: int
    first_cell: int
    dead: int
    won: int
    time_ms: int
    #: Not part of the serialized state: the per-cell keys drawn at reset. They
    #: are a property of the episode's seed, reproduced identically by the
    #: batched implementation from the same (key, step 0, slot, index), so they
    #: never need to travel in a trajectory.
    keys: list[int] = field(default_factory=list)


class MinesweeperReference(ReferenceEnv):
    def __init__(self, H=H_DEFAULT, W=W_DEFAULT, M=M_DEFAULT,
                 latency=LATENCY_DEFAULT, horizon=HORIZON_DEFAULT):
        # spec: Parameters — M <= H*W - 9 so a full 3x3 exclusion always fits
        if M > H * W - 9:
            raise ValueError(f"M={M} exceeds H*W-9={H * W - 9}")
        self.H = H
        self.W = W
        self.M = M
        self.latency = latency
        self.horizon = horizon
        self.K = H * W

    # -- geometry ------------------------------------------------------------

    def neighbours(self, cell: int) -> list[int]:
        """The <=8 neighbours of `cell`, in ascending cell-index order.

        spec: Actions — chord reveals neighbours in ascending cell-index order,
        so this helper returns them already sorted and every caller inherits
        that order.
        """
        r = cell // self.W
        c = cell % self.W
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    out.append(rr * self.W + cc)
        out.sort()
        return out

    def count(self, state: State, cell: int) -> int:
        # spec: State space — counts are derived from mines, never stored
        n = 0
        for j in self.neighbours(cell):
            if state.mines[j] == 1:
                n += 1
        return n

    # -- reset ---------------------------------------------------------------

    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)
        # spec: Reset — H*W draws from MINE_KEYS at step 0, index = cell. The
        # board is NOT laid out yet; only the keys that will decide it.
        keys = []
        for cell in range(self.K):
            # spec: RNG slots — low 32 bits, identical in both backends
            keys.append(rng.draw_bits(self.key, 0, Slots.MINE_KEYS, cell) & 0xFFFFFFFF)
        self.state = State(
            t=0,
            mines=[0] * self.K,
            revealed=[0] * self.K,
            flagged=[0] * self.K,
            generated=0,
            first_cell=-1,
            dead=0,
            won=0,
            time_ms=0,
            keys=keys,
        )
        return self.state

    def place_mines(self, state: State, cell: int) -> None:
        """spec: Reset — Mine placement, at the first reveal on `cell`."""
        # 1. excluded = the closed 3x3 neighbourhood of `cell`
        excluded = set(self.neighbours(cell))
        excluded.add(cell)
        # 2. eligible = every cell not excluded
        eligible = []
        for j in range(self.K):
            if j not in excluded:
                eligible.append(j)
        # 3. the M eligible cells with the smallest keys, ties by smaller index
        eligible.sort(key=lambda j: (state.keys[j], j))
        for j in eligible[:self.M]:
            state.mines[j] = 1

    # -- reveal --------------------------------------------------------------

    def do_reveal(self, state: State, cell: int) -> None:
        """spec: Actions — Reveal sequence on `cell`.

        The caller is responsible for the precondition (not revealed, not
        flagged); a chord checks it per neighbour before calling.
        """
        # 1. first reveal lays out the board
        if state.generated == 0:
            self.place_mines(state, cell)
            state.generated = 1
            state.first_cell = cell
        # 2. reveal it
        state.revealed[cell] = 1
        # 3. a mine ends the episode
        if state.mines[cell] == 1:
            state.dead = 1
            return
        # 4. a zero cell floods, to a fixpoint
        if self.count(state, cell) == 0:
            changed = True
            while changed:
                changed = False
                for j in range(self.K):
                    if state.revealed[j] == 0:
                        continue
                    if self.count(state, j) != 0:
                        continue
                    for k in self.neighbours(j):
                        if state.flagged[k] == 1:
                            continue
                        if state.revealed[k] == 0:
                            state.revealed[k] = 1
                            changed = True

    # -- step ----------------------------------------------------------------

    def step(self, action) -> tuple[State, float, bool, dict]:
        state = self.state
        action = int(action)
        # spec: Actions — kind = a // K, cell = a % K
        kind = action // self.K
        cell = action % self.K

        dead_before = state.dead
        won_before = state.won

        if kind == FLAG:
            # spec: Actions — flag: no-op on a revealed cell, else toggle
            if state.revealed[cell] == 0:
                state.flagged[cell] = 1 - state.flagged[cell]

        elif kind == REVEAL:
            # spec: Actions — reveal: no-op if already revealed or flagged
            if state.revealed[cell] == 0 and state.flagged[cell] == 0:
                self.do_reveal(state, cell)

        elif kind == CHORD:
            # spec: Actions — chord applies only when the cell is revealed,
            # its count is positive, and its flagged neighbours equal it
            if state.revealed[cell] == 1:
                n = self.count(state, cell)
                if n > 0:
                    flags = 0
                    for j in self.neighbours(cell):
                        if state.flagged[j] == 1:
                            flags += 1
                    if flags == n:
                        # ascending cell-index order, per spec
                        for j in self.neighbours(cell):
                            if state.flagged[j] == 1:
                                continue
                            if state.revealed[j] == 1:
                                continue
                            self.do_reveal(state, j)

        # spec: Termination — T2, a win needs a generated board
        if state.generated == 1 and state.dead == 0:
            unrevealed_safe = 0
            for j in range(self.K):
                if state.mines[j] == 0 and state.revealed[j] == 0:
                    unrevealed_safe += 1
            if unrevealed_safe == 0:
                state.won = 1

        # spec: Parameters — one step advances the clock by COST[kind]+LATENCY
        step_cost = COST[kind] + self.latency
        state.time_ms += step_cost

        # spec: Rewards
        dead_now = 1 if (state.dead == 1 and dead_before == 0) else 0
        won_now = 1 if (state.won == 1 and won_before == 0) else 0
        reward = float(-step_cost + WIN_BONUS * won_now - DEATH_PENALTY * dead_now)

        state.t += 1

        # spec: Termination — T1 dead, T2 won, T3 step cap
        terminated = bool(state.dead == 1 or state.won == 1
                          or state.t >= self.horizon)
        return state, reward, terminated, {}

    # -- observation ---------------------------------------------------------

    def observe(self, state: State):
        """spec: Observations — [11, H, W] float32, built as ints then cast."""
        planes = [[[0] * self.W for _ in range(self.H)] for _ in range(11)]
        for j in range(self.K):
            r = j // self.W
            c = j % self.W
            if state.revealed[j] == 1:
                planes[2 + self.count(state, j)][r][c] = 1
            elif state.flagged[j] == 1:
                planes[1][r][c] = 1
            else:
                planes[0][r][c] = 1
        # cast to float32 once, at the end
        return np.array(planes, dtype=np.float32)

    # -- serialization -------------------------------------------------------

    def to_json(self, state: State) -> dict:
        # schema.json $defs/state — field names and integer types exactly
        return {
            "mines": [int(v) for v in state.mines],
            "revealed": [int(v) for v in state.revealed],
            "flagged": [int(v) for v in state.flagged],
            "generated": int(state.generated),
            "first_cell": int(state.first_cell),
            "dead": int(state.dead),
            "won": int(state.won),
            "time_ms": int(state.time_ms),
            "t": int(state.t),
        }

    def from_json(self, obj: dict) -> State:
        # exact inverse of to_json. `keys` is not serialized: it is a pure
        # function of the episode seed, so a replayed state re-derives it from
        # self.key rather than carrying it around.
        keys = []
        for cell in range(self.K):
            # spec: RNG slots — low 32 bits, identical in both backends
            keys.append(rng.draw_bits(self.key, 0, Slots.MINE_KEYS, cell) & 0xFFFFFFFF)
        return State(
            t=int(obj["t"]),
            mines=[int(v) for v in obj["mines"]],
            revealed=[int(v) for v in obj["revealed"]],
            flagged=[int(v) for v in obj["flagged"]],
            generated=int(obj["generated"]),
            first_cell=int(obj["first_cell"]),
            dead=int(obj["dead"]),
            won=int(obj["won"]),
            time_ms=int(obj["time_ms"]),
            keys=keys,
        )
