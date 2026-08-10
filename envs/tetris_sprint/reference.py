"""Readable single-instance reference implementation of tetris_sprint.

Written from spec.md ONLY. Style: dataclass state, explicit ifs, no
vectorization, no premature abstraction, every rule traceable to a spec line.
All randomness via simulacrum.rng scalar draws with slots from
tetris_sprint.Slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulacrum import ReferenceEnv, rng

from tetris_sprint import Slots

# spec: Parameters — frozen constants
G = 1000
LOCK = 500
RESETS_MAX = 15
DAS = 167
ARR = 33
CLEAR_BONUS = 10_000
TOPOUT_PENALTY = 200_000
HORIZON = 1200

# spec: State space — piece ids
I, O, T, J, L, S, Z = 0, 1, 2, 3, 4, 5, 6

# spec: Actions — ids
TAP_LEFT, TAP_RIGHT, DAS_LEFT, DAS_RIGHT = 0, 1, 2, 3
ROTATE_CW, ROTATE_CCW, ROTATE_180 = 4, 5, 6
SOFT_DROP, HARD_DROP, HOLD = 7, 8, 9

# spec: Piece shapes — CELLS[piece][rot] = set of four (dx, dy), dy up
CELLS = {
    I: [
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    O: [
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
    ],
    T: [
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    J: [
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
    L: [
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    S: [
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(1, 0), (0, 1), (1, 1), (0, 2)],
    ],
    Z: [
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
}

# spec: Spawn placement
SPAWN = {
    I: (3, 18),
    O: (3, 19),
    T: (3, 19),
    J: (3, 19),
    L: (3, 19),
    S: (3, 19),
    Z: (3, 19),
}

# spec: SRS kick tables — (kx, ky), ky up, tried strictly left to right.
KICKS_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
KICKS_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}


@dataclass
class State:
    # spec: State space — field per table row
    t: int
    time_ms: int
    board: list  # 24 rows x 10 cols, row 0 = bottom
    piece: int
    rot: int
    x: int
    y: int
    hold: int
    hold_used: int
    queue: list  # 5 entries
    bag: int
    lines: int
    fall_ms: int
    lock_ms: int
    lock_resets: int


class TetrisSprintReference(ReferenceEnv):
    def __init__(self, latency: int = 100):
        # spec: Parameters — LATENCY dial, [17, 2000]
        if not 17 <= latency <= 2000:
            raise ValueError(f"LATENCY must be in [17, 2000], got {latency}")
        self.latency = latency
        self.state = None
        self._terminated = True

    # -- geometry helpers (spec: Actions — "valid") -------------------------

    def _valid(self, board, piece, rot, x, y):
        for dx, dy in CELLS[piece][rot]:
            col = x + dx
            row = y + dy
            if col < 0 or col > 9:
                return False
            if row < 0 or row > 23:
                return False
            if board[row][col] != 0:
                return False
        return True

    def _ghost_d(self, s):
        # spec: Actions — ghost distance
        d = 0
        while self._valid(s.board, s.piece, s.rot, s.x, s.y - (d + 1)):
            d = d + 1
        return d

    # -- the only random primitive (spec: Bag draw) -------------------------

    def _bag_draw(self, s, step, slot, index):
        n = bin(s.bag).count("1")  # >= 1 by invariant I5
        r = rng.draw_randint(self.key, step, slot, n, index)
        seen = 0
        drawn = -1
        for p in range(7):
            if (s.bag >> p) & 1:
                if seen == r:
                    drawn = p
                    break
                seen = seen + 1
        s.bag = s.bag & ~(1 << drawn)
        if s.bag == 0:
            s.bag = 127
        return drawn

    # -- lock sequence (spec: Lock sequence) --------------------------------

    def _lock(self, s, k, consumed, drop_first):
        """Returns (n_cleared, consumed, topped, terminated)."""
        # spec: Lock sequence 1 — hard_drop drops first; auto-lock is at rest
        if drop_first:
            s.y = s.y - self._ghost_d(s)

        # spec: Lock sequence 2 — merge
        rows_touched = []
        for dx, dy in CELLS[s.piece][s.rot]:
            s.board[s.y + dy][s.x + dx] = 1
            rows_touched.append(s.y + dy)

        # spec: Lock sequence 3 — lock-out (T3): ALL four cells in rows >= 20
        if all(row >= 20 for row in rows_touched):
            return 0, consumed, True, True

        # spec: Lock sequence 4 — clear full rows, shift down
        kept = []
        for row in s.board:
            if sum(row) < 10:
                kept.append(row)
        n = 24 - len(kept)
        while len(kept) < 24:
            kept.append([0] * 10)
        s.board = kept
        s.lines = min(s.lines + n, 43)

        # spec: Lock sequence 5 — success (T1)
        if s.lines >= 40:
            return n, consumed, False, True

        # spec: Lock sequence 6
        s.hold_used = 0

        # spec: Lock sequence 7 — spawn from queue, refill via BAG_REFILL
        s.piece = s.queue[0]
        s.queue = s.queue[1:] + [self._bag_draw(s, k, Slots.BAG_REFILL, consumed)]
        consumed = consumed + 1
        s.rot = 0
        s.x, s.y = SPAWN[s.piece]
        s.fall_ms = 0
        s.lock_ms = 0
        s.lock_resets = 0

        # spec: Lock sequence 8 — block-out (T2)
        if not self._valid(s.board, s.piece, s.rot, s.x, s.y):
            return n, consumed, True, True

        return n, consumed, False, False

    # -- ReferenceEnv interface --------------------------------------------

    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)
        # spec: Reset 1
        s = State(
            t=0,
            time_ms=0,
            board=[[0] * 10 for _ in range(24)],
            piece=0,
            rot=0,
            x=0,
            y=0,
            hold=-1,
            hold_used=0,
            queue=[0] * 5,
            bag=127,
            lines=0,
            fall_ms=0,
            lock_ms=0,
            lock_resets=0,
        )
        # spec: Reset 2 — six BAG_RESET draws at step 0, index 0..5
        s.piece = self._bag_draw(s, 0, Slots.BAG_RESET, 0)
        for k in range(5):
            s.queue[k] = self._bag_draw(s, 0, Slots.BAG_RESET, k + 1)
        # spec: Reset 3
        s.rot = 0
        s.x, s.y = SPAWN[s.piece]
        self.state = s
        self._terminated = False
        return s

    def step(self, action) -> tuple[State, float, bool, dict]:
        assert not self._terminated, "step() on a terminated episode"
        s = self.state
        a = int(action)
        k = s.t  # pre-increment counter: RNG draws this step use it

        # spec: Step algorithm 1
        consumed = 0
        n_cleared = 0
        topped = False
        terminated = False
        das_cells = 0

        # spec: Step algorithm 2 — action phase (instantaneous)
        if a == TAP_LEFT or a == TAP_RIGHT:
            dx = -1 if a == TAP_LEFT else 1
            moved = False
            if self._valid(s.board, s.piece, s.rot, s.x + dx, s.y):
                s.x = s.x + dx
                moved = True
            self._maybe_reset_lock(s, moved)
        elif a == DAS_LEFT or a == DAS_RIGHT:
            dx = -1 if a == DAS_LEFT else 1
            while self._valid(s.board, s.piece, s.rot, s.x + dx, s.y):
                s.x = s.x + dx
                das_cells = das_cells + 1
            self._maybe_reset_lock(s, das_cells > 0)
        elif a == ROTATE_CW or a == ROTATE_CCW:
            to = (s.rot + 1) % 4 if a == ROTATE_CW else (s.rot + 3) % 4
            self._maybe_reset_lock(s, self._try_rotate(s, to, kicks=True))
        elif a == ROTATE_180:
            # spec: Actions — rotate_180 at offset (0,0) only
            to = (s.rot + 2) % 4
            self._maybe_reset_lock(s, self._try_rotate(s, to, kicks=False))
        elif a == SOFT_DROP:
            # spec: Actions — soft_drop drops to rest, never locks or resets
            s.y = s.y - self._ghost_d(s)
        elif a == HARD_DROP:
            n_cleared, consumed, topped, terminated = self._lock(
                s, k, consumed, drop_first=True)
        elif a == HOLD:
            # spec: Hold sequence
            if s.hold_used == 0:
                if s.hold == -1:
                    s.hold = s.piece
                    s.piece = s.queue[0]
                    s.queue = s.queue[1:] + [
                        self._bag_draw(s, k, Slots.BAG_REFILL, consumed)]
                    consumed = consumed + 1
                else:
                    s.hold, s.piece = s.piece, s.hold
                s.rot = 0
                s.x, s.y = SPAWN[s.piece]
                s.fall_ms = 0
                s.lock_ms = 0
                s.lock_resets = 0
                if not self._valid(s.board, s.piece, s.rot, s.x, s.y):
                    topped = True
                    terminated = True
                else:
                    s.hold_used = 1
            # hold_used == 1: no-op (spec: Hold sequence 1)

        # spec: Actions — leaving the ground discards lock delay (no reset)
        if not terminated and self._ghost_d(s) >= 1:
            s.lock_ms = 0

        # spec: Step algorithm 3 — cost
        if a == DAS_LEFT or a == DAS_RIGHT:
            table_cost = DAS + max(0, das_cells - 1) * ARR
        else:
            table_cost = 0
        cost = table_cost + self.latency

        # spec: Step algorithm 4 — physics phase, skipped if action terminated
        if not terminated:
            remaining = cost
            while remaining > 0:
                if self._ghost_d(s) >= 1:
                    # airborne: gravity accrues toward the next one-cell fall
                    need = G - s.fall_ms
                    if remaining >= need:
                        s.y = s.y - 1
                        s.fall_ms = 0
                        s.lock_ms = 0
                        remaining = remaining - need
                    else:
                        s.fall_ms = s.fall_ms + remaining
                        remaining = 0
                else:
                    # resting: lock delay accrues toward auto-lock
                    s.fall_ms = 0
                    need = LOCK - s.lock_ms
                    if remaining >= need:
                        n2, consumed, topped2, terminated = self._lock(
                            s, k, consumed, drop_first=False)
                        n_cleared = n_cleared + n2
                        topped = topped or topped2
                        break  # physics ends at the first auto-lock
                    else:
                        s.lock_ms = s.lock_ms + remaining
                        remaining = 0

        # spec: Step algorithm 5
        s.time_ms = s.time_ms + cost

        # spec: Step algorithm 6 — reward
        reward = float(CLEAR_BONUS * n_cleared - cost
                       - (TOPOUT_PENALTY if topped else 0))

        # spec: Step algorithm 7 — increment, then T4
        s.t = s.t + 1
        if s.t == HORIZON and not terminated:
            terminated = True

        self._terminated = terminated
        return s, reward, terminated, {}

    def _maybe_reset_lock(self, s, success):
        # spec: Actions — lock-delay reset on action success (ids 0-6)
        if success and s.lock_ms > 0:
            if s.lock_resets < RESETS_MAX:
                s.lock_ms = 0
                s.lock_resets = s.lock_resets + 1

    def _try_rotate(self, s, to, kicks):
        """Try a rotation per the SRS tables. Returns True on success."""
        if s.piece == O:
            # spec: kick tables — O rotation is the identity and "succeeds"
            s.rot = to
            return True
        if kicks:
            if s.piece == I:
                offsets = KICKS_I[(s.rot, to)]
            else:
                offsets = KICKS_JLSTZ[(s.rot, to)]
        else:
            offsets = [(0, 0)]
        for kx, ky in offsets:
            if self._valid(s.board, s.piece, to, s.x + kx, s.y + ky):
                s.rot = to
                s.x = s.x + kx
                s.y = s.y + ky
                return True
        return False

    def observe(self, state: State):
        # spec: Observations — [9, 24, 10] float32, 0/1 planes, single cast
        planes = np.zeros((9, 24, 10), dtype=np.uint8)
        for row in range(24):
            for col in range(10):
                planes[0][row][col] = state.board[row][col]
        for dx, dy in CELLS[state.piece][state.rot]:
            planes[1][state.y + dy][state.x + dx] = 1
        gy = state.y - self._ghost_d(state)
        for dx, dy in CELLS[state.piece][state.rot]:
            planes[2][gy + dy][state.x + dx] = 1
        if state.hold != -1:
            hx, hy = SPAWN[state.hold]
            for dx, dy in CELLS[state.hold][0]:
                planes[3][hy + dy][hx + dx] = 1
        for k in range(5):
            q = state.queue[k]
            qx, qy = SPAWN[q]
            for dx, dy in CELLS[q][0]:
                planes[4 + k][qy + dy][qx + dx] = 1
        return planes.astype(np.float32)

    def action_mask(self, state: State):
        # spec: Actions — hold masked iff hold_used == 1
        mask = np.ones(10, dtype=bool)
        if state.hold_used == 1:
            mask[HOLD] = False
        return mask

    def to_json(self, state: State) -> dict:
        return {
            "t": int(state.t),
            "board": [[int(c) for c in row] for row in state.board],
            "piece": int(state.piece),
            "rot": int(state.rot),
            "x": int(state.x),
            "y": int(state.y),
            "hold": int(state.hold),
            "hold_used": int(state.hold_used),
            "queue": [int(q) for q in state.queue],
            "bag": int(state.bag),
            "lines": int(state.lines),
            "time_ms": int(state.time_ms),
            "fall_ms": int(state.fall_ms),
            "lock_ms": int(state.lock_ms),
            "lock_resets": int(state.lock_resets),
        }

    def from_json(self, obj: dict) -> State:
        return State(
            t=int(obj["t"]),
            time_ms=int(obj["time_ms"]),
            board=[[int(c) for c in row] for row in obj["board"]],
            piece=int(obj["piece"]),
            rot=int(obj["rot"]),
            x=int(obj["x"]),
            y=int(obj["y"]),
            hold=int(obj["hold"]),
            hold_used=int(obj["hold_used"]),
            queue=[int(q) for q in obj["queue"]],
            bag=int(obj["bag"]),
            lines=int(obj["lines"]),
            fall_ms=int(obj["fall_ms"]),
            lock_ms=int(obj["lock_ms"]),
            lock_resets=int(obj["lock_resets"]),
        )
