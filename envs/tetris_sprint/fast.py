"""Batched torch implementation of tetris_sprint.

Written from spec.md, NOT from reference.py. Branch-free step path: every
data-dependent decision is a mask, every bounded loop has a fixed trip count
(9 for DAS travel, 5 for kicks, 5 for the physics phase), so torch.compile
can capture one graph.

All integer arithmetic end to end; the only float op is the final cast of the
0/1 observation planes to float32, exactly as the spec orders.
"""

from __future__ import annotations

import torch

from simulacrum import BatchedEnv, invariant, rng

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

_MASK63 = (1 << 63) - 1

# spec: Piece shapes — CELLS[piece][rot] as (dx, dy) with dy up.
_CELLS = [
    # I
    [[(0, 2), (1, 2), (2, 2), (3, 2)],
     [(2, 0), (2, 1), (2, 2), (2, 3)],
     [(0, 1), (1, 1), (2, 1), (3, 1)],
     [(1, 0), (1, 1), (1, 2), (1, 3)]],
    # O
    [[(1, 1), (2, 1), (1, 2), (2, 2)]] * 4,
    # T
    [[(0, 1), (1, 1), (2, 1), (1, 2)],
     [(1, 0), (1, 1), (2, 1), (1, 2)],
     [(1, 0), (0, 1), (1, 1), (2, 1)],
     [(1, 0), (0, 1), (1, 1), (1, 2)]],
    # J
    [[(0, 1), (1, 1), (2, 1), (0, 2)],
     [(1, 0), (1, 1), (1, 2), (2, 2)],
     [(2, 0), (0, 1), (1, 1), (2, 1)],
     [(0, 0), (1, 0), (1, 1), (1, 2)]],
    # L
    [[(0, 1), (1, 1), (2, 1), (2, 2)],
     [(1, 0), (2, 0), (1, 1), (1, 2)],
     [(0, 0), (0, 1), (1, 1), (2, 1)],
     [(1, 0), (1, 1), (0, 2), (1, 2)]],
    # S
    [[(0, 1), (1, 1), (1, 2), (2, 2)],
     [(2, 0), (1, 1), (2, 1), (1, 2)],
     [(0, 0), (1, 0), (1, 1), (2, 1)],
     [(1, 0), (0, 1), (1, 1), (0, 2)]],
    # Z
    [[(1, 1), (2, 1), (0, 2), (1, 2)],
     [(1, 0), (1, 1), (2, 1), (2, 2)],
     [(1, 0), (2, 0), (0, 1), (1, 1)],
     [(0, 0), (0, 1), (1, 1), (1, 2)]],
]

# spec: Spawn placement — (x, y) per piece id
_SPAWN = [(3, 18), (3, 19), (3, 19), (3, 19), (3, 19), (3, 19), (3, 19)]

# spec: SRS kick tables — (kx, ky), ky up, tried strictly left to right.
_KICKS_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
_KICKS_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}


def _kick_tensor():
    """[7, 4, 2, 5, 2]: piece, from_rot, direction (0=cw, 1=ccw), try, (kx, ky).

    O gets all-zero kicks: its rotation is the identity on cells, so the
    (0,0) try always succeeds at the current position (spec: kick tables, O).
    """
    table = torch.zeros(7, 4, 2, 5, 2, dtype=torch.int64)
    for piece in range(7):
        for from_rot in range(4):
            for direction, to_rot in ((0, (from_rot + 1) % 4), (1, (from_rot + 3) % 4)):
                if piece == 1:  # O
                    kicks = [(0, 0)] * 5
                elif piece == 0:  # I
                    kicks = _KICKS_I[(from_rot, to_rot)]
                else:
                    kicks = _KICKS_JLSTZ[(from_rot, to_rot)]
                for j, (kx, ky) in enumerate(kicks):
                    table[piece, from_rot, direction, j, 0] = kx
                    table[piece, from_rot, direction, j, 1] = ky
    return table


class TetrisSprintBatched(BatchedEnv):
    def __init__(self, n: int, *, latency: int = 100, **kwargs):
        super().__init__(n, **kwargs)
        # spec: Parameters — LATENCY dial, [17, 2000]
        if not 17 <= latency <= 2000:
            raise ValueError(f"LATENCY must be in [17, 2000], got {latency}")
        self.latency = latency
        dev = self.device
        self.CELLS = torch.tensor(_CELLS, dtype=torch.int64, device=dev)  # [7,4,4,2]
        self.SPAWN = torch.tensor(_SPAWN, dtype=torch.int64, device=dev)  # [7,2]
        self.KICKS = _kick_tensor().to(dev)                               # [7,4,2,5,2]
        self._arange_n = torch.arange(n, dtype=torch.int64, device=dev)
        self._arange7 = torch.arange(7, dtype=torch.int64, device=dev)
        self._arange24 = torch.arange(24, dtype=torch.int64, device=dev)

    # -- geometry (spec: Actions — "valid", ghost distance) -----------------

    def _cells_at(self, piece, rot, x, y):
        """Rows and columns [N, 4] of the four cells of a piece placement."""
        cells = self.CELLS[piece, rot]  # [N, 4, 2]
        cols = x.unsqueeze(1) + cells[..., 0]
        rows = y.unsqueeze(1) + cells[..., 1]
        return rows, cols

    def _valid(self, piece, rot, x, y):
        rows, cols = self._cells_at(piece, rot, x, y)
        in_bounds = (cols >= 0) & (cols <= 9) & (rows >= 0) & (rows <= 23)
        # Clamp BEFORE the gather: masking evaluates both sides, and an
        # out-of-range index would fault (vectorize footgun 1).
        occ = self.board[self._arange_n.unsqueeze(1),
                         rows.clamp(0, 23), cols.clamp(0, 9)] != 0
        return (in_bounds & ~occ).all(dim=1)

    def _can_fall(self):
        return self._valid(self.piece, self.rot, self.x, self.y - 1)

    def _ghost_d(self):
        # Once an instance cannot fall at its current d the same check repeats
        # and stays False — the loop is monotone, so 24 fixed trips are exact.
        d = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        for _ in range(24):
            can = self._valid(self.piece, self.rot, self.x, self.y - d - 1)
            d = d + can.long()
        return d

    # -- the only random primitive (spec: Bag draw) -------------------------

    def _bag_draw(self, steps, slot, index, apply_mask):
        """Draw for the WHOLE batch (discarded draws are safe with a
        counter-based RNG — footgun 2); mutate `bag` only under apply_mask.
        Returns the drawn piece id [N]."""
        bits7 = (self.bag.unsqueeze(1) >> self._arange7) & 1          # [N,7]
        n_remaining = bits7.sum(dim=1)                                # >=1 (I5)
        bits = rng.draw_bits_torch(self.keys, steps, int(slot), index)
        # Match scalar draw_randint exactly: logical shift right 1, then mod.
        r = ((bits >> 1) & _MASK63) % n_remaining
        csum = bits7.cumsum(dim=1)
        match = (csum == (r + 1).unsqueeze(1)) & (bits7 == 1)
        drawn = match.long().argmax(dim=1)
        new_bag = self.bag & ~torch.bitwise_left_shift(
            torch.ones_like(drawn), drawn)
        new_bag = torch.where(new_bag == 0, torch.full_like(new_bag, 127), new_bag)
        self.bag = torch.where(apply_mask, new_bag, self.bag)
        return drawn

    # -- lock sequence (spec: Lock sequence; callers drop first) ------------

    def _lock_sequence(self, mask, consumed):
        """Merge, clear, spawn for instances in `mask` (already at rest).
        Returns (n_cleared [N], topped [N], terminated [N], consumed [N])."""
        rows, cols = self._cells_at(self.piece, self.rot, self.x, self.y)
        rows_c, cols_c = rows.clamp(0, 23), cols.clamp(0, 9)
        bidx = self._arange_n.unsqueeze(1).expand(-1, 4)

        # spec: Lock sequence 2 — merge. For unmasked instances this writes
        # each cell's existing value back (max with 0): a no-op.
        add = mask.unsqueeze(1).to(self.board.dtype).expand(-1, 4)
        self.board[bidx, rows_c, cols_c] = torch.maximum(
            self.board[bidx, rows_c, cols_c], add)

        # spec: Lock sequence 3 — lock-out (T3)
        lockout = mask & (rows >= 20).all(dim=1)
        m2 = mask & ~lockout

        # spec: Lock sequence 4 — clear full rows, stable shift down
        full = (self.board.sum(dim=2) == 10) & m2.unsqueeze(1)        # [N,24]
        n = full.sum(dim=1)
        order = torch.argsort(full.long(), dim=1, stable=True)
        compacted = torch.gather(
            self.board, 1, order.unsqueeze(-1).expand(-1, -1, 10))
        keep = self._arange24.unsqueeze(0) < (24 - n).unsqueeze(1)
        compacted = compacted * keep.unsqueeze(-1).to(self.board.dtype)
        self.board = torch.where(m2.view(-1, 1, 1), compacted, self.board)
        self.lines = torch.where(
            m2, torch.clamp(self.lines + n, max=43), self.lines)

        # spec: Lock sequence 5 — success (T1)
        success = m2 & (self.lines >= 40)
        spawn_m = m2 & ~success

        # spec: Lock sequence 6
        self.hold_used = torch.where(
            spawn_m, torch.zeros_like(self.hold_used), self.hold_used)

        # spec: Lock sequence 7 — spawn from queue, BAG_REFILL at index=consumed
        drawn = self._bag_draw(self.t, Slots.BAG_REFILL, consumed, spawn_m)
        self.piece = torch.where(spawn_m, self.queue[:, 0], self.piece)
        shifted = torch.cat([self.queue[:, 1:], drawn.unsqueeze(1)], dim=1)
        self.queue = torch.where(spawn_m.unsqueeze(1), shifted, self.queue)
        self.rot = torch.where(spawn_m, torch.zeros_like(self.rot), self.rot)
        sp = self.SPAWN[self.piece]
        self.x = torch.where(spawn_m, sp[:, 0], self.x)
        self.y = torch.where(spawn_m, sp[:, 1], self.y)
        zero = torch.zeros_like(self.fall_ms)
        self.fall_ms = torch.where(spawn_m, zero, self.fall_ms)
        self.lock_ms = torch.where(spawn_m, zero, self.lock_ms)
        self.lock_resets = torch.where(spawn_m, zero, self.lock_resets)
        consumed = consumed + spawn_m.long()

        # spec: Lock sequence 8 — block-out (T2)
        blockout = spawn_m & ~self._valid(self.piece, self.rot, self.x, self.y)

        topped = lockout | blockout
        terminated = lockout | success | blockout
        return n, topped, terminated, consumed

    # -- BatchedEnv hooks ---------------------------------------------------

    def _reset_instances(self, mask):
        dev = self.device
        n = self.n
        if not hasattr(self, "board"):
            self.board = torch.zeros(n, 24, 10, dtype=torch.int8, device=dev)
            z = torch.zeros(n, dtype=torch.int64, device=dev)
            self.piece, self.rot = z.clone(), z.clone()
            self.x, self.y = z.clone(), z.clone()
            self.hold, self.hold_used = z.clone(), z.clone()
            self.queue = torch.zeros(n, 5, dtype=torch.int64, device=dev)
            self.bag, self.lines = z.clone(), z.clone()
            self.time_ms, self.fall_ms = z.clone(), z.clone()
            self.lock_ms, self.lock_resets = z.clone(), z.clone()
            self.just_terminated = torch.zeros(n, dtype=torch.bool, device=dev)

        # spec: Reset 1 — computed fresh for everyone, applied under mask
        zero = torch.zeros(n, dtype=torch.int64, device=dev)
        self.board = torch.where(
            mask.view(-1, 1, 1), torch.zeros_like(self.board), self.board)
        self.lines = torch.where(mask, zero, self.lines)
        self.hold = torch.where(mask, torch.full_like(self.hold, -1), self.hold)
        self.hold_used = torch.where(mask, zero, self.hold_used)
        self.time_ms = torch.where(mask, zero, self.time_ms)
        self.fall_ms = torch.where(mask, zero, self.fall_ms)
        self.lock_ms = torch.where(mask, zero, self.lock_ms)
        self.lock_resets = torch.where(mask, zero, self.lock_resets)
        self.just_terminated = self.just_terminated & ~mask

        # spec: Reset 2 — six BAG_RESET draws at step 0, index 0..5. The k-th
        # draw is uniform over 7-k pieces: a SCALAR modulus, since a resetting
        # instance has dealt exactly k pieces from a full bag.
        bag = torch.full((n,), 127, dtype=torch.int64, device=dev)
        dealt = []
        for k in range(6):
            bits7 = (bag.unsqueeze(1) >> self._arange7) & 1
            bits = rng.draw_bits_torch(self.keys, 0, int(Slots.BAG_RESET), k)
            r = ((bits >> 1) & _MASK63) % (7 - k)
            csum = bits7.cumsum(dim=1)
            match = (csum == (r + 1).unsqueeze(1)) & (bits7 == 1)
            drawn = match.long().argmax(dim=1)
            bag = bag & ~torch.bitwise_left_shift(torch.ones_like(drawn), drawn)
            dealt.append(drawn)
        self.piece = torch.where(mask, dealt[0], self.piece)
        fresh_queue = torch.stack(dealt[1:], dim=1)
        self.queue = torch.where(mask.unsqueeze(1), fresh_queue, self.queue)
        self.bag = torch.where(mask, bag, self.bag)

        # spec: Reset 3
        self.rot = torch.where(mask, zero, self.rot)
        sp = self.SPAWN[self.piece]
        self.x = torch.where(mask, sp[:, 0], self.x)
        self.y = torch.where(mask, sp[:, 1], self.y)

    def _step_impl(self, actions):
        a = actions.to(torch.int64)
        zero = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        consumed = zero.clone()
        n_cleared = zero.clone()
        topped = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        terminated = torch.zeros(self.n, dtype=torch.bool, device=self.device)

        # ---- action phase (spec: Step algorithm 2) ----
        # taps
        tap_dx = (a == 1).long() - (a == 0).long()
        tap_m = a <= 1
        moved_tap = tap_m & self._valid(self.piece, self.rot, self.x + tap_dx, self.y)
        self.x = torch.where(moved_tap, self.x + tap_dx, self.x)

        # das: fixed 9-trip loop (max travel across a 10-wide board)
        das_m = (a == 2) | (a == 3)
        das_dx = (a == 3).long() - (a == 2).long()
        das_cells = zero.clone()
        for _ in range(9):
            can = das_m & self._valid(self.piece, self.rot, self.x + das_dx, self.y)
            self.x = torch.where(can, self.x + das_dx, self.x)
            das_cells = das_cells + can.long()

        # rotations: cw, ccw (5 kicks in spec order), then 180 at (0,0) only
        rot_success = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for direction, rot_m in ((0, a == 4), (1, a == 5)):
            to = (self.rot + (1 if direction == 0 else 3)) % 4
            kicks = self.KICKS[self.piece, self.rot, direction]  # [N,5,2]
            done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
            for j in range(5):
                kx, ky = kicks[:, j, 0], kicks[:, j, 1]
                ok = rot_m & ~done & self._valid(
                    self.piece, to, self.x + kx, self.y + ky)
                self.x = torch.where(ok, self.x + kx, self.x)
                self.y = torch.where(ok, self.y + ky, self.y)
                self.rot = torch.where(ok, to, self.rot)
                done = done | ok
            rot_success = rot_success | done
        to180 = (self.rot + 2) % 4
        ok180 = (a == 6) & self._valid(self.piece, to180, self.x, self.y)
        self.rot = torch.where(ok180, to180, self.rot)
        rot_success = rot_success | ok180

        # spec: Actions — lock-delay reset on action success (ids 0-6)
        success = moved_tap | (das_cells > 0) | rot_success
        reset_cond = success & (self.lock_ms > 0) & (self.lock_resets < RESETS_MAX)
        self.lock_ms = torch.where(reset_cond, zero, self.lock_ms)
        self.lock_resets = self.lock_resets + reset_cond.long()

        # soft / hard drop share the ghost computed after all movement
        gd = self._ghost_d()
        drop_m = (a == 7) | (a == 8)
        self.y = torch.where(drop_m, self.y - gd, self.y)

        # hard drop locks (spec: Lock sequence, drop already applied)
        hard_m = a == 8
        n1, topped1, term1, consumed = self._lock_sequence(hard_m, consumed)
        n_cleared = n_cleared + n1
        topped = topped | topped1
        terminated = terminated | term1

        # hold (spec: Hold sequence); disjoint from hard_m by action id
        hold_m = (a == 9) & (self.hold_used == 0)
        first = hold_m & (self.hold == -1)
        drawn = self._bag_draw(self.t, Slots.BAG_REFILL, consumed, first)
        old_piece = self.piece
        self.piece = torch.where(first, self.queue[:, 0],
                                 torch.where(hold_m, self.hold, self.piece))
        self.hold = torch.where(hold_m, old_piece, self.hold)
        shifted = torch.cat([self.queue[:, 1:], drawn.unsqueeze(1)], dim=1)
        self.queue = torch.where(first.unsqueeze(1), shifted, self.queue)
        consumed = consumed + first.long()
        self.rot = torch.where(hold_m, zero, self.rot)
        sp = self.SPAWN[self.piece]
        self.x = torch.where(hold_m, sp[:, 0], self.x)
        self.y = torch.where(hold_m, sp[:, 1], self.y)
        self.fall_ms = torch.where(hold_m, zero, self.fall_ms)
        self.lock_ms = torch.where(hold_m, zero, self.lock_ms)
        self.lock_resets = torch.where(hold_m, zero, self.lock_resets)
        hold_blocked = hold_m & ~self._valid(self.piece, self.rot, self.x, self.y)
        topped = topped | hold_blocked
        terminated = terminated | hold_blocked
        self.hold_used = torch.where(
            hold_m & ~hold_blocked, torch.ones_like(self.hold_used), self.hold_used)

        # spec: Actions — leaving the ground discards lock delay (no reset)
        airborne_now = self._can_fall() & ~terminated
        self.lock_ms = torch.where(airborne_now, zero, self.lock_ms)

        # ---- cost (spec: Step algorithm 3) ----
        table_cost = torch.where(
            das_m, DAS + torch.clamp(das_cells - 1, min=0) * ARR, zero)
        cost = table_cost + self.latency

        # ---- physics phase (spec: Step algorithm 4) ----
        # Fixed 5 trips: at most 3 gravity falls fit in a 2464 ms budget at
        # G=1000, plus one resting trip, plus slack. An instance that lands in
        # trip i is handled by the resting branch in trip i+1.
        remaining = torch.where(terminated, zero, cost)
        auto_lock = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(5):
            can_fall = self._can_fall()
            live = (remaining > 0) & ~auto_lock
            airborne = can_fall & live
            need_f = G - self.fall_ms
            do_fall = airborne & (remaining >= need_f)
            self.y = torch.where(do_fall, self.y - 1, self.y)
            self.lock_ms = torch.where(do_fall, zero, self.lock_ms)
            self.fall_ms = torch.where(
                do_fall, zero,
                torch.where(airborne, self.fall_ms + remaining, self.fall_ms))
            remaining = torch.where(
                do_fall, remaining - need_f,
                torch.where(airborne, zero, remaining))
            resting = ~can_fall & live
            self.fall_ms = torch.where(resting, zero, self.fall_ms)
            need_l = LOCK - self.lock_ms
            do_lock = resting & (remaining >= need_l)
            self.lock_ms = torch.where(
                resting & ~do_lock, self.lock_ms + remaining, self.lock_ms)
            remaining = torch.where(resting & ~do_lock, zero, remaining)
            auto_lock = auto_lock | do_lock
        n2, topped2, term2, consumed = self._lock_sequence(auto_lock, consumed)
        n_cleared = n_cleared + n2
        topped = topped | topped2
        terminated = terminated | term2

        # ---- clock, reward, T4 (spec: Step algorithm 5-7) ----
        self.time_ms = self.time_ms + cost
        rewards = (CLEAR_BONUS * n_cleared - cost
                   - TOPOUT_PENALTY * topped.long()).to(torch.float32)
        terminated = terminated | (self.t + 1 >= HORIZON)
        self.just_terminated = terminated
        return rewards, terminated

    def observe(self):
        # spec: Observations — [N, 9, 24, 10], 0/1 planes, one final cast
        planes = torch.zeros(self.n, 9, 24, 10, dtype=torch.int8, device=self.device)
        planes[:, 0] = self.board
        bidx = self._arange_n.unsqueeze(1).expand(-1, 4)
        one = torch.ones(self.n, 4, dtype=torch.int8, device=self.device)

        rows, cols = self._cells_at(self.piece, self.rot, self.x, self.y)
        planes[bidx, 1, rows.clamp(0, 23), cols.clamp(0, 9)] = one

        gy = self.y - self._ghost_d()
        g_rows, g_cols = self._cells_at(self.piece, self.rot, self.x, gy)
        planes[bidx, 2, g_rows.clamp(0, 23), g_cols.clamp(0, 9)] = one

        has_hold = self.hold >= 0
        hold_c = self.hold.clamp(min=0)
        h_sp = self.SPAWN[hold_c]
        h_rows = h_sp[:, 1].unsqueeze(1) + self.CELLS[hold_c, 0][..., 1]
        h_cols = h_sp[:, 0].unsqueeze(1) + self.CELLS[hold_c, 0][..., 0]
        planes[bidx, 3, h_rows.clamp(0, 23), h_cols.clamp(0, 9)] = (
            has_hold.unsqueeze(1).to(torch.int8).expand(-1, 4))

        for k in range(5):
            q = self.queue[:, k]
            q_sp = self.SPAWN[q]
            q_rows = q_sp[:, 1].unsqueeze(1) + self.CELLS[q, 0][..., 1]
            q_cols = q_sp[:, 0].unsqueeze(1) + self.CELLS[q, 0][..., 0]
            planes[bidx, 4 + k, q_rows.clamp(0, 23), q_cols.clamp(0, 9)] = one

        return planes.to(torch.float32)

    def state_tensors(self):
        return {
            "t": self.t,
            "board": self.board,
            "piece": self.piece,
            "rot": self.rot,
            "x": self.x,
            "y": self.y,
            "hold": self.hold,
            "hold_used": self.hold_used,
            "queue": self.queue,
            "bag": self.bag,
            "lines": self.lines,
            "time_ms": self.time_ms,
            "fall_ms": self.fall_ms,
            "lock_ms": self.lock_ms,
            "lock_resets": self.lock_resets,
        }

    # -- invariants (spec: Invariants I1-I12) -------------------------------

    @invariant("I1_board_binary")
    def _i1(self):
        return ((self.board == 0) | (self.board == 1)).flatten(1).all(dim=1)

    @invariant("I2_active_piece_legal")
    def _i2(self):
        return self._valid(self.piece, self.rot, self.x, self.y) | self.just_terminated

    @invariant("I3_no_full_rows")
    def _i3(self):
        return (self.board.sum(dim=2) < 10).all(dim=1)

    @invariant("I4_line_counter")
    def _i4(self):
        bounded = (self.lines >= 0) & (self.lines <= 43)
        return bounded & ((self.lines < 40) | self.just_terminated)

    @invariant("I5_bag_never_empty")
    def _i5(self):
        return (self.bag >= 1) & (self.bag <= 127)

    @invariant("I6_id_ranges")
    def _i6(self):
        ok = (self.piece >= 0) & (self.piece <= 6)
        ok = ok & ((self.queue >= 0) & (self.queue <= 6)).all(dim=1)
        ok = ok & (self.hold >= -1) & (self.hold <= 6)
        ok = ok & ((self.hold_used == 0) | (self.hold_used == 1))
        return ok & (self.rot >= 0) & (self.rot <= 3)

    @invariant("I7_reset_support")
    def _i7(self):
        six = torch.bitwise_left_shift(torch.ones_like(self.piece), self.piece)
        for k in range(5):
            six = six | torch.bitwise_left_shift(
                torch.ones_like(self.piece), self.queue[:, k])
        popcount_six = ((six.unsqueeze(1) >> self._arange7) & 1).sum(dim=1)
        fresh = (self.board == 0).flatten(1).all(dim=1)
        fresh = fresh & (self.lines == 0) & (self.hold == -1) & (self.hold_used == 0)
        fresh = fresh & (self.time_ms == 0) & (self.fall_ms == 0)
        fresh = fresh & (self.lock_ms == 0) & (self.lock_resets == 0)
        fresh = fresh & (popcount_six == 6) & (self.bag == (127 & ~six))
        return (self.t != 0) | fresh

    @invariant("I8_counter_bound")
    def _i8(self):
        bounded = (self.t >= 0) & (self.t <= HORIZON)
        return bounded & ((self.t < HORIZON) | self.just_terminated)

    @invariant("I9_clock_bounds")
    def _i9(self):
        bounded = (self.time_ms >= 0) & (self.time_ms <= 3_000_000)
        return bounded & (self.time_ms <= self.t * 2464)

    @invariant("I10_gravity_residual")
    def _i10(self):
        bounded = (self.fall_ms >= 0) & (self.fall_ms <= 999)
        resting_ok = self._can_fall() | (self.fall_ms == 0) | self.just_terminated
        return bounded & resting_ok

    @invariant("I11_lock_residual")
    def _i11(self):
        bounded = (self.lock_ms >= 0) & (self.lock_ms <= 499)
        airborne_ok = ~self._can_fall() | (self.lock_ms == 0) | self.just_terminated
        return bounded & airborne_ok

    @invariant("I12_reset_cap")
    def _i12(self):
        return (self.lock_resets >= 0) & (self.lock_resets <= RESETS_MAX)
