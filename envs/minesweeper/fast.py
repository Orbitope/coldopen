"""Batched tensor implementation of minesweeper.

Written from spec.md ONLY — never from reference.py. Independent misreadings of
the spec are exactly what the differential test exists to catch, and copying
the reference's reading would defeat it.

Every per-instance branch in the spec becomes a mask here. The two places that
need care are stated in the spec and handled explicitly below: the flood must
run only for instances that just revealed a zero cell (a global fixpoint every
step would auto-reveal cells when a flag is *removed*, which the rules do not
do), and the mine keys are masked to 32 bits so the signed int64 tensor sorts
in the same order as the reference's unsigned Python ints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from simulacrum import BatchedEnv, invariant, rng

from minesweeper import Slots

# spec: Parameters — defaults (Expert)
H_DEFAULT = 16
W_DEFAULT = 30
M_DEFAULT = 99
LATENCY_DEFAULT = 200
HORIZON = 2000

# spec: Actions — kinds
REVEAL, FLAG, CHORD = 0, 1, 2

# spec: Parameters — action costs in ms, before LATENCY
COST_REVEAL, COST_FLAG, COST_CHORD = 30, 30, 60

# spec: Rewards
WIN_BONUS = 1_000_000
DEATH_PENALTY = 1_000_000

#: Larger than any 32-bit key, so excluded cells sort last. spec: Reset —
#: Mine placement.
KEY_SENTINEL = 1 << 32


def _geometry(H: int, W: int, device):
    """NEIGH[K, 8] neighbour indices and NVALID[K, 8] existence mask.

    The eight (dr, dc) offsets are emitted in the order that makes the
    resulting cell indices ascending — j-W-1, j-W, j-W+1, j-1, j+1, j+W-1,
    j+W, j+W+1 — so slot order *is* ascending cell-index order and the chord's
    ordering requirement (spec: Actions) is satisfied by construction.
    Invalid neighbours point at cell 0 and are masked; they are never read
    through NVALID, but they must be a legal index because `torch.where` and
    gathers evaluate both branches.
    """
    K = H * W
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neigh = torch.zeros(K, 8, dtype=torch.int64)
    valid = torch.zeros(K, 8, dtype=torch.bool)
    for j in range(K):
        r, c = j // W, j % W
        for s, (dr, dc) in enumerate(offsets):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                neigh[j, s] = rr * W + cc
                valid[j, s] = True
    return neigh.to(device), valid.to(device)


class MinesweeperBatched(BatchedEnv):
    def __init__(self, n, H=H_DEFAULT, W=W_DEFAULT, M=M_DEFAULT,
                 latency=LATENCY_DEFAULT, horizon=HORIZON, **kwargs):
        # spec: Parameters — M <= H*W - 9
        if M > H * W - 9:
            raise ValueError(f"M={M} exceeds H*W-9={H * W - 9}")
        self.H, self.W, self.M = H, W, M
        self.K = H * W
        self.latency = latency
        self.horizon = horizon
        super().__init__(n, **kwargs)
        self.NEIGH, self.NVALID = _geometry(H, W, self.device)

    # -- geometry helpers ----------------------------------------------------

    def _dilate(self, x):
        """[N,K] bool -> [N,K] bool: cells adjacent to a True cell."""
        return (x[:, self.NEIGH] & self.NVALID).any(dim=2)

    def _neighbour_sum(self, x):
        """[N,K] bool -> [N,K] int64: number of True neighbours."""
        return (x[:, self.NEIGH] & self.NVALID).sum(dim=2)

    def _counts(self):
        # spec: State space — counts are derived from mines, never stored
        return self._neighbour_sum(self.mines)

    @staticmethod
    def _at(field, cell):
        """field[n, cell[n]] for every instance."""
        return field.gather(1, cell.unsqueeze(1)).squeeze(1)

    # -- reset ---------------------------------------------------------------

    def _reset_instances(self, mask):
        n, K = self.n, self.K
        dev = self.device
        # spec: Reset — H*W draws from MINE_KEYS at step 0, index = cell.
        # Drawn for the whole batch and masked: the RNG is counter-based, so
        # discarded draws cannot desync anything.
        cells = torch.arange(K, device=dev).unsqueeze(0)
        keys = rng.draw_bits_torch(
            self.keys.unsqueeze(1), 0, int(Slots.MINE_KEYS), cells)
        # spec: RNG slots — low 32 bits. draw_bits_torch returns the same 64
        # bits reinterpreted as SIGNED int64, so an unmasked ascending sort
        # would disagree with the reference for every key with bit 63 set.
        keys = keys & 0xFFFFFFFF

        if not hasattr(self, "mines"):
            z = lambda: torch.zeros(n, K, dtype=torch.bool, device=dev)
            self.mines, self.revealed, self.flagged = z(), z(), z()
            self.mine_keys = torch.zeros(n, K, dtype=torch.int64, device=dev)
            self.generated = torch.zeros(n, dtype=torch.bool, device=dev)
            self.dead = torch.zeros(n, dtype=torch.bool, device=dev)
            self.won = torch.zeros(n, dtype=torch.bool, device=dev)
            self.first_cell = torch.full((n,), -1, dtype=torch.int64, device=dev)
            self.time_ms = torch.zeros(n, dtype=torch.int64, device=dev)

        m1 = mask.unsqueeze(1)
        blank = torch.zeros_like(self.mines)
        self.mines = torch.where(m1, blank, self.mines)
        self.revealed = torch.where(m1, blank, self.revealed)
        self.flagged = torch.where(m1, blank, self.flagged)
        self.mine_keys = torch.where(m1, keys, self.mine_keys)
        self.generated = torch.where(mask, torch.zeros_like(self.generated),
                                     self.generated)
        self.dead = torch.where(mask, torch.zeros_like(self.dead), self.dead)
        self.won = torch.where(mask, torch.zeros_like(self.won), self.won)
        self.first_cell = torch.where(mask, torch.full_like(self.first_cell, -1),
                                      self.first_cell)
        self.time_ms = torch.where(mask, torch.zeros_like(self.time_ms),
                                   self.time_ms)

    # -- step ----------------------------------------------------------------

    def _step_impl(self, actions):
        K = self.K
        actions = actions.to(torch.int64)
        # spec: Actions — kind = a // K, cell = a % K
        kind = actions // K
        cell = actions % K
        onehot = F.one_hot(cell, K).bool()

        dead_before = self.dead
        won_before = self.won

        is_reveal = kind == REVEAL
        is_flag = kind == FLAG
        is_chord = kind == CHORD

        # --- flag: no-op on a revealed cell, else toggle --------------------
        can_flag = is_flag & ~self._at(self.revealed, cell)
        self.flagged = self.flagged ^ (onehot & can_flag.unsqueeze(1))

        # --- reveal: no-op if already revealed or flagged -------------------
        rev_ok = (is_reveal & ~self._at(self.revealed, cell)
                  & ~self._at(self.flagged, cell))

        # spec: Reset — Mine placement, at the first reveal
        need_gen = rev_ok & ~self.generated
        # excluded = the closed 3x3 of `cell`; adjacency is symmetric, so a
        # cell j is in it iff j == cell or cell is a neighbour of j.
        excluded = onehot | self._dilate(onehot)
        masked_keys = torch.where(excluded,
                                  torch.full_like(self.mine_keys, KEY_SENTINEL),
                                  self.mine_keys)
        # stable sort => ties resolve to the smaller cell index, per spec
        order = torch.argsort(masked_keys, dim=1, stable=True)
        laid = torch.zeros_like(self.mines).scatter(
            1, order[:, :self.M], torch.ones_like(self.mines))
        self.mines = torch.where(need_gen.unsqueeze(1), laid, self.mines)
        self.first_cell = torch.where(need_gen, cell, self.first_cell)
        self.generated = self.generated | need_gen

        counts = self._counts()

        # --- chord ----------------------------------------------------------
        # spec: Actions — applies only when the cell is revealed, count > 0,
        # and flagged neighbours equal the count.
        around = self._dilate(onehot)                      # the <=8 neighbours
        n_at = self._at(counts, cell)
        flags_at = self._at(self._neighbour_sum(self.flagged), cell)
        chord_ok = (is_chord & self._at(self.revealed, cell)
                    & (n_at > 0) & (flags_at == n_at))
        chord_targets = (around & ~self.flagged & ~self.revealed
                         & chord_ok.unsqueeze(1))

        # --- apply reveals ---------------------------------------------------
        reveal_targets = (onehot & rev_ok.unsqueeze(1)) | chord_targets
        self.revealed = self.revealed | reveal_targets

        # a revealed mine kills; a chord reveals every target even if one of
        # them is a mine, so this is an `any` over the whole target set
        self.dead = self.dead | (reveal_targets & self.mines).any(dim=1)

        # --- flood ------------------------------------------------------------
        # spec: Actions — Reveal sequence step 4. Gated per instance: running
        # the fixpoint unconditionally would reveal cells whenever a flag is
        # REMOVED next to a revealed zero, which is not a reveal at all.
        #
        # `~self.mines` in both the gate and the seed is the Reveal sequence's
        # step 3 ("a mine ends the reveal — and stop") expressed as a mask: a
        # mine can have count 0, since count excludes the cell itself, and the
        # differential test caught this env flooding the ring around such a
        # mine. Note the flood is NOT gated on ~dead: the reference's chord
        # loop keeps revealing later targets after a mine, floods included, so
        # a dead instance can still legitimately flood from a non-mine zero
        # revealed by the same chord.
        zero_seeds = (counts == 0) & ~self.mines
        do_flood = (reveal_targets & zero_seeds).any(dim=1)
        flood_mask = do_flood.unsqueeze(1)
        for _ in range(self.K):
            frontier = (self._dilate(self.revealed & zero_seeds)
                        & ~self.flagged & flood_mask)
            grown = self.revealed | frontier
            changed = (grown != self.revealed).any()
            self.revealed = grown
            if not bool(changed):
                break

        # --- win --------------------------------------------------------------
        # spec: Termination — T2, only on a generated board
        all_safe_revealed = ~(~self.mines & ~self.revealed).any(dim=1)
        self.won = self.won | (self.generated & ~self.dead & all_safe_revealed)

        # --- clock and reward --------------------------------------------------
        cost = torch.where(is_reveal, torch.full_like(cell, COST_REVEAL),
                           torch.where(is_flag, torch.full_like(cell, COST_FLAG),
                                       torch.full_like(cell, COST_CHORD)))
        step_cost = cost + self.latency
        self.time_ms = self.time_ms + step_cost

        dead_now = self.dead & ~dead_before
        won_now = self.won & ~won_before
        reward = (-step_cost.to(torch.float32)
                  + WIN_BONUS * won_now.to(torch.float32)
                  - DEATH_PENALTY * dead_now.to(torch.float32))

        # spec: Termination — T1 dead, T2 won, T3 step cap (t increments after
        # this returns, so the cap compares against t+1)
        terminated = self.dead | self.won | ((self.t + 1) >= self.horizon)
        return reward, terminated

    # -- observation ---------------------------------------------------------

    def observe(self):
        """spec: Observations — [N, 11, H, W] float32, ints cast once."""
        counts = self._counts()
        planes = [(~self.revealed & ~self.flagged), self.flagged]
        for c in range(9):
            planes.append(self.revealed & (counts == c))
        stacked = torch.stack(planes, dim=1)          # [N, 11, K] bool
        return stacked.to(torch.float32).reshape(self.n, 11, self.H, self.W)

    # -- serialization -------------------------------------------------------

    def state_tensors(self):
        # schema.json $defs/state — integers, so masks are widened from bool
        return {
            "mines": self.mines.to(torch.int64),
            "revealed": self.revealed.to(torch.int64),
            "flagged": self.flagged.to(torch.int64),
            "generated": self.generated.to(torch.int64),
            "first_cell": self.first_cell,
            "dead": self.dead.to(torch.int64),
            "won": self.won.to(torch.int64),
            "time_ms": self.time_ms,
            "t": self.t,
        }

    # -- invariants (spec: Invariants) ---------------------------------------

    @invariant("I1_reveal_and_flag_disjoint")
    def _i1(self):
        return ~(self.revealed & self.flagged).any(dim=1)

    @invariant("I2_ungenerated_is_blank")
    def _i2(self):
        blank = (~self.mines.any(dim=1) & ~self.revealed.any(dim=1)
                 & (self.first_cell == -1))
        return self.generated | blank

    @invariant("I3_generated_has_M_mines")
    def _i3(self):
        ok = ((self.mines.sum(dim=1) == self.M)
              & (self.first_cell >= 0) & (self.first_cell < self.K))
        return ~self.generated | ok

    @invariant("I4_opening_guarantee")
    def _i4(self):
        safe_first = self.first_cell.clamp(min=0)
        return ~self.generated | (self._at(self._counts(), safe_first) == 0)

    @invariant("I5_alive_means_no_revealed_mine")
    def _i5(self):
        return self.dead | ~(self.revealed & self.mines).any(dim=1)

    @invariant("I6_dead_means_a_revealed_mine")
    def _i6(self):
        return ~self.dead | (self.revealed & self.mines).any(dim=1)

    @invariant("I7_won_means_all_safe_revealed")
    def _i7(self):
        all_safe = ~(~self.mines & ~self.revealed).any(dim=1)
        return ~self.won | (all_safe & self.generated)

    @invariant("I8_not_won_and_dead")
    def _i8(self):
        return ~(self.won & self.dead)

    @invariant("I9_fresh_episode_is_blank")
    def _i9(self):
        fresh = (~self.generated & (self.time_ms == 0)
                 & ~self.flagged.any(dim=1))
        return (self.t != 0) | fresh

    @invariant("I10_step_in_range")
    def _i10(self):
        return (self.t >= 0) & (self.t <= self.horizon)

    @invariant("I11_clock_in_range")
    def _i11(self):
        cap = self.horizon * (COST_CHORD + self.latency)
        return (self.time_ms >= 0) & (self.time_ms <= cap)

    @invariant("I12_flags_and_reveals_fit")
    def _i12(self):
        return (self.flagged.sum(dim=1)
                <= self.K - self.revealed.sum(dim=1))

    @invariant("I13_live_board_has_work_left")
    def _i13(self):
        unfinished = (~self.mines & ~self.revealed).any(dim=1)
        return ~(self.generated & ~self.dead & ~self.won) | unfinished
