"""Reverse curriculum for minesweeper: start near the end, walk backwards.

The exploration wall, stated with numbers: an expert win is ~381 correct
reveals with zero fatal ones, and uniformly random play dies in a handful of
clicks. The terminal bonus is effectively invisible to from-scratch
exploration — the same wall that held in tetris_sprint, where 3M DQN steps
never cleared a line.

The classic answer (Salimans & Chen 2018, reverse curriculum generation) is to
**reset the agent into states near the goal and move the start line backwards
as it succeeds**. An agent that first learns "finish a board with 3 safe cells
left" sees the win bonus immediately; once it wins those, it gets boards with
6 left, then 12, and the value function propagates backwards through states it
has actually mastered rather than never reaching the reward at all.

Two sources for the start states, one mechanism:

* **Synthetic** (implemented, usable today): generate a board with the env's
  own layout rule, reveal every safe cell except a random k, start there.
  No human data anywhere, so ladders trained this way remain cold-start
  legitimate and comparable in E4/E6.
* **Human replays** (interface here, data later): when scraped
  minesweeper.online replays exist, each click of a replay is a board state a
  real player actually visited. Feeding those as start states shapes the
  curriculum toward the human-visited manifold. **Contamination rules**: a
  ladder trained this way is a separate track, never the cold-start claim;
  replay players must be split from evaluation players; ids pseudonymised at
  ingest. See `coldopen/human/minesweeper_replays.py` for the format.

Design constraint: `envs/minesweeper/fast.py` is differentially validated and
is NOT touched. The curriculum lives in a subclass whose `_reset_instances`
first defers to the validated reset, then overwrites the masked instances'
tensors. Mine placement is re-derived here with the same argsort rule; a
parity test (`tests/test_ms_curriculum.py`) pins the two byte-for-byte so the
copy cannot drift silently.
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "envs"))

from minesweeper.fast import KEY_SENTINEL, MinesweeperBatched


def lay_mines(mine_keys, first_cell, neigh, nvalid, M):
    """The env's placement rule, re-derived for curriculum use.

    spec: Reset — Mine placement: exclude the closed 3x3 around `first_cell`,
    then the M smallest keys among the rest, ties to the smaller index (stable
    argsort). Byte-parity with fast.py's inline copy is pinned by
    tests/test_ms_curriculum.py::test_placement_parity — if either side
    changes, that test fails rather than the two drifting apart silently.
    """
    n, K = mine_keys.shape
    onehot = torch.nn.functional.one_hot(first_cell, K).bool()
    excluded = onehot | (onehot[:, neigh] & nvalid).any(dim=2)
    masked = torch.where(excluded,
                         torch.full_like(mine_keys, KEY_SENTINEL), mine_keys)
    order = torch.argsort(masked, dim=1, stable=True)
    mines = torch.zeros(n, K, dtype=torch.bool, device=mine_keys.device)
    return mines.scatter(1, order[:, :M], torch.ones_like(mines))


class SyntheticStates:
    """Near-finished boards from the env's own generator. Cold-start clean.

    Hides `k` randomly-chosen safe cells (k drawn per instance from
    [1, k_max]) and reveals the rest. The hidden set can be flood-inconsistent
    with real play — a hidden cell may border a revealed zero, which cannot
    happen organically — and that is fine for value learning: such states are
    strictly easier neighbours of real ones, and the win condition and step
    semantics are untouched.
    """

    def __init__(self, k_max=3, seed=0):
        self.k_max = k_max
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, env, mask):
        n, K = env.n, env.K
        # a random safe opening cell per instance; env.mine_keys was just
        # refreshed by the validated reset for masked instances
        first = torch.randint(0, K, (n,), generator=self.generator)
        mines = lay_mines(env.mine_keys, first, env.NEIGH, env.NVALID, env.M)

        # hide k safe cells per instance (never the opening cell), reveal the rest
        k = torch.randint(1, self.k_max + 1, (n, 1), generator=self.generator)
        noise = torch.rand(n, K, generator=self.generator)
        onehot_first = torch.nn.functional.one_hot(first, K).bool()
        safe = ~mines & ~onehot_first
        # rank safe cells by noise; hide the k best-ranked
        ranked = noise.masked_fill(~safe, -1.0)
        order = torch.argsort(ranked, dim=1, descending=True)
        position = torch.empty_like(order)
        position.scatter_(1, order, torch.arange(K).unsqueeze(0).expand(n, K))
        hidden = safe & (position < k)

        revealed = ~mines & ~hidden
        return {
            "mines": mines,
            "revealed": revealed,
            "flagged": torch.zeros_like(mines),
            "first_cell": first,
            "hidden_count": hidden.sum(dim=1),
        }


class CurriculumMinesweeper(MinesweeperBatched):
    """The validated env, with curriculum start states on reset.

    `source(env, mask) -> dict` provides boards; anything it returns is
    written only where `mask` is True, after the validated reset has run. Set
    `source=None` (or use the base class) for plain resets — the curriculum
    is a training utility and never appears in validation or telemetry runs.

    Invariant note: injected states satisfy I1–I13 **except I9** for the
    synthetic source (the parity-pinned layout rule guarantees I3/I4). I9
    defines a fresh `t == 0` episode as ungenerated, and a curriculum start is
    deliberately not fresh — that exception is the whole mechanism, so train
    with debug=False. Foreign sources (human replays) may additionally violate
    I4 if the site's opening rule differs. Validation always runs on the base
    class, where every invariant holds.
    """

    def __init__(self, n, source=None, **kwargs):
        self.source = source
        super().__init__(n, **kwargs)

    def _reset_instances(self, mask):
        super()._reset_instances(mask)
        if self.source is None:
            return
        board = self.source(self, mask)
        m1 = mask.unsqueeze(1)
        self.mines = torch.where(m1, board["mines"], self.mines)
        self.revealed = torch.where(m1, board["revealed"], self.revealed)
        self.flagged = torch.where(m1, board["flagged"], self.flagged)
        self.first_cell = torch.where(mask, board["first_cell"],
                                      self.first_cell)
        self.generated = self.generated | mask


class Pacer:
    """Move the start line backwards as the agent starts winning.

    Tracks a windowed win rate over finished curriculum episodes and raises
    `k_max` (more hidden cells, starts further from the goal) whenever it
    clears `threshold`. One-directional: sliding back down would let the
    curriculum thrash. The cap is the full board, at which point starts are
    ordinary openings and the curriculum has annealed itself away.
    """

    def __init__(self, source, threshold=0.5, window=256, grow=1.6, k_cap=381):
        self.source = source
        self.threshold = threshold
        self.window = window
        self.grow = grow
        self.k_cap = k_cap
        self.results = []

    def update(self, won_flags):
        self.results.extend(bool(w) for w in won_flags)
        if len(self.results) < self.window:
            return False
        rate = sum(self.results[-self.window:]) / self.window
        if rate >= self.threshold:
            new_k = min(self.k_cap, max(self.source.k_max + 1,
                                        int(self.source.k_max * self.grow)))
            if new_k != self.source.k_max:
                self.source.k_max = new_k
                self.results.clear()
                return True
        return False
